"""Persistent daily bars with explicit provider and date coverage.

Provider responses can expire; a historical close must not. A series is
read from one provider at a time to avoid silently mixing adjustment bases.
Backfill requests do not generate memos or evaluate existing outcomes.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, select

from ..database import SessionLocal
from ..models import DailyPrice
from .ticker_symbols import market_data_symbols

log = logging.getLogger(__name__)
SOURCE_ORDER = ("fmp", "tiingo", "polygon", "alpha_vantage")
MINIMUM_HISTORY_YEARS = 2


def minimum_start(today: date | None = None) -> date:
    today = today or date.today()
    try:
        return today.replace(year=today.year - MINIMUM_HISTORY_YEARS)
    except ValueError:
        return today.replace(year=today.year - MINIMUM_HISTORY_YEARS, day=28)


def _number(value: Any, *, positive: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and (not positive or number > 0) else None
    except (TypeError, ValueError, OverflowError):
        return None


def persist_prices(
    ticker: str, rows: list[dict], *, source: str,
    provider_symbol: str | None = None, provenance: dict | None = None,
) -> dict[str, Any]:
    """Upsert bounded batches; invalid input never destroys a good close."""
    ticker = ticker.strip().upper()
    provenance = provenance or {}
    source = source.lower()
    fetched = datetime.utcnow()
    records: dict[date, dict] = {}
    rejected = []
    duplicates = []
    conflicting_dates = set()
    for position, row in enumerate(rows or []):
        if not isinstance(row, dict):
            rejected.append({"position": position, "date": "", "reason": "invalid_row"})
            continue
        raw_date = str(row.get("date") or "")
        close = _number(row.get("close"), positive=True)
        try:
            day = date.fromisoformat(raw_date)
        except ValueError:
            day = None
        if day is None or day > date.today() or close is None:
            rejected.append({"position": position, "date": raw_date, "reason": "invalid_date_or_close"})
            continue
        if day in conflicting_dates:
            rejected.append({"position": position, "date": raw_date, "reason": "conflicting_duplicate"})
            continue
        adjusted = _number(row.get("adjusted_close"), positive=True)
        if day in records:
            duplicates.append(day.isoformat())
            previous = records[day]
            if previous["close"] != close or (previous["adjusted_close"] is not None and adjusted is not None and previous["adjusted_close"] != adjusted):
                rejected.append({"position": position, "date": raw_date, "reason": "conflicting_duplicate"})
                records.pop(day)
                conflicting_dates.add(day)
                continue
        records[day] = {
            "ticker": ticker, "price_date": day, "source": source,
            "provider_symbol": provider_symbol or ticker,
            "open": _number(row.get("open"), positive=True),
            "high": _number(row.get("high"), positive=True),
            "low": _number(row.get("low"), positive=True),
            "close": close,
            "adjusted_close": adjusted,
            "volume": _number(row.get("volume")),
            "currency": row.get("currency") or provenance.get("currency"),
            "close_basis": provenance.get("close_basis", "provider_reported"),
            "adjusted_close_basis": provenance.get("adjusted_close_basis"),
            "fetched_at": fetched,
        }
    if records:
        with SessionLocal() as db:
            dialect = db.get_bind().dialect.name
            if dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert
            elif dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import insert
            else:
                raise RuntimeError("Unsupported daily price database dialect")
            values = list(records.values())
            for offset in range(0, len(values), 200):
                stmt = insert(DailyPrice).values(values[offset:offset + 200])
                nullable = ("open", "high", "low", "adjusted_close", "volume", "currency", "adjusted_close_basis")
                update = {key: func.coalesce(getattr(stmt.excluded, key), getattr(DailyPrice, key)) for key in nullable}
                update.update({key: getattr(stmt.excluded, key) for key in ("close", "provider_symbol", "close_basis", "fetched_at")})
                db.execute(stmt.on_conflict_do_update(
                    index_elements=["ticker", "price_date", "source", "close_basis"], set_=update,
                ))
            db.commit()
    if rejected or duplicates:
        log.warning("price input identities ticker=%s source=%s rejected=%d:%s duplicate_dates=%d:%s", ticker, source, len(rejected), rejected, len(duplicates), duplicates)
    return {"source": source, "rows_received": len(rows), "rows_upserted": len(records), "rejected": rejected, "duplicate_dates": duplicates}


def _last_weekday(today: date) -> date:
    # A conservative completed-session boundary, not an exchange calendar.
    day = today - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


class StoredPriceRows(list):
    """Ordinary bar payload with additive read-selection evidence."""

    def __init__(self, rows=(), *, selection: dict | None = None):
        super().__init__(rows)
        self.selection = selection or {}


def _flat_zero_volume():
    # Require explicit evidence on every field. Unknown volume or missing
    # OHLC does not prove a carry-forward placeholder.
    return and_(DailyPrice.volume == 0, DailyPrice.open == DailyPrice.close,
                DailyPrice.high == DailyPrice.close, DailyPrice.low == DailyPrice.close)


def _usable_price():
    # SQL NULL means unknown, which remains eligible. Only true is excluded.
    return _flat_zero_volume().is_not(True)


def _source_quality(db, ticker: str, *, start: date | None, end: date, days: int | None = None) -> list[dict]:
    where = [DailyPrice.ticker == ticker.upper(), DailyPrice.price_date <= end]
    if start:
        where.append(DailyPrice.price_date >= start)
    groups = {}
    query = select(DailyPrice.source, DailyPrice.close_basis, DailyPrice.price_date,
                   DailyPrice.fetched_at, _flat_zero_volume().label("placeholder")).where(*where)
    for source, basis, day, fetched, placeholder in db.execute(query):
        group = groups.setdefault((source, basis), {"source": source, "close_basis": basis,
            "dates": set(), "excluded_zero_volume_flat_dates": [], "last_fetched_at": fetched})
        group["last_fetched_at"] = max(group["last_fetched_at"], fetched)
        if placeholder:
            group["excluded_zero_volume_flat_dates"].append(day.isoformat())
        else:
            group["dates"].add(day)
    if not groups:
        return []
    # Calendar rows are evidence of sessions, never a second source of prices.
    observed = set(db.execute(select(DailyPrice.price_date).where(
        DailyPrice.ticker == "SPY", DailyPrice.price_date <= end, _usable_price(),
    ).distinct()).scalars())
    for group in groups.values():
        dates = group["dates"]
        oldest, newest = (min(dates), max(dates)) if dates else (None, None)
        # Validate only the bars this read would return. Old unrelated holes
        # cannot demote a complete current suffix; full coverage passes no days.
        selected_dates = set(sorted(dates)[-days:]) if days else dates
        selected_start = min(selected_dates) if selected_dates else None
        span_sessions = {d for d in observed if selected_start is not None and selected_start <= d <= newest}
        missing = sorted(d.isoformat() for d in span_sessions - dates)
        # Availability requires SPY to cover the candidate's own span with
        # reasonable density. A recently listed company need not predate itself.
        weekdays = sum((selected_start + timedelta(days=i)).weekday() < 5
                       for i in range((newest - selected_start).days + 1)) if dates else 0
        calendar_available = bool(span_sessions and min(observed) <= selected_start + timedelta(days=7)
            and max(observed) >= newest - timedelta(days=4)
            and len(span_sessions) >= max(1, int(weekdays * 0.9)))
        # Gaps in even a partial observed calendar are definite; its absence
        # never proves continuity. This is not an independent exchange calendar.
        complete = False if missing else (True if calendar_available else None)
        group.update(oldest=oldest, newest=newest, row_count=len(dates),
            continuity_start=selected_start, continuity_end=newest,
            internal_missing_benchmark_sessions=missing,
            internal_missing_benchmark_session_count=len(missing),
            selection_calendar_available=calendar_available,
            internal_continuity_verified=complete)
        group["excluded_zero_volume_flat_dates"].sort()
    return list(groups.values())


def _quality_evidence(group: dict) -> dict:
    return {key: (value.isoformat() if isinstance(value, (date, datetime)) else value)
            for key, value in group.items() if key != "dates"}


def price_coverage(ticker: str, start: date, end: date | None = None) -> dict:
    end = end or date.today()
    boundary = _last_weekday(end)
    with SessionLocal() as db:
        benchmark_dates = set(db.execute(select(DailyPrice.price_date).where(
            DailyPrice.ticker == "SPY", DailyPrice.price_date >= start,
            DailyPrice.price_date <= boundary, _usable_price(),
        ).distinct()).scalars())
        benchmark_span = db.execute(select(func.min(DailyPrice.price_date), func.max(DailyPrice.price_date)).where(
            DailyPrice.ticker == "SPY", DailyPrice.price_date <= end, _usable_price(),
        )).one()
        benchmark_available = bool(benchmark_span[0] and benchmark_span[0] <= start + timedelta(days=7)
                                   and benchmark_span[1] >= boundary - timedelta(days=4))
        sources = []
        for group in _source_quality(db, ticker, start=None, end=end):
            oldest, newest, fetched = group["oldest"], group["newest"], group["last_fetched_at"]
            have = {d for d in group["dates"] if d >= start}
            missing = []
            cursor = start
            while cursor <= boundary:
                if cursor.weekday() < 5 and cursor not in have:
                    missing.append(cursor.isoformat())
                cursor += timedelta(days=1)
            covers_start = oldest is not None and oldest <= start + timedelta(days=7)
            current = newest is not None and newest >= boundary - timedelta(days=4)
            missing_sessions = sorted(day.isoformat() for day in benchmark_dates - have)
            dense = len(have) >= max(1, int((len(have) + len(missing)) * 0.9))
            complete = covers_start and current and dense and not missing_sessions and (benchmark_available or ticker.upper() == "SPY")
            sources.append({**_quality_evidence(group),
                "rows_in_requested_range": len(have), "covers_start": covers_start,
                "fresh": datetime.utcnow() - timedelta(hours=24) <= fetched <= datetime.utcnow(),
                "current": current, "date_bounds_covered": covers_start and current,
                "coverage_complete": complete, "missing_benchmark_sessions": missing_sessions,
                "missing_benchmark_session_count": len(missing_sessions),
                "potential_missing_weekdays": missing,
                "potential_missing_weekday_count": len(missing),
            })
    return {"ticker": ticker.upper(), "requested_start": start.isoformat(), "requested_end": end.isoformat(),
            "sources": sources, "date_bounds_covered": any(s["date_bounds_covered"] for s in sources),
            "coverage_complete": any(s["coverage_complete"] for s in sources),
            "benchmark_calendar_available": benchmark_available,
            "calendar_note": "Coverage checks usable dates observed for SPY, plus bounds and density; this is not an independent exchange calendar. Potential missing weekdays include holidays. Explicit flat zero-volume OHLC bars are retained in storage but excluded from usable prices."}


def read_prices(ticker: str, *, days: int | None = None, start: date | None = None, end: date | None = None) -> list[dict]:
    """Return one provider/basis; verified continuity outranks newest/longest."""
    end = end or date.today()
    with SessionLocal() as db:
        candidates = _source_quality(db, ticker, start=start, end=end, days=days)
        eligible = [g for g in candidates if g["row_count"]]
        def rank(group):
            count, oldest, newest = group["row_count"], group["oldest"], group["newest"]
            enough = count >= days if days else (start is None or oldest <= start + timedelta(days=7))
            source = group["source"]
            priority = SOURCE_ORDER.index(source) if source in SOURCE_ORDER else len(SOURCE_ORDER)
            quality = {True: 2, None: 1, False: 0}[group["internal_continuity_verified"]]
            return (quality, enough, newest, count, -priority)
        selected = max(eligible, key=rank) if eligible else None
        selection = {"source": selected["source"] if selected else None,
            "close_basis": selected["close_basis"] if selected else None,
            "internal_continuity_verified": selected["internal_continuity_verified"] if selected else None,
            "calendar_note": "Within-series SPY-observed session continuity; unverified when calendar coverage is unavailable. Raw placeholder rows remain stored.",
            "candidate_sources": [_quality_evidence(g) for g in candidates]}
        if any(g["excluded_zero_volume_flat_dates"] or g["internal_missing_benchmark_sessions"] for g in candidates):
            log.warning("price read selection ticker=%s evidence=%s", ticker, selection)
        if selected is None:
            return StoredPriceRows(selection=selection)
        where = [DailyPrice.ticker == ticker.upper(), DailyPrice.price_date <= end, _usable_price()]
        if start:
            where.append(DailyPrice.price_date >= start)
        query = select(DailyPrice).where(*where, DailyPrice.source == selected["source"],
            DailyPrice.close_basis == selected["close_basis"]).order_by(DailyPrice.price_date.desc())
        if days:
            query = query.limit(days)
        prices = db.execute(query).scalars().all()
        return StoredPriceRows([{
            "date": row.price_date.isoformat(), "open": row.open, "high": row.high,
            "low": row.low, "close": row.close, "adjusted_close": row.adjusted_close,
            "volume": row.volume, "source": row.source, "close_basis": row.close_basis,
            "fetched_at": row.fetched_at.isoformat(),
        } for row in reversed(prices)], selection=selection)


def fetch_and_store_prices(ticker: str, days: int, *, service=None, verify_calendar: bool = True) -> dict:
    """Try configured sources until one spans the requested calendar window.

    Partial results remain useful durable data and are reported individually.
    No failed call can erase a previously stored date or provider series.
    """
    from .data_service import get_data_service
    ds = service or get_data_service()
    start = date.today() - timedelta(days=days)
    attempts = []
    for provider in ds._live_chain("prices"):
        for symbol in market_data_symbols(ticker):
            name = str(provider.name).lower()
            try:
                rows = provider.get_price_history(symbol, days) or []
                provenance = getattr(provider, "price_history_provenance", {})
                stored = persist_prices(ticker, rows, source=name, provider_symbol=symbol, provenance=provenance)
                attempt = {"provider": name, "symbol": symbol, **stored}
            except Exception as exc:
                attempt = {"provider": name, "symbol": symbol, "error_type": type(exc).__name__, "rows_upserted": 0}
            attempts.append(attempt)
            coverage = price_coverage(ticker, start)
            basis = getattr(provider, "price_history_provenance", {}).get("close_basis", "provider_reported")
            own_sources = [s for s in coverage["sources"] if s["source"] == name and s["close_basis"] == basis]
            if not verify_calendar:
                with SessionLocal() as db:
                    suffix_quality = {(g["source"], g["close_basis"]): g["internal_continuity_verified"]
                        for g in _source_quality(db, ticker, start=None, end=date.today(), days=days)}
                own_sources = [{**s, "internal_continuity_verified": suffix_quality.get((name, basis))}
                               for s in own_sources]
            accepted = attempt.get("rows_upserted", 0) > 0 and any(
                s["fresh"] and (s["coverage_complete"] if verify_calendar else s["row_count"] >= days and s["current"] and s["internal_continuity_verified"] is True)
                for s in own_sources
            )
            if accepted:
                return {"attempts": attempts, "coverage": coverage, "refresh_complete": True}
    result = {"attempts": attempts, "coverage": price_coverage(ticker, start), "refresh_complete": False}
    log.warning("price history incomplete ticker=%s requested_start=%s attempts=%s", ticker, start, attempts)
    return result


def backfill_prices(ticker: str, start: date, *, force_refresh: bool = False) -> dict:
    before = price_coverage(ticker, start)
    if any(s["coverage_complete"] and s["fresh"] for s in before["sources"]) and not force_refresh:
        return {"ticker": ticker.upper(), "status": "stored", "attempts": [], "coverage": before, "success": True}
    span = max(1, (date.today() - start).days)
    result = fetch_and_store_prices(ticker, span)
    return {"ticker": ticker.upper(), "status": "fetched", **result, "success": result["refresh_complete"]}
