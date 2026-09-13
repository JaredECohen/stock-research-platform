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

from sqlalchemy import func, select

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


def price_coverage(ticker: str, start: date, end: date | None = None) -> dict:
    end = end or date.today()
    with SessionLocal() as db:
        benchmark_dates = set(db.execute(select(DailyPrice.price_date).where(
            DailyPrice.ticker == "SPY", DailyPrice.price_date >= start,
            DailyPrice.price_date <= _last_weekday(end),
        ).distinct()).scalars())
        benchmark_span = db.execute(select(func.min(DailyPrice.price_date), func.max(DailyPrice.price_date)).where(
            DailyPrice.ticker == "SPY", DailyPrice.price_date <= end,
        )).one()
        benchmark_available = bool(benchmark_span[0] and benchmark_span[0] <= start + timedelta(days=7)
                                   and benchmark_span[1] >= _last_weekday(end) - timedelta(days=4))
        rows = db.execute(select(
            DailyPrice.source, DailyPrice.close_basis, func.min(DailyPrice.price_date),
            func.max(DailyPrice.price_date), func.count(DailyPrice.id),
            func.max(DailyPrice.fetched_at),
        ).where(DailyPrice.ticker == ticker.upper(), DailyPrice.price_date <= end).group_by(DailyPrice.source, DailyPrice.close_basis)).all()
        sources = []
        for source, basis, oldest, newest, count, fetched in rows:
            dates = db.execute(select(DailyPrice.price_date).where(
                DailyPrice.ticker == ticker.upper(), DailyPrice.source == source,
                DailyPrice.close_basis == basis,
                DailyPrice.price_date >= start, DailyPrice.price_date <= end,
            ).order_by(DailyPrice.price_date)).scalars().all()
            # Weekdays are reported as potential gaps because exchange
            # holidays/listing suspensions cannot be invented as prices.
            have = set(dates)
            missing = []
            cursor = start
            boundary = _last_weekday(end)
            while cursor <= boundary:
                if cursor.weekday() < 5 and cursor not in have:
                    missing.append(cursor.isoformat())
                cursor += timedelta(days=1)
            covers_start = oldest <= start + timedelta(days=7)
            current = newest >= boundary - timedelta(days=4)
            missing_sessions = sorted(day.isoformat() for day in benchmark_dates - have)
            dense = len(dates) >= max(1, int((len(dates) + len(missing)) * 0.9))
            complete = covers_start and current and dense and not missing_sessions and (benchmark_available or ticker.upper() == "SPY")
            sources.append({
                "source": source, "close_basis": basis, "oldest": oldest.isoformat(), "newest": newest.isoformat(),
                "row_count": count, "rows_in_requested_range": len(dates),
                "last_fetched_at": fetched.isoformat(), "covers_start": covers_start,
                "fresh": fetched >= datetime.utcnow() - timedelta(hours=24),
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
            "calendar_note": "Coverage checks dates observed for SPY, plus bounds and density; this is not an independent exchange calendar. Potential missing weekdays include holidays."}


def read_prices(ticker: str, *, days: int | None = None, start: date | None = None, end: date | None = None) -> list[dict]:
    """Return a single provider's series, oldest first, without an API call."""
    end = end or date.today()
    with SessionLocal() as db:
        where = [DailyPrice.ticker == ticker.upper(), DailyPrice.price_date <= end]
        if start:
            where.append(DailyPrice.price_date >= start)
        sources = db.execute(select(DailyPrice.source, DailyPrice.close_basis, func.count(DailyPrice.id), func.min(DailyPrice.price_date), func.max(DailyPrice.price_date)).where(*where).group_by(DailyPrice.source, DailyPrice.close_basis)).all()
        if not sources:
            return []
        def rank(item):
            source, basis, count, oldest, newest = item
            enough = count >= days if days else (start is None or oldest <= start + timedelta(days=7))
            priority = SOURCE_ORDER.index(source) if source in SOURCE_ORDER else len(SOURCE_ORDER)
            return (enough, newest, count, -priority)
        source, basis = max(sources, key=rank)[:2]
        query = select(DailyPrice).where(*where, DailyPrice.source == source, DailyPrice.close_basis == basis).order_by(DailyPrice.price_date.desc())
        if days:
            query = query.limit(days)
        prices = db.execute(query).scalars().all()
        return [{
            "date": row.price_date.isoformat(), "open": row.open, "high": row.high,
            "low": row.low, "close": row.close, "adjusted_close": row.adjusted_close,
            "volume": row.volume, "source": row.source, "close_basis": row.close_basis,
            "fetched_at": row.fetched_at.isoformat(),
        } for row in reversed(prices)]


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
            accepted = attempt.get("rows_upserted", 0) > 0 and any(
                s["fresh"] and (s["coverage_complete"] if verify_calendar else s["row_count"] >= days and s["current"])
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
