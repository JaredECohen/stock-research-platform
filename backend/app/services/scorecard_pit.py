"""Phase 6 — point-in-time foundation for the Fundamental Factor Scorecard.

Three jobs, all pure bookkeeping (no scoring, no LLM, no provider calls
beyond the one price series the app already caches):

1. `derive_available_at` — the first date a `financial_periods` row could
   have been known. This is what makes a historical score honest: a
   scorecard "as of 2024-03-31" may only use figures that were public by
   then, and the fiscal-year row for a December year end is not public on
   December 31st. The rule chain, in order, with the tag recorded in
   `available_at_source`:

     provider     the statement's own filing date when the provider passed
                  one through (`fmp_provider` adds `filing_date` /
                  `accepted_date` when FMP supplies `fillingDate` /
                  `acceptedDate`);
     filing_doc   a stored 10-K / 10-Q for the same ticker whose
                  `period_end` sits within ±FILING_DOC_MATCH_DAYS of the
                  row's `period_end` — its `filing_date`;
     lag_rule     `period_end` + `scorecard_pit_lag_annual_days` (75 —
                  conservative against the 60/75/90-day 10-K deadlines) or
                  + `scorecard_pit_lag_quarter_days` (45) for rows carrying
                  a `fiscal_quarter`;
     assumed_fye  the demo dataset writes `period="2024"` with no
                  `period_end` at all; the period end is assumed to be
                  `fiscal_year`-12-31 and the annual lag applied on top.
                  Tagged separately because two assumptions stack.

   Whatever the rule, the result is never later than the row's own
   `fetched_at` when one is known — by the time we fetched a figure it was
   public — and never earlier than `period_end` (a provider "filing date"
   before the period closed is garbage and falls through to the next rule).

2. `backfill_available_at` — idempotent fill of NULLs for rows that were
   ingested before the column existed. Restatements never move
   `available_at` (that is `history_service`'s job to honour on upsert); this
   only touches rows where it is NULL, so a second run writes nothing.

3. `sync_price_month_ends` / `snapshot_as_of` — the month-end price store
   fed from the EXISTING 252-day cached series (`data_service.
   get_price_history(days=252)`; no new provider-cache key, no new calls),
   and the point-in-time read that the feature engine builds a snapshot
   from: only rows with `available_at <= as_of`, with everything excluded
   counted so a thin snapshot is explainable rather than silently empty.

Residue that stays documented rather than fixed here: restated values are
served at their ORIGINAL availability date (mild lookahead on the value,
none on the timing); FMP prices are unadjusted for splits.
"""
from __future__ import annotations

import logging
from calendar import monthrange
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import SessionLocal
from ..models import FilingDoc, FinancialPeriod, PriceMonthEnd

log = logging.getLogger(__name__)

SOURCE_PROVIDER = "provider"
SOURCE_FILING_DOC = "filing_doc"
SOURCE_LAG_RULE = "lag_rule"
SOURCE_ASSUMED_FYE = "assumed_fye"
SOURCES = (SOURCE_PROVIDER, SOURCE_FILING_DOC, SOURCE_LAG_RULE, SOURCE_ASSUMED_FYE)


class PriceSeriesUnavailable(RuntimeError):
    """`data_service.get_price_history` returned None: every provider in the
    chain failed (or the ticker is unknown to all of them). Raised rather
    than folded into "zero months" so an outage never records as a clean
    sync — a month-end row that silently never lands drops the ticker from
    that month's evaluation leg with no trace. An EMPTY series is different
    and legitimate (a listing younger than one complete month)."""

# A statement row and the filing that carried it name the same period end,
# give or take the odd day a provider rounds a 52/53-week year to.
FILING_DOC_MATCH_DAYS = 7
# Only periodic reports date a set of financials. 8-Ks and amendments
# (10-K/A) are excluded: an amendment's date would move availability later
# than the figures were actually first public.
_PERIODIC_FILING_TYPES = ("10-K", "10-Q")
# Rows streamed per fetch in the backfill so a 170-ticker × 10-year table
# never sits in memory at once on the 512 MB worker.
_STREAM_BATCH = 500


def _utcnow() -> datetime:
    """Clock seam — tests monkeypatch this instead of freezing time."""
    return datetime.utcnow()


def _ensure_tables(db: Session) -> None:
    """Lazy-create the tables this module reads/writes, mirroring
    `history_service._ensure_tables` so scripts and tests that import this
    module directly work without `init_db()`."""
    bind = db.get_bind()
    FinancialPeriod.__table__.create(bind=bind, checkfirst=True)
    FilingDoc.__table__.create(bind=bind, checkfirst=True)
    PriceMonthEnd.__table__.create(bind=bind, checkfirst=True)


def _coerce_date(value: Any) -> date | None:
    """`date`, `datetime`, or an ISO-ish string (FMP's `acceptedDate` is
    `"2025-08-01 16:30:00"`) → `date`; anything else → None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# available_at derivation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilingDate:
    """The three fields of a stored filing the matcher needs."""
    filing_type: str
    filing_date: date
    period_end: date


def load_filing_dates(db: Session, ticker: str) -> list[FilingDate]:
    """Every dated periodic filing for `ticker`, loaded once per ticker so
    the per-row derivation never queries."""
    rows = db.execute(
        select(FilingDoc.filing_type, FilingDoc.filing_date, FilingDoc.period_end).where(
            FilingDoc.ticker == ticker.upper(),
            FilingDoc.filing_type.in_(_PERIODIC_FILING_TYPES),
            FilingDoc.filing_date.is_not(None),
            FilingDoc.period_end.is_not(None),
        )
    ).all()
    return [FilingDate(filing_type=t, filing_date=f, period_end=p) for t, f, p in rows]


def _match_filing_doc(filings: Iterable[FilingDate], period_end: date) -> date | None:
    """Earliest filing date among periodic filings whose `period_end` is
    within ±FILING_DOC_MATCH_DAYS of `period_end`. Earliest, because the
    figures were public from the first filing that carried them. The
    filing-type filter is applied here as well as in the loader so a
    caller-built list (tests, a future bulk path) obeys the same rule."""
    window = timedelta(days=FILING_DOC_MATCH_DAYS)
    candidates = [
        f.filing_date for f in filings
        if f.filing_type in _PERIODIC_FILING_TYPES
        and abs((f.period_end - period_end).days) <= window.days
        and f.filing_date >= period_end
    ]
    return min(candidates) if candidates else None


def _lag_days(fiscal_quarter: int | None) -> int:
    if fiscal_quarter is None:
        return int(settings.scorecard_pit_lag_annual_days)
    return int(settings.scorecard_pit_lag_quarter_days)


def derive_available_at(
    *,
    ticker: str,
    period_end: date | None,
    fiscal_year: int | None,
    fiscal_quarter: int | None,
    provider_date: Any = None,
    fetched_at: datetime | date | None = None,
    filings: Iterable[FilingDate] | None = None,
    db: Session | None = None,
) -> tuple[date | None, str | None]:
    """Return `(available_at, available_at_source)` for one statement row.

    `filings` is the ticker's periodic filings (see `load_filing_dates`);
    when it is None and `db` is given they are loaded here, and when
    neither is given the `filing_doc` rule is simply skipped. `provider_date`
    accepts whatever the provider row carried (string or date).

    Returns `(None, None)` when nothing can be derived — no period end and
    no fiscal year. That row is stored with NULL availability and is
    excluded (and counted) by `snapshot_as_of`; it is never guessed at.
    """
    period_end = _coerce_date(period_end)
    fetched = _coerce_date(fetched_at)

    def _bounded(d: date, source: str) -> tuple[date, str]:
        # By the time we fetched the figure it was public, so `fetched_at`
        # caps every rule. The source keeps naming the rule that produced
        # the (pre-cap) date so the audit trail stays readable.
        if fetched is not None and d > fetched:
            return fetched, source
        return d, source

    provider = _coerce_date(provider_date)
    if provider is not None and (period_end is None or provider >= period_end):
        return _bounded(provider, SOURCE_PROVIDER)

    if period_end is not None:
        if filings is None and db is not None:
            filings = load_filing_dates(db, ticker)
        if filings:
            matched = _match_filing_doc(filings, period_end)
            if matched is not None:
                return _bounded(matched, SOURCE_FILING_DOC)
        return _bounded(period_end + timedelta(days=_lag_days(fiscal_quarter)), SOURCE_LAG_RULE)

    if fiscal_year is not None:
        assumed_end = date(int(fiscal_year), 12, 31)
        return _bounded(
            assumed_end + timedelta(days=int(settings.scorecard_pit_lag_annual_days)),
            SOURCE_ASSUMED_FYE,
        )

    return None, None


def backfill_available_at(
    *, tickers: Iterable[str] | None = None, db: Session | None = None,
) -> dict[str, Any]:
    """Fill NULL `available_at` on existing `financial_periods` rows.

    Idempotent by construction — only NULL rows are candidates, so a second
    pass scans the still-unresolvable rows and writes nothing. Streams the
    candidate rows in `_STREAM_BATCH` chunks ordered by ticker so each
    ticker's filings are loaded once and dropped before the next.

    Returns `{scanned, filled, unresolved, tickers, by_source}`; `unresolved`
    rows (no period end, no fiscal year) stay NULL and are reported, not
    guessed.
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        stmt = (
            select(FinancialPeriod)
            .where(FinancialPeriod.available_at.is_(None), ~FinancialPeriod.ticker.startswith("~Q"))
            .order_by(FinancialPeriod.ticker, FinancialPeriod.id)
        )
        if tickers is not None:
            wanted = sorted({t.upper() for t in tickers})
            if not wanted:
                return {"scanned": 0, "filled": 0, "unresolved": 0, "tickers": 0, "by_source": {}}
            stmt = stmt.where(FinancialPeriod.ticker.in_(wanted))
        scanned = filled = unresolved = 0
        by_source: dict[str, int] = {}
        seen: set[str] = set()
        current_ticker: str | None = None
        filings: list[FilingDate] = []
        for row in db.execute(stmt.execution_options(yield_per=_STREAM_BATCH)).scalars():
            scanned += 1
            if row.ticker != current_ticker:
                current_ticker = row.ticker
                seen.add(row.ticker)
                filings = load_filing_dates(db, row.ticker)
            available_at, source = derive_available_at(
                ticker=row.ticker, period_end=row.period_end,
                fiscal_year=row.fiscal_year, fiscal_quarter=row.fiscal_quarter,
                fetched_at=row.fetched_at, filings=filings,
            )
            if available_at is None or source is None:
                unresolved += 1
                continue
            row.available_at = available_at
            row.available_at_source = source
            filled += 1
            by_source[source] = by_source.get(source, 0) + 1
        db.commit()
        result = {
            "scanned": scanned, "filled": filled, "unresolved": unresolved,
            "tickers": len(seen), "by_source": by_source,
        }
        if filled or unresolved:
            log.info("scorecard_pit backfill_available_at: %s", result)
        return result
    finally:
        if own:
            db.close()


# ---------------------------------------------------------------------------
# Month-end prices
# ---------------------------------------------------------------------------

def _month_end(d: date) -> date:
    return date(d.year, d.month, monthrange(d.year, d.month)[1])


def select_month_end_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pure: reduce a daily price series to one row per COMPLETE month.

    The last dated row of each calendar month is that month's close. A month
    counts as complete when a later month appears in the series or the row
    sits on the calendar month end — decided from the data alone, no clock,
    so a re-run over the same series is a no-op and a partial current month
    is never written as if it were finished (it fills in on the first sync
    after the month turns). Rows without a parseable date or a close are
    skipped.
    """
    dated: list[tuple[date, dict[str, Any]]] = []
    for r in rows or []:
        d = _coerce_date(r.get("date"))
        close = r.get("close")
        if d is None or close is None:
            continue
        try:
            close_f = float(close)
        except (TypeError, ValueError):
            continue
        adj = r.get("adjusted_close")
        try:
            adj_f = None if adj is None else float(adj)
        except (TypeError, ValueError):
            adj_f = None
        dated.append((d, {"close": close_f, "adjusted_close": adj_f}))
    dated.sort(key=lambda item: item[0])
    if not dated:
        return []
    last_in_month: dict[tuple[int, int], tuple[date, dict[str, Any]]] = {}
    for d, vals in dated:
        last_in_month[(d.year, d.month)] = (d, vals)
    keys = sorted(last_in_month)
    out: list[dict[str, Any]] = []
    for i, key in enumerate(keys):
        d, vals = last_in_month[key]
        complete = i < len(keys) - 1 or d == _month_end(d)
        if not complete:
            continue
        out.append({
            "month_end": _month_end(d), "price_date": d,
            "close": vals["close"], "adjusted_close": vals["adjusted_close"],
        })
    return out


def sync_price_month_ends(
    ticker: str, *, db: Session | None = None, days: int = 252,
) -> dict[str, int]:
    """Upsert `price_month_ends` for `ticker` from the cached price series.

    Reads `data_service.get_price_history(ticker, days=252)` — the SAME
    cache key every other price consumer uses, so this costs no extra
    provider call. Idempotent: `(ticker, month_end)` is unique and an
    unchanged close is not rewritten. Returns `{months, written, skipped}`
    where `skipped` is the count of incomplete trailing months (0 or 1).

    Raises `PriceSeriesUnavailable` when the provider chain returned no
    series at all (None). Callers that loop over a universe (the CLI, the
    worker's pit_prepare step) must count that as a failure for the ticker,
    not as a successful zero-month sync; the DB is not touched in that case.
    """
    from .data_service import get_data_service
    ticker = ticker.upper()
    ds = get_data_service()
    series = ds.get_price_history(ticker, days=days)
    if series is None:
        raise PriceSeriesUnavailable(
            f"no price series for {ticker}: provider chain returned nothing"
        )
    month_rows = select_month_end_rows(series)
    months_seen: set[tuple[int, int]] = set()
    for r in series:
        d = _coerce_date(r.get("date"))
        if d is not None:
            months_seen.add((d.year, d.month))
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        source = ds.mode()
        written = 0
        now = _utcnow()
        for m in month_rows:
            existing = db.execute(
                select(PriceMonthEnd).where(
                    PriceMonthEnd.ticker == ticker,
                    PriceMonthEnd.month_end == m["month_end"],
                )
            ).scalar_one_or_none()
            if existing is not None:
                if (
                    existing.close == m["close"]
                    and existing.adjusted_close == m["adjusted_close"]
                    and existing.price_date == m["price_date"]
                ):
                    continue
                existing.close = m["close"]
                existing.adjusted_close = m["adjusted_close"]
                existing.price_date = m["price_date"]
                existing.source = source
                existing.fetched_at = now
                written += 1
                continue
            db.add(PriceMonthEnd(
                ticker=ticker, month_end=m["month_end"], price_date=m["price_date"],
                close=m["close"], adjusted_close=m["adjusted_close"],
                source=source, fetched_at=now,
            ))
            written += 1
        db.commit()
        return {
            "months": len(month_rows), "written": written,
            "skipped": max(0, len(months_seen) - len(month_rows)),
        }
    finally:
        if own:
            db.close()


def latest_month_end_price(
    ticker: str, as_of: date, *, db: Session,
) -> dict[str, Any] | None:
    """Most recent stored month-end close with `month_end <= as_of`."""
    row = db.execute(
        select(PriceMonthEnd)
        .where(PriceMonthEnd.ticker == ticker.upper(), PriceMonthEnd.month_end <= as_of)
        .order_by(PriceMonthEnd.month_end.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        return None
    return {
        "month_end": row.month_end, "price_date": row.price_date,
        "close": row.close, "adjusted_close": row.adjusted_close, "source": row.source,
    }


# ---------------------------------------------------------------------------
# Point-in-time snapshot
# ---------------------------------------------------------------------------

def _period_sort_key(p: dict[str, Any]) -> tuple[date, str]:
    end = p.get("period_end")
    if end is None and p.get("fiscal_year") is not None:
        end = date(int(p["fiscal_year"]), 12, 31)
    return (end or date.min, str(p.get("period") or ""))


def snapshot_as_of(
    ticker: str, as_of: date, *, db: Session | None = None,
) -> dict[str, Any]:
    """Every `financial_periods` row for `ticker` knowable on `as_of`.

    Shape::

        {
          "ticker", "as_of",
          "periods": [  # newest first
            {"period", "period_end", "fiscal_year", "fiscal_quarter",
             "available_at", "available_at_source",
             "income": {line: value}, "balance": {...}, "cash": {...}}
          ],
          "rows": [ ...flat rows in the same order... ],
          "latest_period": str | None,
          "data_available_at": date | None,   # max available_at used
          "price": {...} | None,              # latest month-end close <= as_of
          "excluded": {"null_available_at": n, "after_as_of": n},
        }

    Rows whose `available_at` is NULL are excluded and counted separately
    from rows that were simply not yet public — the first is a data gap to
    backfill, the second is the point-in-time rule working as intended.
    """
    ticker = ticker.upper()
    as_of = _coerce_date(as_of) or as_of
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        rows = db.execute(
            select(FinancialPeriod).where(FinancialPeriod.ticker == ticker)
        ).scalars().all()
        excluded_null = excluded_future = 0
        by_period: dict[str, dict[str, Any]] = {}
        flat: list[dict[str, Any]] = []
        for r in rows:
            if r.available_at is None:
                excluded_null += 1
                continue
            if r.available_at > as_of:
                excluded_future += 1
                continue
            entry = by_period.get(r.period)
            if entry is None:
                entry = by_period[r.period] = {
                    "period": r.period, "period_end": r.period_end,
                    "fiscal_year": r.fiscal_year, "fiscal_quarter": r.fiscal_quarter,
                    "available_at": r.available_at,
                    "available_at_source": r.available_at_source,
                    "income": {}, "balance": {}, "cash": {},
                }
            else:
                # Lines of one period can carry different availability
                # dates (a restated line re-derived later); the period is
                # knowable only when its LAST line is.
                if r.available_at > entry["available_at"]:
                    entry["available_at"] = r.available_at
                    entry["available_at_source"] = r.available_at_source
            entry.setdefault(r.statement, {})[r.line_item] = r.value
            flat.append({
                "period": r.period, "period_end": r.period_end,
                "fiscal_year": r.fiscal_year, "fiscal_quarter": r.fiscal_quarter,
                "statement": r.statement, "line_item": r.line_item, "value": r.value,
                "available_at": r.available_at, "available_at_source": r.available_at_source,
            })
        periods = sorted(by_period.values(), key=_period_sort_key, reverse=True)
        flat.sort(key=lambda r: (_period_sort_key(r), r["statement"], r["line_item"]), reverse=True)
        data_available_at = max((p["available_at"] for p in periods), default=None)
        return {
            "ticker": ticker,
            "as_of": as_of,
            "periods": periods,
            "rows": flat,
            "latest_period": periods[0]["period"] if periods else None,
            "data_available_at": data_available_at,
            "price": latest_month_end_price(ticker, as_of, db=db),
            "excluded": {"null_available_at": excluded_null, "after_as_of": excluded_future},
        }
    finally:
        if own:
            db.close()
