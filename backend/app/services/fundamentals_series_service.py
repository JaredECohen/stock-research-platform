"""FEAT-001 — build the Fundamentals Explorer series (annual, v1).

`build_series(tickers, metrics, *, years, normalize)` turns stored
`FinancialPeriod` rows into the chart contract in `schemas/fundamentals`:
one series per (ticker, metric), every series aligned to one shared
fiscal-year axis, every missing value carrying a closed-set reason.

What this module deliberately does NOT do:
  - fetch statements from a provider. A ticker with no rows is reported
    as `not_backfilled`; the remedy is the existing research path (the
    memo job backfills history). Page reads never run provider sweeps.
  - call an LLM, import numpy or pandas, or write anything.
  - read quarterly data. v1 only ever ingests annual statements, and the
    live providers label those inconsistently (FMP `FY2024`, or `2024Q4`
    from its month-derived fallback; AlphaVantage `2024Q4` for income and
    `2024-12-31` for balance/cash — see `_fiscal_year_of`), so a row's
    label alone never decides. A fiscal year is derived from the label
    or the stored period end, and only a *real* quarterly signature —
    several distinct periods for one line item in one fiscal year, or a
    lone period ending months away from the fiscal year end — is
    excluded. Exclusions are counted in a WARNING and echoed in the
    response `warnings` so a mislabelled feed never renders as a silent
    gap or a false "not backfilled".

The only provider-touching path is the price series for market-derived
metrics, and that is the same `market_data_service.get_price_series`
read (`provider_cache`, 1-day TTL) that `/api/stocks/{t}/prices` makes.

Provenance (`source`, `fetched_at`, staleness) is aggregated from the raw
rows *before* any pivot — `comps_history._pivot_long_to_per_period`
discards it, and this response must be able to say where a number came
from and how old it is.
"""
from __future__ import annotations

import hashlib
import json
import logging
from calendar import monthrange
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..agents.log_safety import log_safely
from ..database import SessionLocal
from ..finance.comps_history import _closing_price_for
from ..models import Company, FinancialPeriod
from ..schemas.fundamentals import (
    AppliedLimits,
    MetricSeries,
    SeriesCoverage,
    SeriesLimits,
    SeriesPoint,
    SeriesProvenance,
    SeriesResponse,
    UnavailableTicker,
)
from . import fundamentals_catalog as catalog
from . import market_data_service

log = logging.getLogger(__name__)

NOT_BACKFILLED_REMEDY = (
    "No financial history is stored for this company yet. Run research on it — "
    "the memo job backfills its statement history."
)
# Rows exist but none of them is annual-shaped (see `_classify_rows`).
# Saying "nothing is stored" here would be false; the honest remedy is a
# refresh, and the response `warnings` carry the exclusion counts.
NOT_ANNUAL_REMEDY = (
    "Financial history is stored for this company but none of it is annual-shaped "
    "({excluded} row(s) excluded). Re-run research to refresh its statements."
)

# Month names as `Company.fiscal_year_end` stores them (live FMP profiles
# emit the full name; the demo dataset the same). Used only to *estimate*
# a period end when the provider row carried no date.
_MONTHS = {
    m.lower(): i for i, m in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"), start=1,
    )
}


def _utcnow() -> datetime:
    """Clock seam — tests monkeypatch this instead of freezing time."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Raw-row partitioning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _RawRow:
    """One `financial_periods` row as the SELECT returns it."""
    period: str
    period_end: date | None
    fiscal_year: int | None
    fiscal_quarter: int | None
    line: str
    value: float | None
    currency: str | None
    source: str | None
    fetched_at: datetime | None


@dataclass
class _TickerData:
    """Everything the builder needs for one ticker, straight from raw rows."""
    ticker: str
    # fiscal_year -> {line_item: value}
    by_year: dict[int, dict[str, float | None]] = field(default_factory=dict)
    # fiscal_year -> stored period_end (None when the provider gave none)
    period_end: dict[int, date | None] = field(default_factory=dict)
    sources: Counter[str] = field(default_factory=Counter)
    currencies: Counter[str] = field(default_factory=Counter)
    fetched_at: datetime | None = None
    excluded_quarterly: int = 0   # a real quarterly signature (see `_classify_rows`)
    excluded_ambiguous: int = 0   # no fiscal year derivable from label, date or column
    excluded_duplicate: int = 0   # a non-FY label beside an FY label for the same line/year
    # From `companies`: month number of the fiscal year end, current shares.
    fye_month: int | None = None
    shares_outstanding: float | None = None

    @property
    def has_rows(self) -> bool:
        return bool(self.by_year)

    @property
    def excluded(self) -> int:
        """Rows dropped for a reason worth warning about (duplicates are not)."""
        return self.excluded_quarterly + self.excluded_ambiguous

    @property
    def currency(self) -> str | None:
        return self.currencies.most_common(1)[0][0] if self.currencies else None

    @property
    def source(self) -> str:
        # Stable order (alphabetical) so the same rows always render the
        # same provenance string regardless of fetch order.
        return "+".join(sorted(self.sources)) if self.sources else ""

    def period_end_for(self, fy: int) -> tuple[date | None, bool]:
        """Stored period end, else the fiscal year end estimated from the
        company's FYE month (December when unknown). Second value: True
        when estimated."""
        stored = self.period_end.get(fy)
        if stored is not None:
            return stored, False
        month = self.fye_month or 12
        return date(fy, month, monthrange(fy, month)[1]), True


def _annual_label_year(period: str, fiscal_year: int | None, fiscal_quarter: int | None) -> int | None:
    """Fiscal year of a row whose *label* says annual, else None.

    Annual-labelled means: no `fiscal_quarter`, and a label shaped
    `FY2024` / `2024` that agrees with the `fiscal_year` column when one
    is set. This is the only label shape we trust without a date; every
    other shape goes through `_fiscal_year_of`, which needs the stored
    `period_end` to say which fiscal year the row belongs to.
    """
    if fiscal_quarter is not None:
        return None
    s = (period or "").strip().upper().replace("-", "").replace(" ", "")
    if s.startswith("FY"):
        s = s[2:]
    if not (s.isdigit() and len(s) == 4):
        return None
    year = int(s)
    if fiscal_year is not None and fiscal_year != year:
        return None
    return year


def _fiscal_year_of(
    period: str, period_end: date | None, fiscal_year: int | None, fiscal_quarter: int | None,
) -> tuple[int | None, bool]:
    """`(fiscal_year, explicit)` for one stored row; `(None, False)` when
    no fiscal year can be derived at all.

    `explicit` is True only for `FY2024` / `2024` labels. Everything else
    is what the live providers actually write for an *annual* statement:
    AlphaVantage labels annual income rows `2024Q4` (so `fiscal_quarter`
    is 4) and annual balance/cash rows `2024-12-31` (so the `fiscal_year`
    column holds 20241231); FMP's month-derived fallback label is
    `2024Q4` as well. v1 only ever ingests annual statements, so those
    rows ARE the annual data — the fiscal year is the year the period
    ended, exactly as FMP's own `FY<year>` labelling defines it. Whether
    such a row is *really* quarterly is decided from the row set, not
    the label (`_classify_rows`).
    """
    year = _annual_label_year(period, fiscal_year, fiscal_quarter)
    if year is not None:
        return year, True
    if period_end is not None:
        return period_end.year, False
    if fiscal_year is not None and 1900 <= fiscal_year <= 2999:
        return fiscal_year, False
    return None, False


def _month_distance(a: int, b: int) -> int:
    """Circular distance between two month numbers (Dec ↔ Jan is 1)."""
    d = abs(a - b) % 12
    return min(d, 12 - d)


def _classify_rows(td: _TickerData, rows: list[_RawRow]) -> list[tuple[int, _RawRow]]:
    """Decide which rows are annual and count the rest on `td`.

    Three signals, in order:
      1. no derivable fiscal year at all → `ambiguous`;
      2. per (line item, fiscal year): an explicit `FY` label wins and any
         non-explicit label beside it is a `duplicate` (two providers
         wrote the same year); with no explicit label, more than one
         distinct label in the year is a real quarterly signature and
         every one of them is `quarterly`;
      3. a lone non-explicit row whose period end sits more than a month
         (circular) from the company's fiscal-year-end month is a stray
         quarter, not a year → `quarterly`. The reference month is the
         profile's `fiscal_year_end`, else the modal month of the rows
         that survived step 2 (explicit-labelled ones first). The ±1
         tolerance is for 52/53-week years, whose end drifts across a
         month boundary; a real quarter is three months away.
    Explicit labels are never second-guessed by step 3 — an `FY2024`
    row is annual by construction.
    """
    dated: list[tuple[int, bool, _RawRow]] = []
    for r in rows:
        year, explicit = _fiscal_year_of(r.period, r.period_end, r.fiscal_year, r.fiscal_quarter)
        if year is None:
            td.excluded_ambiguous += 1
            continue
        dated.append((year, explicit, r))

    groups: dict[tuple[str, int], list[tuple[bool, _RawRow]]] = {}
    for year, explicit, r in dated:
        groups.setdefault((r.line, year), []).append((explicit, r))

    survivors: list[tuple[int, bool, _RawRow]] = []
    for (_line, year), members in groups.items():
        if any(explicit for explicit, _ in members):
            for explicit, r in members:
                if explicit:
                    survivors.append((year, True, r))
                else:
                    td.excluded_duplicate += 1
            continue
        labels = {r.period for _, r in members}
        if len(labels) > 1:
            td.excluded_quarterly += len(members)
            continue
        for _, r in members:
            survivors.append((year, False, r))

    ref_month = td.fye_month
    if ref_month is None:
        months = Counter(
            r.period_end.month for _, explicit, r in survivors
            if explicit and r.period_end is not None
        ) or Counter(r.period_end.month for _, _, r in survivors if r.period_end is not None)
        ref_month = months.most_common(1)[0][0] if months else None

    accepted: list[tuple[int, _RawRow]] = []
    for year, explicit, r in survivors:
        if (
            not explicit and ref_month is not None and r.period_end is not None
            and _month_distance(r.period_end.month, ref_month) > 1
        ):
            td.excluded_quarterly += 1
            continue
        accepted.append((year, r))
    return accepted


def _load_rows(
    db: Session, tickers: list[str], line_items: tuple[str, ...],
) -> dict[str, _TickerData]:
    """One SELECT for the whole ticker batch, partitioned per ticker."""
    data = {t: _TickerData(ticker=t) for t in tickers}

    # Profile first: the fiscal-year-end month is an input to row
    # classification, not just to period-end estimation.
    for cid, fye, shares in db.execute(
        select(Company.ticker, Company.fiscal_year_end, Company.shares_outstanding)
        .where(Company.ticker.in_(tickers))
    ):
        td = data.get(cid)
        if td is None:  # pragma: no cover
            continue
        td.fye_month = _MONTHS.get((fye or "").strip().lower())
        td.shares_outstanding = float(shares) if shares else None

    stmt = select(
        FinancialPeriod.ticker, FinancialPeriod.period, FinancialPeriod.period_end,
        FinancialPeriod.fiscal_year, FinancialPeriod.fiscal_quarter,
        FinancialPeriod.line_item, FinancialPeriod.value, FinancialPeriod.currency,
        FinancialPeriod.source, FinancialPeriod.fetched_at,
    ).where(
        FinancialPeriod.ticker.in_(tickers),
        FinancialPeriod.line_item.in_(line_items),
    )
    raw: dict[str, list[_RawRow]] = {t: [] for t in tickers}
    for (ticker, period, period_end, fy, fq, line, value, currency, source,
         fetched_at) in db.execute(stmt):
        if ticker not in raw:  # pragma: no cover — IN filter guarantees membership
            continue
        raw[ticker].append(_RawRow(
            period=period or "", period_end=period_end, fiscal_year=fy, fiscal_quarter=fq,
            line=line, value=None if value is None else float(value), currency=currency,
            source=source, fetched_at=fetched_at,
        ))

    for t, td in data.items():
        for year, r in _classify_rows(td, raw[t]):
            bucket = td.by_year.setdefault(year, {})
            # The unique index makes (period, statement, line) unique, but two
            # statements could in principle both carry a line; last write wins
            # only when the earlier value was null so real data is never lost.
            if r.line not in bucket or bucket[r.line] is None:
                bucket[r.line] = r.value
            if r.period_end is not None:
                prev = td.period_end.get(year)
                if prev is None or r.period_end > prev:
                    td.period_end[year] = r.period_end
            else:
                td.period_end.setdefault(year, None)
            if r.source:
                td.sources[r.source] += 1
            if r.currency:
                td.currencies[r.currency] += 1
            if r.fetched_at is not None and (td.fetched_at is None or r.fetched_at > td.fetched_at):
                td.fetched_at = r.fetched_at
        if td.excluded_duplicate:
            log.debug(
                "fundamentals series: %s skipped %d duplicate row(s) already covered by an FY label",
                td.ticker, td.excluded_duplicate,
            )
        if td.excluded:
            # WARNING, not DEBUG: v1 ingests annual rows only, so a real
            # quarterly signature or an undatable row is a mislabelled
            # feed (plan §2.5). `build_series` also surfaces the count in
            # the response so the user does not see a silent gap.
            log.warning(
                "fundamentals series: %s excluded %d non-annual row(s) from annual output "
                "(quarterly-labelled=%d, ambiguous=%d)",
                td.ticker, td.excluded, td.excluded_quarterly, td.excluded_ambiguous,
            )
    return data


def _exclusion_warning(td: _TickerData) -> str:
    return (
        f"{td.ticker}: {td.excluded} stored row(s) excluded from annual output "
        f"(quarterly-labelled={td.excluded_quarterly}, ambiguous={td.excluded_ambiguous}); "
        "some fiscal years may show missing_line"
    )


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

def _months_between(earlier: date, later: date) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def _provenance(td: _TickerData, now: datetime) -> SeriesProvenance:
    if not td.has_rows:
        return SeriesProvenance()
    last_fy = max(td.by_year)
    last_end, _ = td.period_end_for(last_fy)
    today = now.date()
    reasons: list[str] = []
    if _months_between(last_end, today) > catalog.STALE_LAST_PERIOD_MONTHS:
        reasons.append(
            f"last stored period FY{last_fy} ended {last_end.isoformat()}, more than "
            f"{catalog.STALE_LAST_PERIOD_MONTHS} months before {today.isoformat()} — a newer "
            "fiscal year has probably been filed but not ingested"
        )
    if td.fetched_at is not None and (now - td.fetched_at).days > catalog.STALE_FETCHED_AT_DAYS:
        reasons.append(
            f"rows last fetched {td.fetched_at.date().isoformat()}, more than "
            f"{catalog.STALE_FETCHED_AT_DAYS} days ago"
        )
    return SeriesProvenance(
        source=td.source, fetched_at=td.fetched_at,
        stale=bool(reasons), stale_reason="; ".join(reasons) or None,
    )


# ---------------------------------------------------------------------------
# Prices for market-derived metrics
# ---------------------------------------------------------------------------

def _price_rows(td: _TickerData, axis: list[int], now: datetime, warnings: list[str]) -> list[dict[str, Any]]:
    """The cached daily close series reaching back to the oldest period
    end on the axis. Any failure degrades to `no_price` points plus a
    warning — never a 500 for a chart."""
    if not td.has_rows or not axis:
        return []
    oldest, _ = td.period_end_for(min(axis))
    days = max((now.date() - oldest).days + 10, 30)
    try:
        return market_data_service.get_price_series(td.ticker, days) or []
    except Exception as exc:
        log_safely(log, f"fundamentals series: price history unavailable for {td.ticker}", exc)
        warnings.append(f"{td.ticker}: price history unavailable; market-derived metrics show no_price")
        return []


def _market_context(
    td: _TickerData, fy: int, prices: list[dict[str, Any]],
) -> catalog.MarketContext:
    period_end, _ = td.period_end_for(fy)
    price = _closing_price_for(prices, period_end.isoformat())
    shares = (td.by_year.get(fy) or {}).get(catalog.SHARES_LINE)
    if shares is not None:
        return catalog.MarketContext(price=price, shares=shares)
    if td.shares_outstanding:
        # Documented fallback: today's share count for a past period.
        return catalog.MarketContext(price=price, shares=td.shares_outstanding, shares_estimated=True)
    return catalog.MarketContext(price=price, shares=None)


# ---------------------------------------------------------------------------
# Series assembly
# ---------------------------------------------------------------------------

def _points_for(
    td: _TickerData, spec: catalog.MetricSpec, axis: list[int],
    prices: list[dict[str, Any]] | None,
) -> list[SeriesPoint]:
    points: list[SeriesPoint] = []
    for fy in axis:
        label = f"FY{fy}"
        if not td.has_rows:
            points.append(SeriesPoint(period=label, value=None, reason="not_backfilled"))
            continue
        cur = td.by_year.get(fy)
        stored_end = td.period_end.get(fy)
        if cur is None:
            # A year on the shared axis this ticker has no row for.
            points.append(SeriesPoint(period=label, period_end=None, value=None, reason="missing_line"))
            continue
        prior = td.by_year.get(fy - 1) if spec.requires_prior else None
        market: catalog.MarketContext | None = None
        estimated_date = False
        if spec.requires_price:
            market = _market_context(td, fy, prices or [])
            _, estimated_date = td.period_end_for(fy)
        out = catalog.compute(spec.id, cur, prior, market)
        points.append(SeriesPoint(
            period=label, period_end=stored_end, value=out.value, reason=out.reason,  # type: ignore[arg-type]
            # A market metric priced at an *estimated* period end is itself
            # an estimate, even when every line item was reported.
            estimated=out.estimated or (spec.requires_price and out.value is not None and estimated_date),
        ))
    return points


def _apply_indexed(points: list[SeriesPoint]) -> tuple[list[SeriesPoint], bool]:
    """Rebase to 100 at the first valued point. Eligible only when that
    base is positive — indexing off a loss or a zero is meaningless, so
    every point becomes `base_nonpositive` rather than a wrong line."""
    base: float | None = None
    for p in points:
        if p.value is not None:
            base = p.value
            break
    if base is None:
        return points, False
    if base <= 0:
        # Gap rows keep their own (more specific) reason.
        return [
            SeriesPoint(period=p.period, period_end=p.period_end, value=None,
                        reason=p.reason or "base_nonpositive", estimated=p.estimated)
            for p in points
        ], False
    return [
        SeriesPoint(period=p.period, period_end=p.period_end,
                    value=None if p.value is None else p.value / base * 100.0,
                    reason=p.reason, estimated=p.estimated)
        for p in points
    ], True


def _coverage(points: list[SeriesPoint]) -> SeriesCoverage:
    valued = [p.period for p in points if p.value is not None]
    return SeriesCoverage(
        first=valued[0] if valued else None, last=valued[-1] if valued else None,
        n=len(valued), expected=len(points),
    )


def _fingerprint(
    axis: list[int], normalize: str, series: list[MetricSeries], unavailable: list[UnavailableTicker],
) -> str:
    """sha256 over exactly what a chart displays. Provenance timestamps
    and warnings are excluded on purpose: a re-fetch that changes no
    value must not invalidate cached commentary."""
    payload = {
        "catalog_version": catalog.CATALOG_VERSION,
        "frequency": catalog.FREQUENCY,
        "normalize": normalize,
        "periods": axis,
        "series": [
            [s.ticker, s.metric, s.indexed,
             [[p.period, p.value, p.reason, p.estimated] for p in s.points]]
            for s in series
        ],
        "unavailable": [[u.ticker, u.reason] for u in unavailable],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def build_series(
    tickers: list[str], metrics: list[str], *,
    years: int | None = None, normalize: str = "none",
    capped_by_plan: bool = False, db: Session | None = None,
) -> SeriesResponse:
    """The series contract for `tickers × metrics` over the last `years`
    fiscal years (None = everything stored).

    Raises `ValueError` on an unknown metric id or an empty selection so
    the route can answer 422; a ticker with no rows is *not* an error
    (it is listed in `unavailable` with `not_backfilled` points).
    `normalize="indexed"` rebases currency/count series to 100 at their
    first valued period; other unit types are returned as-is.
    `capped_by_plan` is echoed for the route, which applies the plan cap
    to `years` before calling.
    """
    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    metrics = list(dict.fromkeys(m.strip().lower() for m in metrics if m and m.strip()))
    if not tickers or not metrics:
        raise ValueError("tickers and metrics must each have at least one entry")
    if normalize not in ("none", "indexed"):
        raise ValueError(f"unknown normalize mode {normalize!r}")
    if years is not None and years < 1:
        raise ValueError("years must be >= 1")
    specs = [catalog.get(m) for m in metrics]   # raises on an unknown id
    lines = catalog.required_line_items(metrics)

    now = _utcnow()
    warnings: list[str] = []
    own = db is None
    session = db or SessionLocal()
    try:
        data = _load_rows(session, tickers, lines)
    finally:
        if own:
            session.close()
    for t in tickers:
        if data[t].excluded:
            warnings.append(_exclusion_warning(data[t]))

    # Shared axis: every fiscal year from the oldest any selected ticker
    # reports to the newest, *contiguous* so a year nobody filed is a gap
    # in the line rather than two adjacent points, then trimmed to the
    # last `years`. Growth for the first year on the axis still sees the
    # year before it because `by_year` keeps the full history.
    reported = {fy for td in data.values() for fy in td.by_year}
    all_years = list(range(min(reported), max(reported) + 1)) if reported else []
    axis = all_years[-years:] if years is not None and all_years else all_years

    needs_price = any(s.requires_price for s in specs)
    prices: dict[str, list[dict[str, Any]]] = {}
    if needs_price:
        for t in tickers:
            prices[t] = _price_rows(data[t], axis, now, warnings)

    series: list[MetricSeries] = []
    unavailable: list[UnavailableTicker] = []
    for t in tickers:
        td = data[t]
        prov = _provenance(td, now)
        if not td.has_rows:
            remedy = (
                NOT_ANNUAL_REMEDY.format(excluded=td.excluded) if td.excluded
                else NOT_BACKFILLED_REMEDY
            )
            unavailable.append(UnavailableTicker(ticker=t, reason="not_backfilled", remedy=remedy))
        for spec in specs:
            points = _points_for(td, spec, axis, prices.get(t))
            indexed = False
            if normalize == "indexed" and spec.unit_type in ("currency", "count"):
                points, indexed = _apply_indexed(points)
            series.append(MetricSeries(
                ticker=t, metric=spec.id, unit_type=spec.unit_type, kind=spec.kind,  # type: ignore[arg-type]
                currency=td.currency if spec.unit_type == "currency" else None,
                indexed=indexed, points=points, coverage=_coverage(points), provenance=prov,
            ))

    return SeriesResponse(
        catalog_version=catalog.CATALOG_VERSION,
        as_of=now,
        normalize=normalize,  # type: ignore[arg-type]
        periods=[f"FY{fy}" for fy in axis],
        series=series,
        limits=SeriesLimits(
            applied=AppliedLimits(companies=len(tickers), metrics=len(metrics), years=years),
            capped_by_plan=capped_by_plan,
        ),
        unavailable=unavailable,
        warnings=warnings,
        fingerprint=_fingerprint(axis, normalize, series, unavailable),
    )
