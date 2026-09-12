"""FEAT-003 — weekly analytics for one GICS industry group.

``compute_group_stats(code, as_of=...)`` turns the group's constituents
(from the classification table), their price series, screener metrics and
market caps into one ``industry_stats`` row: returns per horizon under
equal and market-cap weighting, breadth, dispersion, valuation medians,
fundamental medians, benchmark-relative returns, leaders/laggards, and —
just as important — who was left out and why.

Rules this module encodes, in order of how often they have bitten:

* **Missing evidence is never a zero.** A ticker with no usable price
  series is listed under ``sample.excluded`` with a reason and drops out
  of every statistic's ``n``; a horizon the price window cannot reach is
  ``{"value": null, "reason": "history_window"}``; a group below the
  sample floor reports ``status = insufficient_sample`` — a labelled
  state, not an error and not a row of zeros — and ``sample.sample_floor``
  says WHICH short state it is: ``prices_not_warmed`` (transient; the
  weekly warm-up clears it) or ``universe_too_small`` (structural; this
  universe holds fewer constituents than the floor and never will cover
  the group without being widened). See ``classify_sample_floor``.
* **Batched reads only.** One SELECT for the group map, one for the
  companies, one for the metrics, one bulk read of cached price rows
  (chunked ``IN`` lists); provider fetches for tickers without a cached
  series are bounded by ``max_fetch`` (default: the weekly warm-up
  budget) and every skipped ticker is reported. Nothing here loops a
  provider call per constituent without a budget.
* **Determinism.** ``inputs_hash`` covers every input the payload depends
  on; recomputing with identical inputs returns the existing row and an
  identical payload (tested). Nothing time-dependent goes into the
  payload — ``compute_ms`` and ``computed_at`` are columns. The benchmark
  cohort is fixed when the period's inputs are loaded, so a group's row
  does not depend on which other groups were computed first (tested with
  two groups sharing one context, in both drain orders).
* **The method is stated in the row.** ``method`` records weighting,
  benchmark definitions with their own sample sizes, the sample floor,
  the missing-data policy, the price window and the as-of, so a report
  written from the row can say exactly what a number means.

Benchmarks (owner decision 2): the universe equal-weight and the sector
equal-weight cohorts computed from the SAME cached price rows, and the
Ken-French daily market factor ``KFR.MKT_RF.D`` (an excess return over
the risk-free rate — labelled as such, never presented as a total
return). No ETF or index-vendor series. A missing benchmark is a
``degraded`` entry, not a silent gap.

The two equal-weight cohorts pass the same eligibility test the group's
own constituents pass — active company, a series present at context load,
not stale at the as-of (``AnalyticsContext.cohort_members``). Comparing a
group against a cohort that still contained the delisted and stale rows
the group had just excluded made ``benchmark_relative`` (and the regime
label the PM reads off it) wrong in the group's favour; every exclusion
is now counted by reason in ``method.benchmarks`` and ``sample``.

Horizons beyond the 252-day price window (``1y`` on a 252-trading-day
cache, ``ytd`` late in the year) return ``history_window`` — a deliberate
v1 cut recorded in the setup doc. Per-ticker weekly closes are persisted
in ``per_ticker`` so a later period can still place a ticker whose cache
row has since expired (source ``stored_weekly_closes``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from statistics import mean, median, pstdev
from typing import Any

from sqlalchemy import select

from ..config import settings
from ..database import SessionLocal
from ..models import Company, IndustryStatSnapshot, ProviderCache, ScreenerMetric
from . import gics_registry, industry_classification
from .gics_registry import NodeInfo, VersionInfo

log = logging.getLogger(__name__)

HORIZONS: tuple[str, ...] = ("1w", "1m", "qtd", "ytd", "1y")
PRICE_WINDOW_DAYS = 252
METHOD_VERSION = "industry-stats-v1"
MARKET_FACTOR_ID = "KFR.MKT_RF.D"
BENCHMARK_UNIVERSE = "universe_ew"
BENCHMARK_SECTOR = "sector_ew"
# Multiples are meaningless when the denominator is negative; they are
# excluded from the median and counted so the reader can see how many.
VALUATION_METRICS: tuple[str, ...] = ("ev_ebitda", "pe_ttm", "ev_revenue")
FUNDAMENTAL_METRICS: tuple[str, ...] = ("revenue_growth_yoy", "op_margin", "fcf_margin")
LEADER_COUNT = 5
MAX_WEEKLY_CLOSES = 60
# A series whose last close is older than this (relative to the as-of) is
# not evidence about the as-of week; the ticker is excluded, not stretched.
STALE_PRICE_DAYS = 10
# `above_50d_mean` is the share of constituents trading above their 50-day
# moving average — 50 TRADING SESSIONS, which is what the name claims and
# what a reader will check. Counting 50 calendar days instead takes about
# 35 bars and silently answers a different question (it once flipped the
# flag to True for a cohort where every member was below its real 50-day
# mean), so the window is taken as the last 50 closes in the series.
MEAN_WINDOW_SESSIONS = 50
# …but only when those 50 closes are a daily series. The stored weekly
# closes a later period reuses would otherwise make "50 sessions" reach
# back a year. 50 trading days span ~70 calendar days; anything wider than
# this is not daily data and the flag is null with a reason.
MEAN_WINDOW_MAX_SPAN_DAYS = 100
REASON_MEAN_WINDOW_SHORT = "window_too_short"
REASON_MEAN_WINDOW_SPARSE = "series_not_daily"
# Size of each `IN (...)` chunk — under SQLite's 999-variable ceiling with
# room for the other predicates.
_CHUNK = 200

PriceSeries = list[tuple[date, float]]

REASON_NO_PRICES = "no_prices"
REASON_FETCH_BUDGET = "fetch_budget_exhausted"
REASON_STALE_PRICES = "stale_prices"
REASON_NO_PRICE_AT_AS_OF = "no_price_at_as_of"
REASON_HISTORY_WINDOW = "history_window"
REASON_INSUFFICIENT = "insufficient_sample"
REASON_INACTIVE = "inactive"

# The three sample-floor states. Below the floor is not one state but
# two, and they are different statements to a reader — see
# ``classify_sample_floor``.
FLOOR_MET = "met"
FLOOR_UNIVERSE_TOO_SMALL = "universe_too_small"
FLOOR_PRICES_NOT_WARMED = "prices_not_warmed"


def _utcnow() -> datetime:
    """Clock seam — tests monkeypatch this instead of freezing time."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Period / horizon arithmetic
# ---------------------------------------------------------------------------


def period_key_for(as_of: date | datetime) -> str:
    """ISO week of the as-of date, ``2026-W36``."""
    d = as_of.date() if isinstance(as_of, datetime) else as_of
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def horizon_start(cutoff: date, horizon: str) -> date:
    """The last date whose close anchors the horizon return ending at
    ``cutoff`` (calendar arithmetic; the anchor is the last close on or
    before this date)."""
    if horizon == "1w":
        return cutoff - timedelta(days=7)
    if horizon == "1m":
        return cutoff - timedelta(days=30)
    if horizon == "qtd":
        quarter_first_month = ((cutoff.month - 1) // 3) * 3 + 1
        return date(cutoff.year, quarter_first_month, 1) - timedelta(days=1)
    if horizon == "ytd":
        return date(cutoff.year - 1, 12, 31)
    if horizon == "1y":
        return cutoff - timedelta(days=365)
    raise ValueError(f"unknown horizon {horizon!r}")


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def close_series(rows: Iterable[dict[str, Any]] | None) -> PriceSeries:
    """``[(date, close)]`` ascending from provider rows. ``adjusted_close``
    is preferred over ``close`` where a provider supplies both; rows
    without a parseable date or a positive close are dropped."""
    out: dict[date, float] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        d = _parse_date(row.get("date"))
        if d is None:
            continue
        raw = row.get("adjusted_close")
        if raw is None:
            raw = row.get("close")
        try:
            px = float(raw)
        except (TypeError, ValueError):
            continue
        if px > 0:
            out[d] = px
    return sorted(out.items())


def _last_close_on_or_before(series: PriceSeries, day: date) -> tuple[date, float] | None:
    found: tuple[date, float] | None = None
    for d, px in series:
        if d > day:
            break
        found = (d, px)
    return found


def horizon_returns(series: PriceSeries, cutoff: date) -> dict[str, dict[str, Any]]:
    """Per-horizon simple returns ending at the last close on or before
    ``cutoff``. A horizon the series cannot anchor is ``value: None`` with
    ``reason: history_window``; a series with no close at the cutoff
    yields ``no_price_at_as_of`` for every horizon."""
    end = _last_close_on_or_before(series, cutoff)
    out: dict[str, dict[str, Any]] = {}
    for h in HORIZONS:
        if end is None:
            out[h] = {"value": None, "reason": REASON_NO_PRICE_AT_AS_OF}
            continue
        start_day = horizon_start(cutoff, h)
        start = _last_close_on_or_before(series, start_day)
        if start is None or start[0] == end[0]:
            out[h] = {"value": None, "reason": REASON_HISTORY_WINDOW, "end_date": end[0].isoformat()}
            continue
        out[h] = {
            "value": end[1] / start[1] - 1.0,
            "start_date": start[0].isoformat(),
            "end_date": end[0].isoformat(),
        }
    return out


def weekly_closes(
    series: PriceSeries, *, limit: int = MAX_WEEKLY_CLOSES, cutoff: date | None = None,
) -> list[list[Any]]:
    """The last close of each ISO week, newest ``limit`` weeks — what a
    later period reuses when the daily cache row has expired.

    Clipped at ``cutoff`` (the row's as-of). Without it a row stamped with
    a Friday could carry closes dated after that Friday, and a later period
    reusing them would be reading the future into a point-in-time row.
    """
    by_week: dict[tuple[int, int], tuple[date, float]] = {}
    for d, px in series:
        if cutoff is not None and d > cutoff:
            continue
        iso = d.isocalendar()
        by_week[(iso[0], iso[1])] = (d, px)
    rows = [[d.isoformat(), px] for d, px in sorted(by_week.values())]
    return rows[-limit:]


def above_mean_window(series: PriceSeries, cutoff: date) -> tuple[bool | None, str | None]:
    """Is the last close on or before ``cutoff`` above the mean of the last
    ``MEAN_WINDOW_SESSIONS`` closes? ``(None, reason)`` when the series
    cannot support the claim — fewer than 50 closes, or closes too sparse
    to be a daily series (the reason travels into ``per_ticker`` so the
    breadth row can say why a constituent is absent rather than counting
    it as "not above")."""
    bars = [(d, px) for d, px in series if d <= cutoff]
    if len(bars) < MEAN_WINDOW_SESSIONS:
        return None, REASON_MEAN_WINDOW_SHORT
    window = bars[-MEAN_WINDOW_SESSIONS:]
    if (window[-1][0] - window[0][0]).days > MEAN_WINDOW_MAX_SPAN_DAYS:
        return None, REASON_MEAN_WINDOW_SPARSE
    return window[-1][1] > mean(px for _, px in window), None


def _series_from_weekly(rows: Iterable[Any]) -> PriceSeries:
    out: dict[date, float] = {}
    for row in rows or []:
        try:
            d = _parse_date(row[0])
            px = float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if d is not None and px > 0:
            out[d] = px
    return sorted(out.items())


# ---------------------------------------------------------------------------
# Loaders — every read is batched; tests inject their own
# ---------------------------------------------------------------------------


def _chunks(items: list[str]) -> Iterable[list[str]]:
    for i in range(0, len(items), _CHUNK):
        yield items[i:i + _CHUNK]


def _db_constituents_by_group(version: VersionInfo) -> dict[str, list[str]]:
    return industry_classification.constituents_by_group(version=version)


def _db_companies(tickers: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not tickers:
        return out
    with SessionLocal() as db:
        for chunk in _chunks(sorted(set(tickers))):
            rows = db.execute(
                select(
                    Company.ticker, Company.company_name, Company.sector, Company.market_cap,
                    Company.shares_outstanding, Company.last_price, Company.is_active,
                ).where(Company.ticker.in_(chunk))
            ).all()
            for ticker, name, sector, cap, shares, last_price, active in rows:
                out[str(ticker).upper()] = {
                    "company_name": name, "sector": sector, "market_cap": cap,
                    "shares_outstanding": shares, "last_price": last_price,
                    "is_active": bool(active) if active is not None else True,
                }
    return out


def _db_metrics(tickers: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not tickers:
        return out
    with SessionLocal() as db:
        for chunk in _chunks(sorted(set(tickers))):
            for row in db.execute(
                select(ScreenerMetric).where(ScreenerMetric.ticker.in_(chunk))
            ).scalars().all():
                out[row.ticker.upper()] = {
                    "pe_ttm": row.pe_ttm, "ev_ebitda": row.ev_ebitda, "ev_revenue": row.ev_revenue,
                    "gross_margin": row.gross_margin, "op_margin": row.op_margin,
                    "fcf_margin": row.fcf_margin, "revenue_growth_yoy": row.revenue_growth_yoy,
                    "market_cap": row.market_cap, "beta": row.beta,
                    "last_updated": row.last_updated.isoformat() if row.last_updated else None,
                }
    return out


def _price_key(ticker: str) -> str:
    # Mirrors `data_service.get_price_history`'s cache key (`<T>:<days>`).
    return f"{ticker.upper()}:{PRICE_WINDOW_DAYS}"


def _db_cached_prices(tickers: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Raw cached price rows for the tickers that have one — one bulk read
    per chunk, no provider call, no TTL check (a stale weekly row is still
    the best evidence about that week; the as-of clip decides usability)."""
    out: dict[str, list[dict[str, Any]]] = {}
    if not tickers:
        return out
    keys = {_price_key(t): t.upper() for t in tickers}
    with SessionLocal() as db:
        for chunk in _chunks(sorted(keys)):
            rows = db.execute(
                select(ProviderCache.key, ProviderCache.payload_json).where(
                    ProviderCache.capability == "prices", ProviderCache.key.in_(chunk),
                )
            ).all()
            for key, payload in rows:
                if isinstance(payload, list) and payload:
                    out[keys[key]] = payload
    return out


def _db_cached_price_tickers(tickers: list[str]) -> set[str]:
    """Which tickers hold a cached series — the warm-up needs presence,
    not payloads, so this reads keys only."""
    have: set[str] = set()
    if not tickers:
        return have
    keys = {_price_key(t): t.upper() for t in tickers}
    with SessionLocal() as db:
        for chunk in _chunks(sorted(keys)):
            for (key,) in db.execute(
                select(ProviderCache.key).where(
                    ProviderCache.capability == "prices", ProviderCache.key.in_(chunk),
                    ProviderCache.payload_json.is_not(None),
                )
            ).all():
                have.add(keys[key])
    return have


def _db_fetch_prices(ticker: str) -> list[dict[str, Any]] | None:
    from .data_service import get_data_service
    return get_data_service().get_price_history(ticker, PRICE_WINDOW_DAYS)


def _db_market_factor() -> list[dict[str, Any]] | None:
    from .data_service import get_data_service
    payload = get_data_service().get_macro_series(MARKET_FACTOR_ID)
    if not payload or not isinstance(payload, dict):
        return None
    points = payload.get("points")
    return list(points) if points else None


def _db_prior_stats(code: str, version: VersionInfo, before_period_key: str) -> dict[str, Any] | None:
    with SessionLocal() as db:
        row = db.execute(
            select(IndustryStatSnapshot).where(
                IndustryStatSnapshot.taxonomy_version_id == version.id,
                IndustryStatSnapshot.industry_group_code == code,
                IndustryStatSnapshot.period_key < before_period_key,
            ).order_by(IndustryStatSnapshot.as_of.desc(), IndustryStatSnapshot.id.desc())
        ).scalars().first()
        return stats_dict(row) if row is not None else None


@dataclass
class Loaders:
    """Every read the analytics performs, as replaceable callables. The
    defaults are the batched DB/provider reads above; tests hand in
    in-memory fixtures and spies."""

    constituents_by_group: Callable[[VersionInfo], dict[str, list[str]]] = _db_constituents_by_group
    companies: Callable[[list[str]], dict[str, dict[str, Any]]] = _db_companies
    metrics: Callable[[list[str]], dict[str, dict[str, Any]]] = _db_metrics
    cached_prices: Callable[[list[str]], dict[str, list[dict[str, Any]]]] = _db_cached_prices
    cached_price_tickers: Callable[[list[str]], set[str]] = _db_cached_price_tickers
    fetch_prices: Callable[[str], list[dict[str, Any]] | None] = _db_fetch_prices
    market_factor: Callable[[], list[dict[str, Any]] | None] = _db_market_factor
    prior_stats: Callable[[str, VersionInfo, str], dict[str, Any] | None] = _db_prior_stats


def default_loaders() -> Loaders:
    return Loaders()


# ---------------------------------------------------------------------------
# Context — inputs shared across the groups of one period
# ---------------------------------------------------------------------------

_UNSET: Any = object()


@dataclass
class AnalyticsContext:
    """Universe-wide inputs for one as-of, loaded once: the group map, the
    cached price series (parsed to ``(date, close)`` immediately so the
    raw provider rows are released), the fetch budget and the market
    factor. Per-group work reads from here; the benchmark cohorts are
    computed over these same series."""

    version: VersionInfo
    as_of: datetime
    cutoff: date
    groups: dict[str, list[str]]
    loaders: Loaders
    max_fetch: int
    prices: dict[str, PriceSeries] = field(default_factory=dict)
    price_source: dict[str, str] = field(default_factory=dict)
    fetches: int = 0
    fetch_failed: set[str] = field(default_factory=set)
    fetch_skipped: set[str] = field(default_factory=set)
    # The tickers that had a series when the period's inputs were loaded,
    # BEFORE any group was computed. The benchmark cohorts are drawn from
    # this set alone (see `cohort_members`), never from `prices`, which
    # grows as groups are drained.
    baseline: set[str] = field(default_factory=set)
    _factor: Any = _UNSET
    _horizons: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    _companies: Any = _UNSET
    _cohort: Any = _UNSET

    @property
    def universe(self) -> list[str]:
        return sorted({t for tickers in self.groups.values() for t in tickers})

    def companies_map(self) -> dict[str, dict[str, Any]]:
        """The company rows for the whole universe, read once per context.
        Per-group work slices this rather than issuing its own SELECT, and
        the benchmark cohort needs ``is_active`` for tickers outside the
        group being computed."""
        if self._companies is _UNSET:
            universe = self.universe
            try:
                self._companies = self.loaders.companies(universe) if universe else {}
            except Exception as exc:  # a missing company row is missing evidence, not a crash
                log.debug("company read failed for the universe: %s", type(exc).__name__)
                self._companies = {}
        return self._companies

    def cohort_members(self) -> dict[str, str | None]:
        """``{ticker: exclusion_reason | None}`` for the whole universe —
        who may stand in a benchmark cohort.

        Two properties this has to hold, both of which it once did not:

        * **The same exclusions the group applies.** A delisted company or
          a series that stops ten days before the as-of is thrown out of
          the group's own numbers; leaving it in the universe cohort made
          ``benchmark_relative`` compare the group against a cohort built
          from the very rows it had just ruled unusable.
        * **Independence from drain order.** Membership is decided from
          ``baseline`` — the series present when the period's inputs were
          loaded — so it cannot change as ``ensure_prices`` fetches series
          for whichever group happens to be computed first. Two groups in
          one period are then measured against the same cohort, and a
          recompute of one group yields the same ``inputs_hash`` whatever
          ran before it.

        A constituent fetched on demand still counts in its own group's
        statistics; it is simply not retro-fitted into the benchmark. The
        counts and the basis are recorded in ``method`` and ``sample``.
        """
        if self._cohort is _UNSET:
            companies = self.companies_map()
            out: dict[str, str | None] = {}
            for t in self.universe:
                company = companies.get(t) or {}
                if company.get("is_active") is False:
                    out[t] = REASON_INACTIVE
                    continue
                series = self.prices.get(t) if t in self.baseline else None
                if not series:
                    out[t] = REASON_NO_PRICES
                    continue
                end = _last_close_on_or_before(series, self.cutoff)
                if end is None:
                    out[t] = REASON_NO_PRICE_AT_AS_OF
                elif (self.cutoff - end[0]).days > STALE_PRICE_DAYS:
                    out[t] = REASON_STALE_PRICES
                else:
                    out[t] = None
            self._cohort = out
        return self._cohort

    def sector_tickers(self, sector_code: str) -> list[str]:
        return sorted({
            t for code, tickers in self.groups.items()
            if code.startswith(sector_code) for t in tickers
        })

    def ensure_prices(self, tickers: list[str]) -> None:
        """Fetch series for tickers without a cached one, within the
        budget; the rest are recorded as skipped so the sample can say so."""
        for ticker in tickers:
            t = ticker.upper()
            if t in self.prices or t in self.fetch_failed or t in self.fetch_skipped:
                continue
            if self.fetches >= self.max_fetch:
                self.fetch_skipped.add(t)
                continue
            self.fetches += 1
            try:
                rows = self.loaders.fetch_prices(t)
            except Exception as exc:  # a provider failure is a missing input, not a crash
                log.debug("price fetch failed for %s: %s", t, type(exc).__name__)
                rows = None
            series = close_series(rows)
            if series:
                self.prices[t] = series
                self.price_source[t] = "fetch"
            else:
                self.fetch_failed.add(t)

    def returns_for(self, ticker: str) -> dict[str, dict[str, Any]] | None:
        t = ticker.upper()
        series = self.prices.get(t)
        if not series:
            return None
        cached = self._horizons.get(t)
        if cached is None:
            cached = horizon_returns(series, self.cutoff)
            self._horizons[t] = cached
        return cached

    def market_factor_points(self) -> list[dict[str, Any]] | None:
        if self._factor is _UNSET:
            try:
                self._factor = self.loaders.market_factor()
            except Exception as exc:
                log.debug("market factor load failed: %s", type(exc).__name__)
                self._factor = None
        return self._factor


def load_context(
    as_of: datetime, *, version: VersionInfo | str | int | None = None,
    max_fetch: int | None = None, loaders: Loaders | None = None,
) -> AnalyticsContext:
    """Load the universe-wide inputs for one as-of. Group membership is
    one SELECT; the cached price rows are one bulk read (chunked); no
    provider fetch happens here — ``ensure_prices`` does that per group,
    inside the budget."""
    info = gics_registry.resolve_version(version)
    ld = loaders or default_loaders()
    budget = settings.industry_price_warmup_budget if max_fetch is None else int(max_fetch)
    groups = {code: sorted({t.upper() for t in tickers}) for code, tickers in ld.constituents_by_group(info).items()}
    ctx = AnalyticsContext(
        version=info, as_of=as_of, cutoff=as_of.date(), groups=groups, loaders=ld, max_fetch=max(0, budget),
    )
    universe = ctx.universe
    if universe:
        for ticker, rows in ld.cached_prices(universe).items():
            series = close_series(rows)
            if series:
                ctx.prices[ticker.upper()] = series
                ctx.price_source[ticker.upper()] = "cache"
    # Frozen here, before any group runs: the benchmark cohorts are drawn
    # from this set so they cannot drift with drain order.
    ctx.baseline = set(ctx.prices)
    return ctx


# ---------------------------------------------------------------------------
# Statistics helpers (pure Python — no numpy on this path)
# ---------------------------------------------------------------------------


def _values(returns_by_ticker: dict[str, dict[str, dict[str, Any]]], horizon: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for ticker, per_h in returns_by_ticker.items():
        entry = per_h.get(horizon) or {}
        if entry.get("value") is not None:
            out[ticker] = float(entry["value"])
    return out


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _r(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _weighted(values: dict[str, float], caps: dict[str, float]) -> tuple[float | None, int]:
    pairs = [(values[t], caps[t]) for t in values if caps.get(t, 0.0) > 0]
    total = sum(w for _, w in pairs)
    if not pairs or total <= 0:
        return None, 0
    return sum(v * w for v, w in pairs) / total, len(pairs)


def _market_cap_for(ticker: str, company: dict[str, Any], metrics: dict[str, Any]) -> tuple[float | None, str | None]:
    """Screener metric first (refreshed nightly), then the company row,
    then price × shares. Non-positive caps are treated as missing."""
    for source, value in (
        ("screener_metrics.market_cap", metrics.get("market_cap")),
        ("companies.market_cap", company.get("market_cap")),
    ):
        try:
            cap = float(value) if value is not None else None
        except (TypeError, ValueError):
            cap = None
        if cap is not None and cap > 0:
            return cap, source
    try:
        px = float(company.get("last_price") or 0.0)
        shares = float(company.get("shares_outstanding") or 0.0)
    except (TypeError, ValueError):
        px, shares = 0.0, 0.0
    if px > 0 and shares > 0:
        return px * shares, "last_price*shares_outstanding"
    return None, None


def _distribution(values: list[float]) -> dict[str, Any]:
    vals = sorted(values)
    return {
        "median": _r(median(vals)),
        "p25": _r(_quantile(vals, 0.25)),
        "p75": _r(_quantile(vals, 0.75)),
        "n": len(vals),
    }


def _cohort_returns(ctx: AnalyticsContext, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Equal-weight mean per horizon over the cohort-eligible tickers — the
    definition of both the universe and sector benchmarks.

    Eligibility comes from ``ctx.cohort_members()``, which applies the same
    exclusions the group's own numbers apply (inactive company, stale or
    absent series) and is fixed at context load. Whoever is dropped is
    counted by reason so the row can say what the benchmark is made of."""
    eligible = ctx.cohort_members()
    per: dict[str, dict[str, dict[str, Any]]] = {}
    excluded: dict[str, int] = {}
    for t in tickers:
        reason = eligible.get(t, REASON_NO_PRICES)
        if reason is not None:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        rets = ctx.returns_for(t)
        if rets is not None:
            per[t] = rets
    out: dict[str, dict[str, Any]] = {}
    for h in HORIZONS:
        vals = _values(per, h)
        if vals:
            out[h] = {"value": _r(mean(vals.values())), "n": len(vals)}
        else:
            out[h] = {"value": None, "n": 0, "reason": "no_constituent_returns"}
    out["_n_with_prices"] = {"value": len(per), "n": len(tickers),
                             "excluded_by_reason": dict(sorted(excluded.items()))}
    return out


def _factor_returns(points: list[dict[str, Any]] | None, cutoff: date) -> dict[str, dict[str, Any]]:
    """Cumulative ``Π(1 + r) − 1`` of the daily market factor over each
    horizon window ``(start, cutoff]``. The factor is Mkt−RF: an excess
    return, which the payload says explicitly."""
    daily: list[tuple[date, float]] = []
    for p in points or []:
        d = _parse_date(p.get("date") if isinstance(p, dict) else None)
        try:
            v = float(p.get("value")) if isinstance(p, dict) and p.get("value") is not None else None
        except (TypeError, ValueError):
            v = None
        if d is not None and v is not None and d <= cutoff:
            daily.append((d, v))
    daily.sort()
    out: dict[str, dict[str, Any]] = {}
    if not daily:
        return {h: {"value": None, "reason": "missing"} for h in HORIZONS}
    first = daily[0][0]
    last = daily[-1][0]
    for h in HORIZONS:
        start = horizon_start(cutoff, h)
        if first > start:
            out[h] = {"value": None, "reason": REASON_HISTORY_WINDOW}
            continue
        if (cutoff - last).days > STALE_PRICE_DAYS:
            out[h] = {"value": None, "reason": REASON_STALE_PRICES, "end_date": last.isoformat()}
            continue
        cum = 1.0
        n = 0
        for d, v in daily:
            if start < d <= cutoff:
                cum *= 1.0 + v
                n += 1
        if n == 0:
            out[h] = {"value": None, "reason": "no_points_in_window"}
        else:
            out[h] = {"value": _r(cum - 1.0), "n_days": n, "end_date": last.isoformat()}
    return out


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------


def _inputs_hash(code: str, ctx: AnalyticsContext, per_ticker: dict[str, dict[str, Any]],
                 metrics: dict[str, dict[str, Any]], benchmarks: dict[str, dict[str, Any]],
                 min_sample: int) -> str:
    """A digest of every input the payload is a function of.

    The identity has to cover the *evidence*, not just its endpoints: a
    provider revision that leaves the last close untouched but rewrites a
    mid-window day still moves ``breadth.above_50d_mean`` and the weekly
    closes a later period reuses, and a metrics refresh still moves the
    valuation medians. Hashing only (last_date, last_close, market_cap)
    let such a recompute collide with the stored row, which ``_persist``
    would then return in place of the new numbers — a stale row wearing a
    fresh timestamp. So the per-ticker entry carries the returns, the
    50-day flag, the weekly closes and the metric values actually read.

    Deliberately excluded: ``price_source`` and ``weight_mcw``. The first
    is provenance (the same series reached from the cache or a fetch is
    the same evidence); the second is derived from the caps already here.
    """
    hashed_metrics = VALUATION_METRICS + FUNDAMENTAL_METRICS + ("market_cap",)
    evidence: list[dict[str, Any]] = []
    for t in sorted(per_ticker):
        row = per_ticker[t]
        evidence.append({
            "ticker": t,
            "last_date": row.get("last_date"),
            "last_close": row.get("last_close"),
            "market_cap": row.get("market_cap"),
            "metrics_last_updated": row.get("metrics_last_updated"),
            "exclusion": row.get("exclusion"),
            "returns": row.get("returns"),
            "return_reasons": row.get("return_reasons"),
            "above_50d_mean": row.get("above_50d_mean"),
            "above_50d_mean_reason": row.get("above_50d_mean_reason"),
            "weekly_closes": row.get("weekly_closes"),
            "metrics": {m: (metrics.get(t) or {}).get(m) for m in hashed_metrics},
        })
    identity = {
        "method": METHOD_VERSION,
        "taxonomy_version_id": ctx.version.id,
        "code": code,
        "as_of": ctx.as_of.isoformat(),
        "min_sample": min_sample,
        "tickers": evidence,
        "benchmarks": {
            bid: {h: (entry.get("value"), entry.get("n"), entry.get("reason")) for h, entry in b.items()
                  if h in HORIZONS}
            for bid, b in benchmarks.items()
        },
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def sample_floor(min_sample: int | None = None) -> int:
    """The configured minimum priced constituents a group needs before its
    statistics mean anything. Public because `/api/industries/taxonomy`
    classifies against the same number and must not carry its own copy."""
    return int(settings.industry_stats_min_sample if min_sample is None else min_sample)


def universe_covers(n_constituents: int, min_sample: int) -> bool:
    """Can this universe EVER put ``min_sample`` priced names in a group
    with ``n_constituents`` classified members? The one definition of
    "structural", shared by the analytics row and the taxonomy endpoint so
    the two can never disagree about which groups are out of reach."""
    return int(n_constituents) >= int(min_sample)


def _count(n: int, singular: str, plural: str | None = None) -> str:
    """``3 companies`` / ``1 company`` — the explanations below are read by
    a person, and "1 constituents" reads as a bug in the number."""
    return f"{n} {singular}" if n == 1 else f"{n} {plural or singular + 's'}"


def classify_sample_floor(*, n_constituents: int, n_with_prices: int, min_sample: int) -> dict[str, Any]:
    """Which sample-floor state a group is in, and why — in words.

    Below the floor is two different situations, and one label for both
    tells a reader "not ready yet" when the truth may be "this universe
    does not hold enough of this industry":

    * ``universe_too_small`` — the classification table holds fewer
      constituents than the floor. Price every one of them and the group
      is still short, so no number of warm-up weeks reaches it; only a
      wider universe does. **Structural.**
    * ``prices_not_warmed`` — the membership clears the floor but this
      period could not price enough of it. The weekly warm-up fetches
      more series every run, so this clears on its own. **Transient.**
    * ``met`` — at or above the floor; the statistics are computed.

    The `explanation` is written here rather than in the UI for the same
    reason every other missing value on this surface carries its reason:
    a page that composes its own sentence is making a claim the server
    never made.
    """
    n_constituents = int(n_constituents)
    n_with_prices = int(n_with_prices)
    floor = int(min_sample)
    constituents_short_by = max(floor - n_constituents, 0)
    priced_short_by = max(floor - n_with_prices, 0)

    if priced_short_by == 0:
        state, explanation = FLOOR_MET, (
            f"{n_with_prices} of {_count(n_constituents, 'classified constituent')} were priced this "
            f"period, at or above the floor of {floor}."
        )
    elif not universe_covers(n_constituents, floor):
        state, explanation = FLOOR_UNIVERSE_TOO_SMALL, (
            f"this universe holds {_count(n_constituents, 'classified constituent')} for the group and the "
            f"floor is {floor}. Priced in full it would still be {constituents_short_by} short, so no amount "
            f"of price warm-up can cover it — the universe would have to add "
            f"{_count(constituents_short_by, 'company', 'companies')}."
        )
    else:
        priced = (
            "none of them had a usable price series this period"
            if n_with_prices == 0
            else f"only {n_with_prices} had a usable price series this period"
        )
        state, explanation = FLOOR_PRICES_NOT_WARMED, (
            f"{_count(n_constituents, 'constituent')} are classified into the group but {priced} — "
            f"{priced_short_by} below the floor of {floor}. The weekly warm-up fetches more series each "
            f"run, so this can clear without changing the universe."
        )
    return {
        "state": state,
        "structural": state == FLOOR_UNIVERSE_TOO_SMALL,
        "clears_with_warm_up": state == FLOOR_PRICES_NOT_WARMED,
        "min_sample": floor,
        "n_constituents": n_constituents,
        "n_with_prices": n_with_prices,
        "priced_short_by": priced_short_by,
        "constituents_short_by": constituents_short_by,
        "explanation": explanation,
    }


def compute_group_stats(
    code: str, *, as_of: datetime, version: VersionInfo | str | int | None = None,
    period_key: str | None = None, max_fetch: int | None = None,
    loaders: Loaders | None = None, context: AnalyticsContext | None = None,
    persist: bool = True, min_sample: int | None = None,
) -> IndustryStatSnapshot:
    """Compute (and by default persist) the stats row for one group.

    ``context`` lets a caller computing several groups for one as-of
    share the universe-wide inputs; otherwise one is loaded here.
    Returns the persisted row, detached — or the existing row when the
    same inputs were already stored for this period (identity by
    ``inputs_hash``). With ``persist=False`` the unsaved instance is
    returned for inspection.
    """
    started = time.perf_counter()
    ctx = context or load_context(as_of, version=version, max_fetch=max_fetch, loaders=loaders)
    group: NodeInfo = gics_registry.group(code, version=ctx.version)
    key = period_key or period_key_for(ctx.as_of)
    floor = sample_floor(min_sample)
    tickers = list(ctx.groups.get(group.code, []))

    # One companies read per context, shared with the benchmark cohort —
    # draining N groups must not re-read the universe N times.
    companies = ctx.companies_map()
    metrics = ctx.loaders.metrics(tickers) if tickers else {}
    ctx.ensure_prices(tickers)

    # Tickers whose cache row is gone and whose fetch was out of budget can
    # still be placed from the closes the prior period stored for them.
    prior: Any = _UNSET
    for t in tickers:
        if t in ctx.prices or (t not in ctx.fetch_skipped and t not in ctx.fetch_failed):
            continue
        if prior is _UNSET:
            try:
                prior = ctx.loaders.prior_stats(group.code, ctx.version, key)
            except Exception as exc:
                log.debug("prior stats read failed for %s: %s", group.code, type(exc).__name__)
                prior = None
        stored = ((prior or {}).get("per_ticker") or {}).get(t, {}).get("weekly_closes") if prior else None
        series = _series_from_weekly(stored or [])
        if series:
            ctx.prices[t] = series
            ctx.price_source[t] = "stored_weekly_closes"

    per_ticker: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, str]] = []
    returns_by_ticker: dict[str, dict[str, dict[str, Any]]] = {}
    caps: dict[str, float] = {}
    for t in tickers:
        company = companies.get(t, {})
        metric = metrics.get(t, {})
        cap, cap_source = _market_cap_for(t, company, metric)
        row: dict[str, Any] = {
            "company_name": company.get("company_name"),
            "market_cap": _r(cap, 2),
            "market_cap_source": cap_source,
            "metrics_last_updated": metric.get("last_updated"),
            "price_source": ctx.price_source.get(t),
            "last_date": None,
            "last_close": None,
            "returns": {},
            "weekly_closes": [],
            "exclusion": None,
        }
        series = ctx.prices.get(t)
        if company and company.get("is_active") is False:
            row["exclusion"] = REASON_INACTIVE
        elif not series:
            row["exclusion"] = REASON_FETCH_BUDGET if t in ctx.fetch_skipped else REASON_NO_PRICES
        else:
            end = _last_close_on_or_before(series, ctx.cutoff)
            if end is None:
                row["exclusion"] = REASON_NO_PRICE_AT_AS_OF
            elif (ctx.cutoff - end[0]).days > STALE_PRICE_DAYS:
                row["exclusion"] = REASON_STALE_PRICES
                row["last_date"], row["last_close"] = end[0].isoformat(), _r(end[1], 4)
            else:
                row["last_date"], row["last_close"] = end[0].isoformat(), _r(end[1], 4)
                rets = ctx.returns_for(t) or {}
                row["returns"] = {h: (_r(e.get("value")) if e.get("value") is not None else None) for h, e in rets.items()}
                row["return_reasons"] = {h: e["reason"] for h, e in rets.items() if e.get("reason")}
                row["weekly_closes"] = weekly_closes(series, cutoff=ctx.cutoff)
                # Above the mean of the last 50 trading sessions — sessions,
                # not calendar days, because that is what "50-day mean" means.
                flag, why = above_mean_window(series, ctx.cutoff)
                row["above_50d_mean"] = flag
                if why:
                    row["above_50d_mean_reason"] = why
                returns_by_ticker[t] = rets
                if cap is not None:
                    caps[t] = cap
        if row["exclusion"]:
            excluded.append({"ticker": t, "reason": row["exclusion"]})
        per_ticker[t] = row

    n_with_prices = len(returns_by_ticker)
    mcw_total = sum(caps.values())
    for t, cap in caps.items():
        per_ticker[t]["weight_mcw"] = _r(cap / mcw_total) if mcw_total > 0 else None

    # Benchmarks over the cohort fixed at context load — same exclusions as
    # the group's own numbers, and independent of which group ran first.
    cohort = ctx.cohort_members()
    cohort_excluded: dict[str, int] = {}
    for reason in cohort.values():
        if reason is not None:
            cohort_excluded[reason] = cohort_excluded.get(reason, 0) + 1
    universe_rets = _cohort_returns(ctx, ctx.universe)
    sector_rets = _cohort_returns(ctx, ctx.sector_tickers(group.sector_code))
    factor_rets = _factor_returns(ctx.market_factor_points(), ctx.cutoff)
    benchmarks: dict[str, dict[str, Any]] = {
        BENCHMARK_UNIVERSE: universe_rets,
        BENCHMARK_SECTOR: sector_rets,
        MARKET_FACTOR_ID: factor_rets,
    }
    degraded: list[str] = []
    for bid, b in benchmarks.items():
        if all(b[h].get("value") is None for h in HORIZONS):
            reason = next((b[h].get("reason") for h in HORIZONS if b[h].get("reason")), "missing")
            degraded.append(f"benchmark:{bid}:{reason}")

    insufficient = n_with_prices < floor
    returns: dict[str, dict[str, Any]] = {}
    benchmark_relative: dict[str, dict[str, dict[str, Any]]] = {bid: {} for bid in benchmarks}
    for h in HORIZONS:
        vals = _values(returns_by_ticker, h)
        if insufficient:
            returns[h] = {"equal_weight": None, "median": None, "market_cap_weight": None,
                          "n": len(vals), "reason": REASON_INSUFFICIENT}
        elif not vals:
            reason = REASON_HISTORY_WINDOW if n_with_prices else REASON_NO_PRICES
            returns[h] = {"equal_weight": None, "median": None, "market_cap_weight": None, "n": 0, "reason": reason}
        else:
            mcw, n_mcw = _weighted(vals, caps)
            returns[h] = {
                "equal_weight": _r(mean(vals.values())),
                "median": _r(median(vals.values())),
                "market_cap_weight": _r(mcw),
                "n": len(vals),
                "n_mcw": n_mcw,
            }
            if n_mcw == 0:
                returns[h]["market_cap_weight_reason"] = "no_market_caps"
        for bid, b in benchmarks.items():
            bench = b[h].get("value")
            ew = returns[h].get("equal_weight")
            if ew is None or bench is None:
                benchmark_relative[bid][h] = {
                    "value": None,
                    "reason": returns[h].get("reason") or b[h].get("reason") or "missing",
                }
            else:
                benchmark_relative[bid][h] = {"value": _r(ew - bench), "n": returns[h]["n"], "benchmark_n": b[h].get("n")}

    breadth: dict[str, Any] = {}
    for h in ("1w", "1m"):
        vals = _values(returns_by_ticker, h)
        if insufficient or not vals:
            breadth[h] = {"pct_positive": None, "n": len(vals),
                          "reason": REASON_INSUFFICIENT if insufficient else REASON_HISTORY_WINDOW}
        else:
            breadth[h] = {"pct_positive": _r(sum(1 for v in vals.values() if v > 0) / len(vals)), "n": len(vals)}
    above = [per_ticker[t]["above_50d_mean"] for t in returns_by_ticker if per_ticker[t].get("above_50d_mean") is not None]
    # A constituent whose series cannot support a 50-session mean is named
    # under its reason, not folded in as "not above".
    above_reasons: dict[str, int] = {}
    for t in returns_by_ticker:
        why = per_ticker[t].get("above_50d_mean_reason")
        if why:
            above_reasons[why] = above_reasons.get(why, 0) + 1
    mean_window_meta: dict[str, Any] = {"window_sessions": MEAN_WINDOW_SESSIONS}
    if above_reasons:
        mean_window_meta["excluded_by_reason"] = dict(sorted(above_reasons.items()))
    if insufficient or not above:
        breadth["above_50d_mean"] = {
            "share": None, "n": len(above),
            "reason": REASON_INSUFFICIENT if insufficient else (
                next(iter(sorted(above_reasons))) if above_reasons else REASON_MEAN_WINDOW_SHORT
            ),
            **mean_window_meta,
        }
    else:
        breadth["above_50d_mean"] = {"share": _r(sum(1 for a in above if a) / len(above)), "n": len(above), **mean_window_meta}

    vals_1m = _values(returns_by_ticker, "1m")
    if insufficient or len(vals_1m) < 2:
        dispersion: dict[str, Any] = {"horizon": "1m", "stdev": None, "iqr": None, "range": None, "n": len(vals_1m),
                                      "reason": REASON_INSUFFICIENT if insufficient else "n<2"}
    else:
        sorted_1m = sorted(vals_1m.values())
        dispersion = {
            "horizon": "1m",
            "stdev": _r(pstdev(sorted_1m)),
            "iqr": _r(_quantile(sorted_1m, 0.75) - _quantile(sorted_1m, 0.25)),
            "range": _r(sorted_1m[-1] - sorted_1m[0]),
            "n": len(sorted_1m),
        }

    valuation: dict[str, Any] = {}
    for m in VALUATION_METRICS:
        raw = [(t, metrics[t].get(m)) for t in tickers if t in metrics and metrics[t].get(m) is not None]
        positive = [float(v) for _, v in raw if float(v) > 0]
        if insufficient or not positive:
            valuation[m] = {"median": None, "n": 0,
                            "reason": REASON_INSUFFICIENT if insufficient else "no_metric_values"}
        else:
            valuation[m] = _distribution(positive)
            valuation[m]["n_excluded_nonpositive"] = len(raw) - len(positive)
    fundamentals: dict[str, Any] = {}
    for m in FUNDAMENTAL_METRICS:
        vals = [float(metrics[t][m]) for t in tickers if t in metrics and metrics[t].get(m) is not None]
        if insufficient or not vals:
            fundamentals[m] = {"median": None, "n": 0,
                               "reason": REASON_INSUFFICIENT if insufficient else "no_metric_values"}
        else:
            fundamentals[m] = _distribution(vals)

    # Fundamental momentum: the change in the median growth rate versus the
    # prior period's row — a proxy, not consensus revisions (none licensed).
    if prior is _UNSET:
        try:
            prior = ctx.loaders.prior_stats(group.code, ctx.version, key)
        except Exception as exc:
            log.debug("prior stats read failed for %s: %s", group.code, type(exc).__name__)
            prior = None
    prior_growth = (((prior or {}).get("payload") or {}).get("fundamentals") or {}).get("revenue_growth_yoy", {}).get("median") if prior else None
    cur_growth = fundamentals["revenue_growth_yoy"].get("median")
    if cur_growth is None or prior_growth is None:
        momentum: dict[str, Any] = {
            "value": None,
            "reason": "no_prior_period" if prior is None else "no_metric_values",
            "basis": "change in median revenue_growth_yoy vs prior period (proxy; not consensus revisions)",
        }
    else:
        momentum = {
            "value": _r(cur_growth - prior_growth),
            "prior_period_key": prior.get("period_key") if prior else None,
            "basis": "change in median revenue_growth_yoy vs prior period (proxy; not consensus revisions)",
        }

    ranked = [] if insufficient else sorted(vals_1m.items(), key=lambda kv: (-kv[1], kv[0]))
    # Disjoint by construction. Taking the top and bottom LEADER_COUNT of
    # one ranking re-listed the same names in both sections whenever a
    # group had fewer than 2 * LEADER_COUNT priced constituents, so a small
    # group appeared to be both leading and lagging on the same company.
    # Splitting the ranking keeps the two halves apart at any size: a
    # 3-name group reports one leader and one laggard, a 1-name group
    # reports neither, and a group of 10 or more is unchanged.
    n_ends = min(LEADER_COUNT, len(ranked) // 2)
    leaders = [
        {"ticker": t, "ret_1m": _r(v), "weight_mcw": per_ticker[t].get("weight_mcw")} for t, v in ranked[:n_ends]
    ]
    laggards = [
        {"ticker": t, "ret_1m": _r(v), "weight_mcw": per_ticker[t].get("weight_mcw")}
        for t, v in sorted(ranked[len(ranked) - n_ends:], key=lambda kv: (kv[1], kv[0]))
    ]

    dates = [row["last_date"] for row in per_ticker.values() if row.get("last_date")]
    floor_state = classify_sample_floor(
        n_constituents=len(tickers), n_with_prices=n_with_prices, min_sample=floor,
    )
    sample = {
        "n_constituents": len(tickers),
        "n_with_prices": n_with_prices,
        # Which floor state this group is in, and whether the shortfall is
        # a warm-up that improves weekly or a universe that cannot hold
        # the floor at all. Two different sentences to a reader, and this
        # is the field that keeps them apart.
        "sample_floor": floor_state,
        "n_with_market_cap": len(caps),
        "n_with_metrics": sum(1 for t in tickers if t in metrics),
        "min_sample": floor,
        "coverage": _r(n_with_prices / len(tickers), 4) if tickers else None,
        "excluded": sorted(excluded, key=lambda e: e["ticker"]),
        "prices_max_date": max(dates) if dates else None,
        "prices_min_date": min(dates) if dates else None,
        "price_sources": {
            src: sum(1 for t in returns_by_ticker if ctx.price_source.get(t) == src)
            for src in sorted({ctx.price_source.get(t, "") for t in returns_by_ticker} - {""})
        },
        "fetches_this_run": ctx.fetches,
        "fetch_budget": ctx.max_fetch,
        "benchmark_cohort": {
            "n_universe": len(cohort),
            "n_eligible": sum(1 for reason in cohort.values() if reason is None),
            "excluded_by_reason": dict(sorted(cohort_excluded.items())),
            # Named because these constituents count in the group's own
            # numbers but not in the cohort it is measured against.
            "group_members_outside_cohort": [
                {"ticker": t, "reason": cohort.get(t) or REASON_NO_PRICES}
                for t in sorted(returns_by_ticker) if cohort.get(t) is not None
            ],
        },
    }
    method = {
        "version": METHOD_VERSION,
        "weighting": ["equal", "market_cap"],
        "benchmarks": [
            {"id": BENCHMARK_UNIVERSE, "definition": "equal-weight mean of every cohort-eligible classified constituent",
             "n": universe_rets["_n_with_prices"]["value"], "n_universe": universe_rets["_n_with_prices"]["n"],
             "excluded_by_reason": universe_rets["_n_with_prices"].get("excluded_by_reason", {})},
            {"id": BENCHMARK_SECTOR, "definition": f"equal-weight mean of the cohort-eligible sector {group.sector_code} constituents",
             "n": sector_rets["_n_with_prices"]["value"], "n_universe": sector_rets["_n_with_prices"]["n"],
             "excluded_by_reason": sector_rets["_n_with_prices"].get("excluded_by_reason", {})},
            {"id": MARKET_FACTOR_ID, "definition": "Ken-French daily market factor, cumulative over the horizon; an EXCESS return over the risk-free rate, not a total return"},
        ],
        "benchmark_cohort_basis": (
            "a classified constituent whose company is active and whose price series was present when the "
            "period's inputs were loaded and is not stale at the as-of. The same exclusions the group's own "
            "numbers apply, fixed before any group is computed — so the benchmark does not depend on the "
            "order groups are drained, and a constituent fetched on demand counts in its group but is not "
            "retro-fitted into the cohort."
        ),
        "breadth_mean_window": {
            "sessions": MEAN_WINDOW_SESSIONS,
            "basis": "the last 50 closes in the series on or before the as-of — trading sessions, not calendar days",
            "max_span_days": MEAN_WINDOW_MAX_SPAN_DAYS,
            "unavailable": "fewer than 50 closes, or closes too sparse to be daily, yield null with a reason — never a default",
        },
        "min_sample": floor,
        "missing_data": "exclude_and_report",
        "horizons": list(HORIZONS),
        "price_window_days": PRICE_WINDOW_DAYS,
        "history_window_note": "horizons the cached price window cannot anchor return null with reason history_window (v1 scope cut)",
        "stale_price_days": STALE_PRICE_DAYS,
        "price_field": "adjusted_close, else close",
        "market_cap_source": "screener_metrics.market_cap, else companies.market_cap, else last_price*shares_outstanding",
        "valuation_note": "non-positive multiples excluded from medians and counted",
        "as_of": ctx.as_of.isoformat(),
        "taxonomy_version": ctx.version.version_key,
    }
    for b in method["benchmarks"]:
        b["available"] = not any(d.startswith(f"benchmark:{b['id']}:") for d in degraded)
    for b in (universe_rets, sector_rets):
        b.pop("_n_with_prices", None)

    payload = {
        "status": REASON_INSUFFICIENT if insufficient else "ok",
        "code": group.code,
        "name": group.name,
        "sector_code": group.sector_code,
        "period_key": key,
        "as_of": ctx.as_of.isoformat(),
        "returns": returns,
        "benchmarks": benchmarks,
        "benchmark_relative": benchmark_relative,
        "breadth": breadth,
        "dispersion": dispersion,
        "valuation": valuation,
        "fundamentals": fundamentals,
        "fundamental_momentum": momentum,
        "leaders": leaders,
        "laggards": laggards,
        "degraded": degraded,
    }
    if insufficient:
        # `status` stays the one label the snapshot and the writer already
        # count; WHICH short state it is rides alongside it, because
        # "the warm-up has not reached this group yet" and "this universe
        # cannot hold this group's floor" are not the same news.
        payload["insufficient_sample"] = {
            "n_with_prices": n_with_prices, "min_sample": floor,
            "n_constituents": len(tickers),
            "state": floor_state["state"],
            "structural": floor_state["structural"],
            "clears_with_warm_up": floor_state["clears_with_warm_up"],
            "explanation": floor_state["explanation"],
            "reasons": sorted({e["reason"] for e in excluded}),
        }

    row = IndustryStatSnapshot(
        taxonomy_version_id=ctx.version.id,
        industry_group_code=group.code,
        period_key=key,
        as_of=ctx.as_of,
        method=method,
        sample=sample,
        payload=payload,
        per_ticker=per_ticker,
        inputs_hash=_inputs_hash(group.code, ctx, per_ticker, metrics, benchmarks, floor),
        compute_ms=int((time.perf_counter() - started) * 1000),
        computed_at=_utcnow(),
    )
    if not persist:
        return row
    return _persist(row)


def _persist(row: IndustryStatSnapshot) -> IndustryStatSnapshot:
    """Insert unless an identical-input row exists for the period; either
    way return a detached row."""
    with SessionLocal() as db:
        existing = db.execute(
            select(IndustryStatSnapshot).where(
                IndustryStatSnapshot.taxonomy_version_id == row.taxonomy_version_id,
                IndustryStatSnapshot.industry_group_code == row.industry_group_code,
                IndustryStatSnapshot.period_key == row.period_key,
                IndustryStatSnapshot.inputs_hash == row.inputs_hash,
            ).order_by(IndustryStatSnapshot.id.desc())
        ).scalars().first()
        if existing is not None:
            db.expunge(existing)
            return existing
        db.add(row)
        db.commit()
        db.refresh(row)
        db.expunge(row)
        return row


# ---------------------------------------------------------------------------
# Warm-up (called by the weekly loop BEFORE it enqueues report jobs)
# ---------------------------------------------------------------------------


def warm_up_prices(
    *, budget: int | None = None, version: VersionInfo | str | int | None = None,
    loaders: Loaders | None = None,
) -> dict[str, Any]:
    """Pre-fetch missing price series, lowest-coverage groups first, within
    ``budget`` provider calls (``INDUSTRY_PRICE_WARMUP_BUDGET``).

    The fetch goes through ``data_service.get_price_history``, which writes
    the cache row the analytics later read. Returns the coverage before
    and after per group so the loop's ``record_run`` note can say what the
    week's reports will actually be built on.
    """
    info = gics_registry.resolve_version(version)
    ld = loaders or default_loaders()
    cap = settings.industry_price_warmup_budget if budget is None else int(budget)
    groups = {code: sorted({t.upper() for t in tickers}) for code, tickers in ld.constituents_by_group(info).items()}
    universe = sorted({t for tickers in groups.values() for t in tickers})
    have = set(ld.cached_price_tickers(universe)) if universe else set()

    def coverage(code: str) -> float:
        tickers = groups[code]
        return (sum(1 for t in tickers if t in have) / len(tickers)) if tickers else 1.0

    before = {code: round(coverage(code), 4) for code in groups}
    order = sorted(groups, key=lambda c: (coverage(c), c))
    fetched: list[str] = []
    failed: list[str] = []
    touched: list[str] = []
    attempted: set[str] = set()
    for code in order:
        missing = [t for t in groups[code] if t not in have and t not in attempted]
        if not missing:
            continue
        for t in missing:
            if len(fetched) + len(failed) >= cap:
                break
            attempted.add(t)
            try:
                rows = ld.fetch_prices(t)
            except Exception as exc:
                log.debug("warm-up fetch failed for %s: %s", t, type(exc).__name__)
                rows = None
            if close_series(rows):
                fetched.append(t)
                have.add(t)
            else:
                failed.append(t)
        if code not in touched:
            touched.append(code)
        if len(fetched) + len(failed) >= cap:
            break
    after = {code: round(coverage(code), 4) for code in groups}
    remaining = sorted(t for t in universe if t not in have)
    return {
        "budget": cap,
        "fetched": len(fetched),
        "failed": sorted(failed),
        "remaining_missing": len(remaining),
        "groups_touched": touched,
        "coverage_before": before,
        "coverage_after": after,
        "lowest_coverage_after": sorted(after.items(), key=lambda kv: (kv[1], kv[0]))[:5],
    }


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def stats_dict(row: IndustryStatSnapshot) -> dict[str, Any]:
    """Detached projection of a stats row."""
    return {
        "id": row.id,
        "taxonomy_version_id": row.taxonomy_version_id,
        "code": row.industry_group_code,
        "period_key": row.period_key,
        "as_of": row.as_of.isoformat() if row.as_of else None,
        "method": dict(row.method or {}),
        "sample": dict(row.sample or {}),
        "payload": dict(row.payload or {}),
        "per_ticker": dict(row.per_ticker or {}),
        "inputs_hash": row.inputs_hash,
        "compute_ms": row.compute_ms,
        "computed_at": row.computed_at.isoformat() if row.computed_at else None,
    }


def latest_stats(code: str, *, version: VersionInfo | str | int | None = None) -> dict[str, Any] | None:
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        row = db.execute(
            select(IndustryStatSnapshot).where(
                IndustryStatSnapshot.taxonomy_version_id == info.id,
                IndustryStatSnapshot.industry_group_code == str(code),
            ).order_by(IndustryStatSnapshot.as_of.desc(), IndustryStatSnapshot.id.desc())
        ).scalars().first()
        return stats_dict(row) if row is not None else None


def stats_for_period(
    code: str, period_key: str, *, version: VersionInfo | str | int | None = None,
) -> dict[str, Any] | None:
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        row = db.execute(
            select(IndustryStatSnapshot).where(
                IndustryStatSnapshot.taxonomy_version_id == info.id,
                IndustryStatSnapshot.industry_group_code == str(code),
                IndustryStatSnapshot.period_key == period_key,
            ).order_by(IndustryStatSnapshot.id.desc())
        ).scalars().first()
        return stats_dict(row) if row is not None else None


def stats_by_id(stats_id: int) -> dict[str, Any] | None:
    with SessionLocal() as db:
        row = db.get(IndustryStatSnapshot, int(stats_id))
        return stats_dict(row) if row is not None else None


def period_stats(period_key: str, *, version: VersionInfo | str | int | None = None) -> dict[str, dict[str, Any]]:
    """``{code: stats_dict}`` — the newest row per group for one period,
    one SELECT. What the cross-industry snapshot reads."""
    info = gics_registry.resolve_version(version)
    out: dict[str, dict[str, Any]] = {}
    with SessionLocal() as db:
        rows = db.execute(
            select(IndustryStatSnapshot).where(
                IndustryStatSnapshot.taxonomy_version_id == info.id,
                IndustryStatSnapshot.period_key == period_key,
            ).order_by(IndustryStatSnapshot.id)
        ).scalars().all()
        for row in rows:
            out[row.industry_group_code] = stats_dict(row)
    return out
