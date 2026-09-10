"""Fundamental Factor Scorecard — scoring runs, version registry, readers
and the frozen export (Phase 6, slice C).

The worker computes; the web process reads. `run_scorecard(version, as_of)`
is the job body the queue executes: it streams one ticker's
`financial_periods` at a time, reduces them to a point-in-time feature
vector (`finance.scorecard_features`), normalises the cross-section
(`finance.scorecard_normalize`) and persists one `scorecard_scores` row per
ticker together with the run's lineage — the spec hash, an inputs hash
over every snapshot that fed the run, counts, and a `key=value` note that
tells cron-health whether an empty run was "nothing due" or "something
broke". Score rows and the run's outcome commit in ONE transaction, so a
failed run leaves no rows and a succeeded run is never missing any.

Point-in-time rules honoured here, not re-derived: only rows with
`available_at <= as_of` reach the feature engine (NULL availability is
excluded and counted, never guessed); the price on the as-of date comes
from the month-end store when the store already holds the AS-OF MONTH
(the exact close the evaluation later joins against) and otherwise from
the SAME 252-day cached series the rest of the app uses — no new provider
key, no new provider calls. A store row from an earlier month is only ever
a last resort, and the row then says so (`price_date`, `price_stale`).
Sector labels and the constituent list are today's (documented residue).
Restated values keep their original availability date.

Two layers in every stored row: `feature_raw` is what was observed (null
with a reason when an input is missing — never zero, never "neutral");
`feature_z` is the model read under the named methodology. The score scale
is 0-100 with 50 = a z of 0, the sector (or universe-fallback) mean — not
the median. Research and education only: a scorecard row is an observed
ranking, not a recommendation.

Storage convention for the JSON columns: every key beginning with `_` is
metadata (reasons, derived context, per-feature basis, unneutralised z,
profiles). Readers split it off with `_split_meta`; the export never emits
it. Keeping it in the same row avoids a seventh table and keeps the
"why is this null" answer next to the null.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
from calendar import monthrange
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, NamedTuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..agents.log_safety import safe_exc
from ..config import settings
from ..database import SessionLocal
from ..finance import scorecard_features, scorecard_normalize, scorecard_spec
from ..finance.factor_scores import _z_to_100
from ..finance.scorecard_evaluation_math import EVALUATION_CAVEATS
from ..models import (
    Company,
    FinancialPeriod,
    PriceMonthEnd,
    ScorecardEvaluation,
    ScorecardRun,
    ScorecardScore,
    ScorecardVersion,
)
from ..schemas.scorecard import (
    ScorecardCategory,
    ScorecardContribution,
    ScorecardDetailOut,
    ScorecardEvaluationItem,
    ScorecardEvaluationOut,
    ScorecardFeatureOut,
    ScorecardHistoryPoint,
    ScorecardSummary,
    ScorecardUniverseOut,
    ScorecardUniverseRow,
)
from . import scorecard_pit, scorecard_queue

log = logging.getLogger(__name__)

VERSION_KEY = scorecard_spec.VERSION_KEY

# Rows streamed per fetch from `financial_periods` — one ticker's rows are
# a few hundred, so a batch of 2000 keeps a handful of tickers in flight
# and nothing like the whole table on the 512 MB worker.
_STREAM_BATCH = 2000

# A daily as-of older than the cached price window cannot be served by
# the series at all; the month-end store is the only source there.
_SERIES_WINDOW_DAYS = 252

EXPORT_CONTRACT = "v1"

# FROZEN. `contract=v1` promises this order byte-for-byte to downstream
# consumers; new columns are only ever appended under `contract=v2`.
EXPORT_COLUMNS_V1: tuple[str, ...] = (
    "contract_version", "version_key", "spec_hash", "run_id", "as_of", "price_date",
    "data_available_at", "ticker", "sector", "coverage", "overall_z", "overall_score",
    "universe_percentile", "sector_percentile",
    "z_valuation", "z_quality", "z_growth", "z_profitability", "z_efficiency", "z_leverage",
    "z_capital_allocation", "z_earnings_quality",
    "pct_valuation", "pct_quality", "pct_growth", "pct_profitability", "pct_efficiency",
    "pct_leverage", "pct_capital_allocation", "pct_earnings_quality",
    "top_positive_1", "top_positive_2", "top_positive_3",
    "top_negative_1", "top_negative_2", "top_negative_3",
    "generated_at",
)
_EXPORT_FAMILY_ORDER: tuple[str, ...] = (
    "valuation", "quality", "growth", "profitability", "efficiency", "leverage",
    "capital_allocation", "earnings_quality",
)

UNIVERSE_SORT_COLUMNS: frozenset[str] = frozenset({
    "overall_score", "overall_z", "universe_percentile", "sector_percentile", "coverage", "ticker",
    *scorecard_spec.FAMILY_NAMES,
})


class UnknownVersion(LookupError):
    """A `version_key` that is neither registered nor the in-code spec."""


class UnsupportedVersion(ValueError):
    """Only the in-code methodology can be *computed*; registry rows for
    older versions are read-only history."""


def _utcnow() -> datetime:
    """Clock seam — tests monkeypatch this instead of freezing time."""
    return datetime.utcnow()


def _today() -> date:
    return _utcnow().date()


_TABLES_READY: set[int] = set()


def _ensure_tables(db: Session) -> None:
    """Lazy-create the scorecard tables once per engine. Every reader
    calls this, and `ticker_detail` three times over, so without the
    per-bind memo each `/api/scorecard/{ticker}` request paid ~18 catalog
    `has_table` round-trips on Postgres before its first data query."""
    bind = db.get_bind()
    key = id(bind)
    if key in _TABLES_READY:
        return
    for model in (ScorecardVersion, ScorecardRun, ScorecardScore, ScorecardEvaluation, FinancialPeriod, Company):
        model.__table__.create(bind=bind, checkfirst=True)
    _TABLES_READY.add(key)


def _month_end(d: date) -> date:
    return date(d.year, d.month, monthrange(d.year, d.month)[1])


def is_month_end(d: date) -> bool:
    return d == _month_end(d)


def _finite(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _split_meta(d: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(values, meta)`: metadata keys start with `_` (see module docstring)."""
    values: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    for k, v in (d or {}).items():
        (meta if str(k).startswith("_") else values)[str(k)] = v
    return values, meta


# ---------------------------------------------------------------------------
# Version registry
# ---------------------------------------------------------------------------

def ensure_version_registered(*, db: Session | None = None) -> dict[str, Any]:
    """Upsert the registry row for the in-code spec and make it active.

    Idempotent. The row's `spec_hash` is recomputed from code on every
    call, so an edit to `fs-v1` that forgot to bump `VERSION_KEY` shows up
    as `changed=True` and a warning — the hash is the only thing that can
    catch that. Called by the worker's seed thread and by the loop at the
    start of every tick; the web process never needs it because readers
    fall back to the in-code spec when no active row exists.
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        spec = scorecard_spec.spec_as_dict()
        digest = scorecard_spec.hash_spec_dict(spec)
        row = db.execute(
            select(ScorecardVersion).where(ScorecardVersion.version_key == VERSION_KEY)
        ).scalar_one_or_none()
        created = changed = False
        if row is None:
            row = ScorecardVersion(
                version_key=VERSION_KEY, spec_hash=digest, spec_json=spec, is_active=True,
                created_at=_utcnow(), notes="registered from code",
            )
            db.add(row)
            created = True
        elif row.spec_hash != digest:
            log.warning(
                "scorecard spec hash changed under version %s (%s -> %s) without a version bump; "
                "rows scored before this point are not comparable to rows scored after it",
                VERSION_KEY, (row.spec_hash or "")[:12], digest[:12],
            )
            row.spec_hash = digest
            row.spec_json = spec
            row.notes = f"spec hash changed at {_utcnow().isoformat()} without a version bump"
            changed = True
        for other in db.execute(
            select(ScorecardVersion).where(ScorecardVersion.version_key != VERSION_KEY, ScorecardVersion.is_active)
        ).scalars().all():
            other.is_active = False
        if not row.is_active:
            row.is_active = True
        db.commit()
        return {"version_key": VERSION_KEY, "spec_hash": digest, "created": created, "changed": changed, "active": True}
    finally:
        if own:
            db.close()


def active_version(*, db: Session | None = None) -> dict[str, Any]:
    """`{version_key, spec_hash, spec, source}` — the registry's active
    row, else the in-code spec (`source="code"`): the web service can boot
    before the worker registers anything and must still answer."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        row = db.execute(
            select(ScorecardVersion).where(ScorecardVersion.is_active).order_by(ScorecardVersion.id.desc())
        ).scalars().first()
        if row is not None and row.spec_json:
            return {"version_key": row.version_key, "spec_hash": row.spec_hash, "spec": dict(row.spec_json),
                    "source": "registry"}
        spec = scorecard_spec.spec_as_dict()
        return {"version_key": VERSION_KEY, "spec_hash": scorecard_spec.hash_spec_dict(spec), "spec": spec,
                "source": "code"}
    finally:
        if own:
            db.close()


def resolve_version(version_key: str | None, *, db: Session | None = None) -> dict[str, Any]:
    """Look a version up by key (None = active). Raises `UnknownVersion`."""
    if not version_key:
        return active_version(db=db)
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        row = db.execute(
            select(ScorecardVersion).where(ScorecardVersion.version_key == version_key)
        ).scalar_one_or_none()
        if row is not None and row.spec_json:
            return {"version_key": row.version_key, "spec_hash": row.spec_hash, "spec": dict(row.spec_json),
                    "source": "registry"}
        if version_key == VERSION_KEY:
            spec = scorecard_spec.spec_as_dict()
            return {"version_key": VERSION_KEY, "spec_hash": scorecard_spec.hash_spec_dict(spec), "spec": spec,
                    "source": "code"}
        raise UnknownVersion(f"unknown scorecard version {version_key!r}")
    finally:
        if own:
            db.close()


# ---------------------------------------------------------------------------
# Universe and price context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Member:
    ticker: str
    sector: str | None
    company_name: str
    shares_outstanding: float | None


def universe_members(*, tickers: Iterable[str] | None = None, db: Session | None = None) -> list[Member]:
    """The scorecard universe: `auto_analysis` companies, or the explicit
    `tickers` (which need not be in the universe — a name without a
    `companies` row scores with no sector and no share fallback, and the
    normaliser notes `sector_missing`). Sorted by ticker so every run
    walks the same order and the run-level hash is order-free anyway."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        if tickers is not None:
            wanted = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
            if not wanted:
                return []
            rows = db.execute(
                select(Company.ticker, Company.sector, Company.company_name, Company.shares_outstanding)
                .where(Company.ticker.in_(wanted))
            ).all()
            known = {r[0]: r for r in rows}
            return [
                Member(t, known[t][1], known[t][2] or "", _finite(known[t][3])) if t in known
                else Member(t, None, "", None)
                for t in wanted
            ]
        rows = db.execute(
            select(Company.ticker, Company.sector, Company.company_name, Company.shares_outstanding)
            .where(Company.universe_tier == "auto_analysis")
            .order_by(Company.ticker)
        ).all()
        return [Member(r[0], r[1], r[2] or "", _finite(r[3])) for r in rows]
    finally:
        if own:
            db.close()


def _price_series(ticker: str) -> list[dict[str, Any]] | None:
    """The app's cached 252-day series (`data_service.get_price_history`,
    same cache key as every other consumer). None when the chain has
    nothing; exceptions are logged by type and treated as None so one
    provider hiccup never aborts a run."""
    try:
        from .data_service import get_data_service
        return get_data_service().get_price_history(ticker.upper(), days=_SERIES_WINDOW_DAYS)
    except Exception as exc:
        log.warning("scorecard price series for %s failed: %s", ticker, type(exc).__name__)
        return None


class StorePrice(NamedTuple):
    """One `price_month_ends` row as the scorer needs it. `month_end` is
    the calendar month the close belongs to — `price_context` compares it
    against the as-of month to decide whether the store is current."""
    close: float
    price_date: date
    month_end: date


def store_prices(db: Session, tickers: list[str], as_of: date) -> dict[str, StorePrice]:
    """Latest stored month-end close on or before `as_of` per ticker, in
    ONE query — loaded before the statement stream opens so no query runs
    against a connection holding a server-side cursor (Postgres). The row
    may belong to an EARLIER month than `as_of` (the store fills a month
    in only after it closes and `pit_prepare` has synced it); the
    `month_end` on each entry lets the caller tell current from stale."""
    if not tickers:
        return {}
    rows = db.execute(
        select(PriceMonthEnd.ticker, PriceMonthEnd.month_end, PriceMonthEnd.price_date, PriceMonthEnd.close)
        .where(PriceMonthEnd.ticker.in_(tickers), PriceMonthEnd.month_end <= as_of)
        .order_by(PriceMonthEnd.ticker, PriceMonthEnd.month_end.desc())
    ).all()
    out: dict[str, StorePrice] = {}
    for t, month_end, price_date, close in rows:
        if t in out:
            continue
        c = _finite(close)
        if c is not None:
            out[t] = StorePrice(c, price_date or month_end, month_end)
    return out


def _as_store_price(value: Any) -> StorePrice | None:
    """Accept a `StorePrice` or a bare `(close, price_date)` pair (the
    month is then the price date's own month, which is what the store
    guarantees anyway)."""
    if value is None:
        return None
    if isinstance(value, StorePrice):
        return value
    close, price_date = value[0], value[1]
    month_end = value[2] if len(value) > 2 and value[2] is not None else _month_end(price_date)
    return StorePrice(float(close), price_date, month_end)


def price_context(
    ticker: str, as_of: date, *, store_price: Any, prefer_store: bool,
    shares_fallback: float | None, today: date | None = None,
) -> tuple[dict[str, Any], str]:
    """`(price_ctx, basis)` for the feature engine.

    `basis` says where the price came from: `month_end_store` (a stored
    month-end close on or before `as_of`, preloaded by `store_prices`),
    `daily_series` (the last cached daily close on or before `as_of`) or
    `none`.

    The store is preferred ONLY when `prefer_store` is set AND the stored
    row belongs to the as-of month — that is the exact close the
    evaluation joins forward returns against, so a month-end run must
    reuse it. A store row from an earlier month is never preferred: on the
    first tick after a month turns (and permanently for months whose last
    trading day is not the calendar month end) the store lags a month
    while the cached series already carries the as-of close, and scoring
    every valuation feature with last month's price would silently poison
    rows that are kept forever. So a lagging store falls behind the series
    and is used only when the series has nothing, with `price_date` and
    `price_stale` saying so.
    """
    ctx: dict[str, Any] = {"price": None, "price_date": None, "shares_fallback": shares_fallback}
    today = today or _today()
    store = _as_store_price(store_price)
    store_is_current = store is not None and store.month_end == _month_end(as_of)

    def from_store() -> tuple[float, date, str] | None:
        if store is None:
            return None
        return store.close, store.price_date, "month_end_store"

    def from_series() -> tuple[float, date, str] | None:
        if (today - as_of).days > _SERIES_WINDOW_DAYS:
            return None
        series = _price_series(ticker)
        if not series:
            return None
        best: tuple[date, float] | None = None
        for r in series:
            d = scorecard_pit._coerce_date(r.get("date"))
            c = _finite(r.get("close"))
            if d is None or c is None or d > as_of:
                continue
            if best is None or d > best[0]:
                best = (d, c)
        return (best[1], best[0], "daily_series") if best else None

    order = (from_store, from_series) if (prefer_store and store_is_current) else (from_series, from_store)
    for source in order:
        got = source()
        if got is not None:
            ctx["price"], ctx["price_date"] = got[0], got[1]
            ctx["price_stale"] = _month_end(got[1]) < _month_end(as_of)
            return ctx, got[2]
    ctx["price_stale"] = False
    return ctx, "none"


# ---------------------------------------------------------------------------
# The scoring run
# ---------------------------------------------------------------------------

def _iter_period_rows(db: Session, tickers: list[str]) -> Iterator[tuple[str, list[tuple[Any, ...]]]]:
    """Stream `financial_periods` grouped by ticker, in the row contract
    `scorecard_features.pit_snapshot` expects, holding one ticker's rows
    at a time. NULL availability rows are passed through so the engine
    can count them (`missing_available_at`) rather than silently dropped."""
    if not tickers:
        return
    stmt = (
        select(
            FinancialPeriod.ticker, FinancialPeriod.statement, FinancialPeriod.line_item,
            FinancialPeriod.period, FinancialPeriod.period_end, FinancialPeriod.fiscal_year,
            FinancialPeriod.fiscal_quarter, FinancialPeriod.value, FinancialPeriod.available_at,
        )
        .where(FinancialPeriod.ticker.in_(tickers))
        .order_by(FinancialPeriod.ticker, FinancialPeriod.id)
        .execution_options(yield_per=_STREAM_BATCH)
    )
    current: str | None = None
    bucket: list[tuple[Any, ...]] = []
    for row in db.execute(stmt):
        t = row[0]
        if t != current:
            if current is not None:
                yield current, bucket
            current, bucket = t, []
        bucket.append(tuple(row[1:]))
    if current is not None:
        yield current, bucket


def _run_inputs_hash(spec_hash: str, as_of: date, per_ticker: dict[str, str]) -> str:
    payload = {"spec_hash": spec_hash, "as_of": as_of.isoformat(), "tickers": dict(sorted(per_ticker.items()))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _contrib_list(pairs: list[tuple[str, float]], feature_z: dict[str, float | None]) -> list[dict[str, Any]]:
    return [
        {"feature": name, "family": scorecard_spec.FEATURES_BY_NAME[name].family,
         "z": feature_z.get(name), "contribution": value}
        for name, value in pairs
    ]


def run_scorecard(
    version_key: str | None,
    as_of: date,
    *,
    run_kind: str = scorecard_queue.KIND_MANUAL,
    tickers: Iterable[str] | None = None,
    run_row_id: int | None = None,
    requested_by: str = "direct",
) -> dict[str, Any]:
    """Score the universe as of `as_of` under `version_key` (None / the
    in-code key). Returns the finished run dict.

    Never raises for a data problem: the run row ends `failed` with the
    exception type and a redacted message, and — because rows and outcome
    share one transaction — no score rows. `skipped` means a succeeded run
    for the same `(version, as_of)` already holds identical inputs.
    """
    as_of = scorecard_pit._coerce_date(as_of) or as_of
    if run_kind not in scorecard_queue.SCORING_KINDS:
        raise ValueError(f"run_scorecard: {run_kind!r} is not a scoring run kind")
    vk = version_key or VERSION_KEY
    explicit = None if tickers is None else [str(t).strip().upper() for t in tickers if str(t).strip()]
    if run_row_id is None:
        run_row_id = scorecard_queue.create_running_row(
            version_key=vk, as_of=as_of, kind=run_kind,
            params={"tickers": explicit} if explicit is not None else {}, requested_by=requested_by,
        )
    try:
        return _run_scorecard_inner(vk, as_of, run_kind=run_kind, explicit=explicit, run_row_id=run_row_id)
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        log.error("scorecard run %d failed: %s", run_row_id, safe_exc(exc))
        return scorecard_queue.mark_failed(run_row_id, exc, note=f"as_of={as_of.isoformat()} version={vk}") or {}


def _run_scorecard_inner(
    vk: str, as_of: date, *, run_kind: str, explicit: list[str] | None, run_row_id: int,
) -> dict[str, Any]:
    if vk != VERSION_KEY:
        raise UnsupportedVersion(
            f"only the in-code methodology {VERSION_KEY!r} can be computed; {vk!r} is read-only history"
        )
    spec_hash = scorecard_spec.spec_hash()
    month_end = is_month_end(as_of)
    prefer_store = month_end or run_kind in (scorecard_queue.KIND_MONTH_END, scorecard_queue.KIND_BACKFILL)
    today = _today()

    with SessionLocal() as db:
        _ensure_tables(db)
        members = universe_members(tickers=explicit, db=db)
        by_ticker = {m.ticker: m for m in members}
        universe = [m.ticker for m in members]

        # NULL availability is a data gap, not a point-in-time fact; the
        # backfill is idempotent and a no-op once every row is dated.
        filled = scorecard_pit.backfill_available_at(tickers=universe, db=db) if universe else {"filled": 0}
        stored = store_prices(db, universe, as_of)

        raw_by_ticker: dict[str, dict[str, float | None]] = {}
        applicable_by_ticker: dict[str, frozenset[str]] = {}
        details: dict[str, dict[str, Any]] = {}
        per_ticker_hash: dict[str, str] = {}
        seen: set[str] = set()
        counts = {"pit_excluded": 0, "null_available_at": 0, "no_price": 0, "no_snapshot": 0,
                  "price_from_store": 0, "price_from_series": 0, "price_stale": 0}

        def _score_one(t: str, rows: list[tuple[Any, ...]]) -> None:
            m = by_ticker[t]
            snapshot = scorecard_features.pit_snapshot(rows, as_of)
            ctx, basis = price_context(t, as_of, store_price=stored.get(t), prefer_store=prefer_store,
                                       shares_fallback=m.shares_outstanding, today=today)
            res = scorecard_features.compute_features_detailed(snapshot, ctx, m.sector)
            raw_by_ticker[t] = res.values
            applicable_by_ticker[t] = res.applicable
            per_ticker_hash[t] = scorecard_features.inputs_hash(snapshot, ctx)
            counts["pit_excluded"] += snapshot.notes.get("pit_excluded", 0)
            counts["null_available_at"] += snapshot.notes.get("missing_available_at", 0)
            if ctx["price"] is None:
                counts["no_price"] += 1
            if snapshot.latest is None:
                counts["no_snapshot"] += 1
            if basis == "month_end_store":
                counts["price_from_store"] += 1
            elif basis == "daily_series":
                counts["price_from_series"] += 1
            # A price from an earlier month than `as_of` is a fallback the
            # row must own up to: counted here so cron-health can see a
            # month-end cross-section that was priced late.
            if ctx.get("price_stale"):
                counts["price_stale"] += 1
            latest = snapshot.latest
            details[t] = {
                "reasons": res.reasons,
                "context": {**res.context, "price_basis": basis, "price_stale": bool(ctx.get("price_stale"))},
                "latest_period": f"FY{latest.fiscal_year}" if latest is not None else "",
                "data_available_at": max((p.available_at for p in snapshot.points if p.available_at), default=None),
                "price_date": ctx["price_date"],
            }

        for t, rows in _iter_period_rows(db, universe):
            seen.add(t)
            _score_one(t, rows)
        for t in universe:
            if t not in seen:
                _score_one(t, [])   # no statements at all: every feature n/a with a reason

        run_hash = _run_inputs_hash(spec_hash, as_of, per_ticker_hash)
        previous = scorecard_queue.latest_succeeded_run(vk, as_of=as_of, db=db)
        if previous is not None and previous["as_of"] == as_of and previous["inputs_hash"] == run_hash:
            note = (f"as_of={as_of.isoformat()} version={vk} skipped=identical_inputs "
                    f"prior_run={previous['run_id']} universe={len(universe)} written=0")
            out = scorecard_queue.finish_run(
                run_row_id, status=scorecard_queue.STATUS_SKIPPED, note=note,
                universe_size=len(universe), scored_count=0, inputs_hash=run_hash,
                params_update={"is_month_end": month_end}, db=db,
            )
            db.commit()
            return out or {}

        norm = scorecard_normalize.normalize_universe(
            raw_by_ticker, {t: by_ticker[t].sector for t in universe}, applicable_by_ticker=applicable_by_ticker,
        )
        now = _utcnow()
        written = 0
        coverage_sum = 0.0
        for t in universe:
            r = norm.rows[t]
            d = details[t]
            pos, neg = scorecard_normalize.top_contributors(r.contributions, k=3)
            db.add(ScorecardScore(
                run_id=run_row_id, version_key=vk, as_of=as_of, ticker=t,
                sector=r.sector or (r.sector_raw or ""),
                overall_z=r.overall_z, overall_score=r.overall_score,
                universe_percentile=r.percentile_universe, sector_percentile=r.percentile_sector,
                coverage=r.coverage,
                category_z={**r.category_z, "_universe": r.category_z_universe, "_coverage": r.category_coverage,
                            "_profiles": r.profiles, "_notes": list(r.notes),
                            "_n": {"available": r.n_available, "applicable": r.n_applicable}},
                category_percentile={**r.category_percentile, "_sector": r.category_percentile_sector},
                feature_raw={**raw_by_ticker[t], "_reasons": d["reasons"], "_context": d["context"]},
                feature_z={**r.feature_z, "_basis": r.feature_basis, "_universe": r.feature_z_universe},
                top_positive=_contrib_list(pos, r.feature_z), top_negative=_contrib_list(neg, r.feature_z),
                latest_period=d["latest_period"], data_available_at=d["data_available_at"],
                price_date=d["price_date"], inputs_hash=per_ticker_hash[t],
                is_month_end=month_end, created_at=now,
            ))
            written += 1
            coverage_sum += r.coverage
        n = norm.notes
        note_parts = [
            f"as_of={as_of.isoformat()}", f"version={vk}", f"kind={run_kind}", f"universe={len(universe)}",
            f"scored={n['n_scored']}", f"insufficient={n['n_insufficient']}", f"written={written}",
            f"month_end={'1' if month_end else '0'}",
            f"mean_coverage={(coverage_sum / written):.3f}" if written else "mean_coverage=n/a",
            f"no_price={counts['no_price']}", f"no_snapshot={counts['no_snapshot']}",
            f"price_store={counts['price_from_store']}", f"price_series={counts['price_from_series']}",
            f"price_stale={counts['price_stale']}",
            f"pit_excluded={counts['pit_excluded']}", f"null_available_at={counts['null_available_at']}",
            f"available_at_filled={filled.get('filled', 0)}",
            f"sector_unmatched={sum(n['sectors_unmatched'].values())}",
            f"sector_fallback_rows={sum(n['sector_fallback_feature_rows'].values())}",
        ]
        out = scorecard_queue.finish_run(
            run_row_id, status=scorecard_queue.STATUS_SUCCEEDED, note=" ".join(note_parts),
            universe_size=len(universe), scored_count=n["n_scored"], inputs_hash=run_hash,
            params_update={"is_month_end": month_end, "spec_hash": spec_hash, "written": written,
                           "sectors_unmatched": n["sectors_unmatched"], "sector_counts": n["sector_counts"]},
            db=db,
        )
        db.commit()
        log.info("scorecard run %d succeeded: %s", run_row_id, " ".join(note_parts))
        return out or {}


# ---------------------------------------------------------------------------
# pit_prepare and retention
# ---------------------------------------------------------------------------

def pit_prepare(*, run_row_id: int | None = None, tickers: Iterable[str] | None = None) -> dict[str, Any]:
    """Backfill `available_at` and sync month-end prices for the universe.

    Per-ticker failures are recorded on the run row and never abort the
    run; a provider outage (`PriceSeriesUnavailable`) is a failed ticker,
    not a zero-month sync. The run fails only when every ticker failed.
    """
    explicit = None if tickers is None else [str(t).strip().upper() for t in tickers if str(t).strip()]
    if run_row_id is None:
        run_row_id = scorecard_queue.create_running_row(
            version_key=VERSION_KEY, as_of=_today(), kind=scorecard_queue.KIND_PIT_PREPARE,
            params={"tickers": explicit} if explicit is not None else {},
        )
    try:
        universe = [m.ticker for m in universe_members(tickers=explicit)]
        filled = scorecard_pit.backfill_available_at(tickers=universe) if universe else {"filled": 0, "unresolved": 0}
        months = written = failed = unavailable = 0
        failed_tickers: list[str] = []
        for t in universe:
            try:
                res = scorecard_pit.sync_price_month_ends(t)
                months += res["months"]
                written += res["written"]
            except scorecard_pit.PriceSeriesUnavailable as exc:
                failed += 1
                unavailable += 1
                failed_tickers.append(t)
                log.warning("pit_prepare: no price series for %s: %s", t, safe_exc(exc))
            except Exception as exc:
                failed += 1
                failed_tickers.append(t)
                log.warning("pit_prepare: price sync failed for %s: %s", t, safe_exc(exc))
        note = (f"tickers={len(universe)} available_at_filled={filled.get('filled', 0)} "
                f"available_at_unresolved={filled.get('unresolved', 0)} months={months} written={written} "
                f"failed={failed} unavailable={unavailable}")
        if failed_tickers:
            note += " failed_tickers=" + ",".join(failed_tickers[:50])
        all_failed = bool(universe) and failed == len(universe)
        return scorecard_queue.finish_run(
            run_row_id,
            status=scorecard_queue.STATUS_FAILED if all_failed else scorecard_queue.STATUS_SUCCEEDED,
            note=note, universe_size=len(universe),
            error_type="PriceSeriesUnavailable" if all_failed else "",
            error_message="no price series for any ticker in the universe" if all_failed else "",
            params_update={"failed_tickers": failed_tickers},
        ) or {}
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        return scorecard_queue.mark_failed(run_row_id, exc) or {}


def gc_daily_rows(*, today: date | None = None, retention_days: int | None = None) -> int:
    """Delete non-month-end score rows older than the retention window.
    Month-end rows are the evaluation's sample and are kept forever."""
    today = today or _today()
    days = int(settings.scorecard_daily_retention_days if retention_days is None else retention_days)
    cutoff = today - timedelta(days=days)
    with SessionLocal() as db:
        _ensure_tables(db)
        n = db.query(ScorecardScore).filter(
            ScorecardScore.is_month_end.is_(False), ScorecardScore.as_of < cutoff,
        ).delete(synchronize_session=False)
        db.commit()
    return int(n or 0)


# ---------------------------------------------------------------------------
# Readers (web process; DB only)
# ---------------------------------------------------------------------------

def _row_dict(row: ScorecardScore, run: ScorecardRun | None) -> dict[str, Any]:
    raw, raw_meta = _split_meta(row.feature_raw)
    fz, fz_meta = _split_meta(row.feature_z)
    cz, cz_meta = _split_meta(row.category_z)
    cp, cp_meta = _split_meta(row.category_percentile)
    return {
        "id": row.id, "run_row_id": row.run_id,
        "run_id": run.run_id if run is not None else "",
        "run_note": run.note if run is not None else "",
        "spec_hash": ((run.params or {}).get("spec_hash") if run is not None else None) or "",
        "generated_at": run.finished_at if run is not None else row.created_at,
        "version_key": row.version_key, "as_of": row.as_of, "ticker": row.ticker, "sector": row.sector or "",
        "overall_z": row.overall_z, "overall_score": row.overall_score,
        "universe_percentile": row.universe_percentile, "sector_percentile": row.sector_percentile,
        "coverage": row.coverage or 0.0,
        "category_z": cz, "category_z_universe": cz_meta.get("_universe") or {},
        "category_coverage": cz_meta.get("_coverage") or {}, "profiles": cz_meta.get("_profiles") or {},
        "notes": list(cz_meta.get("_notes") or []),
        "category_percentile": cp, "category_percentile_sector": cp_meta.get("_sector") or {},
        "feature_raw": raw, "reasons": raw_meta.get("_reasons") or {}, "context": raw_meta.get("_context") or {},
        "feature_z": fz, "feature_basis": fz_meta.get("_basis") or {}, "feature_z_universe": fz_meta.get("_universe") or {},
        "top_positive": list(row.top_positive or []), "top_negative": list(row.top_negative or []),
        "latest_period": row.latest_period or "", "data_available_at": row.data_available_at,
        "price_date": row.price_date, "inputs_hash": row.inputs_hash or "",
        "is_month_end": bool(row.is_month_end), "created_at": row.created_at,
    }


def _is_stale(as_of: date, today: date | None = None) -> bool:
    today = today or _today()
    return (today - as_of).days > int(settings.scorecard_daily_retention_days)


def latest_score(
    ticker: str, *, version_key: str | None = None, as_of: date | None = None, db: Session | None = None,
) -> dict[str, Any] | None:
    """The latest score row for `ticker` from a SUCCEEDED run (`as_of <=
    as_of` when given). None when nothing is on file — the memo then says
    n/a, never neutral."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        vk = version_key or active_version(db=db)["version_key"]
        q = (
            select(ScorecardScore, ScorecardRun)
            .join(ScorecardRun, ScorecardRun.id == ScorecardScore.run_id)
            .where(
                ScorecardScore.ticker == ticker.upper(), ScorecardScore.version_key == vk,
                ScorecardRun.status == scorecard_queue.STATUS_SUCCEEDED,
            )
            .order_by(ScorecardScore.as_of.desc(), ScorecardScore.id.desc())
        )
        if as_of is not None:
            q = q.where(ScorecardScore.as_of <= as_of)
        got = db.execute(q.limit(1)).first()
        if got is None:
            return None
        out = _row_dict(got[0], got[1])
        out["stale"] = _is_stale(out["as_of"])
        return out
    finally:
        if own:
            db.close()


def score_history(
    ticker: str, *, version_key: str, months: int = 36, before: date | None = None, db: Session | None = None,
) -> list[dict[str, Any]]:
    """Month-end rows from succeeded runs, oldest first, at most `months`."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        q = (
            select(ScorecardScore)
            .join(ScorecardRun, ScorecardRun.id == ScorecardScore.run_id)
            .where(
                ScorecardScore.ticker == ticker.upper(), ScorecardScore.version_key == version_key,
                ScorecardScore.is_month_end.is_(True), ScorecardRun.status == scorecard_queue.STATUS_SUCCEEDED,
            )
            .order_by(ScorecardScore.as_of.desc(), ScorecardScore.id.desc())
        )
        if before is not None:
            q = q.where(ScorecardScore.as_of <= before)
        rows = db.execute(q.limit(max(1, months) * 3)).scalars().all()
        seen: set[date] = set()
        points: list[dict[str, Any]] = []
        for r in rows:               # newest first; the first row per as_of is the latest run
            if r.as_of in seen:
                continue
            seen.add(r.as_of)
            cz, _meta = _split_meta(r.category_z)
            points.append({
                "as_of": r.as_of, "is_month_end": bool(r.is_month_end), "overall_z": r.overall_z,
                "overall_score": r.overall_score, "universe_percentile": r.universe_percentile,
                "sector_percentile": r.sector_percentile, "coverage": r.coverage or 0.0, "category_z": cz,
            })
            if len(points) >= months:
                break
        points.reverse()
        return points
    finally:
        if own:
            db.close()


def _categories(row: dict[str, Any]) -> dict[str, ScorecardCategory]:
    by_family = scorecard_spec.features_by_family()
    basis = row.get("feature_basis") or {}
    fz = row.get("feature_z") or {}
    out: dict[str, ScorecardCategory] = {}
    for fam in scorecard_spec.FAMILY_NAMES:
        feats = [f.name for f in by_family.get(fam, ())]
        applicable = [f for f in feats if basis.get(f, "") != f"{scorecard_normalize.BASIS_NA}:excluded"]
        available = [f for f in applicable if fz.get(f) is not None]
        z = (row.get("category_z") or {}).get(fam)
        out[fam] = ScorecardCategory(
            z=z, score=None if z is None else _z_to_100(z),
            percentile=(row.get("category_percentile") or {}).get(fam),
            sector_percentile=(row.get("category_percentile_sector") or {}).get(fam),
            weight=scorecard_spec.FAMILY_WEIGHTS[fam],
            coverage=(row.get("category_coverage") or {}).get(fam),
            n_features=len(applicable), n_available=len(available),
        )
    return out


def _contribs(items: list[dict[str, Any]]) -> list[ScorecardContribution]:
    out: list[ScorecardContribution] = []
    for it in items or []:
        try:
            out.append(ScorecardContribution(**{k: it.get(k) for k in ("feature", "family", "z", "contribution")}))
        except Exception:  # pragma: no cover — a malformed stored list must not 500 a page
            continue
    return out


def summary_from_row(row: dict[str, Any]) -> ScorecardSummary:
    """The memo's compact view of a score row (observed + model read)."""
    return ScorecardSummary(
        version_key=row["version_key"], as_of=row["as_of"], run_id=row.get("run_id") or "",
        overall_z=row.get("overall_z"), overall_score=row.get("overall_score"),
        universe_percentile=row.get("universe_percentile"), sector_percentile=row.get("sector_percentile"),
        coverage=row.get("coverage") or 0.0, categories=_categories(row),
        top_positive=_contribs(row.get("top_positive") or []), top_negative=_contribs(row.get("top_negative") or []),
        profiles=dict(row.get("profiles") or {}), latest_period=row.get("latest_period") or "",
        data_available_at=row.get("data_available_at"), price_date=row.get("price_date"),
        stale=bool(row.get("stale", _is_stale(row["as_of"]))), notes=list(row.get("notes") or []),
    )


def latest_summary(ticker: str, *, as_of: date | None = None, version_key: str | None = None) -> ScorecardSummary | None:
    """`ScorecardSummary` for the memo pipeline, or None when no row exists."""
    row = latest_score(ticker, version_key=version_key, as_of=as_of)
    return summary_from_row(row) if row is not None else None


def _feature_rows(row: dict[str, Any]) -> list[ScorecardFeatureOut]:
    fz = row.get("feature_z") or {}
    basis = row.get("feature_basis") or {}
    applicable = frozenset(
        f.name for f in scorecard_spec.FEATURE_SPEC
        if basis.get(f.name, "") != f"{scorecard_normalize.BASIS_NA}:excluded"
    )
    contributions = scorecard_normalize.composites(fz, applicable=applicable).contributions
    out: list[ScorecardFeatureOut] = []
    for f in scorecard_spec.FEATURE_SPEC:
        out.append(ScorecardFeatureOut(
            name=f.name, family=f.family, sign=f.sign, weight=f.weight, formula=f.formula,
            description=f.description, applicable=f.name in applicable,
            raw=_finite((row.get("feature_raw") or {}).get(f.name)),
            reason=(row.get("reasons") or {}).get(f.name),
            z=_finite(fz.get(f.name)), z_universe=_finite((row.get("feature_z_universe") or {}).get(f.name)),
            basis=str(basis.get(f.name, "")), contribution=contributions.get(f.name),
        ))
    return out


def ticker_detail(
    ticker: str, *, version_key: str | None = None, as_of: date | None = None, months: int = 36,
) -> ScorecardDetailOut | None:
    """Latest score with every feature's observed value and model read,
    plus the month-end history. Raises `UnknownVersion`."""
    with SessionLocal() as db:
        version = resolve_version(version_key, db=db)
        row = latest_score(ticker, version_key=version["version_key"], as_of=as_of, db=db)
        if row is None:
            return None
        company = db.get(Company, ticker.upper())
        history = score_history(ticker, version_key=version["version_key"], months=months, before=as_of, db=db)
    summary = summary_from_row(row)
    return ScorecardDetailOut(
        **summary.model_dump(),
        ticker=ticker.upper(), company_name=(company.company_name if company is not None else "") or "",
        sector=row["sector"], sector_raw=(row.get("context") or {}).get("sector_raw"),
        is_month_end=row["is_month_end"], spec_hash=row.get("spec_hash") or version["spec_hash"],
        inputs_hash=row.get("inputs_hash") or "", features=_feature_rows(row), context=dict(row.get("context") or {}),
        history=[ScorecardHistoryPoint(**p) for p in history],
    )


def _sort_key(sort_by: str):
    def key(r: dict[str, Any]) -> tuple[int, Any]:
        if sort_by == "ticker":
            return (0, r["ticker"])
        v = r.get(sort_by) if sort_by not in scorecard_spec.FAMILY_NAMES else (r.get("category_z") or {}).get(sort_by)
        return (1, 0.0) if v is None else (0, float(v))     # nulls last on both orders
    return key


def universe_table(
    *,
    version_key: str | None = None,
    as_of: date | None = None,
    sector: str | None = None,
    sort_by: str = "overall_score",
    order: str = "desc",
    limit: int = 200,
    min_coverage: float = 0.0,
) -> ScorecardUniverseOut | None:
    """The cross-section from the latest succeeded run (`as_of <= as_of`).
    None when no run exists. `sort_by` must be in `UNIVERSE_SORT_COLUMNS`."""
    if sort_by not in UNIVERSE_SORT_COLUMNS:
        raise ValueError(f"sort_by must be one of {sorted(UNIVERSE_SORT_COLUMNS)}")
    with SessionLocal() as db:
        version = resolve_version(version_key, db=db)
        run = scorecard_queue.latest_succeeded_run(version["version_key"], as_of=as_of, db=db)
        if run is None:
            return None
        rows = db.execute(
            select(ScorecardScore, Company.company_name)
            .outerjoin(Company, Company.ticker == ScorecardScore.ticker)
            .where(ScorecardScore.run_id == run["id"])
            .order_by(ScorecardScore.ticker)
        ).all()
    dicts: list[dict[str, Any]] = []
    for score, company_name in rows:
        d = _row_dict(score, None)
        d["company_name"] = company_name or ""
        dicts.append(d)
    if sector:
        needle = sector.strip().lower()
        dicts = [d for d in dicts if needle in (d["sector"] or "").lower()]
    if min_coverage:
        dicts = [d for d in dicts if (d["coverage"] or 0.0) >= min_coverage]
    reverse = order != "asc"
    nulls = [d for d in dicts if _sort_key(sort_by)(d)[0] == 1]
    valued = [d for d in dicts if _sort_key(sort_by)(d)[0] == 0]
    valued.sort(key=lambda d: _sort_key(sort_by)(d)[1], reverse=reverse)
    ordered = (valued + nulls)[: max(1, limit)]
    out_rows = [
        ScorecardUniverseRow(
            rank=i, ticker=d["ticker"], company_name=d["company_name"], sector=d["sector"],
            overall_z=d["overall_z"], overall_score=d["overall_score"],
            universe_percentile=d["universe_percentile"], sector_percentile=d["sector_percentile"],
            coverage=d["coverage"],
            category_score={fam: (None if z is None else _z_to_100(z)) for fam, z in (d["category_z"] or {}).items()},
            category_z=dict(d["category_z"] or {}),
            top_positive=_contribs(d["top_positive"]), top_negative=_contribs(d["top_negative"]),
            latest_period=d["latest_period"], notes=d["notes"],
        )
        for i, d in enumerate(ordered, start=1)
    ]
    return ScorecardUniverseOut(
        version_key=version["version_key"], spec_hash=(run["params"] or {}).get("spec_hash") or version["spec_hash"],
        as_of=run["as_of"], run_id=run["run_id"], is_month_end=bool((run["params"] or {}).get("is_month_end")),
        universe_size=run["universe_size"] or len(dicts), scored=run["scored_count"] or 0,
        insufficient=max(0, (run["universe_size"] or len(dicts)) - (run["scored_count"] or 0)),
        sort_by=sort_by, order="asc" if not reverse else "desc", stale=_is_stale(run["as_of"]), rows=out_rows,
    )


def spec_view(version_key: str | None = None) -> dict[str, Any]:
    """The methodology as served by `GET /api/scorecard/spec`."""
    version = resolve_version(version_key)
    return {"version_key": version["version_key"], "spec_hash": version["spec_hash"], "source": version["source"],
            "score_scale": "0-100, 50 = z of 0 (the sector or universe-fallback mean); percentiles are rank-based",
            **version["spec"]}


def evaluation_view(version_key: str | None = None, *, kind: str | None = None) -> ScorecardEvaluationOut:
    """Latest persisted evaluation per kind. Caveats are always attached,
    even when no evaluation has run, so a UI cannot render a bare number."""
    version = resolve_version(version_key)
    with SessionLocal() as db:
        _ensure_tables(db)
        q = (
            select(ScorecardEvaluation)
            .where(ScorecardEvaluation.version_key == version["version_key"])
            .order_by(ScorecardEvaluation.created_at.desc(), ScorecardEvaluation.id.desc())
        )
        if kind:
            q = q.where(ScorecardEvaluation.eval_kind == kind)
        rows = db.execute(q.limit(50)).scalars().all()
        items: dict[str, ScorecardEvaluationItem] = {}
        for r in rows:
            if r.eval_kind in items:
                continue
            items[r.eval_kind] = ScorecardEvaluationItem(
                kind=r.eval_kind, run_id=r.run_id or "", created_at=r.created_at, sample_start=r.sample_start,
                sample_end=r.sample_end, n_obs=r.n_obs or 0, params=dict(r.params or {}), result=dict(r.result or {}),
            )
    note = "" if items else "no evaluation has run for this version yet; nothing here is a result"
    if items:
        # An `insufficient` marker on its own reads as "broken"; the row
        # carries the reasons (a thin price store, a short sample), so the
        # view repeats them where the verdict is read.
        shortfalls = []
        for kind_name, item in items.items():
            res = item.result or {}
            insufficient = res.get("insufficient") is True or res.get("verdict") == "insufficient_data"
            if not insufficient:
                continue
            reasons = [str(r) for r in (res.get("reasons") or []) if r]
            if not reasons and res.get("stats_note"):
                reasons = [str(res["stats_note"])]
            shortfalls.append(f"{kind_name}: insufficient — " + ("; ".join(reasons) or "no reason recorded"))
        note = " | ".join(shortfalls)
    return ScorecardEvaluationOut(version_key=version["version_key"], evaluations=items,
                                  caveats=list(EVALUATION_CAVEATS), note=note)


# ---------------------------------------------------------------------------
# Export (contract v1, frozen)
# ---------------------------------------------------------------------------

def export_run(*, version_key: str | None = None, as_of: date | None = None) -> dict[str, Any] | None:
    """The run an export resolves to, with the version attached."""
    with SessionLocal() as db:
        version = resolve_version(version_key, db=db)
        run = scorecard_queue.latest_succeeded_run(version["version_key"], as_of=as_of, db=db)
    if run is None:
        return None
    run["spec_hash"] = (run["params"] or {}).get("spec_hash") or version["spec_hash"]
    return run


def _fmt_num(v: Any) -> str:
    f = _finite(v)
    return "" if f is None else f"{f:.6f}"


def _fmt_date(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, datetime):
        return v.replace(microsecond=0).isoformat()
    return v.isoformat() if isinstance(v, date) else str(v)


def _export_record(run: dict[str, Any], row: ScorecardScore) -> dict[str, Any]:
    """Native-typed record in `EXPORT_COLUMNS_V1` order (JSON form)."""
    cz, _ = _split_meta(row.category_z)
    cp, _ = _split_meta(row.category_percentile)
    pos = [it.get("feature", "") for it in (row.top_positive or [])][:3]
    neg = [it.get("feature", "") for it in (row.top_negative or [])][:3]
    pos += [""] * (3 - len(pos))
    neg += [""] * (3 - len(neg))
    rec: dict[str, Any] = {
        "contract_version": EXPORT_CONTRACT, "version_key": row.version_key, "spec_hash": run["spec_hash"],
        "run_id": run["run_id"], "as_of": _fmt_date(row.as_of), "price_date": _fmt_date(row.price_date),
        "data_available_at": _fmt_date(row.data_available_at), "ticker": row.ticker, "sector": row.sector or "",
        "coverage": _finite(row.coverage), "overall_z": _finite(row.overall_z),
        "overall_score": _finite(row.overall_score), "universe_percentile": _finite(row.universe_percentile),
        "sector_percentile": _finite(row.sector_percentile),
    }
    for fam in _EXPORT_FAMILY_ORDER:
        rec[f"z_{fam}"] = _finite(cz.get(fam))
    for fam in _EXPORT_FAMILY_ORDER:
        rec[f"pct_{fam}"] = _finite(cp.get(fam))
    for i in range(3):
        rec[f"top_positive_{i + 1}"] = pos[i]
    for i in range(3):
        rec[f"top_negative_{i + 1}"] = neg[i]
    rec["generated_at"] = _fmt_date(run.get("finished_at"))
    return {col: rec[col] for col in EXPORT_COLUMNS_V1}


def _iter_export_records(run: dict[str, Any], *, include_features: bool = False) -> Iterator[dict[str, Any]]:
    """Stream the run's rows sorted by ticker, `yield_per(200)`, one
    session held open for the duration of the response."""
    with SessionLocal() as db:
        _ensure_tables(db)
        stmt = (
            select(ScorecardScore)
            .where(ScorecardScore.run_id == run["id"])
            .order_by(ScorecardScore.ticker)
            .execution_options(yield_per=200)
        )
        for row in db.execute(stmt).scalars():
            rec = _export_record(run, row)
            if include_features:
                rec["feature_raw"], _ = _split_meta(row.feature_raw)
                rec["feature_z"], _ = _split_meta(row.feature_z)
            yield rec


def iter_export_csv(run: dict[str, Any]) -> Iterator[str]:
    """CSV text chunks: the frozen header, then one line per ticker. Nulls
    are empty cells, floats 6 dp, no thousands separators, `\\n` line ends."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(EXPORT_COLUMNS_V1)
    yield buf.getvalue()
    for rec in _iter_export_records(run):
        buf.seek(0)
        buf.truncate(0)
        writer.writerow([
            _fmt_num(rec[c]) if isinstance(rec[c], float) else ("" if rec[c] is None else str(rec[c]))
            for c in EXPORT_COLUMNS_V1
        ])
        yield buf.getvalue()


def iter_export_json(run: dict[str, Any], *, include_features: bool = False) -> Iterator[str]:
    """JSON text chunks with the same keys as the CSV columns (plus
    `feature_raw` / `feature_z` objects when asked), streamed row by row."""
    head = {
        "contract": EXPORT_CONTRACT, "version_key": run["version_key"], "spec_hash": run["spec_hash"],
        "run_id": run["run_id"], "as_of": _fmt_date(run["as_of"]), "generated_at": _fmt_date(run.get("finished_at")),
        "columns": list(EXPORT_COLUMNS_V1),
    }
    yield json.dumps(head, separators=(",", ":"))[:-1] + ',"rows":['
    first = True
    for rec in _iter_export_records(run, include_features=include_features):
        yield ("" if first else ",") + json.dumps(rec, separators=(",", ":"), allow_nan=False, default=str)
        first = False
    yield "]}"


__all__ = [
    "EXPORT_COLUMNS_V1",
    "EXPORT_CONTRACT",
    "UNIVERSE_SORT_COLUMNS",
    "VERSION_KEY",
    "Member",
    "UnknownVersion",
    "UnsupportedVersion",
    "active_version",
    "ensure_version_registered",
    "evaluation_view",
    "export_run",
    "gc_daily_rows",
    "is_month_end",
    "iter_export_csv",
    "iter_export_json",
    "latest_score",
    "latest_summary",
    "pit_prepare",
    "price_context",
    "resolve_version",
    "run_scorecard",
    "score_history",
    "spec_view",
    "store_prices",
    "summary_from_row",
    "ticker_detail",
    "universe_members",
    "universe_table",
]
