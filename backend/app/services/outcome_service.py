"""Wave 4A — realized-outcome scoring for memo recommendations.

For each `MemoSnapshot` we compute forward returns at 30 / 90 / 180 / 365
days post-`generated_at` vs. a benchmark (SPY) and persist the result as
a `MemoOutcome` row. The daily evaluator (`monitoring/outcome_loop.py`)
calls `evaluate_all_due()` to fold in any horizons that have come of age
since the last run; per-snapshot evaluation is idempotent on
`(memo_snapshot_id, horizon_days)`.

For the longer horizons (default 90d / 365d, configurable), the
evaluator also writes a reflection entry into the company's long-term
memory file so the next sector run on that ticker can read its own
track record. Shorter horizons stay numeric only — 30d returns are too
noisy to write prose about every quarter.

Backtest snapshots (`as_of_date` set) are skipped — outcomes only make
sense for live recommendations.

W6 / FIX-007: only snapshots the eligibility ledger marks eligible are
scored or counted (`services/outcome_eligibility.py`). Ineligible snapshots —
the 2026-05-04 dev-laptop demo copy, the migrated test fixtures (FIX-003),
post-fix demo memos — are counted by reason and skipped before any price
fetch, body read or reflection. Their existing outcome rows are kept, never
deleted or modified; the track record simply excludes them.
"""
from __future__ import annotations

import logging
import math
import statistics
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date as _date
from datetime import datetime, timedelta
from functools import cached_property
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import Company, MemoOutcome, MemoOutcomeEligibility, MemoSnapshot
from . import outcome_eligibility

log = logging.getLogger(__name__)


# Standard set of forward windows we evaluate. Order matters only for
# cosmetic logging — the DB key is `(memo_snapshot_id, horizon_days)` so
# adding/removing a horizon doesn't disturb prior rows.
DEFAULT_HORIZONS = (30, 90, 180, 365)

# Horizons that get a written reflection entry in long-term memory.
# 30d / 180d are recorded numerically only — too noisy / too redundant for prose.
REFLECTION_HORIZONS = {90, 365}

# Benchmark for alpha calculation. Override via `evaluate_all_due(benchmark=...)`.
DEFAULT_BENCHMARK = "SPY"


# --- Price-window sizing ---------------------------------------------------
#
# `get_price_series(ticker, days)` asks the provider for the last `days`
# bars *ending today* (FMP passes it as `limit=`, Polygon and Tiingo slice
# `[-days:]`). This is the adapter request, not a remote coverage guarantee.
# The window therefore slides forward every night while a
# memo's generation date stays put.
#
# The old sizing was `days = horizon_days + 30`, which is measured off the
# horizon — a quantity that says nothing about how long ago the memo was
# written. Once a memo aged past that many bars it fell out of the window
# permanently, and (worse) a partially-covering window silently supplied a
# baseline from weeks after the memo. Both failure modes are reproduced in
# `app/tests/test_outcome_price_window.py`, which fails against the old
# sizing in exactly the two ways production did.
#
# We size off memo age instead, and deliberately express the request in
# *calendar* days: providers read the number as bars, so a calendar-day
# count over-requests by ~40%. That is correct under either reading and
# costs nothing but payload.
#
# The number is part of the provider cache key (`"{TICKER}:{days}"`), so an
# exact per-memo value would rotate every night — it contains `today` — and
# multiply nightly provider calls by the number of distinct memo dates.
# Rounding up to a short ladder keeps the key stable across nights and
# collapses all four horizons of a snapshot onto a single fetch, which is
# fewer keys for each snapshot. Across many differently aged snapshots a
# ticker may still use all four rungs. Larger responses cost payload; their
# actual dates, not the requested size, determine whether scoring is safe.
PRICE_WINDOW_RUNGS = (120, 260, 400, 800)

# Slack on the memo side of the window so the rung is chosen from a date
# that is comfortably before the memo rather than exactly on it.
MEMO_WINDOW_BUFFER_DAYS = 7

# How far a close may sit from the date it stands in for. Seven calendar
# days allows weekends and multi-day closures while rejecting the large
# drift the unbounded fallback accepted. This is an operational tolerance,
# not an exchange calendar or a guarantee about all historical closures.
PRICE_DATE_TOLERANCE_DAYS = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_table(db: Session) -> None:
    bind = db.get_bind()
    MemoOutcome.__table__.create(bind=bind, checkfirst=True)
    outcome_eligibility.ensure_table(db)


def _window_days_for_memo(generated_date: _date, today: _date) -> int | None:
    """Smallest ladder rung whose fetch reaches back past `generated_date`.

    ``None`` when the memo needs the durable archive beyond these remote
    request rungs. Missing archived history can be repaired by backfill.
    """
    span = (today - generated_date).days + MEMO_WINDOW_BUFFER_DAYS
    for rung in PRICE_WINDOW_RUNGS:
        if rung >= span:
            return rung
    return None


def _dated_close(row: dict[str, Any]) -> tuple[str, float] | None:
    """``(iso_date, close)`` for a price row, or None if either is unusable."""
    d = str((row or {}).get("date") or "")
    if not d:
        return None
    try:
        _date.fromisoformat(d)
        value = row.get("close")
        if value is None:
            value = row.get("adjusted_close")
        close = float(value)
        return (d, close) if math.isfinite(close) and close > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _dated_close_on_or_before(
    rows: list[dict[str, Any]], target: str,
) -> tuple[str, float] | None:
    if not rows or not target:
        return None
    chosen: tuple[str, float] | None = None
    for r in rows:
        d = str(r.get("date") or "")
        if not d:
            continue
        if d <= target:
            hit = _dated_close(r)
            if hit is not None:
                chosen = hit
        else:
            break
    return chosen


def _dated_close_on_or_after(
    rows: list[dict[str, Any]], target: str,
) -> tuple[str, float] | None:
    if not rows or not target:
        return None
    for r in rows:
        d = str(r.get("date") or "")
        if d and d >= target:
            hit = _dated_close(r)
            if hit is not None:
                return hit
    return None


def _close_on_or_before(rows: list[dict[str, Any]], target: str) -> float | None:
    hit = _dated_close_on_or_before(rows, target)
    return hit[1] if hit else None


def _close_on_or_after(rows: list[dict[str, Any]], target: str) -> float | None:
    """First close on or after `target`, with no proximity check.

    Kept for callers that have already bounded the search. Scoring uses
    `_baseline_close`/`_target_close` instead — an unbounded "on or after"
    is what let a window beginning after the memo supply a baseline from
    weeks later.
    """
    hit = _dated_close_on_or_after(rows, target)
    return hit[1] if hit else None


def _within_tolerance(iso_date: str, target: _date) -> bool:
    try:
        return abs((_date.fromisoformat(iso_date) - target).days) <= PRICE_DATE_TOLERANCE_DAYS
    except ValueError:
        return False


def _baseline_close(
    rows: list[dict[str, Any]], memo_date: _date,
) -> tuple[str, float] | None:
    """The close standing in for the memo date, or None if none is near it.

    Preference order is unchanged (first close on or after the memo, then
    the last one before it), but each candidate now has to actually sit
    near the memo date. Without that check a window beginning after the
    memo made *every* row satisfy "on or after", so the earliest row in the
    window — a price from weeks or months later — was used as the memo's
    baseline and the resulting return was written as a real evaluation.
    """
    memo_iso = memo_date.isoformat()
    for hit in (
        _dated_close_on_or_after(rows, memo_iso),
        _dated_close_on_or_before(rows, memo_iso),
    ):
        if hit is not None and _within_tolerance(hit[0], memo_date):
            return hit
    return None


def _target_close(
    rows: list[dict[str, Any]], target_date: _date,
) -> tuple[str, float] | None:
    """The close standing in for the target date.

    Strictly on or before the target — never peek past the horizon — and
    near enough to it to be that day's price. The tolerance matters for a
    series that stops early (a halted or delisted ticker): the last bar
    before the halt would otherwise be scored as if it were the target
    day's close, turning a 90-day return into a 20-day one.
    """
    hit = _dated_close_on_or_before(rows, target_date.isoformat())
    if hit is not None and _within_tolerance(hit[0], target_date):
        return hit
    return None


def _close_on_exact_date(rows: list[dict[str, Any]], target: str) -> tuple[str, float] | None:
    hit = _dated_close_on_or_before(rows, target)
    return hit if hit is not None and hit[0] == target else None


def _thesis_held(rating: str, return_signed: float) -> bool | None:
    """Did the recommendation pay off?

    - Very Bullish / Bullish expect positive return → held if ≥ 0.
    - Very Bearish / Bearish expect negative return → held if ≤ 0.
    - Neutral has no directional bet — return None (no judgment).
    """
    r = (rating or "").strip().lower()
    # "very bullish" matches "bullish" too; "very bearish" matches "bearish";
    # plus the legacy "mixed positive/negative" labels for cached snapshots.
    if "bullish" in r or "mixed positive" in r:
        return return_signed >= 0
    if "bearish" in r or "mixed negative" in r:
        return return_signed <= 0
    return None  # Neutral / unknown


def _direction(rating: str) -> int | None:
    """+1 for a long call, -1 for a short call, None for no directional bet.

    Mirrors `_thesis_held` exactly (a test pins that they agree): the
    direction a rating bets on is what makes alpha "beat the benchmark".
    """
    r = (rating or "").strip().lower()
    if "bullish" in r or "mixed positive" in r:
        return 1
    if "bearish" in r or "mixed negative" in r:
        return -1
    return None


def is_late_evaluation(generated: datetime | None, evaluated: datetime | None, horizon_days: int) -> bool | None:
    """The outcome audit's late-evaluation triage predicate (FIX-007).

    ``evaluated - generated > (horizon + 30) * 7/5 days``, cross-multiplied so
    exact microsecond boundaries survive. None when either stamp is missing
    or the evaluation precedes generation. A candidate is not proof of a bad
    baseline, only a row a reader should know about.
    """
    if generated is None or evaluated is None or horizon_days <= 0:
        return None
    elapsed = evaluated - generated
    if elapsed < timedelta(0):
        return None
    return elapsed * 5 > timedelta(days=(horizon_days + 30) * 7)


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

def _evaluate_one(
    snap: MemoSnapshot, horizon_days: int,
    *, today: _date, benchmark: str,
    db: Session,
) -> tuple[MemoOutcome | None, str]:
    """Score a single ``(snapshot, horizon)`` and name every no-write path.

    The status is load-bearing observability.  Returning only ``None`` used
    to conflate harmless idempotency (future/already-recorded) with missing
    production price data.  The latter left hundreds of due outcomes
    unwritten while the cron job still reported success.

    The no-data statuses are deliberately distinct:

    ``ticker_prices_unavailable``
        neither requested provider history nor durable archive is available;
    ``price_history_too_short``
        the provider answered, but with fewer bars than were asked for, so
        the series begins after the memo — also an outage, just a partial
        one (a fallback leg truncating the series looks like this);
    ``price_window_incomplete``
        prices exist and reach the memo, but none sits near the target
        date — a gap a later run may still fill;
    All three describe repairable coverage shortfalls and drive the loop's
    failure flag. The legacy permanent-window counter is retained in the
    aggregate response for compatibility but old dates now use the archive.
    """
    # Backtest snapshots have `as_of_date` set; outcome scoring is for live memos only.
    if snap.as_of_date is not None:
        return None, "backtest"

    generated = snap.generated_at
    if isinstance(generated, datetime):
        generated_date = generated.date()
    else:
        generated_date = generated  # assume date-like
    target_date = generated_date + timedelta(days=horizon_days)
    if target_date > today:
        return None, "not_due"

    # Skip if we've already evaluated this (snapshot, horizon).
    existing = db.execute(
        select(MemoOutcome).where(
            MemoOutcome.memo_snapshot_id == snap.id,
            MemoOutcome.horizon_days == horizon_days,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return None, "already_recorded"

    # Size the window off the memo's age, not off the horizon: the fetch
    # ends today, so what it has to span is memo date → today.
    window_days = _window_days_for_memo(generated_date, today)
    if window_days is None:
        # A durable backfill can reach beyond the remote response ladder.
        # Never discard an otherwise evaluable old memo merely because its
        # required date is no longer in a provider's rolling cache window.
        from .price_history_service import read_prices
        start = generated_date - timedelta(days=MEMO_WINDOW_BUFFER_DAYS)
        ticker_rows = read_prices(snap.ticker, start=start, end=today)
        if not ticker_rows:
            return None, "ticker_prices_unavailable"
        bench_rows = read_prices(benchmark, start=start, end=today)
    else:
        from .market_data_service import get_price_series
        ticker_rows = get_price_series(snap.ticker, window_days) or []
        bench_rows = get_price_series(benchmark, window_days) or [] if ticker_rows else []
    if not ticker_rows:
        return None, "ticker_prices_unavailable"
    # Same rung for the benchmark: one cache key, and both legs of alpha
    # measured over the same span.

    memo_hit = _baseline_close(ticker_rows, generated_date)
    if memo_hit is None:
        # The rows that came back begin after the memo, so refusing to
        # score is the only honest answer — but that is a statement about
        # the response, not about the memo, and it must not be filed under
        # the permanent bucket that deliberately keeps the loop green.
        #
        # A complete daily series over the requested span reaches the
        # memo, but row count alone proves nothing: duplicate dates, gaps,
        # and invalid prices can make even a full-length response unusable.
        # Truncated fallback responses used to include Tiingo's latest-only
        # request (now fixed with explicit dates). These remain provider
        # shortfalls an operator can investigate, not a permanent memo age.
        return None, "price_history_too_short"
    baseline_date, price_at_memo = memo_hit

    target_hit = _target_close(ticker_rows, target_date)
    if target_hit is None or price_at_memo <= 0:
        return None, "price_window_incomplete"
    price_at_target = target_hit[1]

    forward_return = (price_at_target - price_at_memo) / price_at_memo

    # Alpha compares identical holding periods. Choosing the benchmark
    # independently within a tolerance can subtract returns from different
    # sessions when either tape has missing bars or a trading halt.
    bench_memo_hit = _close_on_exact_date(bench_rows, baseline_date)
    bench_target_hit = _close_on_exact_date(bench_rows, target_hit[0])
    bench_return: float | None = None
    alpha: float | None = None
    if bench_memo_hit and bench_target_hit and bench_memo_hit[1] > 0:
        bench_return = (bench_target_hit[1] - bench_memo_hit[1]) / bench_memo_hit[1]
        alpha = forward_return - bench_return

    memo_dict = snap.memo_json or {}
    rating = memo_dict.get("rating_label") or ""
    confidence = float(memo_dict.get("confidence_score") or 0.0)
    held = _thesis_held(rating, forward_return)
    # Wave 10 — copy the macro regime that was active at memo creation
    # so calibration's regime-conditional dashboards can bucket without
    # re-reading the snapshot blob.
    regime_at_memo = (
        str(memo_dict.get("macro_regime_at_memo") or "").strip()
        or None
    )

    # `baseline=` records which close the return was measured from, so a
    # future audit can tell a genuine memo-date baseline from a drifted one
    # without re-fetching prices.
    note_parts: list[str] = [
        f"horizon={horizon_days}d",
        f"baseline={baseline_date}",
        f"target={target_hit[0]}",
        f"price_window={window_days if window_days is not None else 'durable_history'}",
        f"return={forward_return:+.2%}",
    ]
    for label, rows in (("price", ticker_rows), ("benchmark_price", bench_rows)):
        provenance = {(str(row.get("source")), str(row.get("close_basis"))) for row in rows if row.get("source")}
        if provenance:
            note_parts.append(f"{label}_source=" + ";".join(f"{source}:{basis}" for source, basis in sorted(provenance)))
    if alpha is not None:
        note_parts.extend([
            f"benchmark={benchmark}",
            f"benchmark_baseline={bench_memo_hit[0]}",
            f"benchmark_target={bench_target_hit[0]}",
        ])
        note_parts.append(f"alpha={alpha:+.2%}")
    else:
        note_parts.extend([
            f"benchmark={benchmark}",
            "alpha_unavailable=benchmark_missing_exact_dates",
            f"benchmark_required_baseline={baseline_date}",
            f"benchmark_required_target={target_hit[0]}",
        ])
    if held is not None:
        note_parts.append("thesis_held" if held else "thesis_broken")
    note = ", ".join(note_parts)

    row = MemoOutcome(
        memo_snapshot_id=snap.id,
        ticker=snap.ticker,
        rating_at_memo=rating,
        confidence_at_memo=confidence,
        price_at_memo=price_at_memo,
        horizon_days=horizon_days,
        evaluated_at=datetime.utcnow(),
        forward_return=forward_return,
        benchmark_return=bench_return,
        alpha=alpha,
        thesis_held=held,
        note=note,
        regime_at_memo=regime_at_memo,
    )
    db.add(row)
    return row, "written"


def _maybe_write_reflection(
    snap: MemoSnapshot, outcome: MemoOutcome,
) -> bool:
    """For long horizons, append an outcome entry to the company memory file."""
    if outcome.horizon_days not in REFLECTION_HORIZONS:
        return False
    try:
        from ..config import settings
        from ..memory import CompanyMemory, MemoryEntry
        if not settings.enable_long_term_memory:
            return False
        cm = CompanyMemory.for_ticker(snap.ticker)
        body_parts: list[str] = []
        body_parts.append(
            f"**At memo time (v{snap.version}):** rating={outcome.rating_at_memo}, "
            f"confidence={int(outcome.confidence_at_memo)}, "
            f"price=${outcome.price_at_memo:,.2f}." if outcome.price_at_memo else ""
        )
        body_parts.append(
            f"**{outcome.horizon_days}-day forward return:** "
            f"{outcome.forward_return:+.2%}"
            + (f" (alpha vs SPY: {outcome.alpha:+.2%})" if outcome.alpha is not None else "")
            + "."
        )
        verdict = (
            "thesis HELD." if outcome.thesis_held is True
            else "thesis BROKEN." if outcome.thesis_held is False
            else "neutral call — no directional verdict."
        )
        body_parts.append(f"**Verdict:** {verdict}")
        cm.append_entry(MemoryEntry(
            date=_date.today().isoformat(),
            trigger=f"outcome:{outcome.horizon_days}d",
            body="\n\n".join(p for p in body_parts if p),
        ))
        cm.save()
        return True
    except Exception as exc:  # pragma: no cover — diagnostic only
        log.warning("Outcome reflection write failed for %s/%sd: %s",
                    snap.ticker, outcome.horizon_days, exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class _OutcomeSnapshot:
    """Small immutable-snapshot view, with the body loaded only for scoring."""

    id: int
    ticker: str
    version: int
    generated_at: datetime
    as_of_date: datetime | None
    # Identity-valid ledger answer; None when the snapshot is unclassified.
    eligible: bool | None
    eligibility_reason: str | None
    db: Session = field(repr=False)

    @cached_property
    def memo_json(self) -> Any:
        # _evaluate_one reaches this only after its date, idempotency and
        # price-coverage checks. Do not fetch the unused revision log.
        return self.db.execute(
            select(MemoSnapshot.memo_json).where(MemoSnapshot.id == self.id)
        ).scalar_one()


def _iter_outcome_snapshots(db: Session) -> Iterator[_OutcomeSnapshot]:
    """Page metadata without keeping a server cursor open across commits.

    Outcomes commit per pair and roll back a failed pair. A streamed cursor
    on that same PostgreSQL transaction would be invalidated by either, so
    consume each small page before yielding candidates. The initial ID fence
    excludes snapshots inserted during the pass, as the old single query did.
    """
    max_id = db.execute(select(func.max(MemoSnapshot.id))).scalar_one()
    if max_id is None:
        return
    cursor: tuple[datetime, int] | None = None
    while True:
        stmt = select(
            MemoSnapshot.id, MemoSnapshot.ticker, MemoSnapshot.version,
            MemoSnapshot.generated_at, MemoSnapshot.as_of_date,
            MemoOutcomeEligibility.eligible, MemoOutcomeEligibility.reason,
        ).outerjoin(
            MemoOutcomeEligibility, outcome_eligibility.identity_matches(),
        ).where(
            MemoSnapshot.as_of_date.is_(None),
            MemoSnapshot.id <= max_id,
        )
        if cursor is not None:
            generated_at, snapshot_id = cursor
            stmt = stmt.where(or_(
                MemoSnapshot.generated_at > generated_at,
                and_(
                    MemoSnapshot.generated_at == generated_at,
                    MemoSnapshot.id > snapshot_id,
                ),
            ))
        page = db.execute(
            stmt.order_by(MemoSnapshot.generated_at.asc(), MemoSnapshot.id.asc()).limit(100)
        ).all()
        if not page:
            return
        for row in page:
            yield _OutcomeSnapshot(*row, db=db)
        cursor = (page[-1].generated_at, page[-1].id)


def evaluate_all_due(
    *, horizons: list[int] | None = None,
    benchmark: str = DEFAULT_BENCHMARK,
    today: _date | None = None,
    db: Session | None = None,
) -> dict[str, Any]:
    """Score every (snapshot, horizon) that has come of age and isn't
    already in `memo_outcomes`. Idempotent: re-running on the same day
    yields zero new rows once everything's been scored.

    ``evaluated`` retains its historical meaning (all snapshot/horizon pairs
    scanned).  ``due`` and ``data_unavailable`` distinguish work that should
    have produced a row from harmless future/idempotent skips, and
    ``unevaluable`` retains the legacy date-window status for compatibility;
    coverage classification remains delegated to ``_evaluate_one``.

    W6: the eligibility sweep runs first. Its failure (for example an
    ``ExclusionSetMismatch``) rolls the sweep back and is reported as
    ``classification_error``, which fails the loop, but scoring carries on
    over the ledger rows that already exist: one bad new snapshot must not
    stop every eligible memo from being scored until a redeploy. Readers
    already treat an unclassified snapshot as ineligible, so continuing is
    still fail-closed. Snapshots the ledger marks ineligible are counted, not
    evaluated: ``ineligible`` counts their pairs and
    ``ineligible_snapshots_by_reason`` their snapshots, and neither enters
    ``due`` or ``data_unavailable`` (FIX-003: the fixture snapshots stop
    driving the "left N due rows pending" error). ``unclassified`` counts only
    DUE pairs of snapshots that still have no ledger row after an inline
    sweep — a snapshot that arrives mid-run is classified before it is
    skipped, and one with nothing due is not a failure. Nothing is deleted.
    """
    today = today or _date.today()
    horizons = list(horizons or DEFAULT_HORIZONS)
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        classification_error: str | None = None
        try:
            classification: dict[str, Any] = outcome_eligibility.classify_pending(db=db)
        except Exception as exc:
            # classify_pending has rolled back and logged the named row.
            classification_error = f"{type(exc).__name__}: {exc}"[:500]
            classification = {}
        from .memory_probe import log_rss
        log_rss("outcome_scan_start")
        evaluated = 0
        written = 0
        reflections = 0
        errors = 0
        unevaluable_pairs: list[str] = []
        unavailable_pairs: list[str] = []
        error_pairs: list[str] = []
        statuses: dict[str, int] = {
            "backtest": 0,
            "not_due": 0,
            "already_recorded": 0,
            "ticker_prices_unavailable": 0,
            "price_history_too_short": 0,
            "price_window_incomplete": 0,
            "memo_predates_price_window": 0,
            "ineligible": 0,
            "unclassified": 0,
        }
        ineligible_by_reason: Counter[str] = Counter()
        unclassified_ids: list[int] = []
        for snap in _iter_outcome_snapshots(db):
            if snap.eligible is None and classification_error is None:
                # Inserted after the sweep (the id fence is taken later), or
                # its ledger row belongs to a reused sqlite id. Classify now,
                # before deciding anything about it. Not after a failed
                # sweep: it would fail again, once per unclassified snapshot.
                try:
                    outcome_eligibility.classify_pending(db=db)
                except Exception as exc:
                    classification_error = f"{type(exc).__name__}: {exc}"[:500]
                found = outcome_eligibility.lookup(db, snap.id)
                if found is not None:
                    snap.eligible, snap.eligibility_reason = found.eligible, found.reason
            if snap.eligible is not True:
                # Before `_evaluate_one`: no price fetch, no body, no reflection.
                evaluated += len(horizons)
                if snap.eligible is None:
                    memo_date = snap.generated_at.date()
                    due_now = sum(1 for h in horizons if memo_date + timedelta(days=h) <= today)
                    statuses["unclassified"] += due_now
                    statuses["not_due"] += len(horizons) - due_now
                    if due_now:
                        unclassified_ids.append(snap.id)
                else:
                    statuses["ineligible"] += len(horizons)
                    ineligible_by_reason[snap.eligibility_reason or "unknown"] += 1
                continue
            for h in horizons:
                evaluated += 1
                try:
                    out, status = _evaluate_one(
                        snap, h, today=today, benchmark=benchmark, db=db,
                    )
                except Exception as exc:  # pragma: no cover — defensive
                    errors += 1
                    error_pairs.append(f"{snap.ticker}:snap={snap.id}:{h}d:{type(exc).__name__}")
                    # A database exception (for example schema drift) leaves
                    # PostgreSQL's transaction aborted.  Roll it back so one
                    # bad pair does not turn every later pair into a cascade.
                    db.rollback()
                    log.warning(
                        "Outcome evaluation failed for snap=%s h=%sd: %s",
                        snap.id, h, exc,
                    )
                    continue
                if status != "written":
                    statuses[status] = statuses.get(status, 0) + 1
                if status in {"ticker_prices_unavailable", "price_history_too_short", "price_window_incomplete"}:
                    unavailable_pairs.append(f"{snap.ticker}:snap={snap.id}:{h}d:{status}")
                if status == "memo_predates_price_window":
                    unevaluable_pairs.append(f"{snap.ticker}:snap={snap.id}:{h}d")
                if out is not None:
                    written += 1
                    db.commit()
                    if h in REFLECTION_HORIZONS:
                        try:
                            if _maybe_write_reflection(snap, out):
                                reflections += 1
                        except Exception:  # pragma: no cover
                            pass
        # `data_unavailable` is the *actionable* shortfall — work that
        # should have produced a row and would produce one on a later run
        # if the data arrived. It drives the loop's success flag.
        data_unavailable = (
            statuses["ticker_prices_unavailable"]
            + statuses["price_history_too_short"]
            + statuses["price_window_incomplete"]
        )
        # Preserve the legacy status fields without duplicating date-window
        # policy here. The evaluator can classify archive gaps as repairable
        # data_unavailable even when their dates exceed remote request rungs.
        unevaluable = statuses["memo_predates_price_window"]
        due = (
            written + statuses["already_recorded"]
            + data_unavailable + unevaluable + errors
        )
        if data_unavailable:
            log.error(
                "Outcome evaluation left %s due rows pending: "
                "ticker_prices_unavailable=%s price_history_too_short=%s "
                "price_window_incomplete=%s pairs=%s",
                data_unavailable,
                statuses["ticker_prices_unavailable"],
                statuses["price_history_too_short"],
                statuses["price_window_incomplete"], ",".join(unavailable_pairs),
            )
        if errors:
            log.error("Outcome evaluation failed for %s pairs: %s", errors, ",".join(error_pairs))
        if unclassified_ids:
            log.error(
                "Outcome evaluation found %s due pairs on %s unclassified snapshots "
                "(eligibility sweep did not cover them): ids=%s",
                statuses["unclassified"], len(unclassified_ids),
                ",".join(str(i) for i in unclassified_ids),
            )
        if classification_error:
            log.error(
                "Outcome evaluation scored only already-classified snapshots: "
                "eligibility sweep failed: %s", classification_error,
            )
        if unevaluable:
            log.warning(
                "Outcome evaluation returned %s legacy unevaluable pairs "
                "(memo_predates_price_window). pairs=%s",
                unevaluable, ",".join(unevaluable_pairs),
            )
        log_rss("outcome_scan_end", evaluated=evaluated, written=written, errors=errors)
        return {
            "evaluated": evaluated, "written": written,
            "reflections": reflections, "errors": errors,
            "due": due,
            "already_recorded": statuses["already_recorded"],
            "not_due": statuses["not_due"],
            "data_unavailable": data_unavailable,
            "ticker_prices_unavailable": statuses["ticker_prices_unavailable"],
            "price_history_too_short": statuses["price_history_too_short"],
            "price_window_incomplete": statuses["price_window_incomplete"],
            "unevaluable": unevaluable,
            "memo_predates_price_window": statuses["memo_predates_price_window"],
            "unevaluable_pairs": unevaluable_pairs,
            "unavailable_pairs": unavailable_pairs,
            "error_pairs": error_pairs,
            "ineligible": statuses["ineligible"],
            "ineligible_snapshots_by_reason": dict(sorted(ineligible_by_reason.items())),
            "unclassified": statuses["unclassified"],
            "unclassified_snapshot_ids": unclassified_ids,
            "classification": classification,
            "classification_error": classification_error,
        }
    finally:
        if own:
            db.close()


def get_outcomes_for_snapshot(
    memo_snapshot_id: int, *, db: Session | None = None,
) -> list[dict[str, Any]]:
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        rows = db.execute(
            select(MemoOutcome)
            .where(MemoOutcome.memo_snapshot_id == memo_snapshot_id)
            .order_by(MemoOutcome.horizon_days.asc())
        ).scalars().all()
        return [
            {
                "memo_snapshot_id": r.memo_snapshot_id,
                "ticker": r.ticker,
                "horizon_days": r.horizon_days,
                "rating_at_memo": r.rating_at_memo,
                "confidence_at_memo": r.confidence_at_memo,
                "price_at_memo": r.price_at_memo,
                "forward_return": r.forward_return,
                "benchmark_return": r.benchmark_return,
                "alpha": r.alpha,
                "thesis_held": r.thesis_held,
                "evaluated_at": r.evaluated_at.isoformat(),
                "note": r.note,
                "regime_at_memo": r.regime_at_memo,
            }
            for r in rows
        ]
    finally:
        if own:
            db.close()


# Provisional banner thresholds (owner default, W6 design §8 Q2): the record
# stays provisional until BOTH hold at the selected horizon. 30 companies is
# about a sixth of the universe; several memos on one company are not
# independent calls, which is why companies are counted separately.
PROVISIONAL_MIN_COMPANIES = 30
PROVISIONAL_MIN_DIRECTIONAL = 100


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _rate(hits: int, n: int) -> float | None:
    return hits / n if n else None


@dataclass(frozen=True)
class _TRRow:
    snapshot_id: int
    ticker: str
    horizon_days: int
    rating: str
    forward_return: float | None
    alpha: float | None
    thesis_held: bool | None
    evaluated_at: datetime | None
    eligible: bool | None
    reason: str | None
    sector: str | None
    rating_source: str | None
    snapshot_generated_at: datetime | None


def _universe_companies(db: Session) -> int:
    # The same `companies` table `GET /api/stocks` lists, without ETFs.
    return int(db.execute(
        select(func.count()).select_from(Company).where(
            func.coalesce(Company.is_etf, False).is_(False),
        )
    ).scalar_one())


def track_record(
    *, ticker: str | None = None, sector: str | None = None,
    horizon_days: int = 90, db: Session | None = None,
) -> dict[str, Any]:
    """Aggregate track-record stats over ELIGIBLE evaluated outcomes (W6).

    Filters: `ticker` (single name), `sector` (from the eligibility ledger,
    never from memo bodies), `horizon_days` (which forward window to look
    at). The historical keys (`total`, `thesis_hit_rate`, `avg_alpha`, ...)
    are computed over eligible rows only. Rows on ineligible snapshots are
    excluded, not deleted: they stay in `memo_outcomes` and are counted by
    reason in `eligibility`. `thesis_hit_rate` is an absolute-return measure,
    so it is published beside SPY-relative `alpha`, the always-Bullish
    `base_rate` over the same directional rows, `coverage`, the rating mix by
    who produced the rating, and a `provisional` verdict.
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        led = MemoOutcomeEligibility
        # Small columns only, every horizon (coverage needs them all). The
        # snapshot join reads only id/ticker/generated_at for the ledger's
        # identity guard; no memo body is touched.
        stmt = select(
            MemoOutcome.memo_snapshot_id, MemoOutcome.ticker, MemoOutcome.horizon_days,
            MemoOutcome.rating_at_memo, MemoOutcome.forward_return, MemoOutcome.alpha,
            MemoOutcome.thesis_held, MemoOutcome.evaluated_at,
            led.eligible, led.reason, led.sector, led.rating_source, led.snapshot_generated_at,
        ).outerjoin(
            MemoSnapshot, MemoSnapshot.id == MemoOutcome.memo_snapshot_id,
        ).outerjoin(
            led, outcome_eligibility.identity_matches(led, MemoSnapshot),
        )
        if ticker:
            stmt = stmt.where(MemoOutcome.ticker == ticker.upper())
        rows = [_TRRow(*r) for r in db.execute(stmt).all()]
        if sector:
            wanted = sector.strip().lower()
            rows = [r for r in rows if (r.sector or "").strip().lower() == wanted]
        universe = _universe_companies(db)

        eligible_rows = [r for r in rows if r.eligible is True]
        sel_all = [r for r in rows if r.horizon_days == horizon_days]
        sel = [r for r in eligible_rows if r.horizon_days == horizon_days]

        total = len(sel)
        directional = [r for r in sel if r.thesis_held is not None]
        held = sum(1 for r in directional if r.thesis_held)
        returns = [r.forward_return for r in sel if r.forward_return is not None]
        alphas = [r.alpha for r in sel if r.alpha is not None]

        # Direction-adjusted alpha: a Bearish call that trailed SPY beat it.
        adjusted: list[tuple[str, float]] = []
        for r in sel:
            d = _direction(r.rating)
            if d is not None and r.alpha is not None:
                adjusted.append((r.ticker, d * r.alpha))
        per_company: dict[str, list[float]] = {}
        for t, a in adjusted:
            per_company.setdefault(t, []).append(a)
        # One vote per company damps overlapping memos on one name.
        company_means = [statistics.fmean(v) for v in per_company.values()]

        # Base rate over the SAME directional rows as thesis_hit_rate: what
        # labelling every one of them Bullish would have scored. A Neutral
        # row is not a call, so it is in neither denominator.
        bull_base = [r.forward_return for r in directional if r.forward_return is not None]

        rating_mix: Counter[str] = Counter()
        mix_by_source: dict[str, Counter[str]] = {}
        for r in sel:
            label = (r.rating or "").strip() or "Unrated"
            source = r.rating_source or outcome_eligibility.SOURCE_UNKNOWN
            rating_mix[label] += 1
            mix_by_source.setdefault(source, Counter())[label] += 1

        horizons = sorted(set(DEFAULT_HORIZONS) | {horizon_days})
        coverage_rows: list[dict[str, Any]] = []
        for h in horizons:
            hr = [r for r in eligible_rows if r.horizon_days == h]
            companies_h = len({r.ticker for r in hr})
            coverage_rows.append({
                "horizon_days": h,
                # One outcome per (snapshot, horizon), so memos == evaluations.
                "memos": len({r.snapshot_id for r in hr}),
                "companies": companies_h,
                "directional": sum(1 for r in hr if r.thesis_held is not None),
                "universe_pct": (companies_h / universe) if universe else None,
                "late_evaluation_candidates": sum(
                    1 for r in hr
                    if is_late_evaluation(r.snapshot_generated_at, r.evaluated_at, h) is True
                ),
            })
        sel_coverage = next(c for c in coverage_rows if c["horizon_days"] == horizon_days)

        excluded_by_reason = Counter(r.reason or "unknown" for r in sel_all if r.eligible is False)
        eligible_by_reason = Counter(r.reason or "unknown" for r in sel)
        unclassified = sum(1 for r in sel_all if r.eligible is None)

        companies = int(sel_coverage["companies"])
        provisional_reasons = []
        if companies < PROVISIONAL_MIN_COMPANIES:
            provisional_reasons.append("companies_below_threshold")
        if len(directional) < PROVISIONAL_MIN_DIRECTIONAL:
            provisional_reasons.append("directional_below_threshold")

        return {
            "horizon_days": horizon_days,
            "total": total,
            "directional_evaluations": len(directional),
            "thesis_hit_rate": _rate(held, len(directional)),
            # Mean over rows that have a return (every written row does);
            # 0.0 on an empty selection, as before, for the existing client.
            "avg_forward_return": _mean(returns) if returns else 0.0,
            "avg_alpha": _mean(alphas),
            "ticker_filter": ticker,
            "sector_filter": sector,
            "benchmark": DEFAULT_BENCHMARK,
            "alpha": {
                "n": len(alphas),
                "unavailable": total - len(alphas),
                "mean": _mean(alphas),
                "median": _median(alphas),
                "directional_n": len(adjusted),
                "directional_median": _median([a for _, a in adjusted]),
                "beat_benchmark_rate": _rate(sum(1 for _, a in adjusted if a > 0), len(adjusted)),
                "company_weighted_median": _median(company_means),
            },
            "base_rate": {
                "always_bullish_hit_rate": _rate(sum(1 for v in bull_base if v >= 0), len(bull_base)),
                "always_bullish_n": len(bull_base),
                "positive_alpha_rate": _rate(sum(1 for a in alphas if a > 0), len(alphas)),
                "positive_alpha_n": len(alphas),
            },
            "rating_mix": dict(sorted(rating_mix.items())),
            "rating_mix_by_source": {
                source: dict(sorted(mix.items())) for source, mix in sorted(mix_by_source.items())
            },
            "coverage": {
                "universe_companies": universe,
                "memos_any_horizon": len({r.snapshot_id for r in eligible_rows}),
                "companies_any_horizon": len({r.ticker for r in eligible_rows}),
                "late_evaluation_candidates": sel_coverage["late_evaluation_candidates"],
                "horizons": coverage_rows,
            },
            "eligibility": {
                "rule_version": outcome_eligibility.RULE_VERSION,
                "eligible": total,
                "excluded": sum(excluded_by_reason.values()),
                "unclassified": unclassified,
                "excluded_by_reason": dict(sorted(excluded_by_reason.items())),
                "eligible_by_reason": dict(sorted(eligible_by_reason.items())),
            },
            "provisional": {
                "is_provisional": bool(provisional_reasons),
                "reasons": provisional_reasons,
                "min_companies": PROVISIONAL_MIN_COMPANIES,
                "min_directional": PROVISIONAL_MIN_DIRECTIONAL,
                "companies": companies,
                "directional": len(directional),
            },
        }
    finally:
        if own:
            db.close()
