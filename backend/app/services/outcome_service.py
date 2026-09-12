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
"""
from __future__ import annotations

import logging
from datetime import date as _date
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import MemoOutcome, MemoSnapshot

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
# `[-days:]`). The window therefore slides forward every night while a
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
# strictly fewer provider calls than the four the old sizing made per
# ticker. Slack costs nothing here: over-fetching is free, and correctness
# is guaranteed by the proximity tolerance below, not by exact sizing.
PRICE_WINDOW_RUNGS = (120, 260, 400, 800)

# Slack on the memo side of the window so the rung is chosen from a date
# that is comfortably before the memo rather than exactly on it.
MEMO_WINDOW_BUFFER_DAYS = 7

# How far a close may sit from the date it stands in for. Seven calendar
# days clears every US market closure on record — the longest (Sept 2001)
# left seven days between consecutive sessions — while being far tighter
# than the weeks-to-months drift the unbounded fallback used to accept.
# It is also the boundary of the grey band in the triage predicate used to
# audit rows written before this fix, so the two agree.
PRICE_DATE_TOLERANCE_DAYS = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_table(db: Session) -> None:
    bind = db.get_bind()
    MemoOutcome.__table__.create(bind=bind, checkfirst=True)


def _window_days_for_memo(generated_date: _date, today: _date) -> int | None:
    """Smallest ladder rung whose fetch reaches back past `generated_date`.

    ``None`` when the memo is older than the longest rung — a permanent
    condition, since tomorrow's window starts a day later still.
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
        return d, float(row.get("close") or row.get("adjusted_close"))
    except (TypeError, ValueError):
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

    The three no-data statuses are deliberately distinct:

    ``ticker_prices_unavailable``
        the provider returned nothing for this ticker — an outage;
    ``price_window_incomplete``
        prices exist and reach the memo, but none sits near the target
        date — a gap a later run may still fill;
    ``memo_predates_price_window``
        no price history we can obtain reaches the memo's own date, so the
        pair can never be scored and never will be.
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
        return None, "memo_predates_price_window"

    from .market_data_service import get_price_series
    ticker_rows = get_price_series(snap.ticker, window_days) or []
    if not ticker_rows:
        return None, "ticker_prices_unavailable"
    # Same rung for the benchmark: one cache key, and both legs of alpha
    # measured over the same span.
    bench_rows = get_price_series(benchmark, window_days) or []

    memo_hit = _baseline_close(ticker_rows, generated_date)
    if memo_hit is None:
        # The oldest price we can obtain postdates the memo. Refusing to
        # score is the only honest answer, and it is terminal: tomorrow's
        # window begins a day later still.
        return None, "memo_predates_price_window"
    baseline_date, price_at_memo = memo_hit

    target_hit = _target_close(ticker_rows, target_date)
    if target_hit is None or price_at_memo <= 0:
        return None, "price_window_incomplete"
    price_at_target = target_hit[1]

    forward_return = (price_at_target - price_at_memo) / price_at_memo

    # Benchmark-relative alpha (None if the benchmark's own baseline or
    # target close isn't available at the same dates — a shifted benchmark
    # baseline corrupts alpha exactly the way a shifted ticker baseline
    # corrupts the return).
    bench_memo_hit = _baseline_close(bench_rows, generated_date)
    bench_target_hit = _target_close(bench_rows, target_date)
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
        f"return={forward_return:+.2%}",
    ]
    if alpha is not None:
        note_parts.append(f"alpha={alpha:+.2%}")
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

def evaluate_all_due(
    *, horizons: list[int] | None = None,
    benchmark: str = DEFAULT_BENCHMARK,
    today: _date | None = None,
    db: Session | None = None,
) -> dict[str, int]:
    """Score every (snapshot, horizon) that has come of age and isn't
    already in `memo_outcomes`. Idempotent: re-running on the same day
    yields zero new rows once everything's been scored.

    ``evaluated`` retains its historical meaning (all snapshot/horizon pairs
    scanned).  ``due`` and ``data_unavailable`` distinguish work that should
    have produced a row from harmless future/idempotent skips, and
    ``unevaluable`` separates the permanently unscoreable pairs from the
    ones a later run can still fill in.
    """
    today = today or _date.today()
    horizons = list(horizons or DEFAULT_HORIZONS)
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        snaps = db.execute(
            select(MemoSnapshot).where(
                MemoSnapshot.as_of_date.is_(None),  # skip backtests
            ).order_by(MemoSnapshot.generated_at.asc())
        ).scalars().all()
        evaluated = 0
        written = 0
        reflections = 0
        errors = 0
        statuses: dict[str, int] = {
            "backtest": 0,
            "not_due": 0,
            "already_recorded": 0,
            "ticker_prices_unavailable": 0,
            "price_window_incomplete": 0,
            "memo_predates_price_window": 0,
        }
        for snap in snaps:
            for h in horizons:
                evaluated += 1
                try:
                    out, status = _evaluate_one(
                        snap, h, today=today, benchmark=benchmark, db=db,
                    )
                except Exception as exc:  # pragma: no cover — defensive
                    errors += 1
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
            + statuses["price_window_incomplete"]
        )
        # `unevaluable` is the permanent shortfall: no price history we can
        # obtain reaches this memo's generation date, and the window only
        # moves further away from it each night. Folding it into
        # `data_unavailable` would hold the loop red forever for a reason
        # nobody can fix, which is how two earlier alarms in this codebase
        # ended up ignored. It is counted and logged, not treated as a
        # failure.
        unevaluable = statuses["memo_predates_price_window"]
        due = (
            written + statuses["already_recorded"]
            + data_unavailable + unevaluable + errors
        )
        if data_unavailable:
            log.error(
                "Outcome evaluation left %s due rows pending: "
                "ticker_prices_unavailable=%s price_window_incomplete=%s",
                data_unavailable,
                statuses["ticker_prices_unavailable"],
                statuses["price_window_incomplete"],
            )
        if unevaluable:
            log.warning(
                "Outcome evaluation skipped %s permanently unevaluable pairs "
                "(memo_predates_price_window): no obtainable price history "
                "reaches the memo date.",
                unevaluable,
            )
        return {
            "evaluated": evaluated, "written": written,
            "reflections": reflections, "errors": errors,
            "due": due,
            "already_recorded": statuses["already_recorded"],
            "not_due": statuses["not_due"],
            "data_unavailable": data_unavailable,
            "ticker_prices_unavailable": statuses["ticker_prices_unavailable"],
            "price_window_incomplete": statuses["price_window_incomplete"],
            "unevaluable": unevaluable,
            "memo_predates_price_window": statuses["memo_predates_price_window"],
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


def track_record(
    *, ticker: str | None = None, sector: str | None = None,
    horizon_days: int = 90, db: Session | None = None,
) -> dict[str, Any]:
    """Aggregate track-record stats over evaluated outcomes.

    Filters: `ticker` (single name), `sector` (joined via memo_snapshots),
    `horizon_days` (which forward window to look at). Returns counts +
    hit rate + average alpha.
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = select(MemoOutcome).where(MemoOutcome.horizon_days == horizon_days)
        if ticker:
            stmt = stmt.where(MemoOutcome.ticker == ticker.upper())
        rows = db.execute(stmt).scalars().all()
        # Sector filter requires a join — do it in Python since memo_snapshots
        # already lives in the same DB. Cheap at our scale.
        if sector:
            snap_ids = {r.memo_snapshot_id for r in rows}
            sec_rows = db.execute(
                select(MemoSnapshot.id, MemoSnapshot.memo_json)
                .where(MemoSnapshot.id.in_(snap_ids))
            ).all()
            keep = {
                sid for sid, mj in sec_rows
                if (mj or {}).get("sector", "").lower() == sector.lower()
            }
            rows = [r for r in rows if r.memo_snapshot_id in keep]

        total = len(rows)
        evaluated_directional = [r for r in rows if r.thesis_held is not None]
        held = sum(1 for r in evaluated_directional if r.thesis_held)
        avg_return = (
            sum(r.forward_return for r in rows if r.forward_return is not None) / total
            if total else 0.0
        )
        avg_alpha = None
        alpha_rows = [r.alpha for r in rows if r.alpha is not None]
        if alpha_rows:
            avg_alpha = sum(alpha_rows) / len(alpha_rows)
        return {
            "horizon_days": horizon_days,
            "total": total,
            "directional_evaluations": len(evaluated_directional),
            "thesis_hit_rate": (held / len(evaluated_directional))
                if evaluated_directional else None,
            "avg_forward_return": avg_return,
            "avg_alpha": avg_alpha,
            "ticker_filter": ticker,
            "sector_filter": sector,
        }
    finally:
        if own:
            db.close()
