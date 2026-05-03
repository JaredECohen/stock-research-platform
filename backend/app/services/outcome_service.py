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
from datetime import date as _date, datetime, timedelta
from typing import Any, Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_table(db: Session) -> None:
    bind = db.get_bind()
    MemoOutcome.__table__.create(bind=bind, checkfirst=True)


def _close_on_or_before(rows: List[Dict[str, Any]], target: str) -> Optional[float]:
    if not rows or not target:
        return None
    chosen: Optional[float] = None
    for r in rows:
        d = str(r.get("date") or "")
        if not d:
            continue
        if d <= target:
            try:
                chosen = float(r.get("close") or r.get("adjusted_close"))
            except (TypeError, ValueError):
                continue
        else:
            break
    return chosen


def _close_on_or_after(rows: List[Dict[str, Any]], target: str) -> Optional[float]:
    """First close on or after `target` — used for the price at memo
    generation when we don't have an exact-day match."""
    for r in rows or []:
        d = str(r.get("date") or "")
        if d and d >= target:
            try:
                return float(r.get("close") or r.get("adjusted_close"))
            except (TypeError, ValueError):
                continue
    return None


def _thesis_held(rating: str, return_signed: float) -> Optional[bool]:
    """Did the recommendation pay off?

    - Bullish / Mixed Positive expect positive return → held if ≥ 0.
    - Bearish / Mixed Negative expect negative return → held if ≤ 0.
    - Neutral has no directional bet — return None (no judgment).
    """
    r = (rating or "").strip().lower()
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
) -> Optional[MemoOutcome]:
    """Score a single (snapshot, horizon). Skips when the horizon hasn't
    come of age, the snapshot is a backtest, or the price series is missing."""
    # Backtest snapshots have `as_of_date` set; outcome scoring is for live memos only.
    if snap.as_of_date is not None:
        return None

    generated = snap.generated_at
    if isinstance(generated, datetime):
        generated_date = generated.date()
    else:
        generated_date = generated  # assume date-like
    target_date = generated_date + timedelta(days=horizon_days)
    if target_date > today:
        return None  # not due yet

    # Skip if we've already evaluated this (snapshot, horizon).
    existing = db.execute(
        select(MemoOutcome).where(
            MemoOutcome.memo_snapshot_id == snap.id,
            MemoOutcome.horizon_days == horizon_days,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return None

    from .market_data_service import get_price_series
    days = horizon_days + 30
    ticker_rows = get_price_series(snap.ticker, days) or []
    bench_rows = get_price_series(benchmark, days) or []
    if not ticker_rows:
        return None

    g_iso = generated_date.isoformat()
    t_iso = target_date.isoformat()

    price_at_memo = _close_on_or_after(ticker_rows, g_iso) or _close_on_or_before(ticker_rows, g_iso)
    price_at_target = _close_on_or_before(ticker_rows, t_iso)
    if not (price_at_memo and price_at_target and price_at_memo > 0):
        return None

    forward_return = (price_at_target - price_at_memo) / price_at_memo

    # Benchmark-relative alpha (None if benchmark price unavailable).
    bench_at_memo = _close_on_or_after(bench_rows, g_iso) or _close_on_or_before(bench_rows, g_iso)
    bench_at_target = _close_on_or_before(bench_rows, t_iso)
    bench_return: Optional[float] = None
    alpha: Optional[float] = None
    if bench_at_memo and bench_at_target and bench_at_memo > 0:
        bench_return = (bench_at_target - bench_at_memo) / bench_at_memo
        alpha = forward_return - bench_return

    rating = (snap.memo_json or {}).get("rating_label") or ""
    confidence = float((snap.memo_json or {}).get("confidence_score") or 0.0)
    held = _thesis_held(rating, forward_return)

    note_parts: List[str] = [
        f"horizon={horizon_days}d",
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
    )
    db.add(row)
    return row


def _maybe_write_reflection(
    snap: MemoSnapshot, outcome: MemoOutcome,
) -> None:
    """For long horizons, append an outcome entry to the company memory file."""
    if outcome.horizon_days not in REFLECTION_HORIZONS:
        return
    try:
        from ..config import settings
        from ..memory import CompanyMemory, MemoryEntry
        if not settings.enable_long_term_memory:
            return
        cm = CompanyMemory.for_ticker(snap.ticker)
        body_parts: List[str] = []
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
    except Exception as exc:  # pragma: no cover — diagnostic only
        log.warning("Outcome reflection write failed for %s/%sd: %s",
                    snap.ticker, outcome.horizon_days, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_all_due(
    *, horizons: Optional[List[int]] = None,
    benchmark: str = DEFAULT_BENCHMARK,
    today: Optional[_date] = None,
    db: Optional[Session] = None,
) -> Dict[str, int]:
    """Score every (snapshot, horizon) that has come of age and isn't
    already in `memo_outcomes`. Idempotent: re-running on the same day
    yields zero new rows once everything's been scored.

    Returns `{evaluated, written, reflections, errors}`.
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
        for snap in snaps:
            for h in horizons:
                evaluated += 1
                try:
                    out = _evaluate_one(
                        snap, h, today=today, benchmark=benchmark, db=db,
                    )
                except Exception as exc:  # pragma: no cover — defensive
                    errors += 1
                    log.warning(
                        "Outcome evaluation failed for snap=%s h=%sd: %s",
                        snap.id, h, exc,
                    )
                    continue
                if out is not None:
                    written += 1
                    db.commit()
                    if h in REFLECTION_HORIZONS:
                        try:
                            _maybe_write_reflection(snap, out)
                            reflections += 1
                        except Exception:  # pragma: no cover
                            pass
        return {
            "evaluated": evaluated, "written": written,
            "reflections": reflections, "errors": errors,
        }
    finally:
        if own:
            db.close()


def get_outcomes_for_snapshot(
    memo_snapshot_id: int, *, db: Optional[Session] = None,
) -> List[Dict[str, Any]]:
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
            }
            for r in rows
        ]
    finally:
        if own:
            db.close()


def track_record(
    *, ticker: Optional[str] = None, sector: Optional[str] = None,
    horizon_days: int = 90, db: Optional[Session] = None,
) -> Dict[str, Any]:
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
