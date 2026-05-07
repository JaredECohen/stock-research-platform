"""Wave 10 — calibration + per-agent attribution aggregator.

Reads `memo_outcomes` (realized returns vs SPY) and `memo_postmortems`
(LLM-judged verdicts + per-agent attribution) and computes the
dashboards the design review §13.4-13.5 calls for:

- **Calibration plot data**: per-rating (Very Bullish → Very Bearish)
  realized excess-return distribution. A well-calibrated PM has
  Strong Buy realizations clearly higher than Buy.
- **Per-agent attribution**: average attribution score (-1..1) per
  specialist, plus accuracy stats (% of memos where this agent
  contributed positively to a memo that was right). Identifies
  systematic strengths and weaknesses per specialist.
- **Regime-conditional accuracy**: success rate of memos bucketed by
  the macro regime that was active at memo creation time. Catches
  systematic blind spots ("our model is great in soft-landing
  regimes and terrible in recessions").

All read-only — never mutates the memo / outcome / postmortem
tables. Powers the upcoming track-record dashboard.
"""
from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..database import SessionLocal
from ..models import MemoOutcome, MemoPostmortem, MemoSnapshot

log = logging.getLogger(__name__)


# Stable rating order for consistent display.
RATING_ORDER: List[str] = [
    "Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish",
]

ALL_AGENTS: List[str] = [
    "sector", "earnings", "filing", "valuation",
    "comps", "macro", "risk", "technical",
]


# ---------------------------------------------------------------------------
# Calibration: per-rating realized excess returns
# ---------------------------------------------------------------------------

def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = pct * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def calibration_by_rating(*, horizon_days: int = 90) -> Dict[str, Any]:
    """Bucket realized alpha by memo's rating_label.

    Returns: {
        rating_order: [...],
        buckets: {
            "Very Bullish": {n, mean_alpha, median, p25, p75, win_rate},
            ...
        },
        horizon_days
    }
    win_rate is the fraction of memos with alpha > 0 in this bucket.
    """
    buckets: Dict[str, List[float]] = {r: [] for r in RATING_ORDER}
    with SessionLocal() as db:
        rows = db.execute(
            select(MemoOutcome, MemoSnapshot)
            .join(MemoSnapshot, MemoOutcome.memo_snapshot_id == MemoSnapshot.id)
            .where(MemoOutcome.horizon_days == horizon_days)
        ).all()
        for outcome, snap in rows:
            if outcome.alpha is None:
                continue
            memo = snap.memo_json or {}
            rating = (memo.get("rating_label") or "").strip() if isinstance(memo, dict) else ""
            if rating in buckets:
                buckets[rating].append(float(outcome.alpha))

    out_buckets: Dict[str, Dict[str, Any]] = {}
    for rating in RATING_ORDER:
        vals = buckets[rating]
        if not vals:
            out_buckets[rating] = {
                "n": 0, "mean_alpha": None, "median": None,
                "p25": None, "p75": None, "win_rate": None,
            }
            continue
        wins = sum(1 for v in vals if v > 0)
        out_buckets[rating] = {
            "n": len(vals),
            "mean_alpha": statistics.fmean(vals),
            "median": statistics.median(vals),
            "p25": _percentile(vals, 0.25),
            "p75": _percentile(vals, 0.75),
            "win_rate": wins / len(vals),
        }
    return {
        "horizon_days": horizon_days,
        "rating_order": RATING_ORDER,
        "buckets": out_buckets,
    }


# ---------------------------------------------------------------------------
# Per-agent attribution
# ---------------------------------------------------------------------------

def per_agent_attribution(*, horizon_days: int = 90) -> Dict[str, Any]:
    """Aggregate per-agent attribution scores from `memo_postmortems`.

    Each postmortem stores `agent_attribution: {agent: -1..1}` set by
    the LLM postmortem call (or empty when LLM was unavailable).
    Returns mean attribution per agent + count of postmortems.
    """
    sums: Dict[str, float] = {a: 0.0 for a in ALL_AGENTS}
    counts: Dict[str, int] = {a: 0 for a in ALL_AGENTS}
    contrib_when_right: Dict[str, int] = {a: 0 for a in ALL_AGENTS}
    total_right = 0

    with SessionLocal() as db:
        rows = db.execute(
            select(MemoPostmortem)
            .where(MemoPostmortem.horizon_days == horizon_days)
        ).scalars().all()
        for row in rows:
            attribution = row.agent_attribution or {}
            if not isinstance(attribution, dict):
                continue
            verdict_right = (row.verdict == "right")
            if verdict_right:
                total_right += 1
            for agent, score in attribution.items():
                if agent not in sums or not isinstance(score, (int, float)):
                    continue
                sums[agent] += float(score)
                counts[agent] += 1
                if verdict_right and float(score) > 0.2:
                    contrib_when_right[agent] += 1

    agents_out: Dict[str, Dict[str, Any]] = {}
    for agent in ALL_AGENTS:
        n = counts[agent]
        agents_out[agent] = {
            "n": n,
            "mean_attribution": (sums[agent] / n) if n else None,
            "contrib_to_right": contrib_when_right[agent],
            "contrib_to_right_pct": (
                contrib_when_right[agent] / total_right
                if total_right else None
            ),
        }
    return {
        "horizon_days": horizon_days,
        "total_postmortems": sum(1 for _ in rows),
        "total_right": total_right,
        "agents": agents_out,
    }


# ---------------------------------------------------------------------------
# Regime-conditional accuracy
# ---------------------------------------------------------------------------

def regime_conditional_accuracy(*, horizon_days: int = 90) -> Dict[str, Any]:
    """Bucket memo outcomes by the macro regime at memo-creation time.

    Reads regime_at_memo from `memo_postmortems` (set by the LLM
    postmortem call). Falls back to "unknown" when the field is empty.
    Surfaces systematic regime weaknesses ("model overestimates growth
    in sticky-inflation environments").
    """
    by_regime: Dict[str, Dict[str, Any]] = {}
    with SessionLocal() as db:
        rows = db.execute(
            select(MemoPostmortem)
            .where(MemoPostmortem.horizon_days == horizon_days)
        ).scalars().all()
        for row in rows:
            regime = (row.regime_at_memo or "unknown").strip().lower()
            entry = by_regime.setdefault(regime, {
                "n": 0, "right": 0, "wrong": 0, "mixed": 0,
                "alpha_sum": 0.0,
            })
            entry["n"] += 1
            if row.verdict == "right":
                entry["right"] += 1
            elif row.verdict == "wrong":
                entry["wrong"] += 1
            elif row.verdict == "mixed":
                entry["mixed"] += 1
            if row.realized_return is not None and row.benchmark_return is not None:
                entry["alpha_sum"] += float(
                    row.realized_return - row.benchmark_return
                )

    for regime, entry in by_regime.items():
        n = entry["n"] or 1
        entry["accuracy"] = entry["right"] / n
        entry["mean_alpha"] = entry["alpha_sum"] / n
        del entry["alpha_sum"]
    return {
        "horizon_days": horizon_days,
        "regimes": by_regime,
    }


# ---------------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------------

def summary(*, horizon_days: int = 90) -> Dict[str, Any]:
    """One-call aggregator that returns calibration + per-agent +
    regime stats. Powers the future track-record dashboard."""
    return {
        "calibration": calibration_by_rating(horizon_days=horizon_days),
        "per_agent": per_agent_attribution(horizon_days=horizon_days),
        "regime_conditional": regime_conditional_accuracy(horizon_days=horizon_days),
    }
