"""Quantitative factor scoring used by the screener and portfolio agents.

Scores are 0-100 normalized within the universe so they're directly
comparable across sectors.
"""
from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple


# Threshold for treating an EPS surprise as a real "beat" (not noise).
# 2% margin filters out coin-flip rounding-error beats.
_BEAT_THRESHOLD: float = 0.02


def beat_streak(surprise_history: List[Optional[float]]) -> int:
    """Count consecutive most-recent beats (surprise > _BEAT_THRESHOLD).

    Assumes `surprise_history` is ordered oldest-first (the convention
    every call site uses — the screener pulls from `earnings.quarters`
    where the latest quarter is last). Walks the tail of the list and
    stops at the first miss / None.

    Examples:
      [.01, .02, .03, .05]   →  4   (all beats)
      [.01, .05, -.01, .02]  →  2   (last two)
      [.01, .02, None]       →  0   (last entry unknown, conservative)
    """
    streak = 0
    for s in reversed(surprise_history or []):
        if s is None or s <= _BEAT_THRESHOLD:
            break
        streak += 1
    return streak


def guidance_net_direction(
    guidance_changes: Optional[List[Dict[str, Any]]],
) -> int:
    """Net direction score from the latest call's structured guidance.

    Returns `raised - lowered` across all guidance change entries (the
    LLM extraction emits one entry per metric). Positive net = company
    raised forward guidance materially; negative = cut it; zero = mostly
    reaffirmed or mixed.
    """
    if not guidance_changes:
        return 0
    raised = lowered = 0
    for g in guidance_changes:
        d = ""
        if isinstance(g, dict):
            d = str(g.get("direction") or "").lower()
        else:
            d = str(getattr(g, "direction", "") or "").lower()
        if d == "raised":
            raised += 1
        elif d == "lowered" or d == "withdrawn":
            lowered += 1
    return raised - lowered


def _z_to_100(z: float, *, clip: float = 2.5) -> float:
    z = max(-clip, min(clip, z))
    return round(50 + (z / clip) * 50, 1)


def _zscore(values: List[float], v: float) -> float:
    clean = [x for x in values if x is not None]
    if len(clean) < 2:
        return 0.0
    m, s = mean(clean), pstdev(clean)
    if s == 0:
        return 0.0
    return (v - m) / s


def normalize(universe: List[Dict], field: str, *, higher_better: bool = True) -> Dict[str, float]:
    vals = [r.get(field) for r in universe if r.get(field) is not None]
    out: Dict[str, float] = {}
    for r in universe:
        v = r.get(field)
        if v is None:
            out[r["ticker"]] = 50.0
            continue
        z = _zscore(vals, v)
        if not higher_better:
            z = -z
        out[r["ticker"]] = _z_to_100(z)
    return out


def composite_score(scores: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total_w = sum(weights.values()) or 1.0
    for ticker, sc in scores.items():
        s = 0.0
        for k, w in weights.items():
            s += sc.get(k, 50.0) * w
        out[ticker] = round(s / total_w, 1)
    return out


def quality_score(roic: Optional[float], op_margin: Optional[float], gross_margin: Optional[float]) -> float:
    parts = []
    if roic is not None:
        parts.append(min(100, max(0, (roic - 0.05) / 0.30 * 100)))
    if op_margin is not None:
        parts.append(min(100, max(0, (op_margin - 0.05) / 0.40 * 100)))
    if gross_margin is not None:
        parts.append(min(100, max(0, (gross_margin - 0.20) / 0.60 * 100)))
    return round(sum(parts) / len(parts), 1) if parts else 50.0


def growth_score(revenue_growth: Optional[float], earnings_growth: Optional[float] = None) -> float:
    parts = []
    if revenue_growth is not None:
        parts.append(min(100, max(0, (revenue_growth - 0.0) / 0.30 * 100)))
    if earnings_growth is not None:
        parts.append(min(100, max(0, (earnings_growth - 0.0) / 0.30 * 100)))
    return round(sum(parts) / len(parts), 1) if parts else 50.0


def valuation_score(ev_ebitda: Optional[float], p_fcf: Optional[float], fcf_yield: Optional[float]) -> float:
    """Cheap is good — invert."""
    parts: List[float] = []
    if ev_ebitda is not None and ev_ebitda > 0:
        parts.append(min(100, max(0, (35 - ev_ebitda) / 30 * 100)))
    if p_fcf is not None and p_fcf > 0:
        parts.append(min(100, max(0, (50 - p_fcf) / 45 * 100)))
    if fcf_yield is not None:
        parts.append(min(100, max(0, fcf_yield * 1500)))
    return round(sum(parts) / len(parts), 1) if parts else 50.0


def earnings_momentum_score(
    surprise_history: List[Optional[float]],
    *,
    latest_guidance_changes: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """0-100 earnings momentum score.

    Composition:
    - Surprise mean (existing): `50 + avg(surprises) * 10`, clipped 0-100.
    - Beat-streak bonus (NEW): +5 per consecutive recent beat,
      capped at +20 (4 quarters). Companies that beat earnings often
      are a known momentum factor — Bernard & Thomas (1989) PEAD and
      every quantitative replication since.
    - Beat-AND-raise bonus (NEW): +15 when the latest quarter was a
      beat (>2%) AND the company net-raised forward guidance. The
      combination is the high-conviction tell — management is
      comfortable enough with execution to publicly commit to a
      higher bar, which historically signals durable outperformance.
    - Miss-and-cut penalty (NEW): -15 when the latest was a miss
      AND guidance was net-lowered (the symmetric bear signal).

    Args:
      surprise_history: per-quarter EPS surprise % (oldest first).
        Live providers emit None for forward quarters whose actuals
        haven't reported — those are dropped before averaging.
      latest_guidance_changes: structured guidance change list from
        the most recent call (from `EarningsStructured.guidance_changes`).
        Pass None for paths without an LLM extraction (e.g. screener).
    """
    cleaned = [s for s in (surprise_history or []) if s is not None]
    if not cleaned:
        base = 50.0
    else:
        avg = mean(cleaned)
        base = 50.0 + avg * 10

    streak = beat_streak(surprise_history or [])
    streak_bonus = min(20.0, streak * 5.0)

    raise_signal = guidance_net_direction(latest_guidance_changes)
    latest_surprise = cleaned[-1] if cleaned else None
    pattern_bonus = 0.0
    if latest_surprise is not None:
        if latest_surprise > _BEAT_THRESHOLD and raise_signal >= 2:
            pattern_bonus = 15.0  # beat-and-raise
        elif latest_surprise < -_BEAT_THRESHOLD and raise_signal <= -2:
            pattern_bonus = -15.0  # miss-and-cut

    total = base + streak_bonus + pattern_bonus
    return round(min(100, max(0, total)), 1)


def risk_score(beta: Optional[float], debt_to_ebitda: Optional[float], drawdown: Optional[float]) -> float:
    """Higher score = lower perceived risk."""
    parts = []
    if beta is not None:
        parts.append(min(100, max(0, 100 - abs(beta - 1.0) * 60)))
    if debt_to_ebitda is not None:
        parts.append(min(100, max(0, 100 - max(0, debt_to_ebitda) * 25)))
    if drawdown is not None:
        parts.append(min(100, max(0, 100 + drawdown * 200)))  # drawdown is negative
    return round(sum(parts) / len(parts), 1) if parts else 50.0
