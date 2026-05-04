"""Quantitative factor scoring used by the screener and portfolio agents.

Scores are 0-100 normalized within the universe so they're directly
comparable across sectors.
"""
from __future__ import annotations

from statistics import mean, pstdev
from typing import Dict, List, Optional


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


def earnings_momentum_score(surprise_history: List[Optional[float]]) -> float:
    # Live providers (FMP /stable/, AV) emit None for forward quarters
    # whose actuals haven't reported yet. Drop those before averaging so
    # `statistics.mean` doesn't crash on a NoneType numerator.
    cleaned = [s for s in (surprise_history or []) if s is not None]
    if not cleaned:
        return 50.0
    avg = mean(cleaned)
    return round(min(100, max(0, 50 + avg * 10)), 1)


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
