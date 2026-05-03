"""Comps agent.

Two valuation lenses, both surfaced when available (Wave 3E):
- Peer-relative — target metrics vs. peer median (today's set).
- Self-historical — target metrics vs. the target's own multi-year
  distribution. A name can look cheap vs. peers but expensive vs. its
  own history (or vice versa); the agent calls out divergence
  explicitly because that's where the alpha sits.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas import AgentFinding, CompsHistoryStats, CompsResult


def _peer_relative_points(comps: CompsResult) -> List[str]:
    target, median = comps.target, comps.median
    pts: List[str] = []
    if target.ev_ebitda is not None and median.ev_ebitda is not None:
        pts.append(
            f"EV/EBITDA: target {target.ev_ebitda:.1f}x vs peer median "
            f"{median.ev_ebitda:.1f}x."
        )
    if target.operating_margin is not None and median.operating_margin is not None:
        pts.append(
            f"Op margin: {target.operating_margin:.0%} vs peers "
            f"{median.operating_margin:.0%}."
        )
    if target.revenue_growth is not None and median.revenue_growth is not None:
        pts.append(
            f"Growth: {target.revenue_growth:+.0%} vs peers "
            f"{median.revenue_growth:+.0%}."
        )
    if target.fcf_yield is not None and median.fcf_yield is not None:
        pts.append(
            f"FCF yield: {target.fcf_yield:.1%} vs peers "
            f"{median.fcf_yield:.1%}."
        )
    return pts


def _self_history_points(
    history: CompsHistoryStats, target_ev_ebitda: Optional[float],
    target_op_margin: Optional[float], target_revenue_growth: Optional[float],
) -> List[str]:
    pts: List[str] = []
    label = history.lookback_label
    own_med = history.own_median
    pct = history.current_percentile

    if target_ev_ebitda is not None and own_med.get("ev_ebitda") is not None:
        ev_pct = pct.get("ev_ebitda")
        if ev_pct is not None:
            pts.append(
                f"EV/EBITDA vs own {label} median {own_med['ev_ebitda']:.1f}x — "
                f"currently at {ev_pct * 100:.0f}th percentile of own history."
            )
    if target_op_margin is not None and own_med.get("operating_margin") is not None:
        delta = history.current_vs_own_median.get("operating_margin")
        if delta is not None:
            pts.append(
                f"Op margin vs own {label} median "
                f"{own_med['operating_margin']:.0%} — {delta:+.0%} delta."
            )
    if target_revenue_growth is not None and own_med.get("revenue_growth") is not None:
        rg_pct = pct.get("revenue_growth")
        if rg_pct is not None:
            pts.append(
                f"Revenue growth at {rg_pct * 100:.0f}th percentile of "
                f"own {label} range."
            )
    return pts


def _premium_signal(comps: CompsResult, metric: str) -> Optional[str]:
    """For a given metric, return 'premium', 'discount', or None for in-line."""
    val = comps.premium_discount.get(metric)
    if val is None:
        return None
    if val > 0.05:
        return "premium"
    if val < -0.05:
        return "discount"
    return None


def _own_history_signal(history: CompsHistoryStats, metric: str) -> Optional[str]:
    delta = history.current_vs_own_median.get(metric)
    if delta is None:
        return None
    if delta > 0.05:
        return "premium"
    if delta < -0.05:
        return "discount"
    return None


def run_comps_agent(profile: Dict, comps: Optional[CompsResult]) -> AgentFinding:
    if comps is None:
        return AgentFinding(
            agent="Comps Analyst",
            headline="Insufficient peer data for comps.",
            summary="No peer set defined for this ticker in the demo dataset.",
            key_points=[],
            confidence=0.4,
            sources=[],
        )

    ticker = profile.get("ticker", "")
    history = comps.history
    key_points: List[str] = _peer_relative_points(comps)

    # Wave 3E: when self-historical context is present, combine both lenses.
    confidence = 0.7
    summary_parts: List[str] = [
        f"Peer set: {', '.join(p.ticker for p in comps.peers)}. {comps.interpretation}"
    ]
    headline = f"Peer-relative read for {ticker}"

    if history is not None:
        key_points.extend(_self_history_points(
            history,
            comps.target.ev_ebitda,
            comps.target.operating_margin,
            comps.target.revenue_growth,
        ))
        if history.interpretation:
            summary_parts.append(f"Own-history context: {history.interpretation}")

        # Highest-signal divergence call: same axis on EV/EBITDA showing
        # opposite signals across the two lenses is alpha territory.
        peer_sig = _premium_signal(comps, "ev_ebitda")
        own_sig = _own_history_signal(history, "ev_ebitda")
        if peer_sig and own_sig:
            if peer_sig == own_sig:
                # Both lenses agree — bump confidence.
                confidence = 0.75
                headline = (
                    f"{ticker}: EV/EBITDA at a {peer_sig} on BOTH peer "
                    f"and own-history axes."
                )
            else:
                # They disagree — that's the alpha; surface in headline.
                confidence = 0.65
                headline = (
                    f"{ticker}: peer {peer_sig} but own-history {own_sig} "
                    f"on EV/EBITDA — divergence to investigate."
                )

    # Wave 7C: discretionary notes tagged for the comps agent. Stashed
    # on data so the drill-down report can surface them; the existing
    # Wave 3E `data["history"]` payload is preserved.
    from ..services.research_notes import build_notes_block_for_agent
    notes_block = build_notes_block_for_agent(
        "comps", profile, extra_query="EV EBITDA multiple cohort peer premium discount",
    )
    finding_data: Dict[str, Any] = {}
    if history is not None:
        finding_data["history"] = history.model_dump()
    if notes_block:
        finding_data["research_notes"] = notes_block

    return AgentFinding(
        agent="Comps Analyst",
        headline=headline,
        summary=" ".join(summary_parts),
        key_points=key_points,
        confidence=confidence,
        sources=[f"peer:{p.ticker}" for p in comps.peers]
        + ([f"history:{ticker}"] if history is not None else []),
        data=finding_data,
    )
