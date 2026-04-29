"""Comps agent."""
from __future__ import annotations

import json
from typing import Dict, Optional

from ..schemas import AgentFinding, CompsResult


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

    target = comps.target
    median = comps.median
    summary = (
        f"Peer set: {', '.join(p.ticker for p in comps.peers)}. "
        f"{comps.interpretation}"
    )
    key_points = []
    if target.ev_ebitda is not None and median.ev_ebitda is not None:
        key_points.append(f"EV/EBITDA: target {target.ev_ebitda:.1f}x vs peer median {median.ev_ebitda:.1f}x.")
    if target.operating_margin is not None and median.operating_margin is not None:
        key_points.append(f"Op margin: {target.operating_margin:.0%} vs peers {median.operating_margin:.0%}.")
    if target.revenue_growth is not None and median.revenue_growth is not None:
        key_points.append(f"Growth: {target.revenue_growth:+.0%} vs peers {median.revenue_growth:+.0%}.")
    if target.fcf_yield is not None and median.fcf_yield is not None:
        key_points.append(f"FCF yield: {target.fcf_yield:.1%} vs peers {median.fcf_yield:.1%}.")
    return AgentFinding(
        agent="Comps Analyst",
        headline=f"Peer-relative read for {profile.get('ticker', '')}",
        summary=summary,
        key_points=key_points,
        confidence=0.7,
        sources=[f"peer:{p.ticker}" for p in comps.peers],
    )
