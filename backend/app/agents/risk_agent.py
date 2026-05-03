"""Risk analyst agent."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..schemas import AgentFinding, RiskItem


def run_risk_agent(profile: Dict, ratios: Dict, dcf_summary: Optional[Dict] = None) -> AgentFinding:
    risks: List[str] = profile.get("risks") or []
    summary_lines: List[str] = []

    debt_to_eb = ratios.get("debt_to_ebitda")
    if debt_to_eb and debt_to_eb > 3.5:
        summary_lines.append(f"Leverage at {debt_to_eb:.1f}x debt/EBITDA is elevated.")
    if (ratios.get("EV_EBITDA") or 0) > 30:
        summary_lines.append("Multiple is high — valuation risk elevated to growth slippage.")
    if (ratios.get("FCF_yield") or 0) < 0.02:
        summary_lines.append("Low FCF yield reduces downside support.")
    if not summary_lines:
        summary_lines.append("Quantitative risk profile looks moderate.")

    key_points = [f"Risk: {r}" for r in risks[:5]]
    if dcf_summary and "bear" in dcf_summary:
        key_points.append("DCF bear case maps the downside if growth and margin slip.")

    # Wave 7C: discretionary notes tagged for the risk agent. Risk has no
    # LLM call, so matched notes ride on `data["research_notes"]` for the
    # drill-down report (Wave 3C) to surface.
    from ..services.research_notes import build_notes_block_for_agent
    notes_block = build_notes_block_for_agent(
        "risk", profile, extra_query="thesis breakers downside survivable",
    )
    finding_data = {"research_notes": notes_block} if notes_block else {}

    return AgentFinding(
        agent="Risk Analyst",
        headline=f"Risk profile for {profile.get('ticker', '')}",
        summary=" ".join(summary_lines),
        key_points=key_points,
        confidence=0.7,
        sources=[],
        data=finding_data,
    )


def derive_risk_items(profile: Dict) -> List[RiskItem]:
    risks = profile.get("risks") or []
    items: List[RiskItem] = []
    for r in risks[:6]:
        sev = "high" if any(k in r.lower() for k in ("competition", "regulator", "antitrust")) else "medium"
        type_ = "company"
        rl = r.lower()
        if "regulat" in rl or "antitrust" in rl or "policy" in rl:
            type_ = "regulatory"
        elif "macro" in rl or "rate" in rl or "recession" in rl:
            type_ = "macro"
        elif "valuation" in rl or "multiple" in rl:
            type_ = "valuation"
        items.append(RiskItem(title=r[:80], detail=r, severity=sev, type=type_))
    return items
