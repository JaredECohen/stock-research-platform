"""Valuation + DCF agent."""
from __future__ import annotations

import json
from typing import Dict, Optional

from ..schemas import AgentFinding, DCFResult
from ..services.valuation_service import build_dcf
from . import llm, prompts


def run_valuation_agent(profile: Dict, ratios: Dict, dcf: Optional[DCFResult]) -> AgentFinding:
    payload = {
        "ticker": profile.get("ticker"),
        "current_price": profile.get("last_price"),
        "PE": ratios.get("PE"),
        "EV_EBITDA": ratios.get("EV_EBITDA"),
        "EV_Revenue": ratios.get("EV_Revenue"),
        "PFCF": ratios.get("PFCF"),
        "FCF_yield": ratios.get("FCF_yield"),
        "ROIC": ratios.get("ROIC"),
        "dcf_summary": dcf.summary if dcf else None,
        "dcf_base_implied": dcf.base.implied_share_price if dcf else None,
        "dcf_bull_implied": dcf.bull.implied_share_price if dcf else None,
        "dcf_bear_implied": dcf.bear.implied_share_price if dcf else None,
        "dcf_base_upside": dcf.base.upside_pct if dcf else None,
    }
    llm_out = llm.chat_json(
        prompts.VALUATION_ANALYST_PROMPT + "\n\nContext:\n" + json.dumps(payload, default=str),
        system=prompts.PM_SYSTEM, route="strong",
    )
    if llm_out:
        return AgentFinding(
            agent="Valuation Analyst",
            headline=llm_out.get("headline", "Valuation view"),
            summary=llm_out.get("summary", ""),
            key_points=llm_out.get("key_points", []),
            confidence=float(llm_out.get("confidence", 0.7)),
            sources=["dcf", "ratios"],
        )

    # Deterministic fallback
    pe = ratios.get("PE")
    ev_eb = ratios.get("EV_EBITDA")
    fcf_y = ratios.get("FCF_yield")
    base_up = dcf.base.upside_pct if dcf else None
    summary_parts = []
    if pe:
        summary_parts.append(f"P/E {pe:.1f}x")
    if ev_eb:
        summary_parts.append(f"EV/EBITDA {ev_eb:.1f}x")
    if fcf_y is not None:
        summary_parts.append(f"FCF yield {fcf_y:.1%}")
    if base_up is not None:
        summary_parts.append(f"DCF base implies {base_up:+.0%} vs current")

    headline = "; ".join(summary_parts) if summary_parts else "Valuation snapshot"
    key_points = []
    if dcf:
        key_points.append(f"Base case implied price: ${dcf.base.implied_share_price:,.2f}")
        key_points.append(f"Bull case: ${dcf.bull.implied_share_price:,.2f} | Bear case: ${dcf.bear.implied_share_price:,.2f}")
    if ev_eb and ev_eb > 25:
        key_points.append("Valuation is elevated on EV/EBITDA — rate-sensitive.")
    elif ev_eb and ev_eb < 10:
        key_points.append("EV/EBITDA looks undemanding versus history.")
    if fcf_y and fcf_y > 0.04:
        key_points.append("FCF yield > 4% gives downside support if execution holds.")

    summary = (
        f"Current multiples: {', '.join(summary_parts)}. "
        f"DCF triangulates against multiples; the bull/bear range frames the discount-rate sensitivity. "
        f"Valuation risk increases if terminal growth or margin assumptions slip."
    )
    return AgentFinding(
        agent="Valuation Analyst",
        headline=headline,
        summary=summary,
        key_points=key_points or ["See DCF and comps for detail."],
        confidence=0.7,
        sources=["dcf", "ratios"],
    )
