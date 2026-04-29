"""Macro analyst agent — answers macro questions and produces sector mappings."""
from __future__ import annotations

import json
from typing import Dict, Optional

from ..schemas import AgentFinding, MacroScenarioResult
from ..services.macro_service import macro_snapshot
from . import llm, prompts


SCENARIO_TEMPLATES: Dict[str, MacroScenarioResult] = {
    "soft_landing": MacroScenarioResult(
        scenario="Soft landing",
        narrative=(
            "Growth stays positive, inflation glides toward target, and the Fed delivers measured rate cuts. "
            "Real yields ease without breaking employment. Risk assets benefit but earnings dispersion widens."
        ),
        sector_impacts={
            "Technology": "Constructive — AI capex continues, multiple support from real-rate easing.",
            "Communication Services": "Constructive — AI-leveraged ad platforms benefit from rerating.",
            "Consumer Discretionary": "Improving — wage gains real, financing eases.",
            "Financials": "Mixed — NIM compresses but credit holds; capital markets reopen.",
            "Healthcare": "Stable — defensive support fades but pipeline catalysts matter.",
            "Energy": "Range-bound — slower demand offset by capital discipline.",
            "Industrials": "Constructive — infrastructure spend, capex cycle.",
            "Utilities": "Constructive — long-duration relief if long end falls.",
        },
        favored_sectors=["Technology", "Communication Services", "Industrials", "Consumer Discretionary"],
        pressured_sectors=["Energy", "Defensive Staples"],
        suggested_research_views=[
            "Quality compounders with AI exposure",
            "Cyclicals with operating leverage to PMI recovery",
            "Long-duration assets benefiting from real-rate easing",
        ],
        risks=["Re-acceleration in inflation", "Labor market crack", "Credit-spread blowout"],
    ),
    "recession": MacroScenarioResult(
        scenario="Recession",
        narrative=(
            "Activity slows, unemployment rises, earnings cuts broaden. Defensive cash flows outperform. "
            "Risk assets reprice; the Fed pivots faster but credit spreads widen first."
        ),
        sector_impacts={
            "Consumer Staples": "Outperforms — defensive demand.",
            "Healthcare": "Outperforms — payor mix headwinds offset by demand stability.",
            "Utilities": "Outperforms — bond proxy and rate relief.",
            "Technology": "Mixed — capex digestion offsets duration tailwind.",
            "Financials": "Underperforms — credit losses, NIM compression.",
            "Consumer Discretionary": "Underperforms — discretionary spend retracts.",
            "Energy": "Mixed — demand pressure, capital discipline support.",
        },
        favored_sectors=["Consumer Staples", "Healthcare", "Utilities"],
        pressured_sectors=["Financials", "Consumer Discretionary", "Industrials"],
        suggested_research_views=[
            "Defensive compounders with FCF resilience",
            "Healthcare names with low cyclicality",
            "Mega-cap balance sheets that absorb credit shocks",
        ],
        risks=["Policy lag prolonging downturn", "CRE losses contaminating banks"],
    ),
    "sticky_inflation": MacroScenarioResult(
        scenario="Sticky inflation",
        narrative=(
            "Core inflation lingers above target. Long-end yields stay elevated, dampening duration assets. "
            "Pricing power and asset-light models outperform; rate cuts are pushed out."
        ),
        sector_impacts={
            "Energy": "Outperforms — pricing power, capital discipline.",
            "Materials": "Constructive — pricing power.",
            "Financials": "Mixed — higher-for-longer NIM tailwind, credit risk on the other side.",
            "Technology": "Pressured — multiple compression on long-duration cash flows.",
            "Real Estate": "Pressured — cap rate expansion.",
            "Utilities": "Pressured — bond-proxy headwind.",
        },
        favored_sectors=["Energy", "Materials", "Financials"],
        pressured_sectors=["Real Estate", "Utilities", "Long-duration Tech"],
        suggested_research_views=[
            "Pricing-power compounders",
            "Asset-light franchises",
            "Energy and materials with disciplined capital allocation",
        ],
        risks=["Demand destruction from rates", "Currency strength weighing on multinationals"],
    ),
    "falling_rates": MacroScenarioResult(
        scenario="Falling rates",
        narrative=(
            "Real yields ease as inflation glides lower. Long-duration assets benefit; "
            "rate-sensitive consumers re-engage; capital markets reopen."
        ),
        sector_impacts={
            "Technology": "Constructive — duration tailwind, AI capex continues.",
            "Real Estate": "Constructive — cap rate compression.",
            "Consumer Discretionary": "Improving — financing eases.",
            "Utilities": "Constructive — bond-proxy tailwind.",
            "Financials": "Mixed — NIM headwind offset by capital markets.",
        },
        favored_sectors=["Technology", "Real Estate", "Utilities", "Consumer Discretionary"],
        pressured_sectors=["Energy", "Defensive Staples"],
        suggested_research_views=[
            "Long-duration growth with AI exposure",
            "REIT subsectors with re-rating optionality",
            "Capital-markets-leveraged financials",
        ],
        risks=["Rate cuts arrive because growth cracks", "Credit-spread widening"],
    ),
    "ai_capex_boom": MacroScenarioResult(
        scenario="AI capex boom",
        narrative=(
            "Hyperscaler and sovereign AI infrastructure spend remains strong. Power and data-center supply chain "
            "tightens; semiconductors, networking, and electrical infrastructure benefit. Valuation risk concentrates "
            "in the bottleneck names."
        ),
        sector_impacts={
            "Technology": "Outperforms — semiconductors, networking, data-center systems.",
            "Industrials": "Outperforms — power equipment, HVAC, electrical.",
            "Utilities": "Constructive — load growth tailwind.",
            "Energy": "Constructive — data-center power demand.",
        },
        favored_sectors=["Technology", "Industrials", "Utilities"],
        pressured_sectors=["Defensive Staples"],
        suggested_research_views=[
            "AI compute and networking",
            "Power generation and grid suppliers",
            "Data-center REITs with low pre-leasing risk",
        ],
        risks=["Capex digestion phase", "Sovereign AI policy reversals", "Power constraints delaying deployments"],
    ),
}


def detect_scenario_key(text: str) -> str:
    t = text.lower()
    if "soft landing" in t:
        return "soft_landing"
    if "recession" in t or "downturn" in t:
        return "recession"
    if "sticky" in t or "inflation stays" in t:
        return "sticky_inflation"
    if "falling rate" in t or "rate cut" in t or "rates fall" in t:
        return "falling_rates"
    if "ai capex" in t or "ai infrastructure" in t or "ai boom" in t:
        return "ai_capex_boom"
    return "soft_landing"


def run_macro_scenario(scenario: str) -> MacroScenarioResult:
    key = detect_scenario_key(scenario)
    base = SCENARIO_TEMPLATES[key]
    snapshot = macro_snapshot()
    if snapshot:
        # tag narrative with current macro snapshot
        narrative = base.narrative + (
            f" Current snapshot: Fed Funds {snapshot.get('FEDFUNDS', '—')}%, "
            f"10Y {snapshot.get('DGS10', '—')}%, "
            f"Core sticky CPI {snapshot.get('CORESTICKM159SFRBATL', '—')}%, "
            f"Unemployment {snapshot.get('UNRATE', '—')}%."
        )
        return base.model_copy(update={"narrative": narrative})
    return base


def run_macro_agent(profile: Dict, scenario: str = "soft_landing") -> AgentFinding:
    s = run_macro_scenario(scenario)
    sector = profile.get("sector", "")
    sector_view = s.sector_impacts.get(sector, "Macro impact mapped via sector framework.")

    summary = f"Scenario read: {s.narrative} {sector} positioning: {sector_view}"
    key_points = [
        f"Favored sectors: {', '.join(s.favored_sectors)}.",
        f"Pressured sectors: {', '.join(s.pressured_sectors)}.",
        f"Top scenario risks: {'; '.join(s.risks[:3])}.",
    ]
    return AgentFinding(
        agent="Macro Analyst",
        headline=f"Macro scenario: {s.scenario}",
        summary=summary,
        key_points=key_points,
        confidence=0.75,
        sources=["macro_scenario_framework"],
    )
