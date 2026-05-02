"""Sector specialist agent — does *real research*, not rubric look-up.

Workflow:
1. Call `sector_research_service.run_sector_research(ticker)` which builds the
   sub-industry cohort, computes distributional placement, detects regime,
   aggregates cohort filing themes, and computes multi-year sector trends.
2. Optionally invoke the LLM to write a structured narrative grounded in the
   computed payload. The deterministic fallback uses the same payload to
   produce a meaningful, cohort-grounded summary even without an LLM.
3. Return an `AgentFinding` with the structured research attached as `data`
   so downstream consumers (frontend, critic, screener) can use it.
"""
from __future__ import annotations

import json
from typing import Dict, List

from ..cache import cache_get
from ..config import settings
from ..memory import CompanyMemory, SectorMemory
from ..schemas import AgentFinding
from ..services.data_service import get_data_service
from ..services.sector_research_service import run_sector_research
from . import llm, prompts


def _format_kpi_summary(placements: Dict) -> List[str]:
    """Compact bullet list of the most informative KPI placements."""
    if not placements:
        return []
    # Prioritize quality + valuation + growth placements that have a quartile.
    priority_order = ["revenue_growth", "operating_margin", "ROIC", "fcf_margin",
                      "EV_EBITDA", "FCF_yield", "PFCF", "gross_margin",
                      "capex_pct_revenue"]
    seen = set()
    bullets: List[str] = []
    for kpi in priority_order:
        if kpi not in placements or kpi in seen:
            continue
        p = placements[kpi]
        if p.get("quartile") is None or p.get("target") is None:
            continue
        target_v = p["target"]
        med_v = p["distribution"].get("median")
        # Format depends on whether the metric is a ratio (~0..1) or a multiple.
        is_ratio = abs(target_v) < 5
        target_s = f"{target_v:.1%}" if is_ratio else f"{target_v:.1f}x"
        med_s = f"{med_v:.1%}" if is_ratio else f"{med_v:.1f}x"
        bullets.append(
            f"{kpi}: {target_s} vs cohort median {med_s} — {p.get('interpretation', f'Q{p['quartile']}')}"
        )
        seen.add(kpi)
        if len(bullets) >= 5:
            break
    return bullets


def _format_trends(trends: Dict) -> List[str]:
    out: List[str] = []
    g = trends.get("cohort_revenue_growth_recent")
    if g is not None:
        out.append(f"Cohort revenue growth (recent): {g:+.1%}")
    om_d = trends.get("cohort_op_margin_delta")
    if om_d is not None:
        direction = "expanding" if om_d > 0.01 else "compressing" if om_d < -0.01 else "stable"
        out.append(f"Cohort op margin {direction} ({om_d:+.1%} multi-year)")
    cx_d = trends.get("cohort_capex_delta")
    if cx_d is not None:
        direction = "intensifying" if cx_d > 0.005 else "moderating" if cx_d < -0.005 else "steady"
        out.append(f"Cohort capex intensity {direction} ({cx_d:+.1%} multi-year)")
    return out


def _macro_broadcast_payload() -> Dict:
    """Subscribe to the latest MacroBroadcast (Phase 6) — empty if none yet."""
    snap = cache_get("macro:global", "macro_broadcast")
    return snap.payload if snap and isinstance(snap.payload, dict) else {}


def _pending_news_alerts(ticker: str) -> List[Dict]:
    snap = cache_get(f"news_hot:{ticker}", "news_hot")
    if not snap or not isinstance(snap.payload, dict):
        return []
    return list(snap.payload.get("alerts") or [])[:5]


def _cross_sector_relevance_heuristic(ticker: str, sector: str) -> List[str]:
    """Demo-mode fallback for cross-sector relevance.

    A real model would inspect the value chain and pull through, e.g.,
    NEE for any AI-infrastructure name. We approximate with a tiny
    sector→adjacent-tickers map seeded from the demo universe.
    """
    ds = get_data_service()
    adjacency: Dict[str, List[str]] = {
        # AI infra → grid: NEE (utility) and CAT (industrials power gen).
        "Technology": ["NEE", "CAT"],
        # Banks → tech (digital exposure).
        "Financials": ["NVDA", "MSFT"],
        # Healthcare → consumer staples (insurance/coverage adjacency).
        "Healthcare": ["WMT", "COST"],
        # Energy → industrials.
        "Energy": ["CAT"],
        # Industrials → tech (automation).
        "Industrials": ["NVDA"],
        # Utilities → tech (load growth from data centers).
        "Utilities": ["NVDA", "MSFT"],
        # Consumer → financials (consumer credit).
        "Consumer Discretionary": ["JPM", "V"],
        "Consumer Staples": ["JPM"],
    }
    candidates = adjacency.get(sector, [])
    universe = set(ds.list_tickers())
    return [c for c in candidates if c in universe and c != ticker]


def run_sector_agent(profile: Dict, ratios: Dict) -> AgentFinding:
    """Produce a deeply researched sector view.

    Phase 6: subscribes to the latest MacroBroadcast and pending NewsAlerts,
    and populates `cross_sector_relevance` so PM can pull through related
    names from other sectors.
    """
    ticker = profile.get("ticker")
    if not ticker:
        return AgentFinding(
            agent="Sector Analyst",
            headline="Sector research unavailable.",
            summary="No ticker provided to sector research.",
            confidence=0.3,
        )
    research = run_sector_research(ticker)
    macro_broadcast = _macro_broadcast_payload()
    news_alerts = _pending_news_alerts(ticker)

    # Long-term agent memory (gated by ENABLE_LONG_TERM_MEMORY). Read both
    # files: the company-specific notebook and the sector-wide self-reflection
    # journal (with cross-company patterns filtered to this ticker).
    memory_context = ""
    if settings.enable_long_term_memory:
        try:
            cm = CompanyMemory.for_ticker(ticker)
            sm = SectorMemory.for_sector(profile.get("sector") or "unknown")
            company_block = cm.as_prompt_context(max_chars=2500)
            sector_block = sm.as_prompt_context_for(ticker, max_chars=2500)
            chunks = [b for b in (company_block, sector_block) if b]
            if chunks:
                memory_context = "\n\n".join(chunks)
        except Exception:  # pragma: no cover — memory should never block a memo
            memory_context = ""
    sector = research["sector"]
    sub_industry = research["sub_industry"]
    cohort = research["cohort"]
    regime = research["regime"]
    placements = research["kpi_placements"]
    outliers = research["outliers"]
    trends = research["trends"]
    structure = research["industry_structure"]
    filing_themes = research["cohort_filing_themes"]
    secular = research["sector_secular_trends"]
    sub_themes = research.get("subindustry_themes") or ""
    sub_watch = research.get("subindustry_watch") or ""

    # ---- Build LLM prompt grounded in the computed research ----
    research_for_prompt = {
        "target_ticker": ticker,
        "sector": sector,
        "sub_industry": sub_industry,
        "regime": regime,
        "cohort_peers": cohort["peers"],
        "industry_structure": structure,
        "kpi_placements": {
            k: {
                "target": p.get("target"),
                "median": p.get("distribution", {}).get("median"),
                "quartile": p.get("quartile"),
                "interpretation": p.get("interpretation"),
                "group": p.get("group"),
            }
            for k, p in placements.items()
        },
        "outliers": outliers,
        "trends": trends,
        "secular_trends": secular,
        "subindustry_themes": sub_themes,
        "subindustry_watch": sub_watch,
        "cohort_filing_themes": filing_themes,
        "valuation_lens": research.get("valuation_lens", ""),
    }
    kpi_names = sorted(placements.keys())
    user_prompt = (
        prompts.SECTOR_ANALYST_PROMPT.format(
            sector=sector,
            drivers=", ".join(research["sector_drivers"]),
            kpis=", ".join(kpi_names) if kpi_names else "-",
            valuation_lens=research.get("valuation_lens", ""),
            macro_sensitivities=", ".join(research.get("macro_sensitivities", [])),
            macro_broadcast=json.dumps(macro_broadcast, default=str)[:600] or "{}",
            news_alerts=json.dumps(news_alerts, default=str)[:600] or "[]",
        )
        + ("\n\nPrior context from long-term memory (use to inform but do not over-anchor):\n"
           + memory_context if memory_context else "")
        + "\n\nDeep research payload (cohort math + regime + filing themes):\n"
        + json.dumps(research_for_prompt, default=str)[:4500]
    )

    # Sector agents share OPENAI_SECTOR_MODEL (gpt-5.4 by default).
    llm_out = llm.chat_json(
        user_prompt, system=prompts.PM_SYSTEM, route="cheap",
        model=settings.openai_sector_model,
    )

    if llm_out:
        cross_sector = llm_out.get("cross_sector_relevance") or []
        if not cross_sector:
            cross_sector = _cross_sector_relevance_heuristic(ticker, sector)
        # Stash macro alignment + cross-sector relevance into the finding's
        # data payload so the PM can read both downstream without a second call.
        finding_data = dict(research)
        finding_data["cross_sector_relevance"] = cross_sector
        finding_data["macro_alignment"] = llm_out.get("macro_alignment", "")
        finding_data["macro_broadcast"] = macro_broadcast
        finding_data["pending_news_alerts"] = news_alerts
        finding = AgentFinding(
            agent="Sector Analyst",
            headline=llm_out.get("headline", f"{sub_industry} cohort placement"),
            summary=llm_out.get("summary", ""),
            key_points=llm_out.get("key_points", []),
            confidence=float(llm_out.get("confidence", 0.75)),
            sources=[
                f"sector_config:{sector}",
                *[f"peer:{p}" for p in cohort["peers"][:6]],
                *[f"filing:{p}" for p in cohort["peers"][:3]],
            ],
            data=finding_data,
        )
        return finding

    # ---------- Deterministic fallback grounded in research payload ----------
    headline_bits: List[str] = [f"{sub_industry} cohort, regime: {regime}"]
    if structure.get("concentration_label"):
        headline_bits.append(f"{structure['concentration_label']} (HHI {structure.get('hhi_revenue', 0):.2f})")
    headline = " · ".join(headline_bits)

    summary_lines: List[str] = []
    summary_lines.append(
        f"{sector} / {sub_industry} regime read: {regime}. "
        f"Cohort of {cohort['size']} peers selected on {cohort['selection_basis']} basis."
    )
    if sub_themes:
        summary_lines.append(f"Sub-industry dynamics: {sub_themes}")
    if outliers:
        leader_bits = []
        if outliers.get("growth_leader"):
            leader_bits.append(f"growth leader {outliers['growth_leader']}")
        if outliers.get("margin_leader"):
            leader_bits.append(f"margin leader {outliers['margin_leader']}")
        if outliers.get("valuation_cheapest"):
            leader_bits.append(f"valuation cheapest {outliers['valuation_cheapest']}")
        if leader_bits:
            summary_lines.append("Cohort outliers: " + ", ".join(leader_bits) + ".")
    if secular:
        summary_lines.append("Secular trends: " + "; ".join(secular[:2]) + ".")
    if sub_watch:
        summary_lines.append(f"Watch: {sub_watch}.")

    key_points: List[str] = []
    key_points.extend(_format_kpi_summary(placements))
    key_points.extend(_format_trends(trends))
    if filing_themes:
        top_themes = ", ".join(f"{t['theme']} ({t['cohort_mentions']})" for t in filing_themes[:3])
        key_points.append(f"Cohort-wide filing themes: {top_themes}.")
    if not key_points:
        key_points = ["See cohort research payload for KPI placements."]

    cross_sector = _cross_sector_relevance_heuristic(ticker, sector)
    if macro_broadcast.get("favored_sectors") and sector in macro_broadcast["favored_sectors"]:
        macro_alignment = "favored"
    elif macro_broadcast.get("pressured_sectors") and sector in macro_broadcast["pressured_sectors"]:
        macro_alignment = "pressured"
    else:
        macro_alignment = "neutral"

    finding_data = dict(research)
    finding_data["cross_sector_relevance"] = cross_sector
    finding_data["macro_alignment"] = macro_alignment
    finding_data["macro_broadcast"] = macro_broadcast
    finding_data["pending_news_alerts"] = news_alerts

    return AgentFinding(
        agent="Sector Analyst",
        headline=headline,
        summary=" ".join(summary_lines),
        key_points=key_points,
        confidence=0.78,
        sources=[
            f"sector_config:{sector}",
            *[f"peer:{p}" for p in cohort["peers"][:6]],
            *[f"filing:{p}" for p in cohort["peers"][:3]],
        ],
        data=finding_data,
    )
