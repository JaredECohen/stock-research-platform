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
from typing import Any, Dict, List, Optional

from ..cache import cache_get
from ..config import settings
from ..memory import CompanyMemory, SectorMemory
from ..schemas import (
    AgentFinding,
    BullBearAnalysis,
    BullBearCase,
    FalsifiableTest,
)
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


def _coerce_bull_bear_analysis(raw: Any) -> Optional[BullBearAnalysis]:
    """Best-effort parse of an LLM-emitted bull_bear_analysis block.

    Returns None when the payload is missing required fields or malformed.
    Falls back gracefully (graph.py will use the deterministic builder)
    rather than risk a memo failure on an over-strict schema check.
    """
    if not isinstance(raw, dict):
        return None
    bull = raw.get("bull_case") or {}
    bear = raw.get("bear_case") or {}
    if not isinstance(bull, dict) or not isinstance(bear, dict):
        return None
    if not (bull.get("headline") and bear.get("headline")):
        return None
    tests_raw = raw.get("falsifiable_tests") or []
    falsifiable: List[FalsifiableTest] = []
    for t in tests_raw:
        if not isinstance(t, dict):
            continue
        side = t.get("invalidates_side")
        statement = (t.get("statement") or "").strip()
        if side not in ("bull", "bear") or not statement:
            continue
        falsifiable.append(FalsifiableTest(
            statement=statement, invalidates_side=side,
        ))
    lean = raw.get("sector_lean")
    if lean not in ("bull", "bear", "balanced"):
        lean = "balanced"
    try:
        return BullBearAnalysis(
            bull_case=BullBearCase(
                headline=str(bull.get("headline", ""))[:240],
                key_points=[str(p) for p in (bull.get("key_points") or [])][:8],
            ),
            bear_case=BullBearCase(
                headline=str(bear.get("headline", ""))[:240],
                key_points=[str(p) for p in (bear.get("key_points") or [])][:8],
            ),
            key_disagreement=str(raw.get("key_disagreement", "")).strip(),
            falsifiable_tests=falsifiable,
            sector_synthesis=str(raw.get("sector_synthesis", "")).strip(),
            sector_lean=lean,
        )
    except Exception:
        return None


def _deterministic_bull_bear_analysis(
    profile: Dict, research: Dict,
) -> BullBearAnalysis:
    """Cohort-grounded fallback when the LLM is offline or output is malformed.

    Bear-first by construction (we build it before bull), uses cohort
    placement + risks/drivers from the profile so it's not just
    template prose. Always emits a falsifiable test on each side so the
    contract is upheld.
    """
    sector = profile.get("sector", "")
    sub_industry = profile.get("sub_industry", "")
    drivers = profile.get("drivers") or []
    risks = profile.get("risks") or []
    placements = research.get("kpi_placements") or {}
    trends = research.get("trends") or {}
    regime = research.get("regime") or "mixed"

    # Bear case — written first.
    bear_points: List[str] = []
    if risks:
        bear_points.extend(f"Headwind: {r}" for r in risks[:3])
    om_d = trends.get("cohort_op_margin_delta")
    if om_d is not None and om_d < -0.005:
        bear_points.append(
            f"Cohort op margin compressing ({om_d:+.1%} multi-year) — competitive intensity is rising."
        )
    val = placements.get("EV_EBITDA") or placements.get("PFCF")
    if val and val.get("quartile") in (3, 4):
        bear_points.append(
            f"Valuation in the {'top' if val['quartile'] == 4 else 'upper'} cohort quartile — "
            f"any execution slip re-rates the multiple."
        )
    if not bear_points:
        bear_points = ["Execution risk on the dominant driver."]
    bear_headline = (
        f"Bear case: {sub_industry or sector} thesis breaks on "
        f"{risks[0] if risks else 'execution'}."
    )
    bear = BullBearCase(headline=bear_headline, key_points=bear_points)

    # Bull case — written second so any leftover slack lands here.
    bull_points: List[str] = []
    if drivers:
        bull_points.extend(f"Tailwind: {d}" for d in drivers[:3])
    growth = placements.get("revenue_growth")
    if growth and growth.get("quartile") in (1, 2):
        bull_points.append(
            f"Revenue growth in the {'top' if growth['quartile'] == 1 else 'upper-half'} "
            f"cohort quartile — share-take story is empirical, not narrative."
        )
    margin = placements.get("operating_margin")
    if margin and margin.get("quartile") in (1, 2):
        bull_points.append(
            "Operating margin above cohort median — quality premium is earned."
        )
    if not bull_points:
        bull_points = ["Quality + growth profile supports a premium versus peers."]
    bull_headline = (
        f"Bull case: durable execution against "
        f"{drivers[0] if drivers else 'sector tailwinds'}."
    )
    bull = BullBearCase(headline=bull_headline, key_points=bull_points)

    # Falsifiable tests — one per side, anchored to cohort observable.
    falsifiable = [
        FalsifiableTest(
            statement=(
                f"Cohort revenue growth turns negative for two consecutive quarters "
                f"AND target margin compresses with it."
            ),
            invalidates_side="bull",
        ),
        FalsifiableTest(
            statement=(
                f"{drivers[0] if drivers else 'Core driver'} re-accelerates and the "
                f"target's cohort margin quartile improves."
            ),
            invalidates_side="bear",
        ),
    ]

    # Decide a sector lean from cohort math: positive growth delta + favorable
    # margin placement → bull; the opposite → bear; else balanced.
    lean = "balanced"
    growth_q = (placements.get("revenue_growth") or {}).get("quartile")
    margin_q = (placements.get("operating_margin") or {}).get("quartile")
    if growth_q in (1, 2) and margin_q in (1, 2):
        lean = "bull"
    elif growth_q in (3, 4) and margin_q in (3, 4):
        lean = "bear"

    synthesis = (
        f"Cohort context for {sub_industry or sector} ({regime} regime): "
        f"the bear case rests on {risks[0] if risks else 'execution'}; the bull "
        f"case rests on {drivers[0] if drivers else 'execution against the dominant driver'}. "
        f"Sector lean is {lean} based on cohort placement; PM should weigh other findings."
    )
    key_disagreement = (
        f"Bears price in cohort margin compression flowing through to this name; "
        f"bulls price in this name continuing to outpace cohort on the dominant driver."
    )

    return BullBearAnalysis(
        bull_case=bull,
        bear_case=bear,
        key_disagreement=key_disagreement,
        falsifiable_tests=falsifiable,
        sector_synthesis=synthesis,
        sector_lean=lean,
    )


def run_sector_agent(
    profile: Dict, ratios: Dict, *,
    prior_round_critique: Optional[str] = None,
) -> AgentFinding:
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
    # Wave 7A + 7B + 7C: discretionary research notes via the unified helper.
    # The helper combines summaries (always-on, ~30 tokens each) + top-K
    # body excerpts (BM25-ranked, hard-capped at 4KB combined). `regime`
    # is a useful extra query keyword for the sector pass.
    from ..services.research_notes import build_notes_block_for_agent
    research_notes_block = build_notes_block_for_agent(
        "sector", profile, extra_query=regime,
    )

    # Wave 9: PM follow-up question (when this is a critique-loop re-fire).
    # Prepended to the prompt so the LLM addresses it directly while still
    # producing a complete sector finding.
    critique_block = ""
    if prior_round_critique:
        critique_block = (
            "\n\n## PM FOLLOW-UP (deep-research round)\n"
            "A senior PM has reviewed your prior round's finding and "
            "asked for additional depth on this specific question. "
            "Address it directly with cohort math, ratio references, or "
            "filing/transcript quotes. Do NOT contradict the prior "
            "finding without articulating exactly what changed your "
            "read.\n\n"
            f"PM follow-up: {prior_round_critique}\n"
        )

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
        + critique_block
        + ("\n\nPrior context from long-term memory (use to inform but do not over-anchor):\n"
           + memory_context if memory_context else "")
        + (("\n\n" + research_notes_block) if research_notes_block else "")
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
        # Wave 3A: pull through the structured bull/bear analysis. If the
        # LLM didn't produce a parseable block, fall back to the
        # cohort-grounded deterministic builder so this contract is
        # always satisfied.
        bb = _coerce_bull_bear_analysis(llm_out.get("bull_bear_analysis"))
        if bb is None:
            bb = _deterministic_bull_bear_analysis(profile, research)
        finding_data["bull_bear_analysis"] = bb.model_dump()
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
    # Wave 3A: contract-satisfying bull/bear analysis even on the no-LLM path.
    finding_data["bull_bear_analysis"] = (
        _deterministic_bull_bear_analysis(profile, research).model_dump()
    )

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
