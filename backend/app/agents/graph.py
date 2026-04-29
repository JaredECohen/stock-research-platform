"""Agent graph (LangGraph-style structure, hand-rolled).

The graph orchestrates a single stock memo generation:

  classify_intent
        │
        ▼
  fan-out specialists
   ├─ sector_agent
   ├─ earnings_agent
   ├─ filing_agent
   ├─ valuation_agent (uses DCF)
   ├─ comps_agent
   ├─ macro_agent
   └─ risk_agent
        │
        ▼
  draft_memo
        │
        ▼
  critic_agent  (Risk Committee)
        │
        ▼
  pm_synthesis (final view, rating, confidence)

We don't pull in a full LangGraph dependency to keep the container slim; the
shape and naming match the LangGraph mental model and could be swapped in
trivially.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, List, Optional

from ..config import settings
from ..schemas import (
    AgentFinding,
    AgentTrace,
    BullBearCase,
    CatalystItem,
    CriticReview,
    DCFResult,
    RiskItem,
    StockMemoOut,
)  # CriticReview imported for the safe-runner fallback path  # noqa: F401
from ..services.fundamentals_service import get_full_financials
from ..services.market_data_service import get_basic_stats
from ..services.transcripts_service import latest_transcript
from ..services.filings_service import get_filings
from ..services.valuation_service import build_comps, build_dcf
from . import llm, prompts
from .comps_agent import run_comps_agent
from .critic_agent import run_critic
from .earnings_agent import run_earnings_agent
from .filing_agent import run_filing_agent
from .macro_agent import run_macro_agent
from .risk_agent import derive_risk_items, run_risk_agent
from .safe_runner import (
    DegradationLog,
    safe_call,
    safe_critic,
    safe_finding,
)
from .sector_agents import run_sector_agent
from .tools import evidence_quality
from .valuation_agent import run_valuation_agent


# ---------------------------------------------------------------------------
# Memo construction
# ---------------------------------------------------------------------------

def _bull_case(profile: Dict, valuation: AgentFinding, dcf: Optional[DCFResult]) -> BullBearCase:
    points: List[str] = []
    drivers = profile.get("drivers") or []
    for d in drivers[:3]:
        points.append(f"Tailwind: {d}")
    if dcf:
        points.append(f"DCF bull case implies ${dcf.bull.implied_share_price:,.2f} ({dcf.bull.upside_pct:+.0%}).")
    points.append("Quality + growth profile supports a premium versus peers.")
    return BullBearCase(
        headline=f"Bull case: durable execution against {drivers[0] if drivers else 'core drivers'}.",
        key_points=points,
    )


def _bear_case(profile: Dict, dcf: Optional[DCFResult]) -> BullBearCase:
    points: List[str] = []
    risks = profile.get("risks") or []
    for r in risks[:3]:
        points.append(f"Headwind: {r}")
    if dcf:
        points.append(f"DCF bear case implies ${dcf.bear.implied_share_price:,.2f} ({dcf.bear.upside_pct:+.0%}).")
    return BullBearCase(
        headline=f"Bear case: thesis breaks on {risks[0] if risks else 'execution slip'}.",
        key_points=points,
    )


def _catalysts(profile: Dict, transcript: Optional[Dict]) -> List[CatalystItem]:
    items: List[CatalystItem] = []
    for d in (profile.get("drivers") or [])[:3]:
        items.append(CatalystItem(
            title=d[:80], detail=d,
            horizon="medium_term", impact="medium",
        ))
    if transcript and transcript.get("period"):
        items.append(CatalystItem(
            title="Next earnings update",
            detail=f"Watch for follow-on commentary on themes from {transcript['period']}.",
            horizon="near_term", impact="medium",
        ))
    return items


def _pm_synthesis(profile: Dict, findings: Dict[str, AgentFinding], dcf: Optional[DCFResult]) -> Dict:
    # LLM synthesis if available
    llm_out = llm.chat_json(
        prompts.PM_SYNTHESIS_PROMPT
        + "\n\nFindings:\n"
        + json.dumps({k: v.model_dump() for k, v in findings.items()}, default=str)[: settings.max_agent_context_chars],
        system=prompts.PM_SYSTEM, route="strong",
    )
    if llm_out and "rating_label" in llm_out:
        return llm_out

    # Deterministic synthesis
    upside = dcf.base.upside_pct if dcf else 0.0
    pos_signals = sum(1 for f in findings.values() if any(k in (f.headline + f.summary).lower()
                                                          for k in ("constructive", "premium", "outperform", "tailwind")))
    neg_signals = sum(1 for f in findings.values() if any(k in (f.headline + f.summary).lower()
                                                          for k in ("pressured", "underperform", "elevated", "compress")))
    score = pos_signals - neg_signals + (1 if upside > 0.10 else (-1 if upside < -0.10 else 0))
    if score >= 2:
        rating = "Bullish"
    elif score == 1:
        rating = "Mixed Positive"
    elif score == 0:
        rating = "Neutral"
    elif score == -1:
        rating = "Mixed Negative"
    else:
        rating = "Bearish"
    confidence = max(40, min(85, 55 + 5 * abs(score)))

    drivers = profile.get("drivers") or []
    thesis = (
        f"{profile.get('company_name', profile.get('ticker', ''))} is leveraged to "
        f"{drivers[0] if drivers else 'core sector tailwinds'} with valuation that triangulates around the DCF base."
    )
    pm_view = (
        f"Research view: {rating}. {thesis} "
        f"Sector framing supports the cohort thesis; valuation-relative read is the main swing factor. "
        f"The risk committee flagged the dominant downside scenarios; portfolio fit depends on macro view."
    )
    return dict(
        final_pm_view=pm_view,
        one_sentence_thesis=thesis,
        rating_label=rating,
        confidence_score=confidence,
    )


def _portfolio_fit(profile: Dict, rating: str) -> str:
    sector = profile.get("sector", "")
    return (
        f"In a balanced model portfolio, {profile.get('ticker', '')} fits the '{sector}' sleeve. "
        f"With a '{rating}' research view, sizing is governed by the user's max position size and risk level."
    )


# ---------------------------------------------------------------------------
# Public graph entry point
# ---------------------------------------------------------------------------

def run_stock_memo(
    ticker: str, *, scenario: str = "soft_landing", force_refresh: bool = False,
) -> StockMemoOut:
    """Generate a stock memo. When `force_refresh=True`, every cached snapshot
    in the dependency tree is bypassed; otherwise, fundamentals/sector/comps/DCF
    are read from the snapshot cache when fresh.
    """
    # Fundamentals MUST succeed — without a profile we can't even identify
    # the company, so this is an unrecoverable error and we re-raise.
    fin = get_full_financials(ticker, force_refresh=force_refresh)
    profile = fin["profile"]
    if not profile:
        raise ValueError(f"Unknown ticker: {ticker}")
    ratios = fin.get("ratios", {}) or {}

    # Everything below this point goes through the safe-runner: a failure in
    # any single specialist becomes a typed fallback rather than killing the
    # memo. Failures are accumulated into `degradation` and surfaced on the
    # memo's `degraded_agents` field.
    degradation = DegradationLog()

    transcript = safe_call(latest_transcript, ticker, fallback=None,
                           name="Transcript Service", log_to=degradation)
    filings = safe_call(get_filings, ticker, fallback=[],
                        name="Filings Service", log_to=degradation)
    earnings = fin.get("earnings", {})

    dcf = safe_call(build_dcf, ticker, force_refresh=force_refresh, fallback=None,
                    name="DCF Engine", log_to=degradation)
    comps = safe_call(build_comps, ticker, force_refresh=force_refresh, fallback=None,
                      name="Comps Engine", log_to=degradation)

    sector_finding = safe_finding("Sector Analyst", run_sector_agent,
                                  profile, ratios, log_to=degradation)
    earnings_finding = safe_finding("Earnings Analyst", run_earnings_agent,
                                    profile, transcript, earnings, log_to=degradation)
    filing_finding = safe_finding("Filing Analyst", run_filing_agent,
                                  profile, filings, log_to=degradation)
    valuation_finding = safe_finding("Valuation Analyst", run_valuation_agent,
                                     profile, ratios, dcf, log_to=degradation)
    comps_finding = safe_finding("Comps Analyst", run_comps_agent,
                                 profile, comps, log_to=degradation)
    macro_finding = safe_finding("Macro Analyst", run_macro_agent,
                                 profile, scenario, log_to=degradation)
    risk_finding = safe_finding(
        "Risk Analyst", run_risk_agent,
        profile, ratios, (dcf.summary if dcf else None), log_to=degradation,
    )

    findings = {
        "sector": sector_finding,
        "earnings": earnings_finding,
        "filing": filing_finding,
        "valuation": valuation_finding,
        "comps": comps_finding,
        "macro": macro_finding,
        "risk": risk_finding,
    }

    bull = safe_call(_bull_case, profile, valuation_finding, dcf,
                     fallback=BullBearCase(headline="Bull case unavailable.", key_points=[]),
                     name="Bull Case Builder", log_to=degradation)
    bear = safe_call(_bear_case, profile, dcf,
                     fallback=BullBearCase(headline="Bear case unavailable.", key_points=[]),
                     name="Bear Case Builder", log_to=degradation)
    catalysts = safe_call(_catalysts, profile, transcript, fallback=[],
                          name="Catalyst Builder", log_to=degradation)
    risks = safe_call(derive_risk_items, profile, fallback=[],
                      name="Risk Item Builder", log_to=degradation)
    thesis_breakers = [r for r in risks if r.severity == "high"][:3]

    synth = safe_call(
        _pm_synthesis, profile, findings, dcf,
        fallback={
            "final_pm_view": "PM synthesis unavailable; relying on specialist findings only.",
            "one_sentence_thesis": f"Research draft for {profile.get('ticker', ticker)}.",
            "rating_label": "Neutral",
            "confidence_score": 50,
        },
        name="PM Synthesis", log_to=degradation,
    )
    rating = synth.get("rating_label", "Neutral")
    raw_confidence = float(synth.get("confidence_score", 60))

    dcf_summary = {}
    if dcf:
        dcf_summary = dict(
            base_implied_price=dcf.base.implied_share_price,
            bull_implied_price=dcf.bull.implied_share_price,
            bear_implied_price=dcf.bear.implied_share_price,
            base_upside=dcf.base.upside_pct,
            wacc=dcf.base.assumptions.wacc,
            terminal_growth=dcf.base.assumptions.terminal_growth,
            summary=dcf.summary,
        )

    sources = [
        f"profile:{profile.get('ticker')}",
        f"financials:{profile.get('ticker')}",
    ]
    if transcript:
        sources.append(f"transcript:{transcript.get('period', '')}")
    for f in filings or []:
        sources.append(f"filing:{f.get('accession_number', f.get('type', ''))}")
    if comps:
        for p in comps.peers:
            sources.append(f"peer:{p.ticker}")
    if dcf:
        sources.append("dcf:base")

    # Dampen PM confidence by source-quality. A memo evidenced by filings +
    # transcripts + financials lands near 1.0; one leaning on news/social
    # gets a meaningful penalty. Prevents over-confident takes from thin evidence.
    ev_q = evidence_quality(sources)
    blended_confidence = max(20.0, min(95.0, raw_confidence * (0.6 + 0.4 * ev_q)))

    memo = StockMemoOut(
        ticker=profile.get("ticker"),
        company_name=profile.get("company_name", ticker),
        sector=profile.get("sector", ""),
        final_pm_view=synth.get("final_pm_view", ""),
        rating_label=rating,
        confidence_score=blended_confidence,
        one_sentence_thesis=synth.get("one_sentence_thesis", ""),
        business_summary=profile.get("business_description", ""),
        sector_agent_view=sector_finding,
        earnings_agent_view=earnings_finding,
        filing_agent_view=filing_finding,
        valuation_agent_view=valuation_finding,
        comps_agent_view=comps_finding,
        macro_sensitivity=macro_finding,
        bull_case=bull,
        bear_case=bear,
        catalysts=catalysts,
        key_risks=risks,
        thesis_breakers=thesis_breakers,
        dcf_summary=dcf_summary,
        portfolio_fit=_portfolio_fit(profile, rating),
        # Stub critic seeded here, then replaced by the real critic call below.
        # safe_critic guarantees a typed CriticReview even if the stub raises.
        risk_committee_challenge=safe_critic(run_critic, {}, log_to=None) or CriticReview(
            overall_assessment="Pending critic review.",
        ),
        final_verdict="",
        scores=dict(
            confidence=blended_confidence,
            raw_confidence=raw_confidence,
            evidence_quality=round(ev_q * 100, 1),
            sector_confidence=sector_finding.confidence * 100,
            valuation_confidence=valuation_finding.confidence * 100,
            risk_confidence=risk_finding.confidence * 100,
        ),
        sources_used=sources,
        generated_at=datetime.utcnow(),
        generation_mode="live" if settings.has_llm and settings.enable_live_data else "demo",
        degraded_agents=degradation.degraded_agents(),
    )

    # Run critic on a draft of the memo (pass dict to avoid recursion).
    # safe_critic upgrades exceptions into a typed "critic unavailable" review
    # so a flaky Anthropic call doesn't kill the memo.
    draft_for_critic = memo.model_dump()
    critic = safe_critic(run_critic, draft_for_critic, log_to=degradation)
    if critic:
        memo.risk_committee_challenge = critic
    # Refresh degraded_agents in case the critic recorded a failure.
    memo.degraded_agents = degradation.degraded_agents()

    # Phase 6: pull through cross-sector relevance from the sector agent's
    # finding into the PM memo so users see related-name implications without
    # a second model call. Cohort placement is already in the sector view.
    cross_relevance = []
    if isinstance(sector_finding.data, dict):
        cross_relevance = sector_finding.data.get("cross_sector_relevance") or []
    cross_relevance_blurb = (
        f" Cross-sector pull-through: {', '.join(cross_relevance)}." if cross_relevance else ""
    )
    cohort_blurb = ""
    if isinstance(sector_finding.data, dict):
        kpi_placements = sector_finding.data.get("kpi_placements") or {}
        if kpi_placements:
            cohort_blurb = " Cohort placement: see sector view for KPI quartile context."

    # Final verdict ties together rating, confidence, and PM view succinctly
    memo.final_verdict = (
        f"PM final view: {rating} (confidence {int(memo.confidence_score)}). "
        f"{memo.one_sentence_thesis}"
        f"{cohort_blurb}{cross_relevance_blurb} "
        f"Watch items: {', '.join(r.title for r in thesis_breakers) or 'none flagged.'}"
    )
    if cross_relevance and isinstance(memo.scores, dict):
        memo.scores = {**memo.scores, "cross_sector_relevance_count": float(len(cross_relevance))}
    return memo


# ---------------------------------------------------------------------------
# Agent trace helper
# ---------------------------------------------------------------------------

def default_agent_trace(intent: str) -> List[AgentTrace]:
    base = [
        AgentTrace(agent="PM Orchestrator", status="done", detail=f"Intent classified as {intent}."),
    ]
    if intent in ("single_stock_analysis", "stock_comparison"):
        base += [
            AgentTrace(agent="Sector Analyst", status="done", detail="Sector framework applied."),
            AgentTrace(agent="Earnings Analyst", status="done", detail="Latest transcript reviewed."),
            AgentTrace(agent="Filing Analyst", status="done", detail="10-K/10-Q analyzed."),
            AgentTrace(agent="Valuation Analyst", status="done", detail="DCF + multiples interpreted."),
            AgentTrace(agent="Comps Analyst", status="done", detail="Peer median + premium/discount."),
            AgentTrace(agent="Macro Analyst", status="done", detail="Macro mapping applied."),
            AgentTrace(agent="Risk Committee", status="done", detail="Critic reviewed and flagged challenges."),
        ]
    elif intent == "portfolio_construction":
        base += [
            AgentTrace(agent="Screener Agent", status="done", detail="Universe scored against scenario fit."),
            AgentTrace(agent="Portfolio Construction Agent", status="done", detail="Diversified weights enforced."),
            AgentTrace(agent="Risk Committee", status="done", detail="Concentration + risk reviewed."),
        ]
    elif intent == "thematic_screen":
        base += [AgentTrace(agent="Screener Agent", status="done", detail="Theme bias applied to PM scores.")]
    elif intent == "macro_question":
        base += [AgentTrace(agent="Macro Analyst", status="done", detail="Scenario template + snapshot.")]
    return base
