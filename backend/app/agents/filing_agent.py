"""Filing analyst agent."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import settings
from ..schemas import AgentFinding
from ..services import retrieval_service
from . import llm, prompts


def run_filing_agent(
    profile: Dict, filings: List[Dict],
    *, prior_round_critique: Optional[str] = None,
) -> AgentFinding:
    ticker = profile.get("ticker", "")
    if not filings:
        return AgentFinding(
            agent="Filing Analyst",
            headline="No filings cached for this ticker.",
            summary="No 10-K/10-Q on file in the demo dataset.",
            key_points=[],
            confidence=0.3,
            sources=[],
        )

    # Use retrieval to grab the most thesis-relevant filing chunks
    retrieved = retrieval_service.search(ticker, "risk factors growth strategy thesis", limit=4)
    primary = next((f for f in filings if f.get("type") == "10-K"), filings[0])

    payload = {
        "ticker": ticker,
        "filing_type": primary.get("type"),
        "period_end": primary.get("period_end"),
        "business_description": primary.get("business_description"),
        "mda": primary.get("mda"),
        "risks": primary.get("risk_factors", [])[:5],
        "segments": primary.get("segments", []),
        "retrieved_chunks": [r["text"] for r in retrieved][:4],
    }
    # Wave 7C: discretionary notes tagged for the filing agent.
    from ..services.research_notes import build_notes_block_for_agent
    notes_block = build_notes_block_for_agent(
        "filing", profile, extra_query="risk factors disclosure litigation regulation",
    )
    # Long-doc analyst — uses OPENAI_TOOL_MODEL today; GEMINI_LONGDOC_MODEL
    # is documented as a future Gemini override but not yet routed here.
    from .earnings_agent import _critique_block as _q  # share the helper
    llm_out = llm.chat_json(
        prompts.FILING_ANALYST_PROMPT
        + _q(prior_round_critique)
        + (("\n\n" + notes_block) if notes_block else "")
        + "\n\nFiling context:\n" + json.dumps(payload, default=str)[:3500],
        system=prompts.PM_SYSTEM, route="cheap",
        model=settings.openai_tool_model,
    )
    if llm_out:
        return AgentFinding(
            agent="Filing Analyst",
            headline=llm_out.get("headline", "Filing view"),
            summary=llm_out.get("summary", ""),
            key_points=llm_out.get("key_points", []),
            confidence=float(llm_out.get("confidence", 0.7)),
            sources=[f"filing:{primary.get('accession_number', '')}"],
        )

    # Deterministic fallback
    risks = (primary.get("risk_factors") or [])[:3]
    segments = primary.get("segments", []) or profile.get("segments", []) or []
    seg_text = ", ".join(s if isinstance(s, str) else s.get("name", "") for s in segments)[:200]
    summary = (
        f"{primary.get('type', '10-K')} dated {primary.get('filing_date', '—')}: "
        f"business spans {seg_text or 'core segments'}. "
        f"MD&A reads as: {primary.get('mda', '')[:280]}"
    )
    key_points = [f"Risk: {r}" for r in risks]
    return AgentFinding(
        agent="Filing Analyst",
        headline=f"{ticker} {primary.get('type', '10-K')} highlights",
        summary=summary,
        key_points=key_points or ["See filing for detail."],
        confidence=0.6,
        sources=[f"filing:{primary.get('accession_number', '')}"],
    )
