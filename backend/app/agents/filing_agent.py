"""Filing analyst agent."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import settings
from ..schemas import AgentFinding
from ..services import retrieval_service
from . import llm, prompts


def _flatten_key_points(raw: Any) -> List[str]:
    """Coerce a structured key_points payload into a flat List[str].

    Modern Anthropic / OpenAI prompts sometimes return nested category
    objects (`[{"category": "MD&A Highlights", "items": [...]}]`)
    instead of the flat list our `AgentFinding` schema expects. Walk
    the structure, prefix each item with its category when present,
    and cap the total to a sensible bullet count for the UI."""
    out: List[str] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if isinstance(entry, str):
            if entry.strip():
                out.append(entry.strip())
        elif isinstance(entry, dict):
            cat = str(entry.get("category") or entry.get("title") or "").strip()
            items = entry.get("items") or entry.get("points") or []
            if isinstance(items, list):
                for item in items:
                    text = str(item).strip()
                    if not text:
                        continue
                    out.append(f"{cat}: {text}" if cat else text)
            elif isinstance(items, str) and items.strip():
                out.append(f"{cat}: {items.strip()}" if cat else items.strip())
            else:
                # Sometimes the dict itself is the bullet (no items key).
                txt = str(entry.get("point") or entry.get("text") or "").strip()
                if txt:
                    out.append(f"{cat}: {txt}" if cat else txt)
        if len(out) >= 15:
            break
    return out


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

    # Wave 9b — pass real filing content to the LLM. SEC EDGAR returns
    # full document body for the latest 10-K / 10-Q (see
    # SECEdgarProvider.fetch_filing_text). Truncate per-section so the
    # prompt budget is spent on the highest-value text first: MD&A
    # (where management explains numbers), then risk factors, then the
    # business description. Modern Claude Haiku / GPT-4.1-mini handle
    # 40-50 KB of context comfortably.
    risks_list = primary.get("risk_factors") or []
    if isinstance(risks_list, list):
        risks_text = "\n- ".join(str(r)[:600] for r in risks_list[:8])
    else:
        risks_text = str(risks_list)[:6000]
    payload = {
        "ticker": ticker,
        "filing_type": primary.get("type"),
        "period_end": primary.get("period_end"),
        "filing_date": primary.get("filing_date"),
        "business_description": (primary.get("business_description") or "")[:3000],
        "mda": (primary.get("mda") or "")[:12000],
        "risks": risks_text[:6000],
        "segments": primary.get("segments", []),
        "retrieved_chunks": [str(r.get("text") or "")[:1500] for r in retrieved][:3],
    }
    from ..services.research_notes import build_notes_block_for_agent
    notes_block = build_notes_block_for_agent(
        "filing", profile, extra_query="risk factors disclosure litigation regulation",
    )
    from .earnings_agent import _critique_block as _q
    llm_out = llm.chat_json(
        prompts.FILING_ANALYST_PROMPT
        + _q(prior_round_critique)
        + (("\n\n" + notes_block) if notes_block else "")
        + "\n\nFiling context:\n" + json.dumps(payload, default=str)[:32000],
        system=prompts.PM_SYSTEM, route="cheap",
        # Prefer the per-role tool model when the active provider is
        # OpenAI; the new chat_json router drops provider-foreign
        # model names so this is safe under Anthropic too.
        model=settings.openai_tool_model,
        # Filing analyst routinely emits 4-6KB of JSON (headline +
        # multi-paragraph summary + 8-12 key_points). 1200 tokens was
        # truncating mid-response; the unparseable partial JSON sent
        # the agent into the deterministic stub fallback.
        max_tokens=2400,
    )
    if llm_out:
        # The LLM occasionally emits key_points as a list of category
        # dicts (e.g. `[{"category": "MD&A Highlights", "items":
        # [...]}]`) instead of flat strings. Flatten so
        # `AgentFinding.key_points: List[str]` doesn't reject.
        return AgentFinding(
            agent="Filing Analyst",
            headline=llm_out.get("headline", "Filing view"),
            summary=llm_out.get("summary", ""),
            key_points=_flatten_key_points(llm_out.get("key_points", [])),
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
