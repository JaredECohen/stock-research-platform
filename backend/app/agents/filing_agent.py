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

    Recognized shapes (cumulative — any of these survives):
      - `["bullet 1", "bullet 2", ...]` — already flat.
      - `[{"category": ..., "items": [...]}, ...]` — categorized lists.
      - `[{"point": "..."}]` / `[{"text": "..."}]` — single-text dicts.
      - `[{"detail": "...", "bullet": "..."}]` — common variant keys.
      - Plain `dict`: top-level grouping (e.g.
        `{"highlights": [...], "risks": [...]}`) — flatten each group's
        items with the group key as a category prefix.

    Also recognizes string-only items that look like list separators
    ("•", "*", "-") and strips them.
    """
    BULLET_PREFIXES = ("• ", "* ", "- ", "● ", "‣ ")

    def _clean(s: str) -> str:
        s = str(s).strip()
        for p in BULLET_PREFIXES:
            if s.startswith(p):
                s = s[len(p):]
                break
        return s

    out: List[str] = []

    def _add(cat: str, text: Any) -> None:
        s = _clean(text)
        if not s or len(out) >= 15:
            return
        out.append(f"{cat}: {s}" if cat else s)

    def _walk(entry: Any, cat: str = "") -> None:
        if isinstance(entry, str):
            _add(cat, entry)
        elif isinstance(entry, dict):
            local_cat = (
                str(entry.get("category") or entry.get("title") or
                    entry.get("section") or "").strip()
            ) or cat
            items = (
                entry.get("items") or entry.get("points") or
                entry.get("bullets") or entry.get("highlights")
            )
            if isinstance(items, list):
                for item in items:
                    _walk(item, local_cat)
                return
            if isinstance(items, str):
                _add(local_cat, items)
                return
            # No nested list — try common single-text keys.
            for key in ("point", "text", "detail", "bullet", "summary"):
                if entry.get(key):
                    _add(local_cat, entry[key])
                    return
            # Concatenate any string-valued fields as a last resort.
            text = " — ".join(
                str(v) for v in entry.values()
                if isinstance(v, str) and 5 <= len(v) <= 240
            )
            if text:
                _add(local_cat, text)

    if isinstance(raw, list):
        for entry in raw:
            _walk(entry)
            if len(out) >= 15:
                break
    elif isinstance(raw, dict):
        # Top-level grouping — flatten each group.
        for k, v in raw.items():
            if isinstance(v, list):
                for item in v:
                    _walk(item, str(k))
            elif isinstance(v, str):
                _add(str(k), v)
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
