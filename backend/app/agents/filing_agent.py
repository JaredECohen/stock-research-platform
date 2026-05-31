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

    # Wave 10 — vector retrieval first (when chunks exist), BM25 as a
    # belt-and-suspenders fallback. Question-specific query: when a
    # PM follow-up is supplied, use that as the retrieval query so the
    # chunks line up with the question; otherwise stay on the static
    # thesis-relevance prompt.
    retrieval_query = (
        prior_round_critique
        if prior_round_critique and len(prior_round_critique) > 8
        else "risk factors growth strategy thesis"
    )
    retrieved: List[Dict] = []
    try:
        from ..services import vector_store
        vec_hits = vector_store.search(
            retrieval_query, ticker=ticker, source_types=["filing"], top_k=4,
        )
        retrieved = [{"text": h["text"], "section": h.get("section")} for h in vec_hits]
    except Exception:
        retrieved = []
    if not retrieved:
        retrieved = retrieval_service.search(ticker, retrieval_query, limit=4) or []
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
        # Wave 10 — emit citations for retrieved filing chunks +
        # the primary filing's risk_factors / mda sections.
        from ..schemas import Citation
        accession = primary.get("accession_number", "")
        evidence: List[Citation] = []
        if accession and primary.get("mda"):
            evidence.append(Citation(
                kind="filing", ref=accession, section="mda",
                excerpt=str(primary.get("mda") or "")[:300],
            ))
        if accession and primary.get("risk_factors"):
            risks_list = primary.get("risk_factors") or []
            if isinstance(risks_list, list) and risks_list:
                evidence.append(Citation(
                    kind="filing", ref=accession, section="risk_factors",
                    excerpt=str(risks_list[0])[:300],
                ))
        for chunk in (retrieved or [])[:4]:
            chunk_section = (
                chunk.get("section") if isinstance(chunk, dict) else None
            )
            chunk_text = (
                chunk.get("text") if isinstance(chunk, dict) else str(chunk)
            )
            if chunk_text:
                evidence.append(Citation(
                    kind="filing",
                    ref=accession or ticker,
                    section=chunk_section,
                    excerpt=str(chunk_text)[:300],
                ))
        return AgentFinding(
            agent="Filing Analyst",
            headline=llm_out.get("headline", "Filing view"),
            summary=llm_out.get("summary", ""),
            key_points=_flatten_key_points(llm_out.get("key_points", [])),
            confidence=float(llm_out.get("confidence", 0.7)),
            sources=[f"filing:{accession}"],
            evidence=evidence[:6],
        )

    # Deterministic fallback. Skip past SEC boilerplate openers and
    # prefer retrieved chunks (when the vector store has indexed this
    # filing) over the front-of-section truncation, which routinely
    # serves the "The following discussion should be read in conjunction
    # with our Consolidated Financial Statements..." legalese.
    segments = primary.get("segments", []) or profile.get("segments", []) or []
    seg_text = ", ".join(s if isinstance(s, str) else s.get("name", "") for s in segments)[:200]

    mda_snippet = _substantive_filing_snippet(primary.get("mda", ""), retrieved)
    risks = _substantive_risk_factors(primary.get("risk_factors") or [], top_n=3)

    summary_parts = [
        f"{primary.get('type', '10-K')} dated {primary.get('filing_date', '—')}.",
    ]
    if seg_text:
        summary_parts.append(f"Segments: {seg_text}.")
    if mda_snippet:
        summary_parts.append(f"MD&A: {mda_snippet}")
    else:
        summary_parts.append(
            "LLM analyst couldn't run; filing body indexed for retrieval but "
            "no substantive MD&A snippet was extracted in the deterministic path."
        )
    summary = " ".join(summary_parts)

    key_points = [f"Risk: {r}" for r in risks] or ["See filing for detail."]
    return AgentFinding(
        agent="Filing Analyst",
        headline=f"{ticker} {primary.get('type', '10-K')} highlights",
        summary=summary,
        key_points=key_points,
        confidence=0.6,
        sources=[f"filing:{primary.get('accession_number', '')}"],
    )


# Phrases used to filter out SEC boilerplate from MD&A and Risk Factor
# extracts. These appear verbatim across every 10-K and crowd out any
# real signal when we naively take the first N characters of a section.
_FILING_BOILERPLATE_PREFIXES = (
    "the following discussion should be read in conjunction",
    "discussion regarding our financial condition and results of operations",
    "as previously discussed, our actual results could differ materially",
    "you should carefully consider the risks",
    "the risks and uncertainties described below",
    "in addition to the other information set forth in this report",
    "investing in our common stock involves a high degree of risk",
)

_RISK_GENERIC_PREFIXES = (
    "as previously discussed, our actual results",
    "you should carefully consider the risks",
    "the risks and uncertainties described below",
    "investing in our common stock involves",
    "many factors affect more than one category",
)


def _substantive_filing_snippet(
    mda_text: str, retrieved_chunks: List[Dict],
) -> str:
    """Pick a snippet of MD&A worth showing.

    Preference:
      1. The highest-scoring retrieved chunk that doesn't start with
         boilerplate. Vector retrieval already lands semantically near
         the thesis query, so this is usually the substantive content.
      2. Fall back to scanning past boilerplate prefixes in the raw
         MD&A — split on "Results of Operations" / "Liquidity" / similar
         section markers and pull the next 280 chars.
      3. Last resort: empty string (handled by caller).
    """
    for chunk in retrieved_chunks or []:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        low = text.lower()[:200]
        if any(low.startswith(p) for p in _FILING_BOILERPLATE_PREFIXES):
            continue
        return text[:400]

    if not mda_text:
        return ""

    # Try to skip past the standard MD&A intro by anchoring on
    # substantive headers; pull the next 280 chars after the first hit.
    markers = (
        "Results of Operations",
        "Liquidity and Capital Resources",
        "Revenue", "Operating Income", "Segment",
    )
    for marker in markers:
        idx = mda_text.find(marker)
        if idx > 0:
            return mda_text[idx : idx + 360].strip()

    # No marker hit — just strip leading boilerplate paragraphs and
    # take whatever's left.
    paragraphs = [p.strip() for p in mda_text.split("\n") if p.strip()]
    for p in paragraphs:
        low = p.lower()[:200]
        if not any(low.startswith(b) for b in _FILING_BOILERPLATE_PREFIXES):
            return p[:360]
    return ""


def _substantive_risk_factors(
    raw_risks: List[Any], *, top_n: int = 3,
) -> List[str]:
    """Filter out generic risk-section boilerplate.

    The Risk Factors section in every 10-K opens with several paragraphs
    of "you should carefully consider..." legalese before the actual
    named risks. We skip rows that start with those phrases and prefer
    ones that name a specific business risk.
    """
    keep: List[str] = []
    for r in raw_risks:
        text = (str(r) or "").strip()
        if not text:
            continue
        low = text.lower()[:200]
        if any(low.startswith(p) for p in _RISK_GENERIC_PREFIXES):
            continue
        keep.append(text)
        if len(keep) >= top_n:
            break
    return keep
