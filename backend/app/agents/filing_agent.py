"""Filing analyst agent."""
from __future__ import annotations

import json
import logging
from typing import Any

from ..config import settings
from ..schemas import AgentFinding
from ..services import retrieval_service
from . import llm, prompts
from .log_safety import log_safely, redact, safe_exc
from .safe_runner import note_soft
from .source_ledger import register_source

log = logging.getLogger(__name__)


def _retrieved_source(chunk: dict, position: int) -> dict[str, Any]:
    """Keep a passage's own identity; a primary filing is not its provenance."""
    meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
    accession = meta.get("accession") or meta.get("accession_number")
    source_id = chunk.get("source_id")
    if isinstance(accession, str) and accession.strip():
        ref = accession.strip()
    elif isinstance(source_id, str) and source_id.strip() and source_id.strip() != "10K":
        # BM25 stores the accession directly, unlike vector rows' DB IDs.
        ref = source_id.strip()
    elif isinstance(source_id, int) and not isinstance(source_id, bool):
        ref = f"filing_doc:{source_id}"
    else:
        ref = f"unattributed_chunk:{chunk.get('id') or position}"
    return {
        "ref": ref,
        "chunk_id": chunk.get("id"),
        "source_id": source_id,
        "section": chunk.get("section"),
        "period_end": str(chunk["period_end"]) if chunk.get("period_end") else None,
        "filing_date": meta.get("filing_date"),
        "filing_type": meta.get("filing_type"),
        "url": meta.get("url") or chunk.get("url"),
    }


def _flatten_key_points(raw: Any) -> list[str]:
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

    out: list[str] = []

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
    profile: dict, filings: list[dict],
    *, prior_round_critique: str | None = None,
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
    retrieved: list[dict] = []
    # Flags that ride on `data` whichever path (LLM or deterministic)
    # produces the finding. (b) RP-001: a retrieval failure thins what the
    # reader gets — the LLM sees only the front-of-section truncation and
    # the deterministic path loses its substantive MD&A snippet — so it is
    # recorded on the finding and on the memo banner (`note_soft` no-ops
    # outside a memo run) rather than swallowed.
    finding_flags: dict[str, Any] = {}

    def _retrieval_failed(layer: str, exc: BaseException) -> None:
        log_safely(log, f"Filing Analyst {layer} retrieval failed for {ticker}", exc)
        note_soft(
            "Filing Analyst", f"retrieval unavailable: {redact(exc)}",
            kind=type(exc).__name__,
        )
        finding_flags["retrieval_failed"] = safe_exc(exc)

    try:
        from ..services import vector_store
        # `ticker` is `profile.get("ticker", "")` — empty when the profile
        # lookup missed. Skipping the call keeps the (correct) BM25
        # fallback below without logging a spurious global-scan warning;
        # `vector_store.search` refuses the unscoped query either way.
        vec_hits = vector_store.search(
            retrieval_query, ticker=ticker, source_types=["filing"], top_k=4,
        ) if ticker else []
        retrieved = [
            {
                "text": h["text"],
                "section": h.get("section"),
                "source_type": "filing",
                "id": h.get("id"),
                "source_id": h.get("source_id"),
                "period_end": h.get("period_end"),
                "meta": h.get("meta"),
            }
            for h in vec_hits
        ]
    except Exception as exc:
        _retrieval_failed("vector", exc)
        retrieved = []
    if not retrieved:
        # BM25 fallback returns filings + transcripts + news in one
        # scored list (see retrieval_service._chunks_for_ticker). A
        # high-scoring news article would otherwise leak in as a fake
        # MD&A snippet — observed in prod with MSTR news appearing in
        # an ADBE memo. Filter at the call site so downstream code
        # never sees off-source chunks.
        #
        # The BM25 layer reads the news feed too, so a dead news provider
        # used to surface here as a hard Filing Analyst failure (the whole
        # section replaced by the "unavailable" stub). The filing body is
        # still on hand, so the analyst runs without retrieved chunks and
        # the loss is recorded softly instead.
        try:
            raw = retrieval_service.search(ticker, retrieval_query, limit=8) or []
        except Exception as exc:
            _retrieval_failed("BM25", exc)
            raw = []
        retrieved = [c for c in raw if _is_filing_chunk(c)][:4]
    primary = next((f for f in filings if f.get("type") == "10-K"), filings[0])
    retrieved_sources = [_retrieved_source(c, i) for i, c in enumerate(retrieved, 1)]
    if retrieved_sources:
        finding_flags["retrieved_sources"] = retrieved_sources
    unattributed = [s["ref"] for s in retrieved_sources if s["ref"].startswith("unattributed_chunk:")]
    if unattributed:
        finding_flags["unattributed_retrieved_chunks"] = unattributed
        log.warning("Filing Analyst %s: %d unattributed retrieved chunks: %s",
                    ticker, len(unattributed), ", ".join(unattributed))
    source_refs = list(dict.fromkeys(
        [str(primary.get("accession_number") or "")]
        + [s["ref"] for s in retrieved_sources if s["ref"] not in unattributed]
    ))
    finding_sources = [f"filing:{ref}" for ref in source_refs if ref]

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
    # W2b 7(a): the filing text the analyst is given is a primary source.
    # Each retrieved passage is registered under its own ref, in full (the
    # deterministic path below can quote any of the four).
    filing_ref = f"filing:{primary.get('accession_number') or primary.get('type') or ticker}"
    register_source("filing", filing_ref, payload, exclude_keys=("retrieved_chunks",))
    for chunk, source in zip(retrieved, retrieved_sources):
        register_source("filing", f"chunk:{source.get('chunk_id') or source['ref']}",
                        {"text": str(chunk.get("text") or "")})
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
        evidence: list[Citation] = []
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
        for chunk, source in zip((retrieved or [])[:4], retrieved_sources[:4]):
            chunk_section = (
                chunk.get("section") if isinstance(chunk, dict) else None
            )
            chunk_text = (
                chunk.get("text") if isinstance(chunk, dict) else str(chunk)
            )
            if chunk_text:
                evidence.append(Citation(
                    kind="other" if source["ref"] in unattributed else "filing",
                    ref=source["ref"],
                    section=chunk_section,
                    excerpt=str(chunk_text)[:300],
                ))
        return AgentFinding(
            agent="Filing Analyst",
            headline=llm_out.get("headline", "Filing view"),
            summary=llm_out.get("summary", ""),
            key_points=_flatten_key_points(llm_out.get("key_points", [])),
            confidence=float(llm_out.get("confidence", 0.7)),
            sources=finding_sources,
            evidence=evidence[:6],
            data=dict(finding_flags),
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
    # The snippet can come from past the prompt's 12,000-character cut.
    register_source("filing", filing_ref, {"mda_snippet": mda_snippet, "risk_factors": risks})

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
    if settings.has_llm:
        # (b) RP-001: an LLM was configured and returned nothing usable, so
        # this MD&A/risk-factor extract stands in for the analyst's read.
        # The graph promotes the flag into `degraded_agents`; without keys
        # the extract IS the design and is not flagged.
        finding_flags["deterministic_fallback"] = (
            "Filing LLM returned no usable output; deterministic MD&A / "
            "risk-factor extract shipped instead."
        )
    return AgentFinding(
        agent="Filing Analyst",
        headline=f"{ticker} {primary.get('type', '10-K')} highlights",
        summary=summary,
        key_points=key_points,
        confidence=0.6,
        sources=finding_sources,
        data=dict(finding_flags),
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


# Sections that legitimately belong in a "MD&A snippet" — everything
# else (news articles, transcript chunks) gets filtered out before the
# snippet picker runs. Without this, the BM25 fallback in
# retrieval_service.search mixes filing/transcript/news chunks into one
# scored list, and a high-scoring news article would surface as
# "MD&A: <wrong-ticker headline>" (seen in prod: MSTR news leaking
# into an ADBE memo).
_FILING_SECTIONS = frozenset({
    "mda", "business_description", "risk_factors",
    "legal_or_regulatory", "financial_highlights",
    # SEC Item 1-N headers (Item 1, Item 1A, Item 7, etc.) — produced
    # by sec_edgar_provider's section extractor.
    *(f"item_{i}" for i in range(1, 17)),
})


def _is_filing_chunk(chunk: dict) -> bool:
    """True iff `chunk` is sourced from a filing (10-K / 10-Q / 8-K)
    rather than a transcript or news article. Two signals are
    available depending on which retriever produced the chunk:
      - `source_type` (vector_store path): "filing" / "transcript" / "news"
      - `section` (BM25 path): "mda" / "risk_factors" / "article" / ...
    """
    src = (chunk.get("source_type") or "").lower()
    if src and src != "filing":
        return False
    section = (chunk.get("section") or "").lower()
    if section and section in _FILING_SECTIONS:
        return True
    # If neither signal is present, treat as filing (vector_store path
    # already filtered by source_types=["filing"]).
    return src == "filing" or not section


def _substantive_filing_snippet(
    mda_text: str, retrieved_chunks: list[dict],
) -> str:
    """Pick a snippet of MD&A worth showing.

    Preference:
      1. The highest-scoring retrieved chunk that doesn't start with
         boilerplate AND is from a filing (not a news article or
         transcript). Vector retrieval already lands semantically
         near the thesis query, so this is usually the substantive
         content.
      2. Fall back to scanning past boilerplate prefixes in the raw
         MD&A — split on "Results of Operations" / "Liquidity" / similar
         section markers and pull the next 280 chars.
      3. Last resort: empty string (handled by caller).
    """
    for chunk in retrieved_chunks or []:
        if not _is_filing_chunk(chunk):
            continue
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
    raw_risks: list[Any], *, top_n: int = 3,
) -> list[str]:
    """Filter out generic risk-section boilerplate.

    The Risk Factors section in every 10-K opens with several paragraphs
    of "you should carefully consider..." legalese before the actual
    named risks. We skip rows that start with those phrases and prefer
    ones that name a specific business risk.
    """
    keep: list[str] = []
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
