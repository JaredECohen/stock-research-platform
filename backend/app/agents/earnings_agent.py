"""Earnings call agent."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import settings
from ..schemas import AgentFinding
from . import llm, prompts


def _multi_pass_qa_addendum(
    *, ticker: str, prepared: str, qa: str,
) -> Dict[str, Any]:
    """Wave 10 — chunked second pass over long transcripts.

    The first-pass prompt budgets 10KB prepared + 8KB Q&A. When the
    Q&A alone exceeds 8KB we run a focused second pass over the
    *back half* of Q&A and ask specifically for new themes / unique
    analyst pushback / management deflections not captured by the
    first pass. Returns a dict the main pass can incorporate.

    Returns an empty dict when the transcript fits comfortably or
    when the LLM is unavailable.
    """
    if not settings.has_llm:
        return {}
    qa_text = str(qa or "")
    if len(qa_text) <= 8000:
        return {}
    # Chunk the back half of Q&A — that's where pushback questions
    # tend to land that get truncated by the 8KB budget.
    back_half = qa_text[len(qa_text) // 2:]
    if not back_half.strip():
        return {}
    try:
        out = llm.chat_json(
            "Below is the BACK HALF of an earnings Q&A. The first "
            "pass over this transcript only saw the front half. "
            "Identify themes, reversals, or pushback questions that "
            "appear UNIQUELY in this back half. Be terse — bullet "
            "points only.\n\n"
            "Return JSON: {\n"
            "  back_half_themes: [str, ...],   // 0-4 unique themes\n"
            "  hard_questions: [str, ...],     // analyst pushback "
            "    not addressed earlier\n"
            "  management_deflections: [str, ...]  // questions where "
            "    management dodged or hedged\n}\n\n"
            f"Ticker: {ticker}\n\n"
            f"Back-half Q&A:\n{back_half[:14000]}",
            system="You are an earnings call analyst.",
            route="cheap",
            model=settings.openai_tool_model,
            max_tokens=600,
        )
        if not isinstance(out, dict):
            return {}
        return {
            "back_half_themes": [
                str(t)[:240] for t in (out.get("back_half_themes") or [])
            ][:4],
            "hard_questions": [
                str(t)[:240] for t in (out.get("hard_questions") or [])
            ][:4],
            "management_deflections": [
                str(t)[:240] for t in (out.get("management_deflections") or [])
            ][:4],
        }
    except Exception:  # pragma: no cover — never block the main pass
        return {}


def _coerce_structured_enums(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map LLM-creative enum values onto the schema whitelist.

    The EarningsStructured Pydantic model uses `Literal[...]` for
    `overall_tone`, guidance `direction`, tone-signal `classification`,
    and Q&A `response_quality`. Pydantic v2 fails the entire validation
    on any unknown literal — and the LLM occasionally drifts (e.g.
    "not_addressed_in_prepared_remarks", "clear_but_non_committal").
    Rather than loosening the schema, normalize each off-list value to
    the nearest in-list value before validation. Pure prefix / keyword
    mapping; conservative ("partial" / "measured" when nothing fits).
    """
    if not isinstance(raw, dict):
        return raw
    raw = dict(raw)  # shallow copy so we don't mutate the LLM payload

    def _coerce_one(value: Any, whitelist: tuple[str, ...], default: str) -> str:
        if not isinstance(value, str):
            return default
        v = value.lower().strip()
        if v in whitelist:
            return v
        # Prefix / containment fallback — "clear_but_non_committal" → "clear",
        # "not_addressed_in_prepared_remarks" → "partial" (the closest
        # neutral bucket).
        for allowed in whitelist:
            if v.startswith(allowed) or allowed in v:
                return allowed
        return default

    raw["overall_tone"] = _coerce_one(
        raw.get("overall_tone"), ("constructive", "measured", "cautious"), "measured",
    )

    direction_wl = (
        "raised", "lowered", "reaffirmed", "introduced", "withdrawn", "unclear",
    )
    raw["guidance_changes"] = [
        {**g, "direction": _coerce_one(g.get("direction"), direction_wl, "unclear")}
        for g in (raw.get("guidance_changes") or []) if isinstance(g, dict)
    ]

    classification_wl = (
        "constructive", "measured", "cautious", "defensive", "evasive",
    )
    raw["tone_signals"] = [
        {**t, "classification": _coerce_one(t.get("classification"), classification_wl, "measured")}
        for t in (raw.get("tone_signals") or []) if isinstance(t, dict)
    ]

    qa_wl = ("clear", "partial", "deflected", "evasive")
    raw["qa_themes"] = [
        {**q, "response_quality": _coerce_one(q.get("response_quality"), qa_wl, "partial")}
        for q in (raw.get("qa_themes") or []) if isinstance(q, dict)
    ]
    return raw


def _critique_block(question: Optional[str]) -> str:
    """Wave 9 — prepend this to a specialist's user prompt when the
    deep-research loop is asking a follow-up question."""
    if not question:
        return ""
    return (
        "\n\n## PM FOLLOW-UP (deep-research round)\n"
        "A senior PM reviewed your prior round and asked: "
        f"{question}\n"
        "Address it directly with specific evidence (numbers, "
        "filing/transcript quotes). Do NOT contradict your prior "
        "finding without articulating what changed your read.\n"
    )


def run_earnings_agent(
    profile: Dict, transcript: Optional[Dict], earnings: Optional[Dict],
    *, prior_round_critique: Optional[str] = None,
) -> AgentFinding:
    if not transcript:
        return AgentFinding(
            agent="Earnings Analyst",
            headline="Earnings transcript unavailable.",
            summary="No transcript on file. Re-run after the next earnings call to refresh this view.",
            key_points=["Transcript unavailable"],
            confidence=0.4,
            sources=[],
        )

    # Wave 9b — pass real transcript content to the LLM. Live AV
    # transcripts come back as 40-50 KB strings (or short blocks for
    # demo); 2K truncation was throwing away nearly all the substance.
    # Per-section budget so prepared remarks (where management frames
    # the quarter) and Q&A (where analysts probe) both get airtime.
    prepared = transcript.get("prepared_remarks") or ""
    qa = transcript.get("qa") or ""
    # If the upstream transcript is shape-stored as a list of blocks,
    # join them into the same string view the LLM expects.
    if isinstance(prepared, list):
        prepared = "\n".join(
            (b.get("text") if isinstance(b, dict) else str(b))
            for b in prepared
        )
    if isinstance(qa, list):
        qa = "\n".join(
            (b.get("text") if isinstance(b, dict) else str(b))
            for b in qa
        )
    # Wave 10 — multi-pass over long transcripts. The single-pass cap
    # (10KB prepared + 8KB Q&A = ~18KB total) silently dropped the
    # back half of long calls. When the transcript is materially
    # longer than the budget, we run an additional Q&A-chunk pass
    # that asks specifically for new themes / reversals not yet
    # captured, then merge into the first-pass extraction. Costs ~1
    # extra cheap-tier call for transcripts >20KB; cheaper than
    # missing what an analyst pressed management on at minute 45.
    multipass_addendum = _multi_pass_qa_addendum(
        ticker=profile.get("ticker", ""),
        prepared=str(prepared),
        qa=str(qa),
    )
    # Semantic retrieval over the indexed transcript. When a PM follow-up
    # question is supplied, use it as the query so the most relevant
    # blocks float up; otherwise fall back to a general guidance /
    # margin / demand query. Filing analyst already does the same
    # pattern. Cheap (1 embed call); no-op when no chunks indexed yet.
    retrieved_chunks: List[Dict] = []
    try:
        from ..services import vector_store
        retrieval_query = (
            prior_round_critique
            if prior_round_critique and len(prior_round_critique) > 8
            else "guidance margin segment demand capex commentary"
        )
        vec_hits = vector_store.search(
            retrieval_query,
            ticker=profile.get("ticker"),
            source_types=["transcript"],
            top_k=5,
        )
        retrieved_chunks = [
            {
                "text": (h.get("text") or "")[:1200],
                "section": h.get("section"),
                "speaker": (h.get("meta") or {}).get("speaker", ""),
                "period": (h.get("meta") or {}).get("period", ""),
                "score": round(float(h.get("score") or 0.0), 3),
            }
            for h in vec_hits
        ]
    except Exception:  # pragma: no cover — retrieval is best-effort
        retrieved_chunks = []

    payload = {
        "ticker": profile.get("ticker"),
        "period": transcript.get("period"),
        "tone": transcript.get("management_tone"),
        "prepared": str(prepared)[:10000],
        "qa": str(qa)[:8000],
        "next_earnings": (earnings or {}).get("next_earnings_date"),
        "multi_pass_addendum": multipass_addendum,
        # When a PM follow-up question is in play, these chunks line up
        # with the question. When this is the first pass, they surface
        # the most generally-thesis-relevant blocks. Either way the LLM
        # sees BOTH the front-of-call text (which carries the framing)
        # AND the high-relevance specific blocks.
        "retrieved_chunks": retrieved_chunks,
    }
    from ..services.research_notes import build_notes_block_for_agent
    notes_block = build_notes_block_for_agent(
        "earnings", profile, extra_query="guidance margins capex demand",
    )
    llm_out = llm.chat_json(
        prompts.EARNINGS_ANALYST_PROMPT
        + _critique_block(prior_round_critique)
        + (("\n\n" + notes_block) if notes_block else "")
        + "\n\nTranscript context:\n" + json.dumps(payload, default=str)[:32000],
        system=prompts.PM_SYSTEM, route="cheap",
        model=settings.openai_tool_model,
        # Earnings response is the largest of any agent's: 4-6 sentence
        # summary + 8-12 key_points + structured block (3-6 guidance
        # changes, 4-8 tone_signals with quoted evidence, 4-8 qa_themes,
        # 2 segment objects, forward catalysts). Empirically a real
        # AAPL run hits 4000 tokens dead-on (full structured payload
        # with quoted evidence per tone_signal) which truncates mid-
        # JSON and silently drops the structured block. 6000 gives
        # ~50% headroom; cost impact ~$0.005/memo.
        max_tokens=6000,
    )
    if llm_out:
        # Same flattening defense as filing_agent — prompt-tuned models
        # sometimes emit nested category objects instead of flat strings.
        from .filing_agent import _flatten_key_points
        # Wave 10 — structured extraction lives on `data.structured`.
        # Validate via the typed schema so a partial / malformed LLM
        # response still serializes (Pydantic drops bad fields).
        structured_payload: Dict = {}
        raw_struct = llm_out.get("structured")
        if isinstance(raw_struct, dict):
            # Coerce LLM-creative enum values into the whitelist before
            # Pydantic validation — otherwise one off-schema response
            # (e.g. response_quality="not_addressed_in_prepared_remarks")
            # nukes the entire structured block via cascading literal
            # errors. Empirically the LLM mostly sticks to the allowed
            # values but ~10% of qa_themes drift.
            raw_struct = _coerce_structured_enums(raw_struct)
            try:
                from ..schemas import EarningsStructured
                structured_payload = EarningsStructured(
                    period=str(raw_struct.get("period") or transcript.get("period") or ""),
                    overall_tone=raw_struct.get("overall_tone") or "measured",
                    guidance_changes=raw_struct.get("guidance_changes") or [],
                    tone_signals=raw_struct.get("tone_signals") or [],
                    qa_themes=raw_struct.get("qa_themes") or [],
                    most_defended_segment=raw_struct.get("most_defended_segment") or {},
                    most_pressed_segment=raw_struct.get("most_pressed_segment") or {},
                    forward_catalysts=raw_struct.get("forward_catalysts") or [],
                ).model_dump()
            except Exception:  # pragma: no cover — drop the structure rather than fail
                structured_payload = {}
        # Wave 10 — emit citations for the prepared remarks + Q&A
        # sections + back-half multi-pass when we ran one.
        from ..schemas import Citation
        period = str(transcript.get("period") or "")
        evidence: List[Citation] = []
        if period and prepared:
            evidence.append(Citation(
                kind="transcript", ref=period, section="prepared_remarks",
                excerpt=str(prepared)[:300],
            ))
        if period and qa:
            evidence.append(Citation(
                kind="transcript", ref=period, section="qa",
                excerpt=str(qa)[:300],
            ))
        if multipass_addendum:
            for theme in (multipass_addendum.get("hard_questions") or [])[:2]:
                evidence.append(Citation(
                    kind="transcript", ref=period, section="qa_back_half",
                    excerpt=str(theme)[:300],
                ))
        return AgentFinding(
            agent="Earnings Analyst",
            headline=llm_out.get("headline", "Earnings view"),
            summary=llm_out.get("summary", ""),
            key_points=_flatten_key_points(llm_out.get("key_points", [])),
            confidence=float(llm_out.get("confidence", 0.7)),
            sources=[f"transcript:{transcript.get('period', '')}"],
            evidence=evidence[:6],
            data={"structured": structured_payload} if structured_payload else {},
        )

    # Deterministic fallback — used only when the LLM call fails or
    # returns nothing. Wave 9b: stripped the demo-only `management_tone`
    # / `bullish_takeaways` / `bearish_takeaways` references; live AV
    # transcripts don't carry those fields, which made the prior
    # fallback render as identical canned text for every ticker. New
    # version states the limitation honestly + surfaces the next
    # earnings date when known.
    period = transcript.get("period") or "the most recent quarter"
    text_len = len(str(transcript.get("prepared_remarks") or "")) + len(
        str(transcript.get("qa") or "")
    )
    next_date = (earnings or {}).get("next_earnings_date")
    next_line = (
        f" Next earnings: {next_date}." if next_date else
        " Next earnings date TBD."
    )
    if text_len > 1000:
        summary = (
            f"Transcript for {period} is on file ({text_len:,} chars of prepared "
            "remarks + Q&A) but the LLM analyst couldn't run a full pass on this "
            "request — see source for details." + next_line
        )
    else:
        summary = (
            f"No substantive transcript text available for {period}." + next_line
        )
    return AgentFinding(
        agent="Earnings Analyst",
        headline=f"{profile.get('ticker', '')}: transcript pending LLM analysis.",
        summary=summary,
        key_points=[],
        confidence=0.4,
        sources=[f"transcript:{transcript.get('period', '')}"],
    )
