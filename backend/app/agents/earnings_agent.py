"""Earnings call agent."""
from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import settings
from ..schemas import AgentFinding
from . import llm, prompts


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
    payload = {
        "ticker": profile.get("ticker"),
        "period": transcript.get("period"),
        "tone": transcript.get("management_tone"),
        "prepared": str(prepared)[:10000],
        "qa": str(qa)[:8000],
        "next_earnings": (earnings or {}).get("next_earnings_date"),
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
        # Same truncation tax filing analyst was paying — give the
        # response room for headline + 8-12 categorized points.
        max_tokens=2000,
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
        return AgentFinding(
            agent="Earnings Analyst",
            headline=llm_out.get("headline", "Earnings view"),
            summary=llm_out.get("summary", ""),
            key_points=_flatten_key_points(llm_out.get("key_points", [])),
            confidence=float(llm_out.get("confidence", 0.7)),
            sources=[f"transcript:{transcript.get('period', '')}"],
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
