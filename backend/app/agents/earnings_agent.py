"""Earnings call agent."""
from __future__ import annotations

import json
from typing import Dict, Optional

from ..schemas import AgentFinding
from . import llm, prompts


def run_earnings_agent(profile: Dict, transcript: Optional[Dict], earnings: Optional[Dict]) -> AgentFinding:
    if not transcript:
        return AgentFinding(
            agent="Earnings Analyst",
            headline="Earnings transcript unavailable.",
            summary="No transcript on file. Re-run after the next earnings call to refresh this view.",
            key_points=["Transcript unavailable"],
            confidence=0.4,
            sources=[],
        )

    payload = {
        "ticker": profile.get("ticker"),
        "period": transcript.get("period"),
        "tone": transcript.get("management_tone"),
        "prepared": (transcript.get("prepared_remarks") or "")[:2000],
        "qa": (transcript.get("qa") or "")[:2000],
        "next_earnings": (earnings or {}).get("next_earnings_date"),
    }
    llm_out = llm.chat_json(
        prompts.EARNINGS_ANALYST_PROMPT + "\n\nTranscript context:\n" + json.dumps(payload, default=str),
        system=prompts.PM_SYSTEM, route="cheap",
    )
    if llm_out:
        return AgentFinding(
            agent="Earnings Analyst",
            headline=llm_out.get("headline", "Earnings view"),
            summary=llm_out.get("summary", ""),
            key_points=llm_out.get("key_points", []),
            confidence=float(llm_out.get("confidence", 0.7)),
            sources=[f"transcript:{transcript.get('period', '')}"],
        )

    # Deterministic fallback
    bullish = transcript.get("bullish_takeaways", []) or []
    bearish = transcript.get("bearish_takeaways", []) or []
    tone = transcript.get("management_tone", "constructive")
    summary = (
        f"Management tone read as {tone}. Prepared remarks emphasized core drivers; Q&A reinforced the framework. "
        f"Next earnings: {(earnings or {}).get('next_earnings_date', 'TBD')}."
    )
    key_points = [
        f"Bullish: {b}" for b in bullish[:3]
    ] + [
        f"Watch: {b}" for b in bearish[:2]
    ]
    return AgentFinding(
        agent="Earnings Analyst",
        headline=f"{profile.get('ticker', '')}: management tone {tone}.",
        summary=summary,
        key_points=key_points,
        confidence=0.7,
        sources=[f"transcript:{transcript.get('period', '')}"],
    )
