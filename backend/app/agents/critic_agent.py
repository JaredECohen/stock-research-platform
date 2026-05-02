"""Critic / Risk Committee agent."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import settings
from ..schemas import CriticReview
from . import llm, prompts
from .tools import lint_citations


def run_critic(memo_dict: Dict) -> Optional[CriticReview]:
    if not settings.enable_agent_critic:
        return None

    # Citation discipline — challenge any memo whose evidence is news/social-heavy.
    citation_audit = lint_citations(memo_dict.get("sources_used") or [])

    payload = json.dumps(memo_dict, default=str)[: settings.max_agent_context_chars]
    # Critic intentionally crosses provider families (Phase 4): if Anthropic is
    # configured, force-route through ANTHROPIC_CRITIC_MODEL regardless of
    # LLM_PROVIDER. Falls back to the active provider (or rule-based stub)
    # when Anthropic is absent.
    if settings.has_anthropic:
        provider_override = "anthropic"
        critic_model = settings.anthropic_critic_model
    else:
        provider_override = None
        critic_model = None  # use the active provider's strong-route default
    llm_out = llm.chat_json(
        prompts.CRITIC_PROMPT + "\n\nDraft memo:\n" + payload,
        system=prompts.PM_SYSTEM, route="strong",
        provider_override=provider_override,
        model=critic_model,
    )
    if llm_out:
        review = CriticReview(
            overall_assessment=llm_out.get("overall_assessment", "Reviewed."),
            challenges=llm_out.get("challenges", []),
            underweighted_risks=llm_out.get("underweighted_risks", []),
            suggested_revisions=llm_out.get("suggested_revisions", []),
            advice_compliance_check=llm_out.get(
                "advice_compliance_check", "Output framed as research/education, not personalized advice."
            ),
        )
    else:
        # Deterministic fallback
        challenges: List[str] = []
        rating = memo_dict.get("rating_label", "")
        if rating in ("Bullish", "Bearish"):
            challenges.append(f"Rating ({rating}) is one-sided — list explicit thesis-breakers and the cost of being wrong.")
        if not memo_dict.get("key_risks"):
            challenges.append("Risks list is thin — add specific operational and macro risks.")
        if not memo_dict.get("dcf_summary"):
            challenges.append("No DCF context — re-run with a base/bull/bear range to triangulate valuation.")
        underweighted = []
        if memo_dict.get("sector_agent_view"):
            underweighted.append("Re-emphasize the sector cohort context if quality deteriorates.")
        suggested = [
            "Tag every claim with a source: filing, transcript, or ratio.",
            "Verify the bull case is symmetric to the bear case.",
            "If valuation is elevated, explicitly state what the market is pricing in.",
        ]
        review = CriticReview(
            overall_assessment="Memo is structurally sound; balance and source citations should be tightened.",
            challenges=challenges or ["No major issues detected."],
            underweighted_risks=underweighted,
            suggested_revisions=suggested,
            advice_compliance_check="Output framed as research/education only; no direct buy/sell language detected.",
        )

    # Append citation-discipline findings — surfaced regardless of LLM availability.
    if citation_audit["flag"] == "low_quality":
        review.challenges.append(
            f"Source mix is low-quality (avg trust {citation_audit['quality']}); "
            "thesis leans on news/social rather than filings/financials/transcripts."
        )
        review.suggested_revisions.append(
            "Re-cite the bull/bear case using primary sources (10-K MD&A, transcript Q&A, audited financials)."
        )
    elif citation_audit["flag"] == "thin_primary_evidence":
        review.challenges.append(
            f"Only {int(citation_audit['primary_ratio'] * 100)}% of sources are primary "
            "(filings / financials / transcripts). News / sell-side / social shouldn't dominate the evidence base."
        )

    return review
