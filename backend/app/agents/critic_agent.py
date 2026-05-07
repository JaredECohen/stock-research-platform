"""Critic / Risk Committee agent."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import settings
from ..schemas import CriticReview
from . import llm, prompts
from .tools import lint_citations


def _prior_memo_context(ticker: str) -> str:
    """Wave 10 — pull the prior memo's mispricing thesis + rating for
    cross-version consistency checking. Returns a markdown block ready
    to splice into the critic prompt; empty string when no prior memo
    exists."""
    if not ticker:
        return ""
    try:
        from sqlalchemy import select
        from ..database import SessionLocal
        from ..models import MemoSnapshot
        with SessionLocal() as db:
            rows = db.execute(
                select(MemoSnapshot)
                .where(MemoSnapshot.ticker == ticker.upper())
                .order_by(MemoSnapshot.version.desc())
                .limit(2)
            ).scalars().all()
        if len(rows) < 2:
            return ""
        # rows[0] is the just-written memo; rows[1] is the prior version.
        prior = rows[1].memo_json or {}
        if not isinstance(prior, dict):
            return ""
        prior_mp = prior.get("mispricing_thesis") or {}
        if not isinstance(prior_mp, dict):
            prior_mp = {}
        return (
            "\n\n## PRIOR MEMO (v{ver}, {when}):\n"
            "Prior rating: {rating}; one-sentence thesis: {thesis}\n"
            "Prior mispricing thesis:\n"
            "  consensus_view: {cv}\n  our_view: {ov}\n  gap: {gap}\n"
            "  falsifiers: {fals}\n\n"
            "If the new memo's view DIVERGES from this, the new memo "
            "must explicitly explain what changed. Flag any silent "
            "reversal (rating shift without rationale) as a major "
            "challenge."
        ).format(
            ver=rows[1].version,
            when=(rows[1].generated_at.date().isoformat() if rows[1].generated_at else "—"),
            rating=prior.get("rating_label") or "—",
            thesis=(prior.get("one_sentence_thesis") or "")[:200],
            cv=(prior_mp.get("consensus_view") or "")[:200],
            ov=(prior_mp.get("our_view") or "")[:200],
            gap=(prior_mp.get("gap") or "")[:200],
            fals=", ".join(prior_mp.get("falsifiers") or [])[:300],
        )
    except Exception:  # pragma: no cover — never block the critic
        return ""


def _company_memory_context(ticker: str) -> str:
    """Wave 10 — pull the company memory file as additional critic
    grounding. Lets the critic say 'you said the opposite three months
    ago — what changed?' instead of judging the memo in isolation."""
    if not ticker:
        return ""
    try:
        from ..memory import CompanyMemory
        cm = CompanyMemory.for_ticker(ticker)
        body = cm.as_prompt_context(max_chars=2500)
        if not body or not body.strip():
            return ""
        return (
            f"\n\n## COMPANY MEMORY ({ticker.upper()}):\n{body.strip()}\n\n"
            "Cross-check the new memo against this institutional memory. "
            "If the memo CONTRADICTS prior recorded lessons without a "
            "fresh rationale, raise it as a challenge."
        )
    except Exception:  # pragma: no cover
        return ""


def run_critic(memo_dict: Dict) -> Optional[CriticReview]:
    if not settings.enable_agent_critic:
        return None

    # Citation discipline — challenge any memo whose evidence is news/social-heavy.
    citation_audit = lint_citations(memo_dict.get("sources_used") or [])

    # Wave 10 — feed prior memo + company memory so the critic can
    # spot silent reversals and cross-version inconsistencies.
    ticker = memo_dict.get("ticker") or ""
    prior_block = _prior_memo_context(ticker)
    memory_block = _company_memory_context(ticker)

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
        prompts.CRITIC_PROMPT
        + prior_block
        + memory_block
        + "\n\nDraft memo:\n" + payload,
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
