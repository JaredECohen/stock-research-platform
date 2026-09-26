"""Critic / Risk Committee agent."""
from __future__ import annotations

import json

from ..config import settings
from ..schemas import CriticReview
from ..schemas.agents import CRITIC_REVIEW_ITEM8_FIELDS
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
        from ..services import memo_store
        # Review precedes persistence of the current draft. The latest stored
        # live snapshot is therefore the prior; a single existing memo counts.
        snapshot = memo_store.latest_memo(ticker)
        if snapshot is None:
            return ""
        prior = snapshot.memo_json or {}
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
            ver=snapshot.version,
            when=(snapshot.generated_at.date().isoformat() if snapshot.generated_at else "—"),
            rating=prior.get("rating_label") or "—",
            thesis=(prior.get("one_sentence_thesis") or "")[:200],
            cv=(prior_mp.get("consensus_view") or "")[:200],
            ov=(prior_mp.get("our_view") or "")[:200],
            gap=(prior_mp.get("gap") or "")[:200],
            fals=", ".join(prior_mp.get("falsifiers") or [])[:300],
        )
    except Exception:  # pragma: no cover — never block the critic
        return ""


# W7 inject mode: the critic reads lessons as hypotheses, not a record the
# memo must obey. The legacy wording ("if the memo CONTRADICTS prior recorded
# lessons ... raise it as a challenge") over-indexes on memory, which owner
# decision 9 rules out; a filing observation is a fact, so a silent reversal
# of one is still worth flagging.
CRITIC_PRIORS_INSTRUCTION = (
    "Cross-check: flag a silent reversal of a recorded filing observation. "
    "Lessons are hypotheses: do not challenge a departure from an untested or "
    "contested lesson; challenge a departure from a supported lesson only when "
    "the memo gives no current evidence."
)


def _company_memory_context(ticker: str, sector: str | None = None) -> str:
    """Wave 10 — pull the company memory file as additional critic
    grounding. Lets the critic say 'you said the opposite three months
    ago — what changed?' instead of judging the memo in isolation.

    W7: in inject mode the learned-priors block replaces the file, with the
    reframed instruction above; off, shadow and any call outside a live memo
    run keep the legacy block byte for byte."""
    if not ticker:
        return ""
    from ..learning import context as learning_context
    learned = learning_context.render_safely("critic", ticker=ticker, sector=sector)
    if learned.mode == "inject":
        return f"\n\n{learned.text}\n\n{CRITIC_PRIORS_INSTRUCTION}" if learned.text else ""
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


_ASSESSMENTS = frozenset({"supported", "unsupported"})


def _divergence_block(memo_dict: dict) -> str:
    """The W2b 7(b) block the critic must see, or "".

    Built from `quality.rating_reconciliation` (the PM's rating, the
    evidence verdict and the PM's stated reason). It is PREPENDED before the
    memo dump rather than left inside it: the dump is cut at
    `max_agent_context_chars` (60k) while a live memo's dump runs to ~270k,
    and `quality` is the last field, so inside the dump the critic would
    never read it."""
    from .memo_quality import diverges

    quality = memo_dict.get("quality")
    rec = quality.get("rating_reconciliation") if isinstance(quality, dict) else None
    if not isinstance(rec, dict):
        return ""
    pm_rating = str(rec.get("pm_rating") or memo_dict.get("rating_label") or "")
    verdict = str(rec.get("valuation_verdict") or "")
    if not diverges(pm_rating, verdict):
        return ""
    vv = memo_dict.get("valuation_verdict")
    summary = str(vv.get("summary") or "") if isinstance(vv, dict) else ""
    reason = str(rec.get("reason") or "").strip()
    return (
        f"\n\n## RATING DIVERGENCE (PM rated {pm_rating}; valuation evidence reads "
        f"{verdict.replace('_', ' ')})\n"
        f"Valuation evidence: {summary or 'n/a'}\n"
        f"PM's stated reason: {reason or 'none given'}"
    )


# Each item-8 review field's "not produced" value ("" / [] / None).
_ITEM8_UNWRITTEN = {
    name: CriticReview.model_fields[name].get_default(call_default_factory=True)
    for name in CRITIC_REVIEW_ITEM8_FIELDS
}


def _legacy_critic_draft(memo_dict: dict) -> dict:
    """The draft without the D2 fields nothing has written.

    D2 added `StockMemoOut.debate` and the item-8 reviewer fields expand-only,
    so every draft dump now carries `"debate": null` and eight empty keys on
    the pending review. This critic reads neither, but it serializes the
    whole draft, and those ~180 bytes would change the prompt and push real
    content out of the 60k window. With DEBATE_MODE and REVIEWER_MODE off the
    prompt must stay byte-identical to the pre-D2 pipeline (plan §0.3, P4).
    Only unwritten values are dropped: a field a later writer fills reaches
    the critic, and the rest of the dict keeps its order."""
    out = dict(memo_dict)
    if "debate" in out and out["debate"] is None:
        del out["debate"]
    review = out.get("risk_committee_challenge")
    if isinstance(review, dict):
        out["risk_committee_challenge"] = {
            k: v for k, v in review.items()
            if not (k in _ITEM8_UNWRITTEN and v == _ITEM8_UNWRITTEN[k])
        }
    return out


def run_critic(memo_dict: dict) -> CriticReview | None:
    if not settings.enable_agent_critic:
        return None

    # Citation discipline — challenge any memo whose evidence is news/social-heavy.
    citation_audit = lint_citations(memo_dict.get("sources_used") or [])

    # Wave 10 — feed prior memo + company memory so the critic can
    # spot silent reversals and cross-version inconsistencies.
    ticker = memo_dict.get("ticker") or ""
    prior_block = _prior_memo_context(ticker)
    memory_block = _company_memory_context(ticker, memo_dict.get("sector"))

    divergence_block = _divergence_block(memo_dict)
    payload = json.dumps(_legacy_critic_draft(memo_dict), default=str)[: settings.max_agent_context_chars]
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
        + divergence_block
        + "\n\nDraft memo:\n" + payload,
        system=prompts.PM_SYSTEM, route="strong",
        provider_override=provider_override,
        model=critic_model,
    )
    assessment = llm_out.get("overall_assessment") if isinstance(llm_out, dict) else None
    if isinstance(assessment, str) and assessment.strip():
        review = CriticReview(
            overall_assessment=assessment,
            review_mode="live",
            challenges=llm_out.get("challenges", []),
            underweighted_risks=llm_out.get("underweighted_risks", []),
            suggested_revisions=llm_out.get("suggested_revisions", []),
            advice_compliance_check=llm_out.get(
                "advice_compliance_check", "Output framed as research/education, not personalized advice."
            ),
        )
        # Read only when the critic was actually asked (the block was sent);
        # any other value, or an unasked one, stays "not_assessed".
        raw_assessment = (llm_out.get("valuation_divergence_assessment")
                          if isinstance(llm_out, dict) else None)
        if (divergence_block and isinstance(raw_assessment, str)
                and raw_assessment.strip().lower() in _ASSESSMENTS):
            review.valuation_divergence_assessment = raw_assessment.strip().lower()  # type: ignore[assignment]
    else:
        # This checks a few fields, not factual accuracy or research quality.
        # A missing live result must not read as an independent endorsement.
        if settings.has_llm or (settings.enable_live_data and not settings.use_demo_data):
            from .safe_runner import note_soft
            note_soft("Risk Committee", "Live critic unavailable; only rule-based checks were completed.",
                      kind="CriticUnavailable")
        challenges: list[str] = []
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
            overall_assessment=(
                "Rule-based check only; no live critic review was completed. "
                "Claim accuracy, completeness, and research quality were not independently assessed."
            ),
            review_mode="rule_based",
            challenges=challenges,
            underweighted_risks=underweighted,
            suggested_revisions=suggested,
            advice_compliance_check="Advice compliance was not assessed by this rule-based check.",
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
