"""D2 (2026-09-25) — the item-8 full-report reviewer fields on `CriticReview`.

Owner decision 2026-09-25 item 8 makes the risk reviewer a full-report
reviewer: a verdict, specific issues (with severity, direction and
evidence), the case that the rating is too high AND too low (L5), one
bounded PM revision pass with one re-check, and a "not independently
reviewed" status when the review is not live. The writers are later slices
(D7, R1); this pins the contract they write into:

  * every field is defaulted, so the critic reviews already stored — and
    every review written with `REVIEWER_MODE=legacy` — read back unchanged
    and carry no claim the reviewer never made;
  * the vocabularies are closed: `severity` decides whether an issue drives
    the revision pass and an earned-confidence cap, so an off-list value
    must be refused rather than read as either;
  * the text bounds hold, and a full review round-trips.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.schemas import (
    CriticReview,
    DebateReview,
    ReviewIssue,
    ReviewRatingCase,
    ReviewRecheck,
    ReviewRevision,
    StockMemoOut,
)
from app.schemas.agents import (
    CRITIC_REVIEW_ITEM8_FIELDS,
    REVIEW_FIX_REQUEST_MAX_CHARS,
    REVIEW_ISSUE_TEXT_MAX_CHARS,
)

PRE_S2 = Path(__file__).parent / "fixtures" / "memo_contract" / "pre_s2_memo.json"

NEW_REVIEW_FIELDS = {
    "reviewer_model", "verdict", "issues", "rating_too_high", "rating_too_low",
    "review_status", "revision", "debate_review",
}


def _assert_item8_defaults(review: CriticReview) -> None:
    assert review.reviewer_model == ""
    assert review.verdict == ""
    assert review.issues == []
    assert review.rating_too_high is None
    assert review.rating_too_low is None
    assert review.review_status == ""
    assert review.revision is None
    assert review.debate_review is None


def test_legacy_critic_review_defaults():
    """A stored pre-item-8 review (the pre-S2 memo's) and a minimal new one
    both carry the empty item-8 fields; the stored values read back as they
    were stored, and the dump gains exactly the new keys."""
    stored = json.loads(PRE_S2.read_text())["risk_committee_challenge"]
    assert not NEW_REVIEW_FIELDS & set(stored), "fixture must be a pre-item-8 review"
    review = CriticReview.model_validate(stored)
    _assert_item8_defaults(review)
    dumped = review.model_dump(mode="json")
    assert {k: dumped[k] for k in stored} == stored
    # `valuation_divergence_assessment` is S2's addition, which also post-dates this fixture.
    assert set(dumped) - set(stored) == NEW_REVIEW_FIELDS | {"valuation_divergence_assessment"}
    # Even a stored LIVE review is not relabelled independent: independence
    # is a claim only the item-8 reviewer's writer may make.
    assert review.review_mode == "live" and review.review_status == ""
    _assert_item8_defaults(CriticReview(overall_assessment="ok"))
    memo = StockMemoOut.model_validate(json.loads(PRE_S2.read_text()))
    _assert_item8_defaults(memo.risk_committee_challenge)


def _issue(**over: Any) -> dict[str, Any]:
    return {"id": "R1", "category": "thesis_logic", "severity": "material", "text": "t", **over}


def test_review_issue_defaults():
    issue = ReviewIssue.model_validate(_issue())
    assert issue.direction == "neutral"
    assert issue.status == "open"
    assert (issue.evidence, issue.fix_request) == ([], "")


@pytest.mark.parametrize("category", [
    "thesis_logic", "evidence_quality", "debate_handling", "risk_blind_spot", "valuation_consistency",
])
@pytest.mark.parametrize("severity", ["material", "minor"])
def test_review_issue_accepts_every_listed_value(category, severity):
    for direction in ("too_high", "too_low", "neutral"):
        for status in ("open", "addressed_by_pm", "resolved", "rejected_by_pm"):
            ReviewIssue.model_validate(_issue(category=category, severity=severity,
                                              direction=direction, status=status))


@pytest.mark.parametrize("over", [
    {"category": "risk"},                 # the pre-item-8 critic's only lens
    {"category": "Thesis_Logic"},         # no case folding
    {"severity": "major"},
    {"severity": "Material"},
    {"severity": ""},
    {"direction": "up"},
    {"direction": "too high"},
    {"status": "closed"},
    {"status": "addressed"},
])
def test_review_issue_enums_strict(over):
    with pytest.raises(ValidationError):
        ReviewIssue.model_validate(_issue(**over))


@pytest.mark.parametrize("missing", ["id", "category", "severity", "text"])
def test_review_issue_required_fields(missing):
    payload = _issue()
    payload.pop(missing)
    with pytest.raises(ValidationError):
        ReviewIssue.model_validate(payload)


def test_review_issue_text_bounds():
    ReviewIssue.model_validate(_issue(text="x" * REVIEW_ISSUE_TEXT_MAX_CHARS,
                                      fix_request="y" * REVIEW_FIX_REQUEST_MAX_CHARS))
    with pytest.raises(ValidationError):
        ReviewIssue.model_validate(_issue(text="x" * (REVIEW_ISSUE_TEXT_MAX_CHARS + 1)))
    with pytest.raises(ValidationError):
        ReviewIssue.model_validate(_issue(fix_request="y" * (REVIEW_FIX_REQUEST_MAX_CHARS + 1)))


@pytest.mark.parametrize("model, payload", [
    (CriticReview, {"overall_assessment": "a", "verdict": "mostly_sound"}),
    (CriticReview, {"overall_assessment": "a", "review_status": "live"}),
    (ReviewRevision, {"status": "partial"}),
    (ReviewRecheck, {"status": "done"}),
])
def test_review_vocabularies_strict(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_revision_and_recheck_defaults():
    rev = ReviewRevision()
    assert rev.status == "not_needed"
    assert (rev.rating_before, rev.rating_after) == ("", "")
    assert (rev.confidence_before, rev.confidence_after) == (None, None)
    assert rev.notes == []
    # No re-check ran, so nothing counts as resolved.
    assert rev.recheck == ReviewRecheck(status="not_run", resolved=[], open=[])


def test_full_review_roundtrip():
    review = CriticReview(
        overall_assessment="The thesis follows, but the bear's capex point is unanswered.",
        review_mode="live",
        reviewer_model="openai:gpt-6-astra",
        verdict="sound_with_issues",
        issues=[
            ReviewIssue(id="R1", category="debate_handling", severity="material", direction="too_high",
                        text="The PM did not rule on the capex-duration dispute.",
                        evidence=["E02", "check:one_sided_ruling_share"],
                        fix_request="Rule on D1 with its evidence.", status="addressed_by_pm"),
            ReviewIssue(id="R2", category="evidence_quality", severity="minor",
                        text="A margin figure is untraceable.", evidence=["numcheck:final_pm_view:3"]),
        ],
        rating_too_high=ReviewRatingCase(text="Capex guides are being cut.", evidence=["E02"]),
        rating_too_low=ReviewRatingCase(text="Networking attach is under-modelled.", evidence=["E01"]),
        review_status="independent",
        revision=ReviewRevision(status="revised", rating_before="Bullish", rating_after="Neutral",
                                confidence_before=72.0, confidence_after=61.0,
                                notes=["Ruled D1 for the bear."],
                                recheck=ReviewRecheck(status="complete", resolved=["R1"], open=[])),
        debate_review=DebateReview(dispute_views=[{"dispute": "D1", "view": "questionable"}],
                                   unaddressed=["BEAR-2"], one_sided=""),
    )
    wire = json.loads(review.model_dump_json())
    again = CriticReview.model_validate(wire)
    assert again == review
    assert json.loads(again.model_dump_json()) == wire
    assert NEW_REVIEW_FIELDS <= set(wire)


def test_item8_field_tuple_names_exactly_the_new_fields():
    """The legacy critic strips the fields this tuple names; a reviewer
    field added without it would leak an empty key into the critic prompt."""
    assert set(CRITIC_REVIEW_ITEM8_FIELDS) == NEW_REVIEW_FIELDS
    assert len(CRITIC_REVIEW_ITEM8_FIELDS) == len(NEW_REVIEW_FIELDS)


# ---------------------------------------------------------------------------
# With DEBATE_MODE and REVIEWER_MODE off, the legacy critic's prompt is the
# pre-D2 pipeline's, byte for byte (plan §0.3, P4, design §8.3).
#
# The critic serializes the whole draft memo, cut at 60k chars, so the D2
# expand alone (`"debate": null` plus eight empty reviewer keys on the
# pending review, ~180 chars) changed the prompt and shifted the window.
# These digests were captured at ca08521, the commit before D2, by running
# `run_critic` on the same drafts; a change to them is a change to what the
# live critic reads and needs an owner decision, not a re-pin.
# ---------------------------------------------------------------------------

PRE_D2_CRITIC_PAYLOAD = {
    # The stored pre-S2 memo as it validates (its live review included).
    "stored": (2850, "04fd35c627985b9a9181e21f20e9d3fb736fe3ea932bec65f32e069ee3c97c7a"),
    # The draft the graph hands the critic: graph.py's pending placeholder.
    "pending": (2873, "f2216cd35f0dfecd9a73f183c9b2eab3ab349715688bae2c1ae91c29a9d280a0"),
    # A draft that just fits the 60k window before D2 (59,836 chars); with
    # the D2 keys left in, its tail was cut.
    "long": (59836, "734f1cdbd8334be1c2c7705b3b18fe5a8796490a90e12a1a3f1d96c116796436"),
}


def _critic_draft(case: str) -> dict[str, Any]:
    memo = StockMemoOut.model_validate(json.loads(PRE_S2.read_text()))
    if case == "long":
        memo.final_pm_view = "".join(f"Long draft sentence {i:05d}. " for i in range(2110))
    if case in ("pending", "long"):
        memo.risk_committee_challenge = CriticReview(
            overall_assessment="Pending critic review.", review_mode="pending")
    # Exactly what `graph._review_memo` passes: the plain model dump.
    return memo.model_dump()


def _critic_prompts(monkeypatch, answer: dict[str, Any] | None = None) -> list[str]:
    from app.agents import critic_agent
    from app.config import settings

    prompts: list[str] = []

    def fake_chat_json(prompt: str, **_kw: Any) -> dict[str, Any]:
        prompts.append(prompt)
        return dict(answer or {})

    monkeypatch.setattr(settings, "enable_agent_critic", True)
    monkeypatch.setattr(critic_agent.llm, "chat_json", fake_chat_json)
    # The prior-memo and company-memory blocks read the database and the
    # memory files; the draft payload is what D2 touched.
    monkeypatch.setattr(critic_agent, "_prior_memo_context", lambda _t: "")
    monkeypatch.setattr(critic_agent, "_company_memory_context", lambda _t, _s=None: "")
    return prompts


@pytest.mark.parametrize("case", sorted(PRE_D2_CRITIC_PAYLOAD))
def test_legacy_critic_payload_byte_identical_to_pre_d2(case, monkeypatch):
    from app.agents import critic_agent

    prompts = _critic_prompts(monkeypatch)
    critic_agent.run_critic(_critic_draft(case))
    assert len(prompts) == 1
    payload = prompts[0].split("\n\nDraft memo:\n", 1)[1]
    # Every case fits the window, so the payload parses: say WHAT differs
    # before the digest says that something does.
    sent = json.loads(payload)
    assert "debate" not in sent
    assert not NEW_REVIEW_FIELDS & set(sent["risk_committee_challenge"])
    size, digest = PRE_D2_CRITIC_PAYLOAD[case]
    assert (len(payload), hashlib.sha256(payload.encode()).hexdigest()) == (size, digest)


def test_legacy_critic_draft_keeps_written_fields():
    """Only UNWRITTEN D2 values are dropped: once a writer fills the debate
    or a reviewer field, the critic sees it, and nothing else moves."""
    from app.agents.critic_agent import _legacy_critic_draft

    draft = _critic_draft("pending")
    draft["debate"] = {"status": "complete"}
    draft["risk_committee_challenge"]["verdict"] = "unsound"
    draft["risk_committee_challenge"]["review_status"] = "not_independent"
    out = _legacy_critic_draft(draft)
    assert list(out) == list(draft)
    assert out["debate"] == {"status": "complete"}
    review = out["risk_committee_challenge"]
    assert (review["verdict"], review["review_status"]) == ("unsound", "not_independent")
    assert not {"reviewer_model", "issues", "revision", "debate_review"} & set(review)
    # The caller's dict is not mutated.
    assert draft["risk_committee_challenge"]["reviewer_model"] == ""


@pytest.mark.parametrize("answer", [
    {},  # no model answer: the rule-based stub
    {"overall_assessment": "Live critic answer.", "challenges": ["c"],
     "underweighted_risks": ["r"], "suggested_revisions": ["s"]},
])
def test_legacy_critic_writes_no_item8_field(answer, monkeypatch):
    """With REVIEWER_MODE off the legacy critic, live or rule-based, makes
    none of the item-8 claims: no verdict, no issues and, above all, no
    `review_status` saying the review was independent."""
    from app.agents import critic_agent

    _critic_prompts(monkeypatch, answer)
    review = critic_agent.run_critic(_critic_draft("pending"))
    assert review is not None
    assert review.review_mode == ("live" if answer else "rule_based")
    _assert_item8_defaults(review)


def test_critic_failure_stub_writes_no_item8_field():
    from app.agents.safe_runner import safe_critic

    def boom(_memo: dict[str, Any]) -> CriticReview | None:
        raise RuntimeError("critic down")

    review = safe_critic(boom, {})
    assert review is not None and review.review_mode == "unavailable"
    _assert_item8_defaults(review)


def test_unavailable_review_presented_keeps_item8_provenance():
    """The presenter blanks a not-live review's legacy prose, and must keep
    its provenance: this branch is exactly the review `not_independent`
    labels, and dropping the label would present it as an unlabelled review.

    D4 (the review projection) keeps the item-8 findings too: a verdict and
    issues are structured findings the open-issue cap rests on (R1), so
    blanking them would show a capped confidence with no reason for it."""
    from app.services import memo_sections

    memo = StockMemoOut.model_validate(json.loads(PRE_S2.read_text()))
    memo.risk_committee_challenge = CriticReview(
        overall_assessment="Rule-based review.", review_mode="rule_based",
        review_status="not_independent", reviewer_model="openai:gpt-6-astra",
        verdict="unsound", issues=[ReviewIssue.model_validate(_issue())],
    )
    out = memo_sections.present_memo(memo)
    assert out.section_availability["risk_committee_challenge"].status == "unavailable"
    review = out.risk_committee_challenge
    assert (review.review_status, review.reviewer_model) == ("not_independent", "openai:gpt-6-astra")
    # The legacy critic prose of a review that was not live is blanked...
    assert review.overall_assessment == memo_sections.UNAVAILABLE_TEXT and review.challenges == []
    # ...and its item-8 findings are kept, labelled not independent.
    assert (review.verdict, [i.id for i in review.issues]) == ("unsound", ["R1"])
    assert memo.risk_committee_challenge.review_status == "not_independent"
