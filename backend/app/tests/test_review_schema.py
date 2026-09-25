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
from app.schemas.agents import REVIEW_FIX_REQUEST_MAX_CHARS, REVIEW_ISSUE_TEXT_MAX_CHARS

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
