"""Final research presentation must distinguish the original PM opinion."""
from datetime import date
from types import SimpleNamespace

import pytest

from app.agents import graph
from app.agents.safe_runner import DegradationLog
from app.schemas import CriticReview
from app.tests.factories import make_findings, make_memo


@pytest.mark.parametrize("rating,factor,risk_confidence", [
    ("Bullish", 50.0, True),
    ("Very Bearish", 47.0, False),
    ("Neutral", 50.0, False),
])
def test_review_preserves_rating_math_and_labels_prior_opinion(
    monkeypatch, rating, factor, risk_confidence,
):
    monkeypatch.setattr(graph.settings, "llm_rating_weight", 0.4)
    monkeypatch.setattr(graph, "_checkpointed_critic", lambda _: CriticReview(
        overall_assessment="Draft checked",
    ))
    findings = make_findings()
    findings["risk"].data = {"recommendations": ([{
        "target": "confidence", "direction": "lower", "magnitude": "small",
        "detail": "Call coverage is incomplete", "rationale": "Only prepared remarks retrieved",
    }] if risk_confidence else [])}
    original = f"Research view: {rating}. Original PM economic reasoning."
    memo = make_memo(
        rating_label=rating, confidence_score=60.0, final_pm_view=original,
        scores={"factor_pm_score": factor, "confidence": 60.0},
    )
    expected = memo.model_copy(deep=True)
    graph._apply_risk_recommendations(expected, findings["risk"])
    graph._blend_rating(expected)
    reviewed_drafts = []
    monkeypatch.setattr(graph, "_checkpointed_critic", lambda draft: (
        reviewed_drafts.append(draft) or CriticReview(overall_assessment="Draft checked")
    ))
    result = graph._review_memo(
        memo, SimpleNamespace(degradation=DegradationLog(), run_id="fixture", as_of_date=date(2026, 9, 1)),
        SimpleNamespace(findings=findings),
    )
    assert result.rating_label == expected.rating_label
    assert result.confidence_score == expected.confidence_score
    assert result.scores["blended_pm_score"] == expected.scores["blended_pm_score"]
    assert result.scores["confidence"] == result.confidence_score
    assert reviewed_drafts[0]["final_pm_view"] == original
    if result.rating_label != rating or risk_confidence:
        assert result.final_pm_view.startswith(
            f"Final rating after risk review and factor blend: {result.rating_label} "
        )
        assert f"PM rationale before those adjustments (rating {rating};" in result.final_pm_view
        assert result.final_pm_view.endswith(original)
    else:
        assert result.final_pm_view == original
