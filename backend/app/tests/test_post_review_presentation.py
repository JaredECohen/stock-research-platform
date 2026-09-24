"""Final research presentation must distinguish the original PM opinion.

W2b §6.3: the "Final rating after ..." preface and the final verdict's
"PM final view: <rating> (confidence N)" lead are rendered ONCE, by
`graph._render_final_texts`, after the valuation check and the earned-
confidence caps have run — so neither can quote a rating or a confidence
that a later stage moved.
"""
from datetime import date
from types import SimpleNamespace

import pytest

from app.agents import graph, memo_quality
from app.agents.memo_context import PMOpinion, VerdictOutcome
from app.agents.safe_runner import DegradationLog
from app.schemas import (
    ConfidenceAssessment,
    ConfidenceCap,
    CriticReview,
    MemoQuality,
    MispricingThesis,
    RatingReconciliation,
)
from app.tests.factories import make_findings, make_memo


def _verdict(body: str = "Thesis body. Watch items: none flagged.") -> VerdictOutcome:
    from app.schemas import ValuationVerdict
    return VerdictOutcome(
        valuation_verdict=ValuationVerdict(), one_sentence_thesis="t",
        mispricing_thesis=MispricingThesis(), final_verdict_body=body,
    )


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
    initial = PMOpinion(rating, 60.0, original)
    result = graph._review_memo(
        memo, SimpleNamespace(degradation=DegradationLog(), run_id="fixture", as_of_date=date(2026, 9, 1)),
        SimpleNamespace(findings=findings),
    )
    assert result.rating_label == expected.rating_label
    assert result.confidence_score == expected.confidence_score
    assert result.scores["blended_pm_score"] == expected.scores["blended_pm_score"]
    assert result.scores["confidence"] == result.confidence_score
    assert reviewed_drafts[0]["final_pm_view"] == original
    # The review stage no longer writes the preface: it is rendered once,
    # from final values, after the quality stage.
    assert result.final_pm_view == original
    graph._render_final_texts(result, _verdict(), initial)
    if result.rating_label != rating or risk_confidence:
        assert result.final_pm_view.startswith(
            f"Final rating after risk review and factor blend: {result.rating_label} "
        )
        assert f"PM rationale before those adjustments (rating {rating};" in result.final_pm_view
        assert result.final_pm_view.endswith(original)
    else:
        assert result.final_pm_view == original


def test_render_names_the_valuation_check_and_the_confidence_cap():
    """When the 7(b) check moved the rating and a 7(c) cap moved the
    confidence, the preface says so and quotes the FINAL values."""
    original = "Research view: Bullish. The PM's own reasoning."
    memo = make_memo(
        rating_label="Neutral", confidence_score=40.0, final_pm_view=original,
        quality=MemoQuality(
            rating_reconciliation=RatingReconciliation(
                outcome="downgraded", pm_rating="Bullish", blended_rating="Bullish",
                final_rating="Neutral", valuation_verdict="overvalued", divergence=True,
            ),
            confidence=ConfidenceAssessment(
                raw=62.0, final=40.0, binding="pm_template",
                caps=[ConfidenceCap(code="pm_template", cap=40.0)],
            ),
        ),
    )
    graph._render_final_texts(memo, _verdict("Body."), PMOpinion("Bullish", 62.0, original))
    assert memo.final_pm_view.startswith(
        "Final rating after risk review, factor blend, valuation check and evidence cap on "
        "confidence: Neutral (confidence 40/100).")
    assert "(rating Bullish; confidence 62/100)" in memo.final_pm_view
    assert memo.final_pm_view.endswith(original)
    assert memo.final_verdict == "PM final view: Neutral (confidence 40). Body."


def test_render_leaves_an_unmoved_pm_view_alone():
    memo = make_memo(rating_label="Neutral", confidence_score=55.0, final_pm_view="PM view.")
    graph._render_final_texts(memo, _verdict("Body."), PMOpinion("Neutral", 55.0, "PM view."))
    assert memo.final_pm_view == "PM view."
    assert memo.final_verdict == "PM final view: Neutral (confidence 55). Body."


def test_full_run_texts_quote_the_final_rating_and_capped_confidence():
    """End to end (demo NVDA: the factor blend lifts a Neutral keyword PM to
    Bullish on overvalued evidence, the valuation check sets Neutral, and
    the template-PM cap binds): both confidence-bearing strings quote the
    published values, never an intermediate one."""
    memo = graph.run_stock_memo("NVDA")
    c = memo.confidence_score
    assert memo.final_verdict.startswith(f"PM final view: {memo.rating_label} (confidence {int(c)}). ")
    if memo.final_pm_view.startswith("Final rating after"):
        assert f": {memo.rating_label} (confidence {c:g}/100)." in memo.final_pm_view.split("\n")[0]
    q = memo.quality
    assert q is not None and q.confidence is not None
    if q.confidence.final < q.confidence.raw:
        assert "evidence cap on confidence" in memo.final_pm_view
    rec = q.rating_reconciliation
    assert rec is not None
    if rec.outcome == "downgraded" and rec.final_rating != rec.blended_rating:
        assert "valuation check" in memo.final_pm_view
    assert memo_quality.diverges(memo.rating_label, memo.valuation_verdict.verdict) is False


def test_reflection_sees_the_final_memo(monkeypatch):
    """W2b §6.4: long-term memory is written from the FINAL memo — the
    reconciled rating, the earned confidence and the rendered texts — not
    from the pre-blend draft it used to see."""
    seen: list[dict] = []
    monkeypatch.setattr(graph, "_run_reflection_step", lambda memo: seen.append(
        memo.model_dump(include={"rating_label", "confidence_score", "final_pm_view",
                                 "final_verdict", "quality"})) or ([], []))
    memo = graph.run_stock_memo("NVDA")
    (at_reflection,) = seen
    assert at_reflection["rating_label"] == memo.rating_label
    assert at_reflection["confidence_score"] == memo.confidence_score == memo.quality.confidence.final
    assert at_reflection["final_pm_view"] == memo.final_pm_view
    assert at_reflection["final_verdict"] == memo.final_verdict != ""
    assert at_reflection["quality"]["confidence"]["final"] == memo.confidence_score


def test_backtests_skip_reflection(monkeypatch):
    calls: list = []
    monkeypatch.setattr(graph, "_run_reflection_step", lambda memo: calls.append(memo) or ([], []))
    graph._run_reflection(make_memo(), SimpleNamespace(as_of_date=date(2025, 1, 2),
                                                       degradation=DegradationLog()))
    assert calls == []
