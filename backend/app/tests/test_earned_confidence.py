"""W2b 7(c): earned confidence.

Owner decision 7(c): the PM's confidence is capped when sections are
template-filled, sources are thin, or the critic failed. Caps are
deterministic, the lowest binds, nothing ever raises the PM's number, and
the result is the ONE confidence the memo carries everywhere
(`confidence_score == scores["confidence"] == quality.confidence.final`).
Number-based caps belong to the number-to-source check (S15) and must not
fire while that check has not run.
"""
from __future__ import annotations

import pytest

from app.agents import graph, memo_quality
from app.schemas import AgentFinding, NumberCheck
from app.services.memo_sections import compute_availability
from app.tests.factories import make_finding, make_memo

KNOWN_CODES = {
    "pm_template", "template_sections", "critic_not_live", "no_transcript",
    "no_filing_review", "divergence_unreviewed",
}


def conf(**kw):
    base = dict(raw=90.0, pm_template=False, template_sections=[], critic_mode="live",
                transcript_given=True, filing_reviewed=True, divergence_unreviewed=False)
    base.update(kw)
    return memo_quality.earned_confidence(**base)


@pytest.mark.parametrize("kw, code, cap", [
    (dict(pm_template=True), "pm_template", 40.0),
    (dict(template_sections=["earnings_agent_view"]), "template_sections", 65.0),
    (dict(template_sections=["earnings_agent_view", "sector_agent_view"]), "template_sections", 55.0),
    (dict(template_sections=["a", "b", "c"]), "template_sections", 45.0),
    (dict(template_sections=["a", "b", "c", "d", "e"]), "template_sections", 45.0),
    (dict(critic_mode="rule_based"), "critic_not_live", 60.0),
    (dict(critic_mode="unavailable"), "critic_not_live", 60.0),
    (dict(critic_mode="pending"), "critic_not_live", 60.0),
    (dict(critic_mode="unknown"), "critic_not_live", 60.0),
    (dict(transcript_given=False), "no_transcript", 75.0),
    (dict(filing_reviewed=False), "no_filing_review", 75.0),
    (dict(divergence_unreviewed=True), "divergence_unreviewed", 55.0),
])
def test_each_cap(kw, code, cap):
    out = conf(**kw)
    assert [(c.code, c.cap) for c in out.caps] == [(code, cap)]
    assert out.raw == 90.0 and out.final == cap and out.binding == code


def test_no_caps_leaves_raw_untouched():
    out = conf()
    assert (out.raw, out.final, out.caps, out.binding) == (90.0, 90.0, [], None)


def test_min_of_caps_and_binding():
    out = conf(critic_mode="rule_based", transcript_given=False,
               template_sections=["earnings_agent_view", "filing_agent_view"])
    assert {c.code for c in out.caps} == {"critic_not_live", "no_transcript", "template_sections"}
    assert out.final == 55.0 and out.binding == "template_sections"
    # A tie binds on the first cap in table order.
    tie = conf(template_sections=["x", "y"], divergence_unreviewed=True)
    assert tie.final == 55.0 and tie.binding == "template_sections"


def test_never_raises_and_floor(monkeypatch):
    assert conf(raw=30.0).final == 30.0
    low = conf(raw=30.0, pm_template=True, critic_mode="rule_based")
    assert low.final == 30.0 and low.binding is None      # a cap above raw never binds
    # The floor: a cap can lower confidence to no less than 20 ...
    monkeypatch.setattr(memo_quality, "CAP_PM_TEMPLATE", 5.0)
    assert conf(raw=90.0, pm_template=True).final == 20.0
    # ... and the floor never lifts a raw value that was already lower.
    assert conf(raw=15.0, pm_template=True).final == 15.0


def test_number_caps_skipped_when_unchecked():
    """7(c)'s number-based caps come with the number check (S15). Until it
    runs, no NumberCheck — present, unchecked, or even carrying counts —
    produces a cap."""
    for nc in (None, NumberCheck(), NumberCheck(checked=False, counts={"untraceable": 30}),
               NumberCheck(checked=True, counts={"untraceable": 30, "traced": 0})):
        out = conf(number_check=nc)
        assert out.caps == [] and out.final == 90.0
    full = conf(pm_template=True, template_sections=["a"], critic_mode="unknown",
                transcript_given=False, filing_reviewed=False, divergence_unreviewed=True,
                number_check=NumberCheck(checked=True, counts={"untraceable": 30}))
    assert {c.code for c in full.caps} == KNOWN_CODES


def _llm_memo(**overrides):
    return make_memo(section_provenance={"v": 1, "llm_configured": True}, **overrides)


def test_template_contract_reads_compute_availability():
    """C2: the caps read W2a's presenter verdicts, so the sections a reader
    sees as "unavailable in this version" are exactly the ones capped."""
    earnings = make_finding("Earnings Analyst", data={"deterministic_fallback": "LLM returned nothing"})
    memo = _llm_memo(earnings_agent_view=earnings)
    pm_template, sections = memo_quality.template_filled(compute_availability(memo))
    assert pm_template is False and sections == ["earnings_agent_view"]
    # A retrieval soft note is not a template: the LLM still wrote the read.
    soft = make_finding("Filing Analyst", data={"retrieval_failed": True})
    assert memo_quality.template_filled(compute_availability(_llm_memo(filing_agent_view=soft)))[1] == []
    # A skipped analyst is "skipped", covered by no_filing_review instead.
    skipped = make_finding("Filing Analyst", data={"intake_skipped": True})
    assert memo_quality.template_filled(compute_availability(_llm_memo(filing_agent_view=skipped)))[1] == []


def test_industry_group_template_counts_as_core():
    ig = AgentFinding(agent="Industry Group Analyst", headline="Banks read", summary="s",
                      confidence=0.5, data={"deterministic_fallback": "no usable output"})
    memo = _llm_memo(extra_agent_views={"industry_group": ig})
    assert memo_quality.template_filled(compute_availability(memo))[1] == ["extra_agent_views.industry_group"]
    # Unmapped (no classification) is not a template.
    unmapped = ig.model_copy(update={"data": {"no_mapping": True}})
    memo = _llm_memo(extra_agent_views={"industry_group": unmapped})
    assert memo_quality.template_filled(compute_availability(memo))[1] == []


def test_pm_template_from_the_presenter():
    memo = make_memo(section_provenance={"v": 1, "llm_configured": False})
    assert memo_quality.template_filled(compute_availability(memo))[0] is True
    events = [{"agent": "PM Synthesis", "error_type": "DeterministicFallback", "message": "x"}]
    memo = _llm_memo(degradation_events=events, degraded_agents=["PM Synthesis"])
    assert memo_quality.template_filled(compute_availability(memo))[0] is True
    assert memo_quality.template_filled(compute_availability(_llm_memo()))[0] is False


# ---------------------------------------------------------------------------
# Through the pipeline
# ---------------------------------------------------------------------------

def _agrees(memo) -> None:
    q = memo.quality
    assert q is not None and q.confidence is not None
    assert memo.confidence_score == memo.scores["confidence"] == q.confidence.final
    assert memo.section_provenance["confidence"] == "earned"
    assert memo.final_verdict.startswith(
        f"PM final view: {memo.rating_label} (confidence {int(memo.confidence_score)}). ")


def test_confidence_everywhere_agrees():
    """Demo NVDA: no keys, so the PM and every LLM section are templates and
    the critic is rule-based: capped at 40 (pm_template binds)."""
    memo = graph.run_stock_memo("NVDA")
    _agrees(memo)
    c = memo.quality.confidence
    assert c.final <= 40.0 and c.final <= c.raw
    codes = {cap.code for cap in c.caps}
    assert {"pm_template", "template_sections", "critic_not_live"} <= codes
    if c.final < c.raw:
        assert c.binding == "pm_template"
        assert f"(confidence {memo.confidence_score:g}/100)" in memo.final_pm_view.split("\n")[0]


def test_a_live_looking_run_is_capped_by_what_is_missing(monkeypatch):
    """A PM that answered (no template) with a rule-based critic and no
    transcript: the critic cap binds at 60, never the PM's 88."""
    monkeypatch.setattr(graph, "_pm_synthesis", lambda *a, **kw: {
        "rating_label": "Neutral", "confidence_score": 88, "final_pm_view": "A real PM view.",
        "one_sentence_thesis": "NVDA is fairly priced — share gains are in the multiple.",
    })
    # has_llm patched on (keys stay blank; every model call answers nothing)
    # so the provenance says an LLM was configured and the PM is not a template.
    from app.config import Settings
    monkeypatch.setattr(Settings, "has_llm", property(lambda self: True))
    monkeypatch.setattr(graph, "latest_transcript", lambda t: None)
    from app.agents import llm as llm_mod
    monkeypatch.setattr(llm_mod, "chat_json", lambda *a, **k: None)
    memo = graph.run_stock_memo("NVDA")
    _agrees(memo)
    c = memo.quality.confidence
    codes = {cap.code: cap.cap for cap in c.caps}
    assert "pm_template" not in codes
    assert codes["critic_not_live"] == 60.0 and codes["no_transcript"] == 75.0
    assert c.final <= 60.0


def test_quality_stage_crash_still_caps(monkeypatch):
    def boom(**kw):
        raise RuntimeError("caps exploded")

    monkeypatch.setattr(memo_quality, "earned_confidence", boom)
    memo = graph.run_stock_memo("NVDA")
    assert "Memo Quality" in memo.degraded_agents
    _agrees(memo)
    assert memo.quality.confidence.caps[0].code == "quality_check_failed"
    assert memo.confidence_score <= memo_quality.CAP_TEMPLATE_SECTIONS[3]


# ---------------------------------------------------------------------------
# Stage wiring (`graph._assess_quality`) on fixture memos
# ---------------------------------------------------------------------------

def test_template_caps_fail_closed_when_the_classifier_crashes():
    """The presenter maps a classifier crash to "available / unclassified"
    so a reader's page never 500s. Read as "no template", that silently
    dropped pm_template and template_sections; the caps must refuse it."""
    from app.services import memo_sections
    unclassified = {k: memo_sections._av("available", None, ["unclassified"])
                    for k in memo_sections.SECTION_KEYS}
    with pytest.raises(ValueError, match="unclassified"):
        memo_quality.template_filled(unclassified)
    # A non-core section the classifier could not read does not block the caps.
    av = compute_availability(_llm_memo())
    av["technical_agent_view"] = memo_sections._av("available", None, ["unclassified"])
    assert memo_quality.template_filled(av) == (False, [])


def test_classifier_crash_ships_the_fallback_cap_and_banner(monkeypatch):
    """Demo NVDA with the W2a classifier crashing: the quality stage falls
    back (cap 45, "Memo Quality" on the banner) instead of dropping the
    template caps and shipping the PM's number as "earned"."""
    from app.services import memo_sections

    def boom(*a, **k):
        raise RuntimeError("classifier exploded")

    monkeypatch.setattr(memo_sections, "_classify", boom)
    memo = graph.run_stock_memo("NVDA")
    assert "Memo Quality" in memo.degraded_agents
    _agrees(memo)
    assert [c.code for c in memo.quality.confidence.caps] == ["quality_check_failed"]
    assert memo.confidence_score <= memo_quality.CAP_TEMPLATE_SECTIONS[3]


def _quality_inputs(*, filings=("10-K",), transcript="call", filing_data=None):
    from types import SimpleNamespace

    from app.tests.factories import make_findings
    findings = make_findings()
    if filing_data is not None:
        findings["filing"] = make_finding("Filing Analyst", data=filing_data)
    inputs = SimpleNamespace(transcript=transcript, filings=list(filings))
    return inputs, SimpleNamespace(findings=findings)


def _divergent_memo(*, rating: str, outcome: str, assessment: str = "not_assessed"):
    from app.schemas import CriticReview, MemoQuality, RatingReconciliation
    vv = memo_quality.valuation_evidence_verdict(
        family_pct=None, family_coverage=None, comps_premium=0.44, dcf_initial_upside=-0.54)
    assert vv.verdict == "overvalued"
    return _llm_memo(
        rating_label=rating, confidence_score=90.0, valuation_verdict=vv,
        risk_committee_challenge=CriticReview(overall_assessment="x", review_mode="live",
                                              valuation_divergence_assessment=assessment),
        quality=MemoQuality(rating_reconciliation=RatingReconciliation(
            outcome=outcome, pm_rating=rating, blended_rating="Bullish",
            final_rating=rating, valuation_verdict="overvalued", divergence=True,
            critic_assessment=assessment)),
    )


@pytest.mark.parametrize("rating, outcome, assessment, capped", [
    # Accepted on a reason no live critic assessed: the cap applies.
    ("Bullish", "accepted", "not_assessed", True),
    # Accepted and the live critic supported it: independently reviewed.
    ("Bullish", "accepted", "supported", False),
    # Downgraded to Neutral: no divergence ships.
    ("Neutral", "downgraded", "not_assessed", False),
    # Record mode: downgraded but not applied. The kill switch leaves
    # confidence alone too, and "a reason no live critic reviewed" would
    # misdescribe a missing or critic-rejected reason.
    ("Bullish", "downgraded", "not_assessed", False),
    ("Bullish", "downgraded", "unsupported", False),
])
def test_divergence_unreviewed_cap_wiring(rating, outcome, assessment, capped):
    memo = _divergent_memo(rating=rating, outcome=outcome, assessment=assessment)
    inputs, analysts = _quality_inputs()
    codes = {c.code for c in graph._assess_quality(memo, inputs, analysts).confidence.caps}
    assert ("divergence_unreviewed" in codes) is capped, codes


@pytest.mark.parametrize("filings, filing_data, capped", [
    (("10-K",), None, False),
    ((), None, True),                          # no filing at all
    (("10-K",), {"intake_skipped": True}, True),  # intake skipped the filing analyst
])
def test_no_filing_review_cap_wiring(filings, filing_data, capped):
    memo = _divergent_memo(rating="Neutral", outcome="consistent")
    inputs, analysts = _quality_inputs(filings=filings, filing_data=filing_data)
    caps = {c.code: c.cap for c in graph._assess_quality(memo, inputs, analysts).confidence.caps}
    assert ("no_filing_review" in caps) is capped, caps
    if capped:
        assert caps["no_filing_review"] == 75.0
