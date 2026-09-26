"""W2b 7(b): the rating must match the valuation, from EVIDENCE.

Owner decision 7(b): a Bullish rating on an overvalued valuation verdict
(or the Bearish mirror) needs an explicit stated reason, or it is
downgraded. Before this slice the rule could never fire: the verdict was
derived from the rating (`graph._verdict_word`). These tests pin that the
verdict is an evidence read that never sees the rating, that the centred
two-signal rule (W3 P7) cannot be carried by one signal, that the rule
binds the published (post-blend) rating, that a reason must engage with
the number it overrides, and that every failure fails closed.
"""
from __future__ import annotations

import inspect
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from app.agents import graph, memo_quality
from app.agents.graph import _build_verdict
from app.agents.memo_context import DCFStage
from app.agents.safe_runner import DegradationLog
from app.config import settings
from app.finance import scorecard_spec
from app.schemas import (
    CriticReview,
    DCFResult,
    MemoQuality,
    RatingReconciliation,
    ScorecardCategory,
    ScorecardSummary,
    ValuationVerdict,
)
from app.tests.factories import make_findings, make_inputs, make_memo, make_profile


def ev(**kw: Any) -> ValuationVerdict:
    base: dict[str, Any] = dict(family_pct=None, family_coverage=None, comps_premium=None,
                                dcf_initial_upside=None, dcf_final_upside=None)
    base.update(kw)
    return memo_quality.valuation_evidence_verdict(**base)


# ---------------------------------------------------------------------------
# The evidence verdict
# ---------------------------------------------------------------------------

# Signal values of stored memos from the W3 evidence bundle (numbers only).
@pytest.mark.parametrize("case, kw, verdict", [
    ("AAPL v8", dict(family_pct=52.0, family_coverage=1.0, comps_premium=0.798, dcf_initial_upside=-0.50),
     "overvalued"),
    ("GOOGL v6, no scorecard", dict(comps_premium=0.444, dcf_initial_upside=-0.63), "overvalued"),
    ("ADBE v11", dict(family_pct=93.5, family_coverage=1.0, comps_premium=-0.47, dcf_initial_upside=0.05),
     "undervalued"),
    ("META v1", dict(family_pct=5.0, family_coverage=1.0, comps_premium=-0.15, dcf_initial_upside=0.23),
     "mixed"),
    ("NVDA v231: DCF rich, comps cheap", dict(family_pct=48.0, family_coverage=1.0, comps_premium=-0.184,
                                              dcf_initial_upside=-0.41), "fairly_priced"),
    ("MSFT v110", dict(family_pct=72.0, family_coverage=1.0, comps_premium=-0.006, dcf_initial_upside=-0.19),
     "fairly_priced"),
    ("TSLA v2", dict(family_pct=1.0, family_coverage=1.0, comps_premium=8.5, dcf_initial_upside=-0.98),
     "overvalued"),
])
def test_evidence_verdict_table(case, kw, verdict):
    vv = ev(**kw)
    assert vv.verdict == verdict, case
    assert vv.basis == "evidence" and vv.signals["method"] == memo_quality.METHOD


def test_no_evidence_is_fairly_priced_and_not_applicable():
    vv = ev()
    assert vv.verdict == "fairly_priced" and vv.signals["available"] == []
    assert "valuation evidence unavailable" in vv.summary
    assert memo_quality.valuation_evidence_block(vv) == ""
    rec = memo_quality.reconcile_rating(blended_rating="Very Bullish", verdict=vv, pm=None, critic=None)
    assert rec.outcome == "not_applicable" and rec.final_rating == "Very Bullish"


def test_verdict_is_independent_of_rating(monkeypatch):
    """The critique's core defect: the verdict followed the rating badge.
    The rule has no rating input at all, and a memo's verdict is identical
    whatever the PM rates."""
    from app.agents.intake import IntakeDecision
    from app.agents.memo_context import AnalystRound
    from app.services.valuation_service import build_comps, build_dcf

    params = inspect.signature(memo_quality.valuation_evidence_verdict).parameters
    assert not any("rating" in p for p in params)
    # Same inputs, five PM ratings. (Separate full runs would not do: the
    # persistent DCF store moves the model between runs on its own.)
    dcf = build_dcf("NVDA")
    inputs = make_inputs("NVDA", comps=build_comps("NVDA"), dcf=dcf)
    stage = DCFStage(dcf=dcf, initial_dcf=dcf)
    verdicts = set()
    for rating in ("Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish"):
        monkeypatch.setattr(graph, "_pm_synthesis", lambda *a, _r=rating, **kw: {
            "rating_label": _r, "confidence_score": 60, "final_pm_view": "v",
            "one_sentence_thesis": "NVDA compounds data-center share.",
        })
        memo = graph._compose_memo(inputs, AnalystRound(findings=make_findings(), intake=IntakeDecision()),
                                   stage)
        assert memo.rating_label == rating
        verdicts.add(memo.valuation_verdict.model_dump_json())
    assert len(verdicts) == 1
    assert '"verdict":"overvalued"' in verdicts.pop()


@pytest.mark.parametrize("kw", [
    dict(family_pct=99.0, family_coverage=1.0),
    dict(family_pct=0.5, family_coverage=1.0),
    dict(comps_premium=3.0),
    dict(comps_premium=-0.9),
    dict(dcf_initial_upside=-0.95),
    dict(dcf_initial_upside=2.5),
    dict(dcf_initial_upside=-0.9, dcf_final_upside=-0.9),
])
def test_verdict_never_from_one_signal(kw):
    """No single signal, however extreme, makes a directional verdict."""
    vv = ev(**kw)
    assert vv.verdict == "fairly_priced", (kw, vv.summary)


def test_initial_dcf_votes_only_with_comps_agreement():
    # Comps rich and a large rich DCF agree: two rich votes.
    assert ev(comps_premium=0.20, dcf_initial_upside=-0.50).signals["votes"] == {
        "comps_ev_ebitda": -1, "dcf_initial": -1}
    assert ev(comps_premium=0.20, dcf_initial_upside=-0.50).verdict == "overvalued"
    # Comps inside its band: the DCF has nothing to corroborate.
    assert ev(comps_premium=0.05, dcf_initial_upside=-0.50).signals["votes"]["dcf_initial"] == 0
    # Comps cheap, DCF rich: the DCF never opposes, so no `mixed` from it.
    opposite = ev(comps_premium=-0.20, dcf_initial_upside=-0.50)
    assert opposite.signals["votes"]["dcf_initial"] == 0 and opposite.verdict == "fairly_priced"
    # Below the 40% band.
    assert ev(comps_premium=0.20, dcf_initial_upside=-0.39).signals["votes"]["dcf_initial"] == 0
    # A terminal-value-clamped model never votes.
    clamped = ev(comps_premium=0.20, dcf_initial_upside=-0.60, dcf_initial_tv_clamped=True)
    assert clamped.signals["votes"]["dcf_initial"] == 0 and clamped.verdict == "fairly_priced"
    assert "terminal value clamped" in clamped.summary


def test_family_direction_pinned_to_fs_v1_signs():
    """A HIGH valuation-family percentile is a CHEAP name only because every
    fs-v1 valuation feature is a yield scored higher-is-better. If a feature
    with the other sign joins the family, this rule must be revisited."""
    family = [f for f in scorecard_spec.FEATURE_SPEC if f.family == scorecard_spec.FAMILY_VALUATION]
    assert family, "the fs-v1 spec has no valuation family"
    assert all(f.sign == +1 for f in family), [(f.name, f.sign) for f in family]
    assert all(f.name.endswith("_yield") for f in family)
    assert ev(family_pct=85.0, family_coverage=0.9).signals["votes"]["valuation_family"] == 1
    assert ev(family_pct=15.0, family_coverage=0.9).signals["votes"]["valuation_family"] == -1
    assert ev(family_pct=50.0, family_coverage=0.9).signals["votes"]["valuation_family"] == 0


def test_family_votes_only_at_the_coverage_floor():
    low = ev(family_pct=5.0, family_coverage=0.5, comps_premium=0.30)
    assert "valuation_family" not in low.signals["votes"] and low.verdict == "fairly_priced"
    assert "below the 60% floor" in low.summary
    ok = ev(family_pct=5.0, family_coverage=0.6, comps_premium=0.30)
    assert ok.signals["votes"]["valuation_family"] == -1 and ok.verdict == "overvalued"


def test_absolute_factor_never_votes():
    vv = ev(comps_premium=0.30, factor_valuation=2.0)
    assert vv.factor_valuation == 2.0
    assert "factor" not in " ".join(vv.signals["votes"]) and vv.verdict == "fairly_priced"


@pytest.mark.parametrize("kw, votes", [
    # The spec's bands are inclusive: >= 80 cheap, <= 20 rich, +/-15%, |u| >= 40%.
    (dict(family_pct=80.0, family_coverage=1.0), {"valuation_family": 1}),
    (dict(family_pct=79.9, family_coverage=1.0), {"valuation_family": 0}),
    (dict(family_pct=20.0, family_coverage=1.0), {"valuation_family": -1}),
    (dict(family_pct=20.1, family_coverage=1.0), {"valuation_family": 0}),
    (dict(comps_premium=0.15), {"comps_ev_ebitda": -1}),
    (dict(comps_premium=0.149), {"comps_ev_ebitda": 0}),
    (dict(comps_premium=-0.15), {"comps_ev_ebitda": 1}),
    (dict(comps_premium=0.20, dcf_initial_upside=-0.40), {"comps_ev_ebitda": -1, "dcf_initial": -1}),
    (dict(comps_premium=-0.20, dcf_initial_upside=0.40), {"comps_ev_ebitda": 1, "dcf_initial": 1}),
    (dict(comps_premium=0.20, dcf_initial_upside=-0.399), {"comps_ev_ebitda": -1, "dcf_initial": 0}),
])
def test_vote_thresholds_at_their_boundaries(kw, votes):
    assert ev(**kw).signals["votes"] == votes


@pytest.mark.parametrize("kw, why", [
    # Negative target EBITDA: (-40 - 20) / 20 = -300%, a "discount" that is
    # really a loss-maker. With a +50% DCF it used to read undervalued.
    (dict(comps_premium=-3.0, comps_target_multiple=-40.0, comps_peer_median_multiple=20.0),
     "negative EV/EBITDA multiple"),
    # Negative peer median: (20 - -10) / 10 = +300%, a meaningless "premium".
    (dict(comps_premium=3.0, comps_target_multiple=20.0, comps_peer_median_multiple=-10.0),
     "negative EV/EBITDA multiple"),
    # Stored bodies carry the premium only: <= -100% implies a negative multiple.
    (dict(comps_premium=-1.0), "negative EV/EBITDA multiple"),
    # EV is not meaningful for banks, insurers or REITs (the scorecard's rule).
    (dict(comps_premium=0.40, sector="Financial Services"), "not meaningful for banks"),
    (dict(comps_premium=-0.40, sector="Real Estate"), "not meaningful for banks"),
])
def test_comps_never_votes_on_a_meaningless_multiple(kw, why):
    """Comps is the pivotal vote (the DCF may only agree with it), so a
    meaningless premium plus a large DCF used to make a directional verdict
    that 7(b) then enforced against a Bearish or Bullish rating."""
    dcf = 0.50 if kw["comps_premium"] < 0 else -0.50
    vv = ev(dcf_initial_upside=dcf, **kw)
    assert vv.verdict == "fairly_priced", vv.summary
    assert "comps_ev_ebitda" not in vv.signals["votes"]
    assert vv.signals["votes"]["dcf_initial"] == 0
    entry = vv.signals["comps_ev_ebitda"]
    assert entry["vote"] is None and why in entry["no_vote"] and "display" not in entry
    assert "not meaningful" in vv.summary and "300%" not in vv.summary
    # Not listed as a value in the PM's evidence block, and nothing to enforce.
    assert "EV/EBITDA premium to peers" not in memo_quality.valuation_evidence_block(vv)
    rating = "Bearish" if dcf > 0 else "Bullish"
    rec = _reconcile(rating, vv)
    assert (rec.outcome, rec.final_rating) == ("consistent", rating)


def test_positive_multiples_outside_financials_still_vote():
    vv = ev(comps_premium=-0.30, comps_target_multiple=14.0, comps_peer_median_multiple=20.0,
            sector="Technology", dcf_initial_upside=0.50)
    assert vv.signals["votes"] == {"comps_ev_ebitda": 1, "dcf_initial": 1}
    assert vv.verdict == "undervalued"


def test_compose_stage_passes_the_multiples_and_sector(monkeypatch):
    """`graph._evidence_verdict` hands the rule both multiples and the
    profile's sector, so the guards above bind the pipeline."""
    from app.services.valuation_service import build_comps, build_dcf

    dcf = build_dcf("NVDA")
    comps = build_comps("NVDA")
    assert comps is not None
    stage = DCFStage(dcf=dcf, initial_dcf=dcf)
    neg = comps.model_copy(update={
        "target": comps.target.model_copy(update={"ev_ebitda": -40.0}),
        "premium_discount": {**comps.premium_discount, "ev_ebitda": -3.0},
    })
    vv = graph._evidence_verdict(make_inputs("NVDA", comps=neg, dcf=dcf), stage)
    assert vv.signals["comps_ev_ebitda"]["no_vote"] == "negative EV/EBITDA multiple"
    bank = make_inputs("NVDA", comps=comps, dcf=dcf)
    bank.profile = {**bank.profile, "sector": "Financials"}
    vv = graph._evidence_verdict(bank, stage)
    assert "banks" in vv.signals["comps_ev_ebitda"]["no_vote"]
    assert "no_vote" not in graph._evidence_verdict(make_inputs("NVDA", comps=comps, dcf=dcf),
                                                    stage).signals["comps_ev_ebitda"]


def _dcf_with_base(dcf: DCFResult, upside: float) -> DCFResult:
    data = dcf.model_dump()
    data["base"]["upside_pct"] = upside
    return DCFResult(**data)


def test_bullish_pm_dcf_adjustment_cannot_move_the_verdict():
    """The PM who rates the name also adjusts the DCF; the adjusted model is
    recorded but never votes, so a bullish adjustment cannot talk the
    evidence out of `overvalued`."""
    from app.services.valuation_service import build_comps, build_dcf
    initial = build_dcf("NVDA")
    assert initial is not None
    inputs = make_inputs("NVDA", comps=build_comps("NVDA"))
    base = graph._evidence_verdict(inputs, DCFStage(dcf=initial, initial_dcf=initial))
    bullish = _dcf_with_base(initial, 1.50)
    adjusted = graph._evidence_verdict(inputs, DCFStage(dcf=bullish, initial_dcf=initial,
                                                       pm_adjustments=[{"field": "growth"}]))
    assert adjusted.verdict == base.verdict
    assert adjusted.signals["votes"] == base.signals["votes"]
    assert adjusted.signals["dcf_pm_adjusted"] == {"upside": 1.50, "vote": None, "display": "+150%"}
    # The display number is the adjusted (dcf_summary) one; the vote is not.
    assert adjusted.dcf_base_upside == 1.50
    # And the pure rule, directly: only the initial upside changes the vote.
    assert ev(comps_premium=0.3, dcf_initial_upside=-0.5, dcf_final_upside=0.9).verdict == "overvalued"


def test_scorecard_family_feeds_the_compose_stage_verdict():
    summary = ScorecardSummary(
        version_key="fs-v1", as_of=date(2026, 9, 21), universe_percentile=40.0, coverage=0.9,
        categories={"valuation": ScorecardCategory(z=-1.5, percentile=8.0, weight=0.125, coverage=1.0)},
    )
    inputs = make_inputs("NVDA", scorecard=summary)
    vv = graph._evidence_verdict(inputs, DCFStage(dcf=None, initial_dcf=None))
    assert vv.signals["valuation_family"]["percentile"] == 8.0
    assert vv.signals["votes"] == {"valuation_family": -1}


# ---------------------------------------------------------------------------
# The divergence reason and the reconciliation matrix
# ---------------------------------------------------------------------------

OVERVALUED = dict(comps_premium=0.444, dcf_initial_upside=-0.54, dcf_final_upside=-0.30)
UNDERVALUED = dict(family_pct=91.0, family_coverage=1.0, comps_premium=-0.32)
GOOD_BULL_REASON = (
    "We override the EV/EBITDA premium of 44% to peers: cloud revenue grew 32% in the "
    "latest quarter while operating margin widened to 41%, and the peer multiple does not "
    "yet price that operating leverage."
)
GOOD_BEAR_REASON = (
    "The scorecard valuation-family rank at the 91st percentile looks cheap, but free cash "
    "flow fell 38% as working capital unwound and the cheapness is a value trap rather than an "
    "opportunity."
)


def _reconcile(rating: str, verdict: ValuationVerdict, reason: str = "", *,
               critic: CriticReview | None = None, enforce: bool = True) -> RatingReconciliation:
    return memo_quality.reconcile_rating(
        blended_rating=rating, verdict=verdict,
        pm=RatingReconciliation(pm_rating=rating, reason=reason), critic=critic, enforce=enforce,
    )


LIVE_SUPPORTED = CriticReview(overall_assessment="x", review_mode="live",
                              valuation_divergence_assessment="supported")
LIVE_UNSUPPORTED = CriticReview(overall_assessment="x", review_mode="live",
                                valuation_divergence_assessment="unsupported")
LIVE_SILENT = CriticReview(overall_assessment="x", review_mode="live")
RULE_BASED = CriticReview(overall_assessment="x", review_mode="rule_based",
                          valuation_divergence_assessment="unsupported")  # ignored: not live


@pytest.mark.parametrize("rating, kw, reason, critic, outcome, final", [
    ("Bullish", OVERVALUED, "", None, "downgraded", "Neutral"),
    ("Very Bullish", OVERVALUED, "", None, "downgraded", "Neutral"),
    ("Bullish", OVERVALUED, GOOD_BULL_REASON, LIVE_SUPPORTED, "accepted", "Bullish"),
    ("Bullish", OVERVALUED, GOOD_BULL_REASON, RULE_BASED, "accepted", "Bullish"),
    ("Bullish", OVERVALUED, GOOD_BULL_REASON, LIVE_SILENT, "accepted", "Bullish"),
    ("Bullish", OVERVALUED, GOOD_BULL_REASON, LIVE_UNSUPPORTED, "downgraded", "Neutral"),
    ("Bullish", OVERVALUED, "n/a", None, "downgraded", "Neutral"),
    ("Bullish", OVERVALUED, "Quality deserves a premium versus peers because the franchise is "
                            "durable and the management team executes well every single year.",
     None, "downgraded", "Neutral"),
    ("Bearish", UNDERVALUED, "", None, "downgraded", "Neutral"),
    ("Very Bearish", UNDERVALUED, GOOD_BEAR_REASON, None, "accepted", "Very Bearish"),
    ("Neutral", OVERVALUED, "", None, "consistent", "Neutral"),
    ("Bullish", dict(family_pct=5.0, family_coverage=1.0, comps_premium=-0.3), "", None,
     "consistent", "Bullish"),   # mixed
    ("Bullish", dict(comps_premium=0.3), "", None, "consistent", "Bullish"),  # fairly priced
    ("Bearish", OVERVALUED, "", None, "consistent", "Bearish"),
])
def test_reconcile_matrix(rating, kw, reason, critic, outcome, final):
    rec = _reconcile(rating, ev(**kw), reason, critic=critic)
    assert (rec.outcome, rec.final_rating) == (outcome, final), rec.note
    assert rec.blended_rating == rating
    if outcome == "accepted":
        expected = "critic: supported" if critic is LIVE_SUPPORTED else "not independently reviewed"
        assert expected in rec.note
    if outcome == "downgraded":
        assert rec.note.startswith("Rating set to Neutral")
    if critic is RULE_BASED:
        assert rec.critic_assessment == "not_assessed"


def test_reason_must_quote_the_overridden_signal():
    """Naming the signal is not enough: the reason must quote the value it
    overrides at display precision, checked against the stored signals."""
    vv = ev(**OVERVALUED)   # comps +44.4% rich, initial DCF -54% rich
    checks = memo_quality.assess_divergence_reason(GOOD_BULL_REASON, verdict=vv, rating="Bullish")
    assert checks == {"substantive": True, "names_signal": True, "quotes_value": True}
    wrong_value = GOOD_BULL_REASON.replace("44%", "45%")
    assert memo_quality.assess_divergence_reason(
        wrong_value, verdict=vv, rating="Bullish")["quotes_value"] is False
    more_precise = GOOD_BULL_REASON.replace("44%", "44.4%")
    assert memo_quality.assess_divergence_reason(
        more_precise, verdict=vv, rating="Bullish")["quotes_value"] is True
    # The DCF voted too, so quoting it (sign dropped, as prose does) counts.
    dcf_reason = ("The consensus DCF shows 54% downside, but it anchors on a growth path that "
                  "ignores the 32% cloud growth and 41% operating margin reported this quarter.")
    assert all(memo_quality.assess_divergence_reason(dcf_reason, verdict=vv, rating="Bullish").values())
    # A signal that did NOT vote against the rating is not an override.
    fam_only = ("The scorecard valuation family sits at the 52nd percentile, so the multiple is "
                "fine given 32% cloud growth and a 41% operating margin this year.")
    vv2 = ev(family_pct=52.0, family_coverage=1.0, **OVERVALUED)
    got = memo_quality.assess_divergence_reason(fam_only, verdict=vv2, rating="Bullish")
    assert got["names_signal"] is True  # "multiple" names comps, which did vote
    assert got["quotes_value"] is False  # but 44% is never quoted
    rec = _reconcile("Bullish", vv2, fam_only)
    assert rec.outcome == "downgraded" and "quotes_value" in rec.note


SHORT_QUOTING = "The EV/EBITDA premium of 44% is fine."
LONG_NO_SIGNAL = (
    "Cloud revenue grew 44% in the latest quarter while operating margin widened to 41%, and "
    "management guided to another year of double-digit growth across every segment."
)
PLACEHOLDER_LONG = (
    "n/a — the EV/EBITDA premium of 44% to peers is noted, but no further reason is given here "
    "for the rating, which the team will revisit at the next full review of the name."
)


@pytest.mark.parametrize("reason, expected", [
    (GOOD_BULL_REASON, {"substantive": True, "names_signal": True, "quotes_value": True}),
    # Only `substantive` fails: too short, though it names and quotes the signal.
    (SHORT_QUOTING, {"substantive": False, "names_signal": True, "quotes_value": True}),
    # Only `substantive` fails: long, names and quotes, but opens as a placeholder.
    (PLACEHOLDER_LONG, {"substantive": False, "names_signal": True, "quotes_value": True}),
    # Long, carries the right figure, names no signal (so quotes nothing).
    (LONG_NO_SIGNAL, {"substantive": True, "names_signal": False, "quotes_value": False}),
])
def test_each_reason_check_isolated(reason, expected):
    vv = ev(**OVERVALUED)
    assert memo_quality.assess_divergence_reason(reason, verdict=vv, rating="Bullish") == expected
    rec = _reconcile("Bullish", vv, reason)
    assert rec.reason_checks == expected
    assert rec.outcome == ("accepted" if all(expected.values()) else "downgraded")


def test_quote_must_be_the_signal_not_any_number():
    """A figure counts only as the signal's value: near its name, with the
    rank's unit for the percentile, and without a contradicting sign or
    direction word for comps and the DCF."""
    fam = ev(family_pct=10.0, family_coverage=1.0, **OVERVALUED)   # family rich too
    prose = ("The scorecard rank ignores that this business compounded earnings for 10 years "
             "straight and should keep doing so given its moat and pricing power in every market.")
    assert memo_quality.assess_divergence_reason(prose, verdict=fam, rating="Bullish") == {
        "substantive": True, "names_signal": True, "quotes_value": False}
    ranked = prose.replace("ignores that", "at the 10th percentile ignores that")
    assert memo_quality.assess_divergence_reason(ranked, verdict=fam, rating="Bullish")["quotes_value"]
    # Far from the signal's name: not a quote of it.
    far = ("The consensus DCF is anchored on a stale growth path that the latest quarter already "
           "broke, with cloud revenue accelerating and operating margin widening again to 54% here.")
    vv = ev(**OVERVALUED)                     # DCF -54%, comps +44.4%
    assert not memo_quality.assess_divergence_reason(far, verdict=vv, rating="Bullish")["quotes_value"]
    base = ("The consensus DCF shows {q}, but it anchors on a growth path that ignores the 32% "
            "cloud growth and 41% operating margin reported this quarter.")
    for q, ok in (("54% downside", True), ("-54% to fair value", True), ("54% upside", False),
                  ("+54% to fair value", False)):
        got = memo_quality.assess_divergence_reason(base.format(q=q), verdict=vv, rating="Bullish")
        assert got["quotes_value"] is ok, q
    comps = ("We override the EV/EBITDA {q} to peers: cloud revenue grew 32% in the latest quarter "
             "while operating margin widened to 41%, which the peer multiple does not price.")
    assert memo_quality.assess_divergence_reason(
        comps.format(q="premium of 44%"), verdict=vv, rating="Bullish")["quotes_value"]
    assert not memo_quality.assess_divergence_reason(
        comps.format(q="level, a 44% discount"), verdict=vv, rating="Bullish")["quotes_value"]


def test_record_mode_records_without_enforcing():
    rec = _reconcile("Bullish", ev(**OVERVALUED), "", enforce=False)
    assert rec.outcome == "downgraded" and rec.final_rating == "Bullish"
    assert rec.note.startswith("Recorded only (rating_reconciliation_mode=record)")


def test_reason_check_fails_closed(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("checker exploded")

    monkeypatch.setattr(memo_quality, "assess_divergence_reason", boom)
    rec = _reconcile("Bullish", ev(**OVERVALUED), GOOD_BULL_REASON, critic=LIVE_SUPPORTED)
    assert rec.outcome == "downgraded" and rec.final_rating == "Neutral"
    assert rec.reason_checks.get("check_failed") is True
    assert "could not be verified" in rec.note
    # Through the review stage: a soft "Rating Check" banner entry.
    memo = _reviewable_memo("Bullish", 90.0, verdict=ev(**OVERVALUED), reason=GOOD_BULL_REASON)
    log = DegradationLog()
    _review(memo, log)
    assert memo.rating_label == "Neutral"
    assert "Rating Check" in log.degraded_agents()


def _reviewable_memo(rating: str, factor: float, *, verdict: ValuationVerdict, reason: str = ""):
    return make_memo(
        rating_label=rating, confidence_score=60.0, valuation_verdict=verdict,
        scores={"factor_pm_score": factor, "confidence": 60.0},
        quality=MemoQuality(rating_reconciliation=RatingReconciliation(
            pm_rating=rating, pm_confidence=60.0, valuation_verdict=verdict.verdict, reason=reason)),
    )


def _review(memo, log: DegradationLog, critic: CriticReview | None = None):
    findings = make_findings()
    findings["risk"].data = {"recommendations": []}
    original = graph._checkpointed_critic
    graph._checkpointed_critic = lambda _d: critic or CriticReview(overall_assessment="ok")  # type: ignore[assignment]
    try:
        return graph._review_memo(
            memo, SimpleNamespace(degradation=log, run_id="fixture", as_of_date=date(2026, 9, 1)),
            SimpleNamespace(findings=findings),
        )
    finally:
        graph._checkpointed_critic = original  # type: ignore[assignment]


def test_blend_up_without_pm_reason_is_downgraded(monkeypatch):
    """The rule binds the PUBLISHED rating: a Neutral PM blended up to
    Bullish by the factor score (0.4*50 + 0.6*72 = 63.2) on overvalued
    evidence never argued Bullish, so it has no reason and lands Neutral."""
    monkeypatch.setattr(settings, "llm_rating_weight", 0.4)
    memo = _reviewable_memo("Neutral", 72.0, verdict=ev(**OVERVALUED))
    _review(memo, DegradationLog())
    rec = memo.quality.rating_reconciliation
    assert rec.pm_rating == "Neutral" and rec.blended_rating == "Bullish"
    assert (rec.outcome, rec.final_rating, memo.rating_label) == ("downgraded", "Neutral", "Neutral")
    assert memo.scores["blended_pm_score"] == pytest.approx(63.2)


def test_bullish_pm_blended_down_is_consistent_and_keeps_its_reason(monkeypatch):
    monkeypatch.setattr(settings, "llm_rating_weight", 0.4)
    memo = _reviewable_memo("Bullish", 45.0, verdict=ev(**OVERVALUED), reason=GOOD_BULL_REASON)
    _review(memo, DegradationLog())
    rec = memo.quality.rating_reconciliation
    assert memo.rating_label == "Neutral" and rec.outcome == "consistent"
    assert rec.reason == GOOD_BULL_REASON


def test_review_uses_the_live_critic_assessment(monkeypatch):
    monkeypatch.setattr(settings, "llm_rating_weight", 1.0)
    memo = _reviewable_memo("Bullish", 50.0, verdict=ev(**OVERVALUED), reason=GOOD_BULL_REASON)
    _review(memo, DegradationLog(), critic=LIVE_UNSUPPORTED)
    assert memo.rating_label == "Neutral"
    assert memo.quality.rating_reconciliation.critic_assessment == "unsupported"
    memo = _reviewable_memo("Bullish", 50.0, verdict=ev(**OVERVALUED), reason=GOOD_BULL_REASON)
    _review(memo, DegradationLog(), critic=LIVE_SUPPORTED)
    assert memo.rating_label == "Bullish"
    assert memo.quality.rating_reconciliation.outcome == "accepted"


def test_kill_switch_record_mode_leaves_the_published_rating(monkeypatch):
    monkeypatch.setattr(settings, "llm_rating_weight", 1.0)
    monkeypatch.setattr(settings, "rating_reconciliation_mode", "record")
    memo = _reviewable_memo("Bullish", 50.0, verdict=ev(**OVERVALUED))
    _review(memo, DegradationLog())
    assert memo.rating_label == "Bullish"
    assert memo.quality.rating_reconciliation.outcome == "downgraded"


def test_review_stage_fails_closed_when_the_rule_itself_crashes(monkeypatch):
    """`graph._reconcile_rating`'s own guard, not the reason checker's:
    a crash in the rule leaves a divergent rating Neutral and says so."""
    monkeypatch.setattr(settings, "llm_rating_weight", 1.0)

    def boom(**kw):
        raise RuntimeError("rule exploded")

    monkeypatch.setattr(memo_quality, "reconcile_rating", boom)
    memo = _reviewable_memo("Bullish", 50.0, verdict=ev(**OVERVALUED), reason=GOOD_BULL_REASON)
    log = DegradationLog()
    _review(memo, log, critic=LIVE_SUPPORTED)
    rec = memo.quality.rating_reconciliation
    assert memo.rating_label == "Neutral"
    assert (rec.outcome, rec.final_rating, rec.blended_rating) == ("downgraded", "Neutral", "Bullish")
    assert rec.reason_checks.get("check_failed") is True and rec.divergence is True
    assert "Rating Check" in log.degraded_agents()
    # A non-divergent rating is left alone, still with the banner entry.
    memo = _reviewable_memo("Bearish", 50.0, verdict=ev(**OVERVALUED))
    log = DegradationLog()
    _review(memo, log)
    assert memo.rating_label == "Bearish"
    assert memo.quality.rating_reconciliation.outcome == "not_applicable"
    assert "Rating Check" in log.degraded_agents()
    # Record mode: recorded as downgraded, published as blended.
    monkeypatch.setattr(settings, "rating_reconciliation_mode", "record")
    memo = _reviewable_memo("Bullish", 50.0, verdict=ev(**OVERVALUED))
    _review(memo, DegradationLog())
    assert memo.rating_label == "Bullish"
    assert memo.quality.rating_reconciliation.outcome == "downgraded"


def test_config_rejects_an_unknown_reconciliation_mode():
    from pydantic import ValidationError

    from app.config import Settings
    with pytest.raises(ValidationError):
        Settings(rating_reconciliation_mode="off")
    assert Settings().rating_reconciliation_mode == "enforce"
    assert Settings().number_check_withhold is True


# ---------------------------------------------------------------------------
# End to end (demo NVDA: comps +25% rich and the consensus DCF -51% rich)
# ---------------------------------------------------------------------------

def _scripted_pm(monkeypatch, rating: str, reason: str = "", thesis: str = ""):
    def pm(*a, **kw):
        out = {"rating_label": rating, "confidence_score": 80, "final_pm_view": "PM view.",
               "one_sentence_thesis": thesis or "NVDA is undervalued — data-center share gains."}
        if reason:
            out["valuation_divergence_reason"] = reason
        return out
    monkeypatch.setattr(graph, "_pm_synthesis", pm)


NVDA_REASON = (
    "We override the EV/EBITDA premium of 25% to peers because data-center revenue grew 94% "
    "year over year and operating margin reached 62%, which the peer multiple does not yet price."
)


def test_bullish_on_overvalued_without_reason_ships_neutral(monkeypatch):
    _scripted_pm(monkeypatch, "Very Bullish")
    memo = graph.run_stock_memo("NVDA")
    assert memo.valuation_verdict.verdict == "overvalued"
    rec = memo.quality.rating_reconciliation
    assert rec.outcome == "downgraded" and memo.rating_label == "Neutral"
    assert memo.final_verdict.startswith("PM final view: Neutral ")
    assert "valuation check" in memo.final_pm_view.split("\n")[0]
    # The thesis's "undervalued" contradicts both the (final) rating and the
    # evidence: rewritten to the evidence word.
    assert "overvalued" in memo.one_sentence_thesis
    assert memo.section_provenance["thesis"] == "rewrite"


def test_accepted_divergence_keeps_rating_thesis_and_caps_confidence(monkeypatch):
    _scripted_pm(monkeypatch, "Very Bullish", NVDA_REASON)
    memo = graph.run_stock_memo("NVDA")
    rec = memo.quality.rating_reconciliation
    assert rec.outcome == "accepted", rec.note
    assert rec.reason_checks == {"substantive": True, "names_signal": True, "quotes_value": True}
    assert memo_quality.rating_direction(memo.rating_label) == 1
    assert "not independently reviewed" in rec.note
    # An accepted divergence is never rewritten back to the evidence word.
    assert memo.one_sentence_thesis.startswith("NVDA is undervalued")
    assert memo.section_provenance["thesis"] == "pm"
    caps = {c.code: c.cap for c in memo.quality.confidence.caps}
    assert caps["divergence_unreviewed"] == 55.0
    assert memo.confidence_score <= 55.0


def test_critic_unsupported_downgrades_end_to_end(monkeypatch):
    _scripted_pm(monkeypatch, "Very Bullish", NVDA_REASON)
    seen: list[dict] = []

    def critic(draft):
        seen.append(draft)
        return CriticReview(overall_assessment="x", review_mode="live",
                            valuation_divergence_assessment="unsupported")
    monkeypatch.setattr(graph, "_checkpointed_critic", critic)
    memo = graph.run_stock_memo("NVDA")
    assert memo.rating_label == "Neutral"
    assert memo.quality.rating_reconciliation.critic_assessment == "unsupported"
    # The critic saw the PM's side of the reconciliation on the draft.
    assert seen[0]["quality"]["rating_reconciliation"]["reason"] == NVDA_REASON


# ---------------------------------------------------------------------------
# Thesis guard
# ---------------------------------------------------------------------------

def _verdict_out(memo):
    return _build_verdict(memo, comps=None, dcf=None, profile=make_profile(memo.ticker),
                          findings=make_findings(), ticker=memo.ticker)


@pytest.mark.parametrize("rating, kw, thesis, rewrite, word", [
    # Evidence overvalued, rating Neutral.
    ("Neutral", OVERVALUED, "TEST is overvalued — the multiple is rich.", False, None),
    ("Neutral", OVERVALUED, "TEST is fairly priced — steady compounder.", False, None),
    ("Neutral", OVERVALUED, "TEST is undervalued — great franchise.", True, "overvalued"),
    # Mixed evidence contradicts no word.
    ("Neutral", dict(family_pct=5.0, family_coverage=1.0, comps_premium=-0.3),
     "TEST is undervalued — great franchise.", False, None),
    # Bullish on fairly-priced evidence stating "overvalued": contradicts both.
    ("Bullish", dict(comps_premium=0.02), "TEST is overvalued — rich.", True, "fairly priced"),
])
def test_thesis_rewrite_only_when_both_disagree(rating, kw, thesis, rewrite, word):
    memo = make_memo(rating_label=rating, valuation_verdict=ev(**kw), one_sentence_thesis=thesis)
    out = _verdict_out(memo)
    assert out.thesis_rewrite_fired is rewrite
    if rewrite:
        assert word in out.one_sentence_thesis
    else:
        assert out.one_sentence_thesis.startswith(thesis.rstrip("."))


def test_accepted_divergence_is_never_rewritten():
    vv = ev(**OVERVALUED)
    memo = make_memo(
        rating_label="Bullish", valuation_verdict=vv,
        one_sentence_thesis="TEST is undervalued — cloud leverage the multiple misses.",
        quality=MemoQuality(rating_reconciliation=RatingReconciliation(
            outcome="accepted", pm_rating="Bullish", blended_rating="Bullish", final_rating="Bullish",
            valuation_verdict="overvalued", divergence=True, reason=GOOD_BULL_REASON)),
    )
    out = _verdict_out(memo)
    assert not out.thesis_rewrite_fired
    assert out.one_sentence_thesis.startswith("TEST is undervalued")


def test_accepted_divergence_thesis_contradicting_both_is_kept():
    """The case the guard exists for: an accepted Bullish-on-overvalued memo
    whose thesis says "fairly priced" contradicts both the rating's word
    (undervalued) and the evidence (overvalued), and would be rewritten if
    the accepted divergence were not exempt."""
    thesis = "TEST is fairly priced — the multiple already reflects cloud leverage."
    memo = make_memo(
        rating_label="Bullish", valuation_verdict=ev(**OVERVALUED), one_sentence_thesis=thesis,
        quality=MemoQuality(rating_reconciliation=RatingReconciliation(
            outcome="accepted", pm_rating="Bullish", blended_rating="Bullish", final_rating="Bullish",
            valuation_verdict="overvalued", divergence=True, reason=GOOD_BULL_REASON)),
    )
    out = _verdict_out(memo)
    assert out.thesis_rewrite_fired is False
    assert out.one_sentence_thesis.startswith(thesis.rstrip("."))
    # Without the accepted outcome the same thesis IS rewritten.
    plain = memo.model_copy(update={"quality": None})
    assert _verdict_out(plain).thesis_rewrite_fired is True


def test_anti_pattern_still_rewrites_with_the_evidence_word():
    anti = "TEST Corp — Technology / Software, AI hook; DCF base case +25% suggests material upside."
    memo = make_memo(rating_label="Neutral", valuation_verdict=ev(**OVERVALUED), one_sentence_thesis=anti)
    out = _verdict_out(memo)
    assert out.thesis_rewrite_fired and out.thesis_rewritten
    assert "overvalued" in out.one_sentence_thesis


def test_mixed_builder_states_no_single_word():
    text = graph._build_thesis_from_findings(
        make_profile("TEST"), {}, None, "TEST", verdict_word=None)
    assert text.startswith("TEST: valuation signals are mixed")
    assert not any(w in text.split(".")[0] for w in ("undervalued", "overvalued", "fairly priced"))


def test_mispricing_fallback_says_evidence_not_blend():
    for kw, needle in ((OVERVALUED, "the valuation evidence reads overvalued"),
                       (dict(comps_premium=0.02), "does not point either way"),
                       (dict(family_pct=5.0, family_coverage=1.0, comps_premium=-0.3),
                        "Valuation signals conflict")):
        memo = make_memo(valuation_verdict=ev(**kw), one_sentence_thesis="t")
        gap = graph._build_mispricing_fallback(memo).gap
        assert needle in gap and "blended" not in gap, gap


# ---------------------------------------------------------------------------
# PM prompt (contract C7)
# ---------------------------------------------------------------------------

def _spy_pm(monkeypatch) -> list[str]:
    from app.agents import llm as llm_mod
    from app.agents import prompts
    seen: list[str] = []

    def spy(prompt, **kw):
        if prompt.startswith(prompts.PM_SYNTHESIS_PROMPT):
            seen.append(prompt)
            return {"rating_label": "Neutral", "confidence_score": 50, "final_pm_view": "v",
                    "one_sentence_thesis": "t"}
        return None
    monkeypatch.setattr(llm_mod, "chat_json", spy)
    return seen


@pytest.mark.parametrize("pm_ctx", ["PM-CONTEXT", ""])
def test_pm_prompt_assembly_order_with_the_evidence_block(monkeypatch, pm_ctx):
    """C7, extended by S14: template + pm_ctx, then the industry digest,
    then the valuation-evidence block, then the capped JSON. The evidence
    block is volatile: the cached static prefix is still exactly the
    template."""
    import json

    from app.agents import pm_context, prompts
    from app.schemas import AgentFinding
    monkeypatch.setattr(pm_context, "build_pm_context", lambda **kw: pm_ctx)
    sector = AgentFinding(agent="Sector Analyst", headline="Sector read", summary="Neutral.")
    findings = {"sector": sector}
    monkeypatch.setattr(graph, "_pm_view", lambda f: graph.PMView(["## Industry group read — DIGEST"], f, f))
    prefixes: list[int] = []
    real_ctx = graph.llm.llm_call_context

    def ctx(**kw):
        prefixes.append(kw.get("static_prefix_chars"))
        return real_ctx(**kw)
    monkeypatch.setattr(graph.llm, "llm_call_context", ctx)
    seen = _spy_pm(monkeypatch)
    vv = ev(**OVERVALUED)
    graph._pm_synthesis({"ticker": "TEST"}, findings, None, valuation_evidence=vv)
    block = memo_quality.valuation_evidence_block(vv)
    assert block.startswith("## Valuation evidence") and "Verdict: overvalued." in block
    assert "EV/EBITDA premium to peers 44% premium" in block
    assert seen == [
        prompts.PM_SYNTHESIS_PROMPT
        + (f"\n\n{pm_ctx}" if pm_ctx else "")
        + "\n\n## Industry group read — DIGEST"
        + "\n\n" + block
        + "\n\nFindings:\n"
        + json.dumps({"sector": sector.model_dump()}, default=str)[: settings.max_agent_context_chars]
    ]
    assert prefixes == [len(prompts.PM_SYNTHESIS_PROMPT) + 2]


def test_pm_prompt_assembly_order_digest_news_evidence_refs_findings(monkeypatch):
    """C7 as extended by G1 (FIX-018): template + pm_ctx, then the industry
    digest, then the NEWS block, then the valuation evidence, then the
    source refs (S15), then the capped JSON. The news block sits outside the
    cached prefix like every other volatile block."""
    import json

    from app.agents import news_context, pm_context, prompts
    from app.agents.source_ledger import SourceLedger
    from app.schemas import AgentFinding
    monkeypatch.setattr(pm_context, "build_pm_context", lambda **kw: "PM-CONTEXT")
    news = news_context.from_alerts("TEST", [{"title": "TEST Corp wins an order", "severity": "material",
                                              "summary": "A large one.", "source": "news_service"}])
    sector = AgentFinding(agent="Sector Analyst", headline="Sector read", summary="Neutral.",
                          data={"pending_news_alerts": news.alerts()})
    monkeypatch.setattr(graph, "_pm_view", lambda f: graph.PMView(["## Industry group read — DIGEST"], f, f))
    seen = _spy_pm(monkeypatch)
    vv = ev(**OVERVALUED)
    ledger = SourceLedger()
    with ledger.activate():
        ledger.register("financials", "financials:TEST", {"revenue": 1.0})
        news_context.register(news)
        graph._pm_synthesis({"ticker": "TEST"}, {"sector": sector}, None, valuation_evidence=vv, news=news)
        refs = graph._source_refs_block()
    assert "news_alerts:TEST" in refs
    (prompt,) = seen
    parts = [
        "## Industry group read — DIGEST",
        news_context.render_block(news, "pm"),
        memo_quality.valuation_evidence_block(vv),
        refs,
    ]
    assert prompt == (
        prompts.PM_SYNTHESIS_PROMPT + "\n\nPM-CONTEXT"
        + "".join("\n\n" + p for p in parts)
        + "\n\nFindings:\n"
        + json.dumps({"sector": {**sector.model_dump(), "data": {}}}, default=str)[
            : settings.max_agent_context_chars]
    )
    offsets = [prompt.index(p) for p in parts] + [prompt.index("\n\nFindings:\n")]
    assert offsets == sorted(offsets)


def test_pm_synthesis_call_is_attributed(monkeypatch):
    from app.agents import llm as llm_mod
    kwargs_seen: list[dict] = []
    monkeypatch.setattr(llm_mod, "chat_json", lambda p, **kw: kwargs_seen.append(kw))
    graph._pm_synthesis({"ticker": "TEST"}, {}, None)
    assert kwargs_seen[-1]["action"] == "pm.synthesis" and kwargs_seen[-1]["ticker"] == "TEST"


def test_thesis_log_has_no_text(caplog):
    """REGRESSION (attribution critique #15): the thesis-rewrite line logged
    the PM's thesis verbatim. The line keeps its signal (ticker, words,
    length, sha1) and never the model's text."""
    import hashlib
    import logging

    anti = ("TEST Corp — Technology / Software, AI hook SENTINEL-PROSE; "
            "DCF base case +25% suggests material upside.")
    memo = make_memo(rating_label="Neutral", valuation_verdict=ev(**OVERVALUED), one_sentence_thesis=anti)
    with caplog.at_level(logging.INFO, logger="app.agents.graph"):
        out = _verdict_out(memo)
    assert out.thesis_rewrite_fired
    (line,) = [r.getMessage() for r in caplog.records if "thesis rewrite fired" in r.getMessage()]
    assert "SENTINEL-PROSE" not in line
    assert f"original_len={len(anti)}" in line
    assert f"original_sha1={hashlib.sha1(anti.encode('utf-8')).hexdigest()}" in line
    assert not [r for r in caplog.records if "SENTINEL-PROSE" in r.getMessage()]


def test_pm_prompt_is_byte_identical_without_evidence(monkeypatch):
    import json

    from app.agents import pm_context, prompts
    from app.schemas import AgentFinding
    monkeypatch.setattr(pm_context, "build_pm_context", lambda **kw: "PM-CONTEXT")
    sector = AgentFinding(agent="Sector Analyst", headline="Sector read", summary="Neutral.")
    expected = (prompts.PM_SYNTHESIS_PROMPT + "\n\nPM-CONTEXT" + "\n\nFindings:\n"
                + json.dumps({"sector": sector.model_dump()}, default=str)[: settings.max_agent_context_chars])
    for vv in (None, ev(), ValuationVerdict(verdict="overvalued", summary="legacy rating-derived")):
        seen = _spy_pm(monkeypatch)
        graph._pm_synthesis({"ticker": "TEST"}, {"sector": sector}, None, valuation_evidence=vv)
        assert seen == [expected]


def test_pm_prompt_asks_for_the_reason_statically():
    from app.agents import prompts
    text = " ".join(prompts.PM_SYNTHESIS_PROMPT.split())
    assert "valuation_divergence_reason" in text and "quote its value as the block prints it" in text
    critic = " ".join(prompts.CRITIC_PROMPT.split())
    assert '"RATING DIVERGENCE" block' in critic and "valuation_divergence_assessment" in critic
