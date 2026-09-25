"""W2b (S16) — the Research-checks UI fixture is the pipeline's own output.

`frontend/src/test/fixtures/memo_quality.wire.json` is what every Vitest
assertion about the Research checks panel, the inline figure marks, the
rating-check note and the capped-confidence line is made against. It was
captured by `app/scripts/capture_memo_quality_fixture.py`: a real
`graph.run_stock_memo("NVDA")` on demo data with the PM and valuation
answers scripted, presented and serialized as the memo route serves it.

A capture is only worth anything while it matches what the producer
writes. `StockMemoOut.model_validate` ignores unknown keys and fills
defaults for absent ones, so a renamed `quality` field would validate
happily while the UI kept reading a field the API no longer sends. So this
test compares KEY SETS both ways, at every level of `quality` the UI reads,
and re-runs the capture to check the committed record still is what the
pipeline produces (`test_fixture_is_what_the_pipeline_produces`).

It deliberately does not pin incidental values (registry sizes, run ids,
timestamps); the scenario assertions pin only what the UI tests rely on.
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from app.agents import memo_quality, number_check
from app.schemas import (
    CatalystItem,
    ConfidenceAssessment,
    ConfidenceCap,
    CriticReview,
    MemoQuality,
    NumberCheck,
    NumberClaim,
    RatingReconciliation,
    RiskItem,
    StockMemoOut,
    WithheldItem,
)
from app.scripts import capture_memo_quality_fixture as capture

FIXTURE = capture.DEFAULT_OUT
PM_TEMPLATE_FIXTURE = capture.PM_TEMPLATE_OUT
RECAPTURE = (
    "Re-capture it, from `backend/` and against a throwaway database:\n"
    '  ENABLE_LIVE_DATA=false USE_DEMO_DATA=true OPENAI_API_KEY="" ANTHROPIC_API_KEY="" '
    'GEMINI_API_KEY="" DATABASE_URL="sqlite:////tmp/memo-quality-fixture.db" '
    "python -m app.scripts.capture_memo_quality_fixture\n"
    "then update `frontend/src/types/index.ts` and the tests that read the changed field."
)

# The tallies `number_check.summarize` writes, plus `withheld`, which the
# quality stage adds after withholding (`memo_context.QualityOutcome`).
COUNT_KEYS = {
    *number_check.FACT_STATUSES, "assumption", "threshold", "claims_total", "flagged_distinct",
    "fields_checked", "registry_facts", "registry_sources", "withheld",
}
# Keys of a declared assumption (`memo_quality.parse_forecast_assumptions`
# plus the status the summary stamps).
ASSUMPTION_KEYS = {"value", "unit", "basis_ref", "horizon", "status"}


@pytest.fixture(scope="module")
def wire() -> dict[str, Any]:
    assert FIXTURE.exists(), f"{FIXTURE} is missing. {RECAPTURE}"
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def pm_template_wire() -> dict[str, Any]:
    assert PM_TEMPLATE_FIXTURE.exists(), f"{PM_TEMPLATE_FIXTURE} is missing. {RECAPTURE}"
    return json.loads(PM_TEMPLATE_FIXTURE.read_text())


def _keys(model: type[BaseModel]) -> set[str]:
    return set(model.model_json_schema(by_alias=True)["properties"])


def _same_keys(obj: dict[str, Any], model: type[BaseModel], where: str) -> None:
    got, want = set(obj), _keys(model)
    assert got == want, (
        f"{where}: the fixture carries {sorted(got - want)} the model does not, and lacks "
        f"{sorted(want - got)}. {RECAPTURE}"
    )


def test_fixture_revalidates_and_round_trips(wire):
    """The stored body is a StockMemoOut as the route serializes it: it
    validates, and re-serializing the validated model gives the same JSON
    (no unknown key dropped, no absent key defaulted)."""
    memo = StockMemoOut.model_validate(wire["memo"])
    assert json.loads(memo.model_dump_json()) == wire["memo"], RECAPTURE
    _same_keys(wire["memo"], StockMemoOut, "memo")


def test_quality_key_sets_match_the_models(wire):
    _assert_quality_shape(wire["memo"]["quality"])


def _assert_quality_shape(q: dict[str, Any] | None, *, require_claims: bool = True) -> None:
    assert q is not None, RECAPTURE
    _same_keys(q, MemoQuality, "quality")
    nc = q["number_check"]
    _same_keys(nc, NumberCheck, "quality.number_check")
    assert set(nc["counts"]) == COUNT_KEYS, (sorted(set(nc["counts"]) ^ COUNT_KEYS), RECAPTURE)
    if require_claims:
        assert nc["claims"], "the fixture must carry stored claims for the UI to mark"
    for c in nc["claims"]:
        _same_keys(c, NumberClaim, "quality.number_check.claims[]")
    assert nc["withheld"], "the fixture must carry a withheld item for the disclosure"
    for w in nc["withheld"]:
        _same_keys(w, WithheldItem, "quality.number_check.withheld[]")
        for c in w["claims"]:
            _same_keys(c, NumberClaim, "quality.number_check.withheld[].claims[]")
    for a in nc["assumptions"]:
        assert set(a) == ASSUMPTION_KEYS, (sorted(set(a) ^ ASSUMPTION_KEYS), RECAPTURE)
    _same_keys(q["rating_reconciliation"], RatingReconciliation, "quality.rating_reconciliation")
    _same_keys(q["confidence"], ConfidenceAssessment, "quality.confidence")
    for cap in q["confidence"]["caps"]:
        _same_keys(cap, ConfidenceCap, "quality.confidence.caps[]")


def test_fixture_carries_every_state_the_ui_tests_read(wire):
    memo = StockMemoOut.model_validate(wire["memo"])
    expect = wire["meta"]["expect"]
    q = memo.quality
    assert q is not None and q.number_check is not None and q.number_check.checked
    nc = q.number_check
    by_raw = {(c.field, c.raw): c.status for c in nc.claims}
    assert by_raw[("final_pm_view", expect["fabricated_pm_figure"])] == "untraceable"
    assert by_raw[("mispricing_thesis.our_view", expect["fabricated_view_figure"])] == "untraceable"
    assert by_raw[("final_pm_view", expect["declared_assumption"])] == "assumption"
    assert [w.text for w in nc.withheld] == [expect["withheld_point"]]
    # The presented body keeps the renderer's contract: every stored claim
    # indexes its field's text, so CheckedText marks it rather than skip it.
    for c in nc.claims:
        text = number_check.resolve_field(memo, c.field)
        assert isinstance(text, str) and text[c.start:c.end] == c.raw, c
    for w in nc.withheld:
        for c in w.claims:
            assert w.text[c.start:c.end] == c.raw, (w, c)
    # The withheld point is nowhere but the quality record.
    dump = wire["memo"].copy()
    dump.pop("quality")
    assert expect["withheld_point"] not in json.dumps(dump)
    rec = q.rating_reconciliation
    assert rec is not None and rec.outcome == "downgraded" and rec.final_rating == memo.rating_label
    conf = q.confidence
    assert conf is not None and conf.final < conf.raw and conf.binding
    assert memo.confidence_score == conf.final
    # Presented: the map is there, and confidence is shown (the hidden case
    # is derived from this body by the frontend fixture loader).
    assert memo.section_availability
    assert memo.section_availability["confidence_score"].status == "available"


def test_fixture_meta_matches_the_script(wire):
    """The scripted answers recorded in the fixture are the script's."""
    assert wire["meta"]["scripted"]["pm_synthesis"] == capture.PM_ANSWER, RECAPTURE
    assert wire["meta"]["scripted"]["valuation_analyst"] == capture.VALUATION_ANSWER, RECAPTURE
    assert wire["meta"]["ticker"] == capture.TICKER


# The variant name -> (rating shown, outcome, final_rating) each rating-check
# variant must reproduce. The UI tests read these states; if the producer
# stops writing one, the fixture no longer shows what the tests claim.
VARIANT_STATES = {
    "record_mode": ("Bullish", "downgraded", "Bullish"),
    "failed_reason": ("Neutral", "downgraded", "Neutral"),
    "accepted_unreviewed": ("Bullish", "accepted", "Bullish"),
    "accepted_supported": ("Bullish", "accepted", "Bullish"),
    "critic_unsupported": ("Neutral", "downgraded", "Neutral"),
    "patch_kept_record": ("Bearish", "downgraded", "Neutral"),
    "patch_guard": ("Neutral", "downgraded", "Neutral"),
    "patch_guard_record_mode": ("Very Bullish", "downgraded", "Very Bullish"),
    "patch_after_accepted": ("Very Bullish", "accepted", "Bullish"),
}


def test_rating_check_variants_are_real_records(wire):
    """`variants` holds the rating-check states one scripted run cannot
    reach, each written by `reconcile_rating` / `enforce_after_patch` on the
    captured memo's verdict. They overlay the captured body in the frontend
    loader, so each must be a full record with the models' key sets."""
    variants = wire["variants"]
    assert set(variants) == set(VARIANT_STATES), RECAPTURE
    for name, v in variants.items():
        assert set(v) == {"rating_label", "confidence_score", "rating_reconciliation", "confidence"}
        _same_keys(v["rating_reconciliation"], RatingReconciliation, f"variants.{name}.rating_reconciliation")
        RatingReconciliation.model_validate(v["rating_reconciliation"])
        _same_keys(v["confidence"], ConfidenceAssessment, f"variants.{name}.confidence")
        ConfidenceAssessment.model_validate(v["confidence"])
        rec = v["rating_reconciliation"]
        assert (v["rating_label"], rec["outcome"], rec["final_rating"]) == VARIANT_STATES[name], name
    # The note shapes the panel must word without printing machine terms.
    assert "failed:" in variants["failed_reason"]["rating_reconciliation"]["note"]
    assert "rating_reconciliation_mode=record" in variants["record_mode"]["rating_reconciliation"]["note"]
    assert variants["failed_reason"]["rating_reconciliation"]["reason_checks"] == {
        k: False for k in memo_quality.REASON_CHECKS}
    # A patch-guard record carries no reason checks (a patch states no
    # reason); the frontend tells it from a full-run record that way.
    assert variants["patch_guard"]["rating_reconciliation"]["reason_checks"] == {}
    assert variants["failed_reason"]["rating_reconciliation"]["reason_checks"]


def test_pm_template_fixture_is_the_template_confidence_state(pm_template_wire):
    """With the PM unscripted the template PM ships. The presenter hides its
    view but shows the confidence (the quality stage stamps it `earned`),
    so the PM's raw confidence in `quality.confidence` is a template's
    number that no guard on `confidence_score` availability catches. The
    panel keys on the `pm_template` cap instead; this pins that the
    producer still writes exactly this state."""
    memo = StockMemoOut.model_validate(pm_template_wire["memo"])
    assert json.loads(memo.model_dump_json()) == pm_template_wire["memo"], RECAPTURE
    _assert_quality_shape(pm_template_wire["memo"]["quality"], require_claims=False)
    q = memo.quality
    assert q is not None and q.confidence is not None
    assert q.confidence.binding == "pm_template" and q.confidence.final < q.confidence.raw
    sa = memo.section_availability or {}
    assert sa["final_pm_view"].status == "unavailable"
    assert sa["confidence_score"].status == "available"


# The item-8 reviewer fields as a review carries them when nothing wrote
# them: with DEBATE_MODE and REVIEWER_MODE both off (the capture's state,
# and production's until waves H and I) the debate is null and every
# reviewer field holds its "not produced" value.
ITEM8_UNWRITTEN = {
    "reviewer_model": "", "verdict": "", "issues": [], "rating_too_high": None,
    "rating_too_low": None, "review_status": "", "revision": None, "debate_review": None,
}


@pytest.mark.parametrize("which", ["wire", "pm_template_wire"])
def test_fixtures_carry_the_d2_contract_with_both_modes_off(which, request):
    """D2: both captures are the pipeline with the debate and the item-8
    reviewer off. The UI tests read `debate` and the review fields from
    these bodies, so the keys must be on the wire (as null / empty, not
    absent), and the review's key set must be the model's both ways."""
    memo = request.getfixturevalue(which)["memo"]
    assert "debate" in memo and memo["debate"] is None, RECAPTURE
    review = memo["risk_committee_challenge"]
    _same_keys(review, CriticReview, "memo.risk_committee_challenge")
    assert {k: review[k] for k in ITEM8_UNWRITTEN} == ITEM8_UNWRITTEN, RECAPTURE


def test_field_paths_the_renderers_build(wire):
    """The renderers look claims up by path strings they build themselves
    (`claimsFor(memo, "key_risks[0].title")` and so on). Pin the formats
    the frontend tests use against `number_check.iter_fields`, so a change
    of path format fails here instead of silently dropping every mark."""
    memo = StockMemoOut.model_validate(wire["memo"]).model_copy(deep=True)
    memo.bull_case.key_points = ["a"]
    memo.bear_case.key_points = ["a"]
    memo.catalysts = [CatalystItem(title="a", detail="b")]
    memo.key_risks = [RiskItem(title="a", detail="b")]
    memo.thesis_breakers = [RiskItem(title="a", detail="b")]
    memo.valuation_agent_view.key_points = ["a"]
    paths = {spec.path for spec in number_check.iter_fields(memo)}
    want = {
        "final_pm_view", "one_sentence_thesis",
        "mispricing_thesis.consensus_view", "mispricing_thesis.our_view", "mispricing_thesis.gap",
        "bull_case.headline", "bull_case.key_points[0]", "bear_case.headline", "bear_case.key_points[0]",
        "valuation_agent_view.headline", "valuation_agent_view.summary",
        "valuation_agent_view.key_points[0]",
        "catalysts[0].title", "catalysts[0].detail",
        "key_risks[0].title", "key_risks[0].detail",
        "thesis_breakers[0].title", "thesis_breakers[0].detail",
    }
    assert want <= paths, sorted(want - paths)


def _scenario(memo: dict[str, Any]) -> dict[str, Any]:
    """The scripted scenario the UI tests read: the stored (flagged,
    assumption) claims with their offsets, the withheld point, the rating
    check's outcome and the caps.

    Deliberately NOT compared: tallies, cited sources, the section map and
    the reconciliation note's numbers. Those depend on what else the
    database holds (a DCF, scorecard or macro snapshot another test wrote
    moves the traced count, the cited refs and the DCF figure in the note),
    so in a full-suite run they differ from a fresh-database capture for
    reasons that have nothing to do with the contract."""
    q = memo["quality"]
    nc = q["number_check"]
    return {
        "rating_label": memo["rating_label"],
        "confidence_score": memo["confidence_score"],
        "claims": [(c["field"], c["start"], c["end"], c["raw"], c["status"]) for c in nc["claims"]],
        "withheld": [(w["field"], w["index"], w["text"]) for w in nc["withheld"]],
        "assumptions": nc["assumptions"],
        "reconciliation": {k: q["rating_reconciliation"][k] for k in (
            "outcome", "pm_rating", "blended_rating", "final_rating", "valuation_verdict")},
        "caps": [(c["code"], c["cap"]) for c in q["confidence"]["caps"]],
        "binding": q["confidence"]["binding"],
    }


def _variant_scenario(variants: dict[str, Any]) -> dict[str, Any]:
    """Each variant's outcome fields (the notes' numbers are DB-dependent)."""
    keys = ("outcome", "pm_rating", "blended_rating", "final_rating", "valuation_verdict",
            "divergence", "reason", "reason_checks", "critic_assessment")
    return {
        name: (v["rating_label"], {k: v["rating_reconciliation"][k] for k in keys},
               [c["code"] for c in v["confidence"]["caps"]])
        for name, v in variants.items()
    }


def test_fixture_is_what_the_pipeline_produces(wire):
    """Re-run the capture (demo data, scripted answers, no keys): the
    pipeline must still write the same `quality` SHAPE, and the scripted
    scenario must still come out the same. A change to the number check,
    the reconciliation or the caps that alters what the UI reads fails here
    with the instruction to re-capture, instead of leaving the UI tested
    against a record the pipeline no longer writes."""
    fresh = capture.capture()
    _assert_quality_shape(fresh["memo"]["quality"])
    assert set(fresh["memo"]) == set(wire["memo"]), RECAPTURE
    assert (set(fresh["memo"]["risk_committee_challenge"])
            == set(wire["memo"]["risk_committee_challenge"])), RECAPTURE
    assert fresh["memo"]["debate"] == wire["memo"]["debate"], RECAPTURE
    assert _scenario(fresh["memo"]) == _scenario(wire["memo"]), RECAPTURE
    assert _variant_scenario(fresh["variants"]) == _variant_scenario(wire["variants"]), RECAPTURE


def test_pm_template_fixture_is_what_the_pipeline_produces(pm_template_wire):
    fresh = capture.capture_pm_template()
    _assert_quality_shape(fresh["memo"]["quality"], require_claims=False)
    assert set(fresh["memo"]) == set(pm_template_wire["memo"]), RECAPTURE
    got, want = fresh["memo"]["quality"]["confidence"], pm_template_wire["memo"]["quality"]["confidence"]
    assert (got["binding"], [c["code"] for c in got["caps"]]) == (
        want["binding"], [c["code"] for c in want["caps"]]), RECAPTURE
    for key in ("final_pm_view", "confidence_score"):
        assert (fresh["memo"]["section_availability"][key]["status"]
                == pm_template_wire["memo"]["section_availability"][key]["status"]), (key, RECAPTURE)
