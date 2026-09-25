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

from app.agents import number_check
from app.schemas import (
    ConfidenceAssessment,
    ConfidenceCap,
    MemoQuality,
    NumberCheck,
    NumberClaim,
    RatingReconciliation,
    StockMemoOut,
    WithheldItem,
)
from app.scripts import capture_memo_quality_fixture as capture

FIXTURE = capture.DEFAULT_OUT
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
    q = wire["memo"]["quality"]
    assert q is not None, RECAPTURE
    _same_keys(q, MemoQuality, "quality")
    nc = q["number_check"]
    _same_keys(nc, NumberCheck, "quality.number_check")
    assert set(nc["counts"]) == COUNT_KEYS, (sorted(set(nc["counts"]) ^ COUNT_KEYS), RECAPTURE)
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


def _essentials(memo: dict[str, Any]) -> dict[str, Any]:
    """What the UI tests rely on, minus incidental run-to-run values."""
    q = memo["quality"]
    nc = q["number_check"]
    counts = {k: v for k, v in nc["counts"].items() if k not in ("registry_facts", "registry_sources")}
    return {
        "rating_label": memo["rating_label"],
        "confidence_score": memo["confidence_score"],
        "counts": counts,
        "claims": [(c["field"], c["start"], c["end"], c["raw"], c["status"]) for c in nc["claims"]],
        "withheld": [(w["field"], w["index"], w["text"]) for w in nc["withheld"]],
        "assumptions": nc["assumptions"],
        "sources_cited": nc["sources_cited"],
        "primary_kinds_cited": nc["primary_kinds_cited"],
        "reconciliation": {k: q["rating_reconciliation"][k] for k in (
            "outcome", "pm_rating", "blended_rating", "final_rating", "valuation_verdict", "note")},
        "caps": [(c["code"], c["cap"]) for c in q["confidence"]["caps"]],
        "binding": q["confidence"]["binding"],
        "availability": {k: (v["status"], v["reason"]) for k, v in memo["section_availability"].items()},
        "final_pm_view": memo["final_pm_view"],
    }


def test_fixture_is_what_the_pipeline_produces(wire):
    """Re-run the capture (demo data, scripted answers, no keys) and compare
    what the UI relies on. A change to the number check, the reconciliation,
    the caps or the presenter that alters the served record fails here with
    the instruction to re-capture, instead of leaving the UI tested against
    a record the pipeline no longer writes."""
    fresh = capture.capture()
    assert _essentials(fresh["memo"]) == _essentials(wire["memo"]), RECAPTURE
