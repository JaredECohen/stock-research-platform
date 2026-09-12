"""FEAT-003 slice 6 — the UI fixture is the API's own output, and stays so.

`frontend/src/test/fixtures/industry.wire.json` was captured by running
the real weekly path against the demo universe and reading every public
route through `TestClient`. Every Vitest assertion about the Industry
Analysis page is made against that capture, which is only worth anything
while it still matches what the API serves.

So this test re-validates each stored body through the pydantic model
that produced it, and — the part that catches the drift a validation
pass would not — compares the stored TOP-LEVEL KEY SET with the model's.
`model_validate` ignores unknown keys and fills defaults for absent ones,
so a renamed or added field passes validation happily while the UI keeps
rendering a field the API no longer sends. The key-set comparison is what
fails in that case, with the instruction to re-capture.

What this deliberately does NOT do:

* assert on values. The fixture is a snapshot of one demo universe on one
  week; pinning `4530` or a return here would make an unrelated data
  change fail a backend test.
* require the taxonomy to be imported, or touch the database. It reads a
  JSON file and some pydantic models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from app.schemas.industry import (
    IndustryChangesOut,
    IndustryCompaniesOut,
    IndustryHistoryOut,
    IndustryReportOut,
    TaxonomyOut,
)

FIXTURE = (
    Path(__file__).resolve().parents[3] / "frontend" / "src" / "test" / "fixtures" / "industry.wire.json"
)

# Which stored response each model answers for.
BODIES: tuple[tuple[str, type[BaseModel]], ...] = (
    ("taxonomy", TaxonomyOut),
    ("report", IndustryReportOut),
    ("report_warming_up", IndustryReportOut),
    ("report_universe_short", IndustryReportOut),
    ("companies", IndustryCompaniesOut),
    ("companies_warming_up", IndustryCompaniesOut),
    ("history", IndustryHistoryOut),
    ("changes", IndustryChangesOut),
)

GENERATOR = Path(__file__).resolve().parents[1] / "scripts" / "capture_industry_ui_fixture.py"
RECAPTURE = (
    "Re-capture it, from `backend/` and against a throwaway database:\n"
    '  ENABLE_LIVE_DATA=false USE_DEMO_DATA=true OPENAI_API_KEY="" ANTHROPIC_API_KEY="" '
    'GEMINI_API_KEY="" DATABASE_URL="sqlite:////tmp/industry-fixture.db" '
    "python -m app.scripts.capture_industry_ui_fixture\n"
    "then update `frontend/src/types/industries.ts` and the tests that read the changed field."
)


@pytest.fixture(scope="module")
def wire() -> dict[str, Any]:
    if not FIXTURE.exists():  # pragma: no cover - the file is checked in
        pytest.skip(f"UI fixture not present at {FIXTURE}")
    return json.loads(FIXTURE.read_text())


def _serialised_keys(model: type[BaseModel]) -> set[str]:
    """The keys the model actually puts on the wire, aliases included
    (`IndustryChangesOut.from_` serialises as `from`)."""
    return set(model.model_json_schema(by_alias=True)["properties"])


@pytest.mark.parametrize(("name", "model"), BODIES, ids=[n for n, _ in BODIES])
def test_stored_response_still_validates(wire, name, model):
    """The capture is a body this model could produce today."""
    model.model_validate(wire[name])


@pytest.mark.parametrize(("name", "model"), BODIES, ids=[n for n, _ in BODIES])
def test_stored_response_carries_exactly_the_models_fields(wire, name, model):
    """No field added, renamed or removed since the capture.

    A one-sided check would not do: an ADDED field means the UI has never
    seen it (and its TypeScript mirror lacks it), a REMOVED one means the
    UI is rendering something the API stopped sending.
    """
    stored = set(wire[name])
    expected = _serialised_keys(model)
    assert stored == expected, (
        f"`{name}` in the UI fixture no longer matches {model.__name__}: "
        f"missing from the fixture {sorted(expected - stored)}, "
        f"not on the model {sorted(stored - expected)}. {RECAPTURE}"
    )


def test_nested_row_shapes_match_their_models(wire):
    """The rows the UI iterates — a group in the picker, a company in the
    table, an edition in the history — are shapes too, and the field-set
    check above only sees the top level."""
    from app.schemas.industry import (
        IndustryCompanyRowOut,
        IndustryGroupNodeOut,
        IndustryReportHistoryItemOut,
        SectorNodeOut,
    )

    sectors = wire["taxonomy"]["sectors"]
    assert sectors, "the captured taxonomy has no sectors; re-capture with an imported taxonomy"
    assert set(sectors[0]) == _serialised_keys(SectorNodeOut)

    groups = [g for s in sectors for g in s["industry_groups"]]
    assert groups, "the captured taxonomy has no industry groups"
    assert set(groups[0]) == _serialised_keys(IndustryGroupNodeOut)

    rows = wire["companies"]["items"]
    assert rows, "the captured companies response has no rows; the UI table fixture would be empty"
    assert set(rows[0]) == _serialised_keys(IndustryCompanyRowOut)

    editions = wire["history"]["items"]
    assert editions, "the captured history has no editions"
    assert set(editions[0]) == _serialised_keys(IndustryReportHistoryItemOut)


def test_the_capture_carries_all_three_sample_floor_states(wire):
    """The UI has to word "the warm-up has not reached this group yet" and
    "this universe cannot cover this group" differently, so the fixture it
    is tested against must contain both — captured, not hand-written."""
    from app.services import industry_analytics as ia

    states = {
        name: wire[name]["coverage"]["sample_floor"]
        for name in ("report", "report_warming_up", "report_universe_short")
    }
    assert states["report"]["state"] == ia.FLOOR_MET
    assert states["report_warming_up"]["state"] == ia.FLOOR_PRICES_NOT_WARMED
    assert states["report_universe_short"]["state"] == ia.FLOOR_UNIVERSE_TOO_SMALL
    # Different words, not just a different enum — the enum never reaches
    # the page.
    assert len({s["explanation"] for s in states.values()}) == 3
    assert states["report_warming_up"]["structural"] is False
    assert states["report_universe_short"]["structural"] is True
    assert states["report_universe_short"]["clears_with_warm_up"] is False

    # And the taxonomy's own structural count agrees about the thin group.
    groups = {g["code"]: g for s in wire["taxonomy"]["sectors"] for g in s["industry_groups"]}
    short_code = wire["meta"]["short_code"]
    assert groups[short_code]["universe_coverage"]["coverable"] is False
    assert short_code in wire["taxonomy"]["universe_coverage"]["not_coverable_codes"]


def test_the_capture_carries_a_real_disagreement_between_two_reads(wire):
    """The page prints both the edition's priced count and the live
    membership read when they differ, and the fixture has to make them
    differ WITHOUT either body being edited into it.

    `/companies` prices from the LATEST statistics row, so the capture
    reads it twice: once before any price existed, once after. Each body
    is internally consistent — the early one counts 0 priced and marks
    every row unpriced with a reason — and the disagreement lives between
    the two reads, which is where the real one lives.
    """
    early = wire["companies_warming_up"]
    edition = wire["report"]["payload"]["sections"]["companies"]["facts"]

    assert wire["report"]["is_latest_good"] is True, "the page compares only on the published edition"
    assert early["n_priced"] != edition["n_priced"], (
        "the two captured reads agree, so the fixture no longer exercises the disagreement notice — "
        "re-capture, and do not edit one of them into disagreeing"
    )
    # Each body agrees with ITSELF: the count matches its own rows, and
    # every unpriced row says why.
    assert early["n_priced"] == sum(1 for item in early["items"] if item["priced"])
    assert early["count"] == wire["companies"]["count"]
    for item in early["items"]:
        if not item["priced"]:
            assert item["unpriced_reason"], item["ticker"]
    assert wire["companies"]["n_priced"] == sum(1 for item in wire["companies"]["items"] if item["priced"])


def test_the_recapture_instruction_names_something_that_exists():
    """The failure messages above tell a reader to re-capture. That is only
    useful while the thing they name is in the repo — the first version of
    this fixture pointed at a generator in one developer's scratch
    directory, which nobody else could run."""
    assert GENERATOR.is_file(), f"{GENERATOR} is missing, but every failure here tells the reader to run it"


def test_the_fixture_says_what_generated_it(wire):
    """`meta.generated_by` is the fixture's own account of where it came
    from, and a capture whose provenance line names a file that does not
    exist is worse than none."""
    generated_by = wire["meta"]["generated_by"]
    assert "app.scripts.capture_industry_ui_fixture" in generated_by, generated_by


WEEKLY_CLOSES_PATH = "payload.sections.companies.facts.per_ticker[].weekly_closes"


def _per_ticker_rows(body: object) -> list[dict]:
    """The per-ticker rows of a stored body, or ``[]`` for one that has
    none (the 404 detail, `/companies`, `/history`). Tolerant on purpose:
    this is the walk that has to visit bodies nothing declared."""
    if not isinstance(body, dict):
        return []
    facts = (((body.get("payload") or {}).get("sections") or {}).get("companies") or {}).get("facts") or {}
    rows = facts.get("per_ticker")
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def test_the_fixture_declares_what_was_edited_after_capture(wire):
    """The honesty contract applies to the fixture itself: the only edit
    made after capture (emptying per-ticker weekly closes) is declared and
    COUNTED, and every row that lost points says how many it lost."""
    meta = wire["meta"]
    assert meta["trimmed"], "meta.trimmed is empty — either nothing was trimmed (drop the key) or the count is missing"
    assert meta["trimmed_note"]
    assert meta["omitted_responses"] and meta["omitted_reason"]

    # `meta.trimmed` is keyed `<response>.<path within it>`: the capture
    # stores several report bodies and each carries its own count.
    declared: dict[str, int] = {}
    for key, n in meta["trimmed"].items():
        body, path = key.split(".", 1)
        assert path == WEEKLY_CLOSES_PATH, key
        assert body in wire, f"meta.trimmed names `{body}`, which is not in the fixture"
        declared[key] = int(n)

    # EVERY body, not only the declared ones. Walking `meta.trimmed`
    # alone checks the fixture's own account against itself and never
    # looks at a body missing from it — so a hand-edit to a capture that
    # had no closes to drop (`report_warming_up` has none) would ship
    # undeclared, which is the edit this test is named for.
    for name, body in wire.items():
        if name == "meta":
            continue
        rows = _per_ticker_rows(body)
        dropped = sum(int(row.get("weekly_closes_dropped", 0) or 0) for row in rows)
        key = f"{name}.{WEEKLY_CLOSES_PATH}"
        assert dropped == declared.get(key, 0), (
            f"`{name}` reports {dropped} dropped weekly closes and meta.trimmed declares "
            f"{declared.get(key, 0)} — a truncated artifact must count what it dropped, and an edit "
            "the fixture does not declare is a hand-edit that has to be re-captured instead"
        )
        for row in rows:
            if row.get("weekly_closes_dropped"):
                assert row["weekly_closes"] == [], "a row that reports dropped closes still carries some"


def test_the_captured_edition_is_the_shape_the_ui_renders(wire):
    """The structural promises the page is built on, checked against the
    capture rather than against a literal: thirteen sections in the
    writer's frozen order, each `{facts, interpretation}`, with the
    facts-only sections carrying a null interpretation by design."""
    from app.agents.industry_report_validator import INTERPRETED_SECTIONS, SECTION_ORDER

    payload = wire["report"]["payload"]
    assert payload["section_order"] == list(SECTION_ORDER)
    assert set(payload["sections"]) == set(SECTION_ORDER)
    for name, section in payload["sections"].items():
        assert set(section) == {"facts", "interpretation"}, name
        assert isinstance(section["facts"], dict), name
        if name in INTERPRETED_SECTIONS:
            assert section["interpretation"] and section["interpretation"].get("text"), name
        else:
            assert section["interpretation"] is None, name


def test_the_capture_covers_the_states_the_ui_has_to_render(wire):
    """A fixture in which every name is priced and every source is the
    same would let the table's coverage and provenance columns rot
    untested. These are properties of the capture, not of the product."""
    companies = wire["companies"]
    assert companies["n_priced"] < companies["count"], (
        "the captured group prices every member; the UI's unpriced-reason path would be untested"
    )
    sources = {row["classification"]["source"] for row in companies["items"]}
    assert len(sources) > 1, f"the captured group has one classification source ({sources}); re-capture a mixed group"
    assert any(row["sub_industry_name"] for row in companies["items"]), "no row carries a sub-industry name"
    assert any(row["sub_industry_name"] is None for row in companies["items"]), "every row carries a sub-industry name"

    deltas = wire["changes"]["facts_delta"].values()
    assert any(d["delta"] is None and d["reason"] for d in deltas), (
        "no fact in the captured diff is missing on one side; the 'never differenced to zero' path would be untested"
    )
