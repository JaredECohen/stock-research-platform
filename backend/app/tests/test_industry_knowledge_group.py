"""FEAT-003 slice 2 — the Industry Group mandate aggregation.

The encyclopedia is written per 6-digit industry in prose; the group
mandate is a mechanical, auditable roll-up. These tests pin the rules on a
synthetic two-industry group (dedupe, frequency rank, per-item provenance,
the `(\\d)/5` priority parse, the bounded sub-industry block) and then run
the same aggregation over every real group so no code in the knowledge
base produces an empty mandate. Counts come from the data — nothing here
knows how many groups exist.
"""
from __future__ import annotations

import pytest

from app.services import industry_group_knowledge as igk
from app.services import industry_knowledge as ik


def _industry(code: str, name: str, **fields: str) -> dict:
    base = {
        "economic_engine": "Engine A; engine B.",
        "core_kpis": "Bookings; backlog; utilization.",
        "leading_indicators": "Budgets; wage trends.",
        "typical_moats": "Scale.",
        "capital_cycle_supply_response": "Labor scales quickly.",
        "valuation_lenses": "P/E; EV/EBITDA.",
        "accounting_data_traps": "Capitalized costs.",
        "common_failure_modes": "Overbuilding.",
        "ideal_compounder_setup": "Sticky services.",
        "ideal_inflection_setup": "Utilization crosses threshold.",
        "research_priority": "3/5.",
        "cadence": "Quarterly.",
        "preferred_archetype": "Compounder.",
        "highest_evi_question": f"What breaks {name}?",
    }
    base.update(fields)
    return {"code": code, "name": name, "fields": base, "sub_industries": []}


def _group(industries: list[dict]) -> dict:
    return {"code": "9910", "name": "Synthetic Group", "sector_code": "99",
            "sector_name": "Synthetic Sector", "industries": industries}


@pytest.fixture()
def synthetic(monkeypatch):
    a = _industry("991010", "Alpha",
                  core_kpis="Bookings; ARR; utilization.", research_priority="3/5.")
    b = _industry("991020", "Beta",
                  core_kpis="bookings; GRR/NRR; ARR.", research_priority="5/5.",
                  preferred_archetype="Inflection.")
    a["sub_industries"] = [{
        "code": "99101010", "name": "Alpha One",
        "fields": {"economics": "Alpha One earns day rates. Second sentence is dropped.",
                   "advantage_test": "Scarce rigs protect returns! Ignored tail."},
        "source_ids": ["S01", "S29"],
    }]
    monkeypatch.setattr(ik, "get_industry_group", lambda code: _group([a, b]) if code == "9910" else None)
    igk.clear_cache()
    yield
    igk.clear_cache()


# --- parsing primitives -------------------------------------------------------


def test_split_items_strips_periods_and_blanks():
    assert igk.split_items("Rig count; frac spreads; utilization.") == ["Rig count", "frac spreads", "utilization"]
    assert igk.split_items(" ;; ") == []
    assert igk.split_items(None) == []


def test_parse_priority_reads_n_over_5_only():
    assert igk.parse_priority("5/5.") == 5
    assert igk.parse_priority("Priority 3 / 5 (rising)") == 3
    assert igk.parse_priority("high") is None
    assert igk.parse_priority(None) is None


def test_first_sentence_is_bounded():
    assert igk.first_sentence("One. Two.") == "One."
    assert igk.first_sentence("No terminator here") == "No terminator here"
    assert igk.first_sentence("x" * 300, max_chars=10).endswith("…")


# --- aggregation rules on the synthetic group ---------------------------------


def test_dedupes_case_insensitively_and_ranks_by_frequency_with_provenance(synthetic):
    m = igk.group_mandate("9910", version_key="test")
    kpis = m.lists["core_kpis"]
    assert kpis[0].text == "Bookings" and kpis[0].count == 2
    assert kpis[0].industry_codes == ("991010", "991020")
    assert kpis[1].text == "ARR" and kpis[1].count == 2
    singles = [k for k in kpis if k.count == 1]
    assert {k.text for k in singles} == {"utilization", "GRR/NRR"}
    # Every single-industry item names exactly the industry that supplied it.
    assert all(len(k.industry_codes) == 1 for k in singles)
    assert next(k for k in singles if k.text == "GRR/NRR").industry_codes == ("991020",)


def test_ties_interleave_across_industries_instead_of_exhausting_the_first():
    industries = [
        _industry("000010", "First", core_kpis="a1; a2; a3; a4"),
        _industry("000020", "Second", core_kpis="b1; b2"),
    ]
    ranked = [r.text for r in igk.aggregate_field(industries, "core_kpis")]
    assert ranked == ["a1", "b1", "a2", "b2", "a3", "a4"]


def test_research_priority_is_the_max_with_its_source_named(synthetic):
    m = igk.group_mandate("9910", version_key="test")
    assert m.research_priority == 5
    assert m.research_priority_source == "991020 Beta"


def test_highest_evi_questions_are_kept_per_industry_never_merged(synthetic):
    m = igk.group_mandate("9910", version_key="test")
    assert m.highest_evi_questions == (("991010", "What breaks Alpha?"), ("991020", "What breaks Beta?"))


def test_sub_industries_carry_one_line_economics_advantage_test_and_provenance(synthetic):
    m = igk.group_mandate("9910", version_key="test")
    assert len(m.sub_industries) == 1
    sub = m.sub_industries[0]
    assert sub.code == "99101010" and sub.industry_code == "991010"
    assert sub.economics == "Alpha One earns day rates."
    assert sub.advantage_test == "Scarce rigs protect returns!"
    assert sub.source_ids == ("S01", "S29")
    assert sub.attribution == ik.BRIEF_ATTRIBUTION
    block = m.sub_industry_block()
    assert "99101010 Alpha One [industry 991010; sources S01,S29]" in block
    assert "original analyst briefs" in block


def test_prompt_block_is_bounded_and_carries_provenance_on_each_line(synthetic):
    m = igk.group_mandate("9910", version_key="test")
    block = m.as_prompt_block(max_chars=400)
    assert len(block) <= 400 and block.endswith("…")
    full = m.as_prompt_block()
    assert "Core KPIs: Bookings [991010,991020]" in full
    assert "Research priority: 5/5 (set by 991020 Beta)" in full
    assert "Preferred archetypes: Compounder [991010]; Inflection [991020]" in full
    assert full.rstrip().endswith(ik.BRIEF_ATTRIBUTION)


def test_checklists_keep_the_two_mandates_distinct(synthetic):
    checklists = igk.group_mandate("9910", version_key="test").as_checklists()
    assert [c["text"] for c in checklists["compounder"]] == ["Sticky services"]
    assert [c["text"] for c in checklists["inflection"]] == ["Utilization crosses threshold"]
    assert checklists["compounder"][0]["industry_codes"] == ["991010", "991020"]


def test_mandate_is_cached_per_code_and_version_key(synthetic):
    a = igk.group_mandate("9910", version_key="v1")
    assert igk.group_mandate("9910", version_key="v1") is a
    assert igk.group_mandate("9910", version_key="v2") is not a


def test_unknown_group_raises_rather_than_returning_an_empty_mandate(synthetic):
    with pytest.raises(igk.UnknownIndustryGroup):
        igk.group_mandate("0000", version_key="test")


# --- the real knowledge base --------------------------------------------------


def test_every_real_group_aggregates_to_a_non_empty_mandate():
    groups = ik.list_industry_groups()
    assert groups, "knowledge base carries no industry groups"
    for g in groups:
        m = igk.group_mandate(g["code"])
        assert m.name == g["name"] and m.sector_code == g["sector_code"]
        assert len(m.industries) == g["industry_count"]
        assert m.lists["core_kpis"], g["code"]
        assert m.research_priority is not None, g["code"]
        assert m.research_priority_source.startswith(tuple(m.industry_codes))
        assert len(m.as_prompt_block()) <= igk.PROMPT_BLOCK_MAX_CHARS
        # Provenance on every ranked item resolves to one of the group's industries.
        for ranked in m.lists.values():
            for item in ranked:
                assert set(item.industry_codes) <= set(m.industry_codes)
                assert item.count == len(item.industry_codes)
        # Sub-industries listed (bounded) are the group's own.
        assert all(s.industry_code in m.industry_codes for s in m.sub_industries)
        assert len(m.sub_industries) + m.sub_industries_truncated == len(ik.list_sub_industries(g["code"]))


def test_sub_industry_budget_truncates_honestly_instead_of_dropping(monkeypatch):
    monkeypatch.setattr(igk, "SUB_INDUSTRY_BLOCK_MAX_CHARS", 200)
    igk.clear_cache()
    try:
        biggest = max(ik.list_industry_groups(), key=lambda g: len(ik.list_sub_industries(g["code"])))
        m = igk.group_mandate(biggest["code"], version_key="budget-test")
        total = len(ik.list_sub_industries(biggest["code"]))
        assert total >= 2
        assert len(m.sub_industries) >= 1
        assert len(m.sub_industries) + m.sub_industries_truncated == total
        assert m.sub_industries_truncated >= 1
        assert "omitted for length" in m.sub_industry_block()
    finally:
        igk.clear_cache()


def test_universal_rules_and_sources_are_loaded_not_retyped():
    payload = ik.load_industry_knowledge()
    assert igk.universal_research_rules() == list(payload["universal_research_rules"])
    assert igk.primary_sources() == list(payload["primary_sources"])
    block = igk.universal_rules_prompt_block()
    for i, rule in enumerate(payload["universal_research_rules"], start=1):
        assert f"{i}. {rule}" in block


def test_methodology_block_carries_the_eight_stages_in_order():
    stages = igk.thesis_stages()
    expected = sorted(ik.governing_methodology()["thesis_construction_order"], key=lambda s: s["order"])
    assert [s["id"] for s in stages] == [s["id"] for s in expected]
    assert stages[0]["id"] == "world_change" and stages[-1]["id"] == "valuation"
    block = igk.methodology_prompt_block()
    for s in expected:
        assert f"{s['order']}. {s['id']}: {s['question']}" in block
    for key in ik.governing_methodology()["operating_rules"]:
        assert f"- {key}:" in block
