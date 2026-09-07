"""The GICS industry knowledge base: generated JSON, its parser, and the loader.

Three things can rot independently here, and each has a test:

- the checked-in JSON drifting from the markdown it was generated from
  (someone edits the encyclopedia and forgets to rebuild);
- the parser silently emitting a partial file when the source is malformed
  (the failure mode ``build_industry_knowledge`` exists to refuse);
- the loader hard-coding taxonomy facts the data should own.

Counts are read from the data and cross-checked structurally; the literal
11 / 25 / 74 appears only as the pin for the April 2026 GICS structure.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.scripts import build_industry_knowledge as builder
from app.services import industry_knowledge as ik

BACKEND_DIR = Path(__file__).resolve().parents[2]

needs_source = pytest.mark.skipif(
    not builder.DEFAULT_SOURCE.is_file(),
    reason="markdown encyclopedia not present (e.g. installed package, not a checkout)",
)

EXPECTED_FIELDS = {
    "economic_engine",
    "core_kpis",
    "leading_indicators",
    "typical_moats",
    "capital_cycle_supply_response",
    "valuation_lenses",
    "accounting_data_traps",
    "common_failure_modes",
    "ideal_compounder_setup",
    "ideal_inflection_setup",
    "research_priority",
    "cadence",
    "preferred_archetype",
    "highest_evi_question",
}


@pytest.fixture(scope="module")
def payload() -> dict:
    return ik.load_industry_knowledge()


def _industries(payload: dict) -> list[tuple[dict, dict, dict]]:
    return [
        (sector, group, industry)
        for sector in payload["sectors"]
        for group in sector["industry_groups"]
        for industry in group["industries"]
    ]


# --- the generated file -------------------------------------------------


def test_json_loads_with_provenance(payload):
    assert payload["taxonomy_version"] == builder.TAXONOMY_VERSION
    assert payload["generated_from"].endswith("GICS_74_Industry_Research_Encyclopedia.md")
    assert len(payload["source_sha256"]) == 64
    assert "not licensed GICS content" in payload["_doc"]


def test_declared_counts_match_structure_and_gics_2026(payload):
    sectors = payload["sectors"]
    groups = [g for s in sectors for g in s["industry_groups"]]
    industries = [i for g in groups for i in g["industries"]]

    # Declared counts must describe the tree they sit next to...
    assert payload["sector_count"] == len(sectors)
    assert payload["industry_group_count"] == len(groups)
    assert payload["industry_count"] == len(industries)
    # ...and the tree must be the April 2026 GICS structure.
    assert (len(sectors), len(groups), len(industries)) == (11, 25, 74)


def test_codes_are_six_digits_with_prefixes_matching_hierarchy(payload):
    seen: set[str] = set()
    for sector, group, industry in _industries(payload):
        code = industry["code"]
        assert len(code) == 6 and code.isdigit(), (industry["name"], code)
        assert code[:4] == group["code"], (industry["name"], group["name"])
        assert code[:2] == sector["code"], (industry["name"], sector["name"])
        assert len(group["code"]) == 4 and group["code"][:2] == sector["code"]
        assert code not in seen, f"duplicate industry code {code}"
        seen.add(code)


def test_every_industry_carries_the_full_uniform_field_set(payload):
    assert set(payload["field_names"]) == EXPECTED_FIELDS
    for _, _, industry in _industries(payload):
        fields = industry["fields"]
        assert set(fields) == EXPECTED_FIELDS, industry["name"]
        for name, value in fields.items():
            assert isinstance(value, str) and value.strip(), (industry["name"], name)
            assert "**" not in value, (industry["name"], name)
        assert fields["economic_engine"] and fields["core_kpis"]


def test_universal_rules_and_sources_travel_with_the_knowledge(payload):
    assert len(payload["universal_research_rules"]) == 10
    assert all(rule.strip() for rule in payload["universal_research_rules"])
    sources = payload["primary_sources"]
    assert len(sources) >= 20
    for source in sources:
        assert set(source) == {"domain", "name", "url", "description"}
        assert source["url"].startswith("https://")


# --- the parser ---------------------------------------------------------


@needs_source
def test_parser_is_idempotent_and_checked_in_json_is_current(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    builder.build(builder.DEFAULT_SOURCE, first)
    builder.build(builder.DEFAULT_SOURCE, second)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == builder.DEFAULT_OUTPUT.read_bytes(), (
        "checked-in JSON is stale — rerun `python -m app.scripts.build_industry_knowledge`"
    )
    assert first.read_bytes().endswith(b"}\n")


@needs_source
def test_module_entrypoint_builds_and_checks(tmp_path):
    output = tmp_path / "out" / "gics.json"
    cmd = [sys.executable, "-m", "app.scripts.build_industry_knowledge", "--output", str(output)]

    built = subprocess.run(cmd, cwd=BACKEND_DIR, capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stderr
    assert "11 sectors / 25 industry groups / 74 industries" in built.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["industry_count"] == 74

    checked = subprocess.run(cmd + ["--check"], cwd=BACKEND_DIR, capture_output=True, text=True, timeout=60)
    assert checked.returncode == 0, checked.stderr

    output.write_text("{}", encoding="utf-8")
    stale = subprocess.run(cmd + ["--check"], cwd=BACKEND_DIR, capture_output=True, text=True, timeout=60)
    assert stale.returncode == 1
    assert "stale" in stale.stderr


@needs_source
def test_parser_refuses_an_industry_without_a_code(tmp_path):
    text = builder.DEFAULT_SOURCE.read_text(encoding="utf-8")
    broken = tmp_path / "broken.md"
    broken.write_text(
        text.replace("### Energy Equipment & Services (101010)", "### Energy Equipment & Services"),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="without a 6-digit GICS code"):
        builder.build_payload(broken)


@needs_source
def test_parser_refuses_wrong_counts_instead_of_writing_partial_file(tmp_path):
    text = builder.DEFAULT_SOURCE.read_text(encoding="utf-8")
    start = text.index("### Oil, Gas & Consumable Fuels (101020)")
    end = text.index("# Materials")
    truncated = tmp_path / "truncated.md"
    truncated.write_text(text[:start] + text[end:], encoding="utf-8")
    output = tmp_path / "never.json"

    with pytest.raises(SystemExit, match="73 industries"):
        builder.build(truncated, output)
    assert not output.exists()


@needs_source
def test_parser_refuses_a_mistyped_label(tmp_path):
    text = builder.DEFAULT_SOURCE.read_text(encoding="utf-8")
    mistyped = tmp_path / "mistyped.md"
    mistyped.write_text(text.replace("**Core KPIs.** Rig count;", "**Core KPI.** Rig count;", 1),
                        encoding="utf-8")
    with pytest.raises(SystemExit, match="core_kpis"):
        builder.build_payload(mistyped)


# --- the loader ---------------------------------------------------------


def test_loader_reads_once_and_returns_the_same_object():
    assert ik.load_industry_knowledge() is ik.load_industry_knowledge()


def test_get_industry_known_code():
    industry = ik.get_industry("101010")
    assert industry is not None
    assert industry["name"] == "Energy Equipment & Services"
    assert industry["industry_group_code"] == "1010"
    assert industry["sector_code"] == "10"
    assert industry["sector_name"] == "Energy"
    assert industry["fields"]["research_priority"].startswith("5/5")
    assert ik.get_industry(" 101010 ") == industry  # whitespace tolerated


def test_get_industry_unknown_or_blank_code():
    assert ik.get_industry("999999") is None
    assert ik.get_industry("") is None
    assert ik.get_industry("1010") is None  # a group code is not an industry


def test_get_industry_group_includes_its_industries(payload):
    group = ik.get_industry_group("1010")
    assert group is not None
    assert group["name"] == "Energy"
    assert group["sector_code"] == "10"
    assert [i["code"] for i in group["industries"]] == ["101010", "101020"]
    assert group["industries"][0]["fields"]["economic_engine"]
    assert ik.get_industry_group("0000") is None
    assert ik.get_industry_group("101010") is None  # an industry code is not a group


def test_list_industry_groups_is_derived_from_data(payload):
    groups = ik.list_industry_groups()
    assert len(groups) == payload["industry_group_count"]
    assert sum(g["industry_count"] for g in groups) == payload["industry_count"]
    assert {g["sector_code"] for g in groups} == {s["code"] for s in payload["sectors"]}
    assert [g["code"] for g in groups] == sorted(g["code"] for g in groups)
    assert groups[0] == {
        "code": "1010", "name": "Energy", "sector_code": "10",
        "sector_name": "Energy", "industry_count": 2,
    }


def test_search_industries_matches_names_case_insensitively():
    reits = ik.search_industries("reit")
    assert "402040" in reits  # Mortgage REITs (Financials)
    assert "601010" in reits  # Diversified REITs (Real Estate)
    assert all("reit" in ik.get_industry(code)["name"].lower() for code in reits)
    assert ik.search_industries("SEMICONDUCTORS") == ["453010"]
    assert ik.search_industries("no such industry") == []
    assert ik.search_industries("") == []
    assert ik.search_industries("   ") == []


def test_lookups_return_copies_so_the_cache_cannot_be_mutated():
    ik.get_industry("101010")["fields"]["core_kpis"] = "poisoned"
    assert ik.get_industry("101010")["fields"]["core_kpis"] != "poisoned"
    ik.get_industry_group("1010")["industries"].clear()
    assert len(ik.get_industry_group("1010")["industries"]) == 2
