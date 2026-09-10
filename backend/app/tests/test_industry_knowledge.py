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


# --- the sub-industry layer (universe map, merged by the builder) -------------


def _sub_industries(payload: dict) -> list[tuple[dict, dict]]:
    return [
        (industry, sub)
        for _, _, industry in _industries(payload)
        for sub in industry["sub_industries"]
    ]


def test_sub_industry_layer_counts_and_provenance_come_from_the_data(payload):
    subs = _sub_industries(payload)
    # The declared count is the length of the merged list, never a literal.
    assert payload["sub_industry_count"] == len(subs)
    assert payload["sub_industry_count"] == len({sub["code"] for _, sub in subs})
    assert payload["map_generated_from"].endswith("Investment_Universe_163_Map.json")
    assert len(payload["map_source_sha256"]) == 64
    assert payload["map_as_of"] and payload["map_metadata"]["as_of"] == payload["map_as_of"]
    assert payload["map_metadata"]["taxonomy_structure_effective"]
    assert "not licensed issuer GICS assignments" in payload["_doc"]
    for industry, sub in subs:
        assert len(sub["code"]) == 8 and sub["code"].isdigit(), sub
        assert sub["code"][:6] == industry["code"], (sub["code"], industry["code"])
        assert sub["name"].strip()
        assert set(sub["fields"]) == set(payload["sub_industry_field_names"])
        assert all(isinstance(v, str) and v.strip() for v in sub["fields"].values()), sub["code"]
        assert isinstance(sub["source_ids"], list) and sub["source_ids"]
        assert isinstance(sub["cross_industry_themes"], list)
        assert sub["framework_status"] and sub["last_framework_review"]
        # Listing fields never travel with the brief — symbols live in one place.
        assert "provider_observed_us_symbols" not in sub and "unresolved_us_symbol_leads" not in sub
    for _, _, industry in _industries(payload):
        assert industry["sub_industries"], industry["code"]
        assert [s["code"] for s in industry["sub_industries"]] == sorted(
            s["code"] for s in industry["sub_industries"]
        )


def test_retired_rows_are_carried_but_never_active(payload):
    active = {sub["code"] for _, sub in _sub_industries(payload)}
    retired = payload["retired_sub_industries"]
    assert retired
    assert [r["code"] for r in retired] == sorted(r["code"] for r in retired)
    for row in retired:
        assert len(row["code"]) == 8 and row["code"] not in active
        assert row["parent_code"] == row["code"][:6]
        assert row["status"] == "discontinued" and row["name"] and row["parent_name"]


def test_relationships_sources_and_security_reference_resolve(payload):
    active = {sub["code"] for _, sub in _sub_industries(payload)}
    relationships = payload["cross_industry_relationships"]
    assert relationships
    for rel in relationships:
        assert set(rel) == {"theme", "mechanism", "codes", "monitor", "failure_of_inference", "source_ids"}
        assert rel["codes"] and set(rel["codes"]) <= active
    source_ids = {s["id"] for s in payload["sources"]}
    assert source_ids and all(rel["source_ids"] for rel in relationships)
    assert all(set(rel["source_ids"]) <= source_ids for rel in relationships)
    reference = payload["security_reference"]
    assert reference
    for symbol, entry in reference.items():
        assert symbol == symbol.upper().strip()
        assert set(entry) == {"codes", "as_of", "source_id"}, symbol
        assert entry["codes"] and set(entry["codes"]) <= active, symbol
        assert entry["source_id"] in source_ids
    assert list(reference) == sorted(reference)


def test_governing_methodology_is_an_ordered_causal_chain(payload):
    method = payload["governing_methodology"]
    order = method["thesis_construction_order"]
    assert [step["order"] for step in order] == list(range(1, len(order) + 1))
    assert len(order) >= 2 and all(step["id"] and step["question"] for step in order)
    assert "world change" in order[0]["id"].replace("_", " ") or "world" in order[0]["question"].lower()
    assert "valuation" in order[-1]["id"] or "valuation" in order[-1]["question"].lower()
    assert method["operating_rules"]


@needs_source
def test_builder_refuses_a_sub_industry_under_the_wrong_industry(tmp_path):
    document = json.loads(builder.DEFAULT_MAP.read_text(encoding="utf-8"))
    entry = document["sub_industries"][0]
    entry["industry_code"] = entry["industry_code"][:-1] + ("9" if entry["industry_code"][-1] != "9" else "8")
    broken = tmp_path / "map.json"
    broken.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SystemExit, match="disagrees with the code prefix"):
        builder.build_payload(builder.DEFAULT_SOURCE, broken)


@needs_source
def test_builder_refuses_a_map_whose_declared_count_disagrees(tmp_path):
    document = json.loads(builder.DEFAULT_MAP.read_text(encoding="utf-8"))
    document["metadata"]["active_sub_industries"] = int(document["metadata"]["active_sub_industries"]) + 1
    inconsistent = tmp_path / "map.json"
    inconsistent.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SystemExit, match="internally inconsistent"):
        builder.build_payload(builder.DEFAULT_SOURCE, inconsistent)


@needs_source
def test_builder_refuses_a_security_with_an_unknown_code(tmp_path):
    document = json.loads(builder.DEFAULT_MAP.read_text(encoding="utf-8"))
    symbol = next(iter(document["security_reference"]))
    document["security_reference"][symbol]["economic_reference_codes"] = ["99999999"]
    bad = tmp_path / "map.json"
    bad.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown sub-industry codes"):
        builder.build_payload(builder.DEFAULT_SOURCE, bad)


# --- the loader's sub-industry API ----------------------------------------------


def test_get_sub_industry_carries_its_parent_chain_and_attribution(payload):
    industry, sub = _sub_industries(payload)[0]
    entry = ik.get_sub_industry(sub["code"])
    assert entry is not None
    assert entry["name"] == sub["name"] and entry["fields"] == sub["fields"]
    assert entry["industry_code"] == industry["code"] and entry["industry_name"] == industry["name"]
    assert entry["industry_group_code"] == sub["code"][:4]
    assert entry["sector_code"] == sub["code"][:2]
    assert entry["attribution"] == ik.BRIEF_ATTRIBUTION
    assert "not licensed GICS content" in entry["attribution"]
    assert ik.get_sub_industry(f" {sub['code']} ") == entry
    assert ik.get_sub_industry(payload["retired_sub_industries"][0]["code"]) is None
    assert ik.get_sub_industry(industry["code"]) is None and ik.get_sub_industry("") is None


def test_list_sub_industries_narrows_by_prefix_at_any_level(payload):
    everything = ik.list_sub_industries()
    assert len(everything) == payload["sub_industry_count"]
    assert everything[0].keys() == {"code", "name", "industry_code", "industry_group_code", "sector_code"}
    group = payload["sectors"][0]["industry_groups"][0]
    in_group = ik.list_sub_industries(group["code"])
    assert in_group and all(s["code"].startswith(group["code"]) for s in in_group)
    assert len(in_group) == sum(len(i["sub_industries"]) for i in group["industries"])
    assert len(ik.list_sub_industries(payload["sectors"][0]["code"])) >= len(in_group)
    assert ik.list_sub_industries(group["industries"][0]["code"]) == [
        s for s in in_group if s["industry_code"] == group["industries"][0]["code"]
    ]
    assert ik.list_sub_industries("") == [] and ik.list_sub_industries("0000") == []


def test_security_reference_lookup_is_symbol_exact_and_caveated(payload):
    symbol, raw = next(iter(payload["security_reference"].items()))
    entry = ik.security_reference(symbol.lower())
    assert entry == {
        "symbol": symbol, "codes": raw["codes"], "as_of": raw["as_of"],
        "source_id": raw["source_id"], "caveat": ik.SECURITY_REFERENCE_CAVEAT,
    }
    assert "not official licensed issuer GICS mapping" in entry["caveat"]
    assert ik.security_reference("ZZ-NOT-A-SYMBOL") is None and ik.security_reference("") is None
    assert ik.security_reference_as_of() == payload["map_as_of"]


def test_map_companions_are_copies(payload):
    assert ik.retired_sub_industries() == payload["retired_sub_industries"]
    assert ik.cross_industry_relationships() == payload["cross_industry_relationships"]
    assert ik.governing_methodology() == payload["governing_methodology"]
    assert ik.sources() == payload["sources"]
    assert ik.map_metadata() == payload["map_metadata"]
    ik.cross_industry_relationships().clear()
    ik.governing_methodology()["thesis_construction_order"].clear()
    ik.get_sub_industry(_sub_industries(payload)[0][1]["code"])["fields"]["economics"] = "poisoned"
    assert ik.cross_industry_relationships() == payload["cross_industry_relationships"]
    assert ik.governing_methodology()["thesis_construction_order"]
    assert ik.get_sub_industry(_sub_industries(payload)[0][1]["code"])["fields"]["economics"] != "poisoned"


def test_existing_industry_lookups_still_expose_the_sub_industries(payload):
    industry = ik.get_industry("101010")
    assert [s["code"] for s in industry["sub_industries"]] == [
        s["code"] for s in ik.list_sub_industries("101010")
    ]
    assert set(EXPECTED_FIELDS) == set(industry["fields"])  # the upper layer is unchanged


# --- the Atlas dependency graph (checked in, dev-time generated) ------------------


from app.scripts import build_atlas_dependencies as atlas_builder  # noqa: E402


@pytest.fixture(scope="module")
def atlas() -> dict:
    return json.loads(atlas_builder.DEFAULT_OUTPUT.read_text(encoding="utf-8"))


def test_atlas_dependencies_json_carries_both_sources_resolved_to_groups(payload, atlas):
    active = {sub["code"] for _, sub in _sub_industries(payload)}
    assert atlas["knowledge_taxonomy_version"] == payload["taxonomy_version"]
    assert atlas["knowledge_map_source_sha256"] == payload["map_source_sha256"]
    assert atlas["knowledge_map_as_of"] == payload["map_as_of"]
    assert len(atlas["atlas_sha256"]) == 64 and atlas["atlas_snapshot_date"]
    assert atlas["edge_count"] == len(atlas["edges"]) > 0
    assert atlas["relationship_count"] == len(atlas["relationships"]) == len(payload["cross_industry_relationships"])
    assert "not estimated correlations" in atlas["_doc"]

    ids = [e["edge_id"] for e in atlas["edges"]]
    assert len(ids) == len(set(ids))
    for edge in atlas["edges"]:
        assert edge["source"] == atlas_builder.SOURCE_ATLAS == atlas["edge_source"]
        assert edge["origin"] and edge["destination"] and edge["transmission"] and edge["status"]
        assert set(edge["sub_industry_codes"]) <= active
        assert edge["industry_group_codes"] == sorted({c[:4] for c in edge["sub_industry_codes"]})
        resolved_codes = {c for codes in edge["resolved_exposures"].values() for c in codes}
        assert resolved_codes == set(edge["sub_industry_codes"])
        # Every exposure label is accounted for: resolved or listed, never dropped.
        assert set(edge["resolved_exposures"]) | {u.upper() for u in edge["unresolved_exposures"]} == {
            x.upper() for x in edge["example_exposures"]
        }
        for symbol, codes in edge["resolved_exposures"].items():
            assert payload["security_reference"][symbol]["codes"] == codes
    for rel, source in zip(atlas["relationships"], payload["cross_industry_relationships"]):
        assert rel["source"] == atlas_builder.SOURCE_MAP == atlas["relationship_source"]
        assert rel["theme"] == source["theme"] and rel["codes"] == source["codes"]
        assert rel["industry_group_codes"] == sorted({c[:4] for c in rel["codes"]})
        assert rel["failure_of_inference"] == source["failure_of_inference"]


def test_atlas_builder_exits_cleanly_without_openpyxl(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(atlas_builder, "openpyxl_available", lambda: False)
    output = tmp_path / "never.json"
    assert atlas_builder.main(["--output", str(output)]) == 0
    assert "openpyxl is not importable" in capsys.readouterr().out
    assert not output.exists()


@pytest.mark.skipif(
    not atlas_builder.openpyxl_available() or not atlas_builder.DEFAULT_ATLAS.is_file(),
    reason="openpyxl (dev-only) or the Atlas workbook is not available",
)
def test_atlas_builder_is_idempotent_and_checked_in_json_is_current(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    atlas_builder.build(output=first)
    atlas_builder.build(output=second)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == atlas_builder.DEFAULT_OUTPUT.read_bytes(), (
        "checked-in atlas_dependencies.json is stale — rerun "
        "`python -m app.scripts.build_atlas_dependencies`"
    )
    assert atlas_builder.main(["--check"]) == 0
    stale = tmp_path / "stale.json"
    stale.write_text("{}", encoding="utf-8")
    assert atlas_builder.main(["--check", "--output", str(stale)]) == 1
