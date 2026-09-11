"""`_resolve_sector_block` — the config a sector's cohort research runs on.

Before the fix an unknown sector label fell through to the FIRST config in
`sector_configs.json`, which is Technology: every `Financial Services`
company (FMP's label for banks, insurers and payments) was researched with
Technology's drivers, KPIs and valuation lens. An unknown sector now gets a
neutral block that says so, and the known-label paths are pinned so the
fix cannot regress them.
"""
from __future__ import annotations

from app.services import sector_research_service as srs


def test_exact_and_substring_matches_still_resolve():
    assert srs._resolve_sector_block("Financials") is srs._sector_config()["Financials"]
    assert srs._resolve_sector_block("Healthcare") is srs._sector_config()["Healthcare"]
    # Substring heuristic: "Information Technology" contains "Technology".
    assert srs._resolve_sector_block("Information Technology") is srs._sector_config()["Technology"]


def test_unknown_sector_gets_the_neutral_block_not_technology():
    block = srs._resolve_sector_block("Financial Services")
    tech = srs._sector_config()["Technology"]
    assert block is not tech
    assert block["key_drivers"] != tech["key_drivers"]
    assert block.get("neutral_default") is True
    assert "Financial Services" in block["reason"]
    # Same shape as a real config so downstream readers need no special case.
    for key in ("key_drivers", "kpi_groups", "valuation_lens", "macro_sensitivities",
                "common_risks", "secular_trends", "subindustry_overrides"):
        assert key in block, key
    assert set(block["kpi_groups"]) == set(tech["kpi_groups"])


def test_blank_sector_is_neutral_too():
    block = srs._resolve_sector_block("")
    assert block.get("neutral_default") is True
    assert block["key_drivers"] == []


def test_neutral_block_is_a_fresh_copy_each_time():
    a = srs._resolve_sector_block("Made Up Sector")
    b = srs._resolve_sector_block("Made Up Sector")
    assert a == b and a is not b
    a["key_drivers"].append("mutated")
    assert srs._resolve_sector_block("Made Up Sector")["key_drivers"] == []
