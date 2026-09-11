"""`_resolve_sector_block` — the config a sector's cohort research runs on.

Two defects live here, and both are the same mistake: putting a sector in
front of the wrong framework and saying nothing.

The first was `next(iter(cfg.values()))` — an unmatched label fell through
to the FIRST config in `sector_configs.json`, which is Technology.

The second was subtler, and survived the first fix: the neutral block
introduced to replace that fallback still COPIED Technology's KPI groups
under a "generic" label, and `Financial Services` — FMP's own label for
banks, insurers and payments, the label the fix was named for — still
failed to match the `Financials` config that was sitting in the file, so
those companies got the neutral block instead of the right one.

Pinned here: the provider labels resolve to their configs, the neutral
block is neutral rather than a copy, and a guess is not made when the
label is genuinely ambiguous.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import sector_research_service as srs

_ALIASES = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "industry_knowledge" / "provider_aliases.json").read_text()
)


def test_exact_matches_still_resolve():
    assert srs._resolve_sector_block("Financials") is srs._sector_config()["Financials"]
    assert srs._resolve_sector_block("Healthcare") is srs._sector_config()["Healthcare"]
    # GICS' own spelling reaches the config file's shorter one through the
    # alias table (it used to arrive by substring, which is a coincidence).
    assert srs._resolve_sector_block("Information Technology") is srs._sector_config()["Technology"]


def test_an_fmp_financial_services_company_researches_against_financials():
    """REGRESSION: "Financial Services" is neither an exact key nor a
    substring of "Financials" in either direction, so every bank, insurer
    and payment network FMP labels that way got the neutral block while the
    correct config sat in the same file."""
    from app.tests.fixtures.demo_dataset import COMPANY_PROFILES
    profile = dict(COMPANY_PROFILES["JPM"])
    assert profile["sector"] == "Financials"      # the app's own vocabulary
    profile["sector"] = "Financial Services"      # what the provider returns

    block = srs._resolve_sector_block(profile["sector"])
    assert block is srs._sector_config()["Financials"]
    assert block.get("neutral_default") is None
    # A bank's lens, not a software company's.
    assert "PB" in block["kpi_groups"]["valuation"]
    assert "EV_EBITDA" not in block["kpi_groups"]["valuation"]
    assert "rd_pct_revenue" not in json.dumps(block["kpi_groups"])


@pytest.mark.parametrize("label", sorted(_ALIASES["sectors"]))
def test_every_provider_sector_label_reaches_a_real_config(label):
    """The other near-miss labels, from the crosswalk the classifier already
    maintains: "Consumer Cyclical", "Consumer Defensive", "Health Care",
    "Basic Materials", "Telecom". None of them may land on the neutral
    block — a config exists for every one."""
    block = srs._resolve_sector_block(label)
    assert block.get("neutral_default") is not True, label
    assert block in list(srs._sector_config().values()), label


def test_an_unknown_label_gets_a_neutral_block_that_is_not_technologys():
    """REGRESSION: the neutral block copied `kpi_groups` from the first
    config — Technology's — and relabelled them "generic". A bank placed
    against `rd_pct_revenue` and `EV_EBITDA` is not being measured
    neutrally; it is being measured as a software company under another
    name. The neutral groups are now the ones EVERY config agrees on."""
    block = srs._resolve_sector_block("Made Up Sector")
    tech = srs._sector_config()["Technology"]
    assert block is not tech
    assert block["key_drivers"] != tech["key_drivers"]
    assert block.get("neutral_default") is True
    assert "Made Up Sector" in block["reason"]
    # Same shape as a real config so downstream readers need no special case.
    for key in ("key_drivers", "kpi_groups", "valuation_lens", "macro_sensitivities",
                "common_risks", "secular_trends", "subindustry_overrides"):
        assert key in block, key

    # The assertion this test used to make was `set(block["kpi_groups"]) ==
    # set(tech["kpi_groups"])` — it PINNED the copy as correct behaviour, so
    # it is inverted rather than deleted: the neutral groups must not be
    # Technology's, and every metric they do carry must be one that every
    # single sector config agrees on.
    assert set(block["kpi_groups"]) != set(tech["kpi_groups"])
    assert "capital_intensity" not in block["kpi_groups"]   # Technology's, not everyone's
    assert "valuation" not in block["kpi_groups"]           # the lens IS sector-specific
    for group, kpis in block["kpi_groups"].items():
        for cfg in srs._sector_config().values():
            assert set(kpis) <= set((cfg.get("kpi_groups") or {}).get(group) or []), (group, kpis)
    assert block["kpi_groups"], "a neutral block with no KPI at all cannot place anything"

    # House rule: a group that could not be filled is reported with a
    # reason, not silently missing.
    omitted = block["kpi_groups_omitted"]
    assert set(omitted) == {"capital_intensity", "leverage", "valuation"}
    assert all(reason.strip() for reason in omitted.values())
    assert "sector-specific" in block["valuation_lens"]


def test_blank_sector_is_neutral_too():
    block = srs._resolve_sector_block("")
    assert block.get("neutral_default") is True
    assert block["key_drivers"] == []


def test_an_ambiguous_substring_is_not_a_guess():
    """"Consumer" is a substring of two configs. Returning whichever came
    first in the file is the original defect in miniature, so an ambiguous
    label goes neutral; an unambiguous one still resolves."""
    assert srs._resolve_sector_block("Consumer").get("neutral_default") is True
    assert srs._resolve_sector_block("Technology Hardware") is srs._sector_config()["Technology"]


def test_neutral_block_is_a_fresh_copy_each_time():
    a = srs._resolve_sector_block("Made Up Sector")
    b = srs._resolve_sector_block("Made Up Sector")
    assert a == b and a is not b
    a["key_drivers"].append("mutated")
    assert srs._resolve_sector_block("Made Up Sector")["key_drivers"] == []
    a["kpi_groups"]["growth"].append("mutated")
    assert "mutated" not in srs._resolve_sector_block("Made Up Sector")["kpi_groups"]["growth"]
