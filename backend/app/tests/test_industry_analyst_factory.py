"""FEAT-003 slice 2 — Industry Group analysts and their place in the memo.

Pinned here:
- the factory memoises per (code, taxonomy version id) and reads the
  active version from the database on every call;
- the analyst's system prompt loads the methodology, the universal rules
  and the group mandate from the knowledge base (never retyped), and the
  company-context block names the sub-industry and the classification's
  source label;
- with routing OFF (the default) a demo memo is byte-for-byte what the
  base commit produced — a golden captured at 3038e78 — and the sector
  prompt still formats with an empty `{industry_group_block}`;
- with routing ON a memo constructs at most one analyst, its finding lands
  in `extra_agent_views["industry_group"]` carrying `data.industry_group`,
  and an unmapped ticker records the "no mapping" soft degradation and
  runs no analyst at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents import graph, prompts
from app.agents import industry_analysts as ia
from app.config import settings
from app.services import gics_registry as reg
from app.services import industry_classification as ic
from app.services import industry_knowledge as ik
from app.services.industry_group_knowledge import thesis_stages, universal_research_rules
from app.tests.gating_helpers import seed_demo_universe

GOLDEN = Path(__file__).parent / "fixtures" / "industry_routing_off_golden_msft.json"


@pytest.fixture(scope="module", autouse=True)
def _universe():
    seed_demo_universe()
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None
    ic.classify_all(tickers=["MSFT", "NVDA", "JPM"])
    yield info


@pytest.fixture(autouse=True)
def _fresh_cache():
    ia.clear_cache()
    yield
    ia.clear_cache()


# --- the factory ---------------------------------------------------------------


def test_factory_caches_per_code_and_version(_universe):
    a = ia.get_industry_analyst("4530")
    assert ia.get_industry_analyst("4530") is a
    assert ia.get_industry_analyst("4530", version=_universe.version_key) is a
    assert ia.get_industry_analyst("4530", version=_universe.id) is a
    assert ia.construction_count() == 1
    b = ia.get_industry_analyst("4510")
    assert b is not a and ia.construction_count() == 2
    assert a.code == "4530" and a.taxonomy_version_id == _universe.id
    assert a.taxonomy_version_key == _universe.version_key
    assert a.mandate.code == "4530" and a.mandate.version_key == _universe.version_key
    assert a.display_name == "Industry Group Analyst 4530"


def test_factory_reads_the_active_version_from_the_database_on_every_call(monkeypatch):
    calls: list[int] = []
    real = reg.active_version

    def counted():
        calls.append(1)
        return real()

    monkeypatch.setattr(reg, "active_version", counted)
    ia.get_industry_analyst("4530")
    ia.get_industry_analyst("4530")
    assert len(calls) == 2  # never a process-local answer for the active version
    assert ia.construction_count() == 1


def test_factory_rejects_a_code_the_registry_does_not_carry():
    with pytest.raises(reg.UnknownNode):
        ia.get_industry_analyst("0000")
    with pytest.raises(reg.UnknownNode):
        ia.get_industry_analyst("453010")  # an industry, not a group


def test_system_prompt_loads_methodology_rules_and_mandate_from_the_knowledge_base():
    prompt = ia.get_industry_analyst("4530").system_prompt()
    assert "Industry Group Analyst at MarketMosaic" in prompt  # prompts/industry_analyst.md
    for stage in thesis_stages():
        assert stage["question"] in prompt
    for rule in universal_research_rules():
        assert rule in prompt
    for key in ik.governing_methodology()["operating_rules"]:
        assert f"- {key}:" in prompt
    assert "## Industry Group mandate — 4530" in prompt
    assert ik.BRIEF_ATTRIBUTION in prompt


def test_company_context_block_names_sub_industry_and_source_label():
    analyst = ia.get_industry_analyst("4510")
    row = ic.current_for(["MSFT"])["MSFT"]
    assert row["source"] == "research_map"
    block = analyst.company_context_block({"ticker": "MSFT", "company_name": "Microsoft"}, row)
    sub = ik.get_sub_industry(row["sub_industry_code"])
    assert f"Sub-industry: {row['sub_industry_code']} {sub['name']}" in block
    assert "Classification: research map (Investment_Universe_163_Map.json@" in block
    assert "derived from provider classification, not licensed GICS security assignments" in block
    assert "## Industry Group mandate — 4510" in block

    alias_row = {**row, "source": "provider_alias", "author": "fmp-aliases-2026-09",
                 "sub_industry_code": None, "state": "mapped"}
    block = analyst.company_context_block({"ticker": "MSFT", "company_name": "Microsoft"}, alias_row)
    assert "Classification: derived from provider classification (fmp-aliases-2026-09)" in block
    assert "Sub-industry: n/a (classified at group level only)" in block
    assert ia.classification_source_label(None) == "unclassified"
    assert ia.classification_source_label({"state": "missing", "source": "none"}) == "unmapped (missing)"


def test_industry_group_summary_carries_provenance_and_caveat():
    analyst = ia.get_industry_analyst("4530")
    row = ic.current_for(["NVDA"])["NVDA"]
    summary = ia.industry_group_summary(row, analyst)
    assert summary["code"] == "4530" and summary["name"] == analyst.name
    assert summary["state"] == "mapped" and summary["source"] == "research_map"
    assert summary["source_label"].startswith("research map (")
    assert summary["sub_industry"]["code"] == row["sub_industry_code"]
    assert summary["taxonomy_version"] == analyst.taxonomy_version_key
    assert summary["report_version"] is None
    assert summary["mapping_caveat"] == reg.MAPPING_CAVEAT


# --- the prompt placeholder ------------------------------------------------------


def _format_sector_prompt(block: str) -> str:
    return prompts.SECTOR_ANALYST_PROMPT.format(
        sector="Technology", drivers="d", kpis="k", valuation_lens="v", macro_sensitivities="m",
        industry_group_block=block, macro_broadcast="{}", news_alerts="[]",
    )


def test_sector_prompt_formats_with_the_empty_placeholder():
    text = _format_sector_prompt("")
    assert "Macro sensitivities: m.\n\nMacro broadcast" in text
    assert "industry_group_block" not in text and "Industry group" not in text
    with_block = _format_sector_prompt("\n\n## Industry group context for X")
    assert "Macro sensitivities: m.\n\n## Industry group context for X\n\nMacro broadcast" in with_block


def test_sector_prompt_block_is_empty_for_a_non_routable_row():
    block, summary, analyst = ia.sector_prompt_block({"ticker": "X"}, {"state": "fallback", "source": "provider_alias",
                                                                        "industry_group_code": None})
    assert block == "" and analyst is None
    assert summary["state"] == "fallback" and summary["code"] is None
    block, summary, analyst = ia.sector_prompt_block({"ticker": "X"}, None)
    assert block == "" and summary is None and analyst is None


# --- routing off: the memo is unchanged versus the base commit -----------------


def _subset(memo) -> dict:
    sv = memo.sector_agent_view
    return {
        "sector_agent_view": {
            "agent": sv.agent, "headline": sv.headline, "summary": sv.summary,
            "key_points": sv.key_points, "sources": sv.sources,
            "data_keys": sorted(sv.data.keys()) if isinstance(sv.data, dict) else None,
        },
        "extra_agent_views": sorted(memo.extra_agent_views.keys()),
        # The scorecard entry depends on whether another suite scored MSFT
        # in this database; it is not this slice's concern.
        "degraded_agents": [a for a in memo.degraded_agents if a != "Fundamental Scorecard"],
        "degradation_events": [e for e in memo.degradation_events if e["agent"] != "Fundamental Scorecard"],
        "rating_label": memo.rating_label,
    }


def test_routing_off_demo_memo_matches_the_base_commit_golden():
    """Golden captured at 3038e78 (before this slice) from the same demo
    run; with the flag off the roster, the sector prompt and the memo
    views must be exactly what they were."""
    assert settings.enable_industry_analyst_routing is False
    golden = json.loads(GOLDEN.read_text())
    golden["degraded_agents"] = [a for a in golden["degraded_agents"] if a != "Fundamental Scorecard"]
    golden["degradation_events"] = [e for e in golden["degradation_events"] if e["agent"] != "Fundamental Scorecard"]
    memo = graph.run_stock_memo("MSFT")
    assert _subset(memo) == golden
    assert "industry_group" not in memo.sector_agent_view.data
    assert ia.construction_count() == 0


# --- routing on ------------------------------------------------------------------


def test_routing_on_constructs_at_most_one_analyst_and_lands_in_extra_views(monkeypatch):
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    memo = graph.run_stock_memo("MSFT")
    assert ia.construction_count() == 1
    assert set(memo.extra_agent_views) == {"industry_group"}
    finding = memo.extra_agent_views["industry_group"]
    assert finding.agent == "Industry Group Analyst"
    assert finding.confidence > 0.0
    ig = finding.data["industry_group"]
    assert ig["code"] == "4510" and ig["state"] == "mapped"
    assert ig["source_label"].startswith("research map (")
    assert ig["sub_industry"]["code"] == "45103020"
    assert ig["mapping_caveat"] == reg.MAPPING_CAVEAT
    assert [c["stage"] for c in finding.data["causal_chain"]] == [s["id"] for s in thesis_stages()]
    assert "industry_knowledge:4510" in finding.sources
    # The sector analyst saw the same mapping and says so on its finding.
    assert memo.sector_agent_view.data["industry_group"]["code"] == "4510"
    assert "Industry Group Analyst" not in memo.degraded_agents
    assert finding.long_form_report  # the long-form pass covered the new spec too


def test_unmapped_ticker_records_no_mapping_and_runs_no_analyst(monkeypatch):
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    monkeypatch.setattr(ia, "lookup_classification", lambda ticker: {
        "ticker": ticker, "state": "missing", "source": "none", "industry_group_code": None,
    })
    memo = graph.run_stock_memo("MSFT")
    assert "industry_group" not in memo.extra_agent_views
    assert ia.construction_count() == 0
    events = [e for e in memo.degradation_events if e["agent"] == "Industry Group Analyst"]
    assert events == [{"agent": "Industry Group Analyst", "error_type": "NoMapping",
                       "message": "no mapping: classification state missing"}]
    assert "Industry Group Analyst" in memo.degraded_agents
    # The sector analyst carries the state so the reader sees why.
    assert memo.sector_agent_view.data["industry_group"]["state"] == "missing"


def test_no_classification_row_is_reported_as_such(monkeypatch):
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    monkeypatch.setattr(ia, "lookup_classification", lambda ticker: None)
    memo = graph.run_stock_memo("MSFT")
    events = [e for e in memo.degradation_events if e["agent"] == "Industry Group Analyst"]
    assert events[0]["message"] == "no mapping: no classification row"
    assert "industry_group" not in memo.sector_agent_view.data


# --- the runner on its own -----------------------------------------------------------


def test_run_industry_group_agent_deterministic_read_is_mandate_grounded():
    row = ic.current_for(["NVDA"])["NVDA"]
    profile = {"ticker": "NVDA", "company_name": "NVIDIA", "sector": "Technology"}
    finding = ia.run_industry_group_agent(profile, {"roic": 0.51, "ev_ebitda": 30.8}, classification=row)
    analyst = ia.get_industry_analyst("4530")
    assert finding.headline.startswith(f"{analyst.name} (4530): mandate read for NVDA")
    assert "Deterministic edition" in finding.summary
    assert finding.data["industry_group"]["sub_industry"]["code"] == "45301020"
    assert finding.data["kpis_to_watch"] and all(k["industry_code"] in analyst.mandate.industry_codes
                                                for k in finding.data["kpis_to_watch"])
    assert finding.data["causal_chain"][0]["text"].startswith("n/a:")
    assert "deterministic_fallback" not in finding.data  # no LLM configured: the design, not a degradation
    assert finding.evidence[0].ref == "gics:4530"


def test_run_industry_group_agent_looks_the_row_up_when_not_given():
    finding = ia.run_industry_group_agent({"ticker": "NVDA"}, {})
    assert finding.data["industry_group"]["code"] == "4530"


def test_run_industry_group_agent_on_an_unknown_symbol_says_no_mapping():
    finding = ia.run_industry_group_agent({"ticker": "ZZNOSUCH"}, {})
    assert finding.confidence == 0.0
    assert finding.data["no_mapping"] is True
    assert finding.data["industry_group"]["state"] == "missing"
    assert "no mapping" in finding.headline.lower()


def test_applies_to_is_false_with_routing_off_and_records_nothing():
    from app.tests.factories import make_inputs
    inputs = make_inputs("MSFT", industry_group={"state": "mapped", "industry_group_code": "4510"})
    assert ia.applies_to(inputs) is False
    assert inputs.degradation.failures == []
