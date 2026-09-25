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
import re
from pathlib import Path

import pytest

from app.agents import deep_research as dr
from app.agents import graph, intake, prompts, roster, safe_runner
from app.agents import industry_analysts as ia
from app.config import settings
from app.schemas import AgentFinding
from app.services import gics_registry as reg
from app.services import industry_classification as ic
from app.services import industry_group_knowledge as igk
from app.services import industry_knowledge as ik
from app.services import industry_labels as il
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
    # `display_name` is the LLMCallLog telemetry key and keeps its code so
    # cost-by-agent continuity survives; readers get `public_display_name`.
    assert a.display_name == "Industry Group Analyst 4530"
    assert a.public_display_name == f"Industry Group Analyst ({il.label('4530')})"


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
    # The PUBLIC mandate (owner decision 2026-09-24): our label, no code.
    assert f"## Industry Group mandate — {il.label('4530')} ({il.label('45')})" in prompt
    assert il.PUBLIC_BRIEF_ATTRIBUTION in prompt
    assert "4530" not in prompt and "gics" not in prompt.lower()


def test_every_group_system_prompt_keeps_the_attribution_and_the_sub_industry_layer():
    """A long mandate must never push the sub-industry layer or the
    attribution line off the end of the bounded prompt (it did for 9 of
    the real groups before the budget counted rendered lines)."""
    profile = {"ticker": "NVDA", "company_name": "NVIDIA"}
    for g in ik.list_industry_groups():
        prompt = ia.get_industry_analyst(g["code"]).system_prompt()
        assert prompt.rstrip().endswith(il.PUBLIC_BRIEF_ATTRIBUTION), g["code"]
        assert igk.PUBLIC_SUB_INDUSTRY_HEADER in prompt, g["code"]
        # Public briefs are unnamed lines after the header; count them there.
        briefs = prompt.split(igk.PUBLIC_SUB_INDUSTRY_HEADER, 1)[1].split("\nAttribution:", 1)[0]
        shown = [ln for ln in briefs.strip("\n").split("\n") if ln.startswith("- ") and "| Advantage test:" in ln]
        omitted = re.search(r"… (\d+) more sub-industr", prompt)
        assert not any(line.endswith("…") for line in shown), g["code"]
        assert len(shown) + (int(omitted.group(1)) if omitted else 0) == len(ik.list_sub_industries(g["code"]))
        block = ia.get_industry_analyst(g["code"]).company_context_block(profile, None)
        assert len(block) <= 4000 and il.PUBLIC_BRIEF_ATTRIBUTION in block, g["code"]


def test_a_long_header_shortens_the_mandate_instead_of_cutting_its_attribution():
    """REGRESSION: the mandate's budget was a fixed `max_chars - 700`, so a
    long company name or a stale row's provenance clause pushed the block
    past `max_chars` and the hard truncation cut the tail — which is the
    BRIEF_ATTRIBUTION line `as_prompt_block` deliberately reserves budget
    for. Sub-industry briefs are original analysis and must never be shown
    without that label."""
    row = ic.current_for(["NVDA"])["NVDA"]
    stale = {**row, "state": "stale", "evidence": {
        **(row.get("evidence") or {}), "previous_state": "mapped",
        "stale_reason": "inputs_changed", "stale_detected_at": "2026-09-08T03:40:00+00:00",
    }}
    for g in ik.list_industry_groups():
        analyst = ia.get_industry_analyst(g["code"])
        for name_len in (10, 85, 95, 100, 105, 215, 400):
            for cls in (None, row, stale):
                block = analyst.company_context_block(
                    {"ticker": "NVDA", "company_name": "N" * name_len}, cls,
                )
                assert len(block) <= 4000, (g["code"], name_len)
                assert il.PUBLIC_BRIEF_ATTRIBUTION in block, (g["code"], name_len, cls and cls["state"])
                assert not block.endswith("…"), (g["code"], name_len)


def test_a_header_that_leaves_no_budget_says_the_mandate_was_omitted():
    """The floor: when `max_chars` cannot hold the header plus a mandate,
    the block announces the omission rather than shipping a mandate whose
    provenance has been sliced off."""
    analyst = ia.get_industry_analyst("4530")
    profile = {"ticker": "NVDA", "company_name": "NVIDIA"}
    # The labels-only header is shorter than the old code-and-version one,
    # so the floor is probed with a tighter ceiling.
    block = analyst.company_context_block(profile, None, max_chars=500)
    assert "mandate omitted: no prompt budget left" in block
    assert "## Industry Group mandate" not in block
    assert len(block) <= 500
    # A ceiling too small for a mandate is the same story, not a silent cut.
    block = analyst.company_context_block(profile, None, mandate_chars=50)
    assert "mandate omitted: no prompt budget left" in block


def test_company_context_block_names_sub_industry_and_source_label():
    analyst = ia.get_industry_analyst("4510")
    row = ic.current_for(["MSFT"])["MSFT"]
    assert row["source"] == "research_map"
    block = analyst.company_context_block({"ticker": "MSFT", "company_name": "Microsoft"}, row)
    # Sub-industries are never named publicly: the provider's own industry
    # string stands in, and neither the 8-digit code nor its registry name
    # reaches the prompt.
    sub = ik.get_sub_industry(row["sub_industry_code"])
    provider_industry = row["source_industry"]      # the row's data, not the function under test
    assert provider_industry and f"Provider industry: {provider_industry}." in block
    assert f"Provider industry: {sub['name']}." not in block
    assert row["sub_industry_code"] not in block and f"{row['sub_industry_code']} {sub['name']}" not in block
    assert "Classification: research map (Investment_Universe_163_Map.json@" in block
    assert il.PUBLIC_MAPPING_CAVEAT in block and "gics" not in block.lower()
    assert f"sits in our {il.label('4510')} industry group ({il.label('45')} sector)" in block
    assert f"## Industry Group mandate — {il.label('4510')}" in block

    alias_row = {**row, "source": "provider_alias", "author": "fmp-aliases-2026-09",
                 "sub_industry_code": None, "state": "mapped", "source_industry": None,
                 "source_sub_industry": None, "evidence": {}}
    block = analyst.company_context_block({"ticker": "MSFT", "company_name": "Microsoft"}, alias_row)
    assert "Classification: derived from provider classification (fmp-aliases-2026-09)" in block
    assert "Provider industry: n/a (not reported by the provider)" in block
    assert ia.classification_source_label(None) == "unclassified"
    assert ia.classification_source_label({"state": "missing", "source": "none"}) == "unmapped (missing)"


def test_a_stale_row_still_routes_and_the_provenance_says_it_is_stale():
    """`industry_classification` flips a drifted row to `stale` IN PLACE —
    the group code survives, the old state moves to
    `evidence.previous_state`. Dropping those rows would make the analyst
    disappear from memos between the drift flag and the next classification
    run, so they route on the previous state and every display says so."""
    row = ic.current_for(["NVDA"])["NVDA"]
    assert row["state"] == "mapped"
    stale = {**row, "state": "stale", "evidence": {
        **(row.get("evidence") or {}),
        "previous_state": "mapped",
        "stale_reason": "inputs_changed",
        "stale_detected_at": "2026-09-08T03:40:00+00:00",
    }}
    assert ia.routed_state(stale) == "mapped"
    assert ia.is_routable(stale)
    analyst = ia.analyst_for_classification(stale)
    assert analyst is not None and analyst.code == row["industry_group_code"]

    label = ia.classification_source_label(stale)
    assert "mapping STALE since 2026-09-08T03:40:00+00:00 (inputs_changed)" in label
    assert "routed on its previous state" in label
    summary = ia.industry_group_summary(stale, analyst)
    # The raw state stays truthful; `routed_state` says what routing used.
    assert summary["state"] == "stale" and summary["routed_state"] == "mapped"
    assert "STALE" in summary["source_label"]
    block = analyst.company_context_block({"ticker": "NVDA", "company_name": "NVIDIA"}, stale)
    assert "mapping STALE since" in block

    # A stale row whose PREVIOUS state was never routable stays unroutable:
    # staleness does not promote a fallback row into a group mandate.
    was_fallback = {**stale, "evidence": {**stale["evidence"], "previous_state": "fallback"}}
    assert ia.routed_state(was_fallback) == "fallback"
    assert not ia.is_routable(was_fallback)
    assert ia.analyst_for_classification(was_fallback) is None
    # …and the banner names the state it actually routed on, so the reader
    # is not told "stale" when the row was only ever a sector-level fallback.
    assert ia._no_mapping_reason(was_fallback) == "no mapping: classification state stale (was fallback)"
    assert ia._no_mapping_reason({"state": "missing"}) == "no mapping: classification state missing"
    assert ia._no_mapping_reason(None) == "no mapping: no classification row"


def test_prompt_header_carries_no_version_key_but_the_summary_keeps_the_public_one():
    """The header used to name both editions (registry taxonomy key and
    bundled knowledge edition). Both are internal keys ("gics-2026-04"), so
    the labels-only header names neither; provenance keeps the PUBLIC key
    on `industry_group_summary`. The mandate prose is still the single
    bundled knowledge base for every taxonomy version."""
    analyst = ia.get_industry_analyst("4530")
    row = ic.current_for(["NVDA"])["NVDA"]
    block = analyst.company_context_block({"ticker": "NVDA", "company_name": "NVIDIA"}, row)
    knowledge = analyst.mandate.knowledge_version
    assert knowledge == ik.load_industry_knowledge()["taxonomy_version"]
    assert analyst.taxonomy_version_key not in block and knowledge not in block
    summary = ia.industry_group_summary(row, analyst)
    assert summary["taxonomy_version"] == il.public_version_key(analyst.taxonomy_version_key)
    assert summary["taxonomy_version"] == il.PUBLIC_TAXONOMY_KEY
    # A mandate built under any other taxonomy version still carries the
    # bundled edition, because there is only one knowledge document.
    # `version_key` is a cache key, not a mandate.
    other = igk.group_mandate("4530", version_key="gics-2099-99")
    assert other.version_key == "gics-2099-99"
    assert other.knowledge_version == knowledge
    assert other.as_prompt_block() != analyst.mandate.as_prompt_block()  # only the header line differs
    assert other.as_prompt_block().split("\n", 1)[1] == analyst.mandate.as_prompt_block().split("\n", 1)[1]


def test_industry_group_summary_carries_provenance_and_caveat():
    analyst = ia.get_industry_analyst("4530")
    row = ic.current_for(["NVDA"])["NVDA"]
    summary = ia.industry_group_summary(row, analyst)
    # Public values only: slug + label, never the code or registry name.
    assert "code" not in summary and summary["slug"] == il.slug("4530") == analyst.slug
    assert summary["label"] == summary["name"] == analyst.label != analyst.name
    assert summary["sector_label"] == il.label("45")
    assert summary["state"] == "mapped" and summary["source"] == "research_map"
    assert summary["source_label"].startswith("research map (")
    assert "sub_industry" not in summary and summary["provider_industry"] == row["source_industry"]
    assert summary["taxonomy_version"] == il.PUBLIC_TAXONOMY_KEY
    assert summary["report_version"] is None
    assert summary["mapping_caveat"] == il.PUBLIC_MAPPING_CAVEAT


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
    assert summary["state"] == "fallback" and summary["slug"] is None and summary["label"] is None
    block, summary, analyst = ia.sector_prompt_block({"ticker": "X"}, None)
    assert block == "" and summary is None and analyst is None


# --- routing off: the memo is unchanged versus the base commit -----------------


# WHAT THE GOLDEN COMPARES, AND WHY EACH EXCLUDED FIELD IS EXCLUDED
#
# The claim under test is narrow and total: with ENABLE_INDUSTRY_ANALYST_
# ROUTING off, this slice changes NOTHING a demo memo says. The slice can
# reach the memo through exactly two doors — the sector analyst's
# `{industry_group_block}` prompt placeholder (and the `industry_group`
# key the routed path adds to its finding data) and the new roster entry
# (a new `extra_agent_views` key, a new name in the degradation lists).
# So the projection below carries every field either door opens onto, and
# excludes only fields whose value is a function of what the shared
# database happens to hold on this run:
#
#   COMPARED, exactly, against the capture:
#     sector_agent_view.agent / headline / summary / sources  — computed
#       from the demo fixtures; the prompt placeholder is the only way
#       this slice could move them.
#     sector_agent_view.data_keys — the sorted key set, which is how an
#       `industry_group` payload leaking onto the finding would show up.
#     sector_agent_view.key_points, minus the macro tail (see below).
#     extra_agent_views — a new roster entry lands here or nowhere.
#     degraded_agents / degradation_events, restricted to _BLAST_RADIUS.
#     rating_label — the memo's own verdict, the end of the pipeline.
#
#   EXCLUDED, by name, with the reason:
#     the macro-overlay tail of key_points. The sector agent appends up to
#       four `narrative_hints` read off `sector_data_context.overlays`, and
#       those quote live macro series ("Sticky core CPI at 3.6% YoY") whose
#       values move with whichever suite last seeded the database. They are
#       not excluded on trust: the tail is re-derived from THIS memo's own
#       overlays every run, and every excluded entry must be one of them,
#       so foreign content cannot hide there.
#     every degradation outside _BLAST_RADIUS. Whether the Fundamental
#       Scorecard has a row for MSFT, whether price history was seeded —
#       none of it is reachable from this flag, and comparing it makes the
#       golden fail for reasons that have nothing to do with the slice.
#       Scoped out by name, not filtered per incident.
#
# `test_routing_off_leaves_no_trace_anywhere_in_the_memo_json` then sweeps
# the WHOLE serialised memo for industry-group markers, so a leak into a
# field this projection does not name is still caught.
_BLAST_RADIUS = {"Sector Analyst", "Industry Group Analyst", "Long-form (Sector)",
                 "Long-form (Industry Group)"}

# Where the capture's `key_points` splits. `sector_agents` builds the list
# in one order — KPI placements, cohort trends, filing themes (all computed
# from the demo fixtures, all invariant) — and only then appends the macro
# overlay hints. The capture holds nine of the former and three of the
# latter. It is a constant here only because the golden is a flat list of
# strings with no data payload to re-derive the boundary from; the MEMO
# side re-derives it from its own overlays on every run, and the test
# asserts the two agree on where the boundary falls.
_GOLDEN_N_COHORT_KEY_POINTS = 9


def _overlay_hints(view) -> set[str]:
    """The key_points the sector agent appended from the macro overlays,
    read back off the finding's own `sector_data_context`."""
    data = view.data if isinstance(view.data, dict) else {}
    bundles = ((data.get("sector_data_context") or {}).get("overlays") or {}).get("bundles") or {}
    hints: set[str] = set()
    for bundle in bundles.values():
        if isinstance(bundle, dict) and bundle.get("available"):
            hints.update(str(h) for h in (bundle.get("narrative_hints") or []))
    return hints


def _invariant_projection(memo) -> dict:
    """The part of the memo that is the same on every run with routing off."""
    sv = memo.sector_agent_view
    hints = _overlay_hints(sv)
    cohort_points = [p for p in sv.key_points if p not in hints]
    return {
        "sector_agent_view": {
            "agent": sv.agent, "headline": sv.headline, "summary": sv.summary,
            "cohort_key_points": cohort_points, "sources": sv.sources,
            "data_keys": sorted(sv.data.keys()) if isinstance(sv.data, dict) else None,
        },
        "extra_agent_views": sorted(memo.extra_agent_views.keys()),
        "degraded_agents": [a for a in memo.degraded_agents if a in _BLAST_RADIUS],
        "degradation_events": [e for e in memo.degradation_events if e["agent"] in _BLAST_RADIUS],
        "rating_label": memo.rating_label,
    }


def test_routing_off_demo_memo_matches_the_base_commit_golden():
    """Golden captured at 3038e78 (before this slice) from the same demo
    run; with the flag off the roster, the sector prompt and the memo views
    must be exactly what they were. See the comment above `_BLAST_RADIUS`
    for what is compared and why each excluded field is excluded."""
    assert settings.enable_industry_analyst_routing is False
    golden = json.loads(GOLDEN.read_text())
    gsv = golden["sector_agent_view"]
    gsv["cohort_key_points"] = gsv.pop("key_points")[:_GOLDEN_N_COHORT_KEY_POINTS]
    golden["degraded_agents"] = [a for a in golden["degraded_agents"] if a in _BLAST_RADIUS]
    golden["degradation_events"] = [e for e in golden["degradation_events"] if e["agent"] in _BLAST_RADIUS]

    memo = graph.run_stock_memo("MSFT")
    projection = _invariant_projection(memo)
    assert projection == golden

    # The exclusion is bounded on all sides, so it cannot become a hole a
    # leak could sit in:
    sv = memo.sector_agent_view
    hints = _overlay_hints(sv)
    cohort_points = projection["sector_agent_view"]["cohort_key_points"]
    #  - the two sides agree on where the macro tail begins;
    assert len(cohort_points) == _GOLDEN_N_COHORT_KEY_POINTS
    #  - the excluded entries are a suffix, never interleaved;
    assert sv.key_points[:len(cohort_points)] == cohort_points
    #  - and every excluded entry is an overlay hint carried by this very
    #    memo, so nothing else can ride along in the part not compared.
    assert all(p in hints for p in sv.key_points[len(cohort_points):])

    assert "industry_group" not in memo.sector_agent_view.data
    assert ia.construction_count() == 0


def test_routing_off_leaves_no_trace_anywhere_in_the_memo_json():
    """The projection above names the fields this slice can reach. This
    names none of them: the whole serialised memo must not contain a single
    industry-group marker while the flag is off, so a leak into a field
    nobody thought to project is caught anyway."""
    assert settings.enable_industry_analyst_routing is False
    blob = graph.run_stock_memo("MSFT").model_dump_json()
    for marker in ("industry_group", "Industry Group", "industry_knowledge:", "IndustryAnalyst",
                   "industry group mandate", "gics:"):
        assert marker not in blob, marker
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
    assert ig["slug"] == il.slug("4510") and ig["state"] == "mapped"
    assert ig["name"] == ia.get_industry_analyst("4510").label
    assert ig["source_label"].startswith("research map (")
    assert ig["provider_industry"] and "sub_industry" not in ig
    assert ig["mapping_caveat"] == il.PUBLIC_MAPPING_CAVEAT
    assert [c["stage"] for c in finding.data["causal_chain"]] == [s["id"] for s in thesis_stages()]
    assert f"industry_knowledge:{il.slug('4510')}" in finding.sources
    # The sector analyst saw the same mapping and says so on its finding.
    assert memo.sector_agent_view.data["industry_group"]["slug"] == il.slug("4510")
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


@pytest.mark.parametrize("malformed", [
    [{"headline": "h"}],                      # a top-level JSON array
    ["causal_chain", "kpis_to_watch"],
    "a bare string",
])
def test_a_malformed_llm_response_degrades_instead_of_raising(monkeypatch, malformed):
    """REGRESSION: `run_industry_group_agent` read `.get` straight off
    whatever `chat_json` handed back. `chat_json` is annotated `dict | None`,
    but a provider in JSON mode can answer with a top-level array, and
    `list.get` is an AttributeError that failed the whole memo. Every other
    agent in this repo degrades to its deterministic fallback on a malformed
    response; this one now does too — and records WHICH malformation, rather
    than swallowing it or filing it as "no usable output"."""
    monkeypatch.setattr(ia.llm, "chat_json", lambda *a, **k: malformed)
    monkeypatch.setattr(type(settings), "has_llm", property(lambda self: True))
    row = ic.current_for(["MSFT"])["MSFT"]

    log = safe_runner.DegradationLog()
    with log.activate():
        finding = ia.run_industry_group_agent({"ticker": "MSFT", "company_name": "Microsoft"},
                                              {"ROIC": 0.5}, classification=row)

    assert finding.agent == "Industry Group Analyst"
    assert finding.confidence == 0.55                      # the deterministic read
    assert finding.data["industry_group"]["slug"] == il.slug("4510")
    shape = type(malformed).__name__
    assert finding.data["deterministic_fallback"] == (
        f"Industry Group LLM returned a JSON {shape}, not an object; "
        "mandate-grounded deterministic read shipped instead."
    )
    # Degradation recorded on the run, not swallowed.
    assert log.events() == [{
        "agent": "Industry Group Analyst", "error_type": "DeterministicFallback",
        "message": f"Industry Group LLM returned a JSON {shape}, not an object; deterministic read shipped",
    }]


def test_an_empty_llm_response_still_reports_the_original_outcome(monkeypatch):
    """The wrong-shape wording must not leak onto the pre-existing path: a
    model that answers with nothing is a different degradation from one that
    answers with the wrong container."""
    monkeypatch.setattr(ia.llm, "chat_json", lambda *a, **k: None)
    monkeypatch.setattr(type(settings), "has_llm", property(lambda self: True))
    row = ic.current_for(["MSFT"])["MSFT"]
    finding = ia.run_industry_group_agent({"ticker": "MSFT", "company_name": "Microsoft"},
                                          {"ROIC": 0.5}, classification=row)
    assert finding.data["deterministic_fallback"].startswith(
        "Industry Group LLM returned no usable output;")


# --- the PM dialog can reach the new analyst ----------------------------------------


def test_pm_critique_can_target_the_industry_analyst_and_the_refire_lands(monkeypatch):
    """`deep_research` hard-coded the eight legacy specialist keys, so a PM
    question aimed at `industry_group` was silently dropped and
    `run_industry_group_agent`'s `prior_round_critique` path was
    unreachable from a memo run. The accepted set is now the roster."""
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    round0 = {
        "sector": AgentFinding(agent="Sector Analyst", headline="h", summary="s"),
        "industry_group": AgentFinding(agent="Industry Group Analyst", headline="h", summary="s"),
    }
    prompts_seen: list[str] = []

    def fake_chat_json(prompt, **kwargs):
        prompts_seen.append(prompt)
        return {"questions": [
            {"target_agent": "industry_group", "question": "Which sub-industry brief applies?",
             "why_it_matters": "placement drives the KPI set"},
            {"target_agent": "not_a_specialist", "question": "q", "why_it_matters": "w"},
        ], "no_further_questions": False, "rationale": "r"}

    monkeypatch.setattr(dr.llm, "chat_json", fake_chat_json)
    out = dr.pm_critique(round_num=0, current_findings=round0, rounds_so_far=[], run_id="t")
    # The roster key is offered to the model and accepted back; a key that
    # is not on the roster is still rejected.
    assert "industry_group" in prompts_seen[0]
    assert [q.target_agent for q in out.questions] == ["industry_group"]

    # …and the question reaches the analyst's own runner as its critique.
    asked: list[str] = []

    def refire(q: str) -> AgentFinding:
        asked.append(q)
        return ia.run_industry_group_agent(
            {"ticker": "NVDA", "company_name": "NVIDIA"}, {"roic": 0.51},
            prior_round_critique=q, classification=ic.current_for(["NVDA"])["NVDA"],
        )

    current, rounds = dr.run_dialog_loop(
        run_id="t", initial_findings=round0,
        re_fire={"sector": lambda q: round0["sector"], "industry_group": refire},
        max_rounds=1,
    )
    assert asked == ["Which sub-industry brief applies?"]
    assert rounds[1].pm_questions[0].target_agent == "industry_group"
    assert current["industry_group"].agent == "Industry Group Analyst"


def test_the_intake_prompt_and_the_critique_targets_come_from_the_roster():
    """Neither list is spelled out: adding a roster entry must not require
    editing prose in two other modules (it did, and the analyst was lost)."""
    assert intake.ALL_SPECIALISTS == [spec.key for spec in roster.AGENTS]
    assert "industry_group" in intake.ALL_SPECIALISTS
    assert dr._addressable({}) == tuple(intake.ALL_SPECIALISTS)
    assert dr._addressable({"macro": None, "sector": None}) == ("sector", "macro")  # roster order


def test_intake_is_offered_only_the_specialists_this_run_will_run(monkeypatch):
    """REGRESSION: the intake prompt listed the WHOLE roster, so with
    ENABLE_INDUSTRY_ANALYST_ROUTING off the PM was still offered
    `industry_group` — an agent `roster.applicable` had already excluded.
    One of the three skips could be spent on it (letting a specialist the
    PM wanted deprioritized run anyway) and the memo's `intake_decision`
    audit line named an agent that was never on the run."""
    seen: dict[str, str] = {}

    def fake_chat_json(prompt, **kwargs):
        seen["prompt"] = prompt
        return {"skip": ["industry_group", "technical"], "rationale": "r"}

    monkeypatch.setattr(intake.llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(intake.settings, "openai_api_key", "sk-test")
    offered = [k for k in intake.ALL_SPECIALISTS if k != "industry_group"]
    decision = intake.run_intake({"ticker": "MSFT"}, specialists=offered)
    assert "industry_group" not in seen["prompt"]
    assert f"{len(offered)} specialists are available" in seen["prompt"]
    # A key the model names anyway is not on this run, so it cannot consume
    # one of the three skips.
    assert decision.skipped == {"technical"}
    # No argument still means the whole roster.
    intake.run_intake({"ticker": "MSFT"})
    assert "industry_group" in seen["prompt"]


def test_the_memo_hands_intake_exactly_this_runs_roster(monkeypatch):
    """The flag is a no-op end to end: with routing off the PM never sees
    the industry analyst; with it on, it does."""
    seen: list[list[str]] = []

    def spy(profile, news_alerts=None, **kwargs):
        seen.append(list(kwargs.get("specialists") or []))
        return intake.IntakeDecision()

    monkeypatch.setattr(intake, "run_intake", spy)
    graph.run_stock_memo("MSFT")
    assert seen[-1] == [k for k in intake.ALL_SPECIALISTS if k != "industry_group"]
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    graph.run_stock_memo("MSFT")
    assert seen[-1] == intake.ALL_SPECIALISTS


# --- the runner on its own -----------------------------------------------------------


def test_run_industry_group_agent_deterministic_read_is_mandate_grounded():
    row = ic.current_for(["NVDA"])["NVDA"]
    profile = {"ticker": "NVDA", "company_name": "NVIDIA", "sector": "Technology"}
    finding = ia.run_industry_group_agent(profile, {"roic": 0.51, "ev_ebitda": 30.8}, classification=row)
    analyst = ia.get_industry_analyst("4530")
    assert finding.headline.startswith(f"{analyst.label}: mandate read for NVDA")
    assert "Deterministic edition" in finding.summary
    # Still grounded in the row's own sub-industry brief — matched on the
    # internal code, shown without it.
    brief = next(s for s in analyst.mandate.sub_industries if s.code == row["sub_industry_code"])
    assert brief.economics in finding.summary and row["sub_industry_code"] not in finding.summary
    assert finding.data["industry_group"]["provider_industry"] == row["source_industry"]
    core = set(analyst.mandate.items("core_kpis"))
    assert finding.data["kpis_to_watch"] and all(k["kpi"] in core and "industry_code" not in k
                                                for k in finding.data["kpis_to_watch"])
    assert finding.data["causal_chain"][0]["text"].startswith("n/a:")
    assert "deterministic_fallback" not in finding.data  # no LLM configured: the design, not a degradation
    assert finding.evidence[0].ref == f"industry_group:{analyst.slug}"
    assert finding.evidence[0].excerpt == analyst.label
    assert finding.sources == [f"industry_knowledge:{analyst.slug}"]


def test_run_industry_group_agent_looks_the_row_up_when_not_given():
    finding = ia.run_industry_group_agent({"ticker": "NVDA"}, {})
    assert finding.data["industry_group"]["slug"] == il.slug("4530")


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


# --- routing ON: the PM reads the routed read (S13, owner decision 2026-09-24) ------
#
# The PM synthesis reads findings as ONE JSON blob cut at
# `max_agent_context_chars`, built in roster order, and the Industry Group
# Analyst is the roster's tail: on JPM's demo memo its entry started at offset
# 62,791 against the 60,000 cap, and on every stored live memo (META v1's
# views alone are 86,518 chars) it is never read. The PM now reads a bounded
# digest ahead of the JSON instead (C7), and a stand-in read not at all.

# An LLM-shaped read: what the analyst returns when its model answers. The
# summary says "premium" on purpose — the deterministic PM heuristic counts
# that word as a bullish vote, so a leak into the heuristic shows up.
_ROUTED_READ: dict = {
    "headline": "Deposit franchise sets this bank's rate leverage",
    "summary": "Net interest margin follows deposit beta more than loan growth; "
               "the funding franchise earns a premium on cost of deposits.",
    "key_points": [f"Routed point {n}" for n in range(1, 8)],
    "confidence": 0.7,
    "mandate_type": "compounder",
    "placement": "Money-centre franchise with a low-beta deposit base.",
    "causal_chain": [{"stage": s["id"], "text": f"Routed stage {s['id']}"} for s in thesis_stages()],
    "kpis_to_watch": [{"kpi": "Deposit beta", "why": "drives net interest margin"}],
    "falsifiers": ["Deposit beta above peers", "Loan losses outrun reserves",
                   "Fee income stalls", "A fourth falsifier the digest leaves out"],
    "traps": ["Reserve releases flatter EPS"],
}
_DIGEST_HEAD = "## Industry group read"
_JSON_ENTRY = '"industry_group": {"agent"'  # the finding as a JSON entry, not the bare key
_FINDINGS = "\n\nFindings:\n"


def _routed_finding(monkeypatch, ticker: str = "JPM", **overrides) -> AgentFinding:
    """The analyst's real LLM path, with its model stubbed to `_ROUTED_READ`."""
    read = {**_ROUTED_READ, **overrides}
    with monkeypatch.context() as m:
        m.setattr(ia.llm, "chat_json", lambda *a, **k: dict(read))
        return ia.run_industry_group_agent({"ticker": ticker, "company_name": ticker}, {},
                                           classification=ic.current_for([ticker])[ticker])


def _spy_llm(monkeypatch, *, industry_read: dict | None = None, pm_reply: dict | None = None) -> dict:
    """Capture the PM Synthesis and Risk Committee prompts, and answer the
    Industry Group Analyst's call with `industry_read` when given. Anything
    else goes to the real (keyless, demo) client."""
    from app.agents import llm as llm_mod
    seen: dict[str, list] = {"pm": [], "critic": [], "industry": []}
    # A second spy in the same test must wrap the client, not the first spy.
    real = getattr(llm_mod.chat_json, "_unspied", llm_mod.chat_json)

    def spy(prompt, **kwargs):
        if prompt.startswith(prompts.PM_SYNTHESIS_PROMPT):
            seen["pm"].append(prompt)
            return dict(pm_reply) if pm_reply else None
        if prompt.startswith(prompts.CRITIC_PROMPT):
            seen["critic"].append(prompt)
            return None
        if llm_mod.current_call_context().get("agent_name") == ia.AGENT_NAME:
            seen["industry"].append(kwargs)
            if industry_read is not None:
                return dict(industry_read)
        return real(prompt, **kwargs)

    spy._unspied = real  # type: ignore[attr-defined]
    monkeypatch.setattr(llm_mod, "chat_json", spy)
    return seen


_PROFILE = {"ticker": "JPM", "sector": "Financials"}
_PM_REPLY = {"rating_label": "Neutral", "confidence_score": 50, "final_pm_view": "v",
             "one_sentence_thesis": "t"}


def test_pm_synthesis_reads_the_routed_read_ahead_of_the_capped_json(monkeypatch):
    cap = settings.max_agent_context_chars
    routed = _routed_finding(monkeypatch)
    findings = {
        # A sector finding alone overflows the cap, as on every live memo.
        "sector": AgentFinding(agent="Sector Analyst", headline="Sector read", summary="s",
                               data={"research": "x" * (cap + 5_000)}),
        "industry_group": routed,
    }
    seen = _spy_llm(monkeypatch, pm_reply=_PM_REPLY)
    graph._pm_synthesis(_PROFILE, findings, None)
    (prompt,) = seen["pm"]

    assert prompt.index(_DIGEST_HEAD) < prompt.index(_FINDINGS)
    assert prompt.index(routed.headline) < prompt.index(_FINDINGS)
    label = routed.data["industry_group"]["label"]
    assert f"{_DIGEST_HEAD} — {label} (Industry Group Analyst)" in prompt
    # The capped JSON is still the cap. (Its content cannot show whether the
    # read was left out: the padded sector entry fills it before any entry
    # after it. The unpadded case below does.)
    assert len(prompt.split(_FINDINGS, 1)[1]) == cap

    # On a small memo, where the whole JSON fits, the read is still never a
    # JSON entry: it would be read twice, once as the digest and once raw.
    small = {"sector": AgentFinding(agent="Sector Analyst", headline="Sector read", summary="s"),
             "industry_group": routed}
    graph._pm_synthesis(_PROFILE, small, None)
    prompt = seen["pm"][-1]
    assert prompt.index(_DIGEST_HEAD) < prompt.index(_FINDINGS)
    assert _JSON_ENTRY not in prompt
    assert prompt.count(routed.headline) == 1


def test_a_digested_read_casts_no_keyword_vote_in_the_no_llm_rating(monkeypatch):
    """Design §3.2: the no-LLM keyword heuristic counts only the JSON set.
    A routed read reaches the PM model as its digest, but it must not move
    the deterministic rating: `_ROUTED_READ.summary` says "premium", a
    bullish keyword, so a digested read that voted would lift a Neutral
    fallback (it did on the base commit: Neutral 55 → Bullish 60)."""
    routed = _routed_finding(monkeypatch)
    assert ia.pm_digest(routed)                     # digested, not withheld
    sector = AgentFinding(agent="Sector Analyst", headline="Sector read", summary="Neutral.")
    _spy_llm(monkeypatch)                           # the PM model answers nothing
    alone = graph._pm_synthesis(_PROFILE, {"sector": sector}, None)
    with_read = graph._pm_synthesis(_PROFILE, {"sector": sector, "industry_group": routed}, None)
    assert (with_read["rating_label"], with_read["confidence_score"]) == (
        alone["rating_label"], alone["confidence_score"])


def test_pm_digest_is_bounded_and_shaped(monkeypatch):
    # Per-field bounds, checked against literal fixture content rather than
    # against pm_digest's own output: at most 5 key points, at most 3
    # falsifiers, the summary clipped to 600 characters.
    digest = ia.pm_digest(_routed_finding(monkeypatch))
    assert "Routed point 5" in digest and "Routed point 6" not in digest
    assert "Fee income stalls" in digest and "A fourth falsifier" not in digest
    clipped = ia.pm_digest(_routed_finding(monkeypatch, summary="w" * 1_000, key_points=["a"]))
    (summary_line,) = [ln for ln in clipped.splitlines() if ln.startswith("Summary: ")]
    assert len(summary_line) == len("Summary: ") + 600 and summary_line.endswith("…")
    assert len(clipped) < ia.PM_DIGEST_MAX_CHARS

    long = AgentFinding(
        agent=ia.AGENT_NAME, headline="H" * 300, summary="S" * 5_000,
        key_points=["k" * 400] * 8, confidence=0.7,
        data={"industry_group": {"label": "Lbl"}, "mandate_type": "inflection",
              "falsifiers": ["f" * 300] * 5},
    )
    digest = ia.pm_digest(long)
    assert len(digest) == ia.PM_DIGEST_MAX_CHARS and digest.endswith("…")
    assert digest.startswith(f"{_DIGEST_HEAD} — Lbl (Industry Group Analyst)\nHeadline: ")
    short = ia.pm_digest(long.model_copy(update={
        "headline": "Head", "summary": "Short.", "key_points": ["a", "b"],
        "data": {**long.data, "falsifiers": ["x", "y"]},
    }))
    assert short == ("## Industry group read — Lbl (Industry Group Analyst)\nHeadline: Head\n"
                     "Summary: Short.\n- a\n- b\nMandate type: inflection\nFalsifiers: x; y")
    # A read that says nothing is withheld rather than sent as an empty header.
    assert ia.pm_digest(long.model_copy(update={"headline": " ", "summary": ""})) == ""


def test_pm_prompt_is_byte_identical_without_a_digest_spec(monkeypatch):
    """With no digest spec in the findings (routing off, or an unmapped
    company) the PM's input is exactly the pre-change expression."""
    findings = {
        "sector": AgentFinding(agent="Sector Analyst", headline="Sector read", summary="Neutral."),
        "earnings": AgentFinding(agent="Earnings Analyst", headline="Beat", summary="Guidance held."),
    }
    view = graph._pm_view(findings)
    assert view.digests == [] and view.findings == findings and view.visible == findings
    assert list(view.findings) == ["sector", "earnings"]

    from app.agents import pm_context
    monkeypatch.setattr(pm_context, "build_pm_context", lambda **kw: "PM-CONTEXT")
    seen = _spy_llm(monkeypatch, pm_reply=_PM_REPLY)
    graph._pm_synthesis(_PROFILE, findings, None)
    expected = (
        prompts.PM_SYNTHESIS_PROMPT
        + "\n\nPM-CONTEXT"
        + "\n\nFindings:\n"
        + json.dumps({k: v.model_dump() for k, v in findings.items()}, default=str)[
            : settings.max_agent_context_chars]
    )
    assert seen["pm"] == [expected]


@pytest.mark.parametrize("pm_ctx", ["PM-CONTEXT", ""])
def test_pm_prompt_assembly_order(monkeypatch, pm_ctx):
    """Contract C7: template + pm_ctx, then the industry digest, then the
    capped JSON. Later slices insert their blocks between the digest and
    "Findings:"; this pins the order they extend."""
    routed = _routed_finding(monkeypatch)
    sector = AgentFinding(agent="Sector Analyst", headline="Sector read", summary="Neutral.")
    from app.agents import pm_context
    monkeypatch.setattr(pm_context, "build_pm_context", lambda **kw: pm_ctx)
    seen = _spy_llm(monkeypatch, pm_reply=_PM_REPLY)
    graph._pm_synthesis(_PROFILE, {"sector": sector, "industry_group": routed}, None)
    digest = ia.pm_digest(routed)
    assert digest
    assert seen["pm"] == [
        prompts.PM_SYNTHESIS_PROMPT
        + (f"\n\n{pm_ctx}" if pm_ctx else "")
        + "\n\n" + digest
        + "\n\nFindings:\n"
        + json.dumps({"sector": sector.model_dump()}, default=str)[: settings.max_agent_context_chars]
    ]


_WITHHELD = {
    "no_mapping": True,
    "deterministic_fallback": "Industry Group LLM returned no usable output; mandate-grounded "
                              "deterministic read shipped instead.",
    "degraded": True,
    "intake_skipped": True,
}


@pytest.mark.parametrize("flag", sorted(_WITHHELD))
def test_withheld_industry_reads_never_reach_the_pm_or_influence(monkeypatch, flag):
    """A stand-in read (the flags `memo_sections.finding_is_template` reads)
    must not reach the rating dressed as an analyst view: no digest, no JSON
    entry, no keyword vote in the no-LLM fallback, and no stored pull in
    `agent_influence`."""
    routed = _routed_finding(monkeypatch)
    withheld = routed.model_copy(update={
        "summary": routed.summary + " Constructive tailwind, premium outperform.",
        "data": {**routed.data, flag: _WITHHELD[flag]},
    })
    assert ia.pm_digest(withheld) == ""

    # The deterministic PM path: the rating is what it is without the read.
    base = {"sector": AgentFinding(agent="Sector Analyst", headline="Sector read", summary="Neutral.")}
    seen = _spy_llm(monkeypatch)
    alone = graph._pm_synthesis(_PROFILE, dict(base), None)
    with_it = graph._pm_synthesis(_PROFILE, {**base, "industry_group": withheld}, None)
    assert (with_it["rating_label"], with_it["confidence_score"]) == (
        alone["rating_label"], alone["confidence_score"])
    assert len(seen["pm"]) == 2
    assert all(_DIGEST_HEAD not in p and _JSON_ENTRY not in p for p in seen["pm"])

    # End to end: stored on the memo as history, absent from its influence.
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    monkeypatch.setattr(roster, "run_industry_group_agent", lambda *a, **k: withheld)
    memo = graph.run_stock_memo("JPM")
    assert "industry_group" in memo.extra_agent_views
    assert "industry_group" not in memo.agent_influence
    assert len(seen["pm"]) == 3 and _JSON_ENTRY not in seen["pm"][-1]
    assert _DIGEST_HEAD not in seen["pm"][-1]


def test_routed_demo_memo_reaches_the_pm_and_degrades_nothing(monkeypatch):
    """Routing happens in demo mode too, and no keys is the design there,
    not a degradation. With the analyst's model answering, its read reaches
    the PM as the digest and is scored for influence; the keyless
    deterministic mandate read (a template) is stored but withheld."""
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    seen = _spy_llm(monkeypatch, industry_read=_ROUTED_READ)
    memo = graph.run_stock_memo("JPM")

    assert seen["industry"], "the Industry Group Analyst's model was never asked"
    (prompt,) = seen["pm"]
    label = ia.analyst_for_classification(ic.current_for(["JPM"])["JPM"]).label
    assert prompt.index(f"{_DIGEST_HEAD} — {label}") < prompt.index(_FINDINGS)
    assert _ROUTED_READ["headline"] in prompt and _JSON_ENTRY not in prompt
    assert ia.construction_count() == 1
    assert memo.extra_agent_views["industry_group"].headline == _ROUTED_READ["headline"]
    assert memo.sector_agent_view.data["industry_group"]["label"] == label
    assert "industry_group" in memo.agent_influence
    assert not [a for a in memo.degraded_agents if "Industry Group" in a]

    seen = _spy_llm(monkeypatch)                  # keyless: the deterministic mandate read
    memo = graph.run_stock_memo("JPM")
    (prompt,) = seen["pm"]
    assert _DIGEST_HEAD not in prompt and _JSON_ENTRY not in prompt
    assert "industry_group" in memo.extra_agent_views
    assert "industry_group" not in memo.agent_influence
    assert not [a for a in memo.degraded_agents if "Industry Group" in a]


def _lookup_raises(monkeypatch):
    def boom(ticker):
        raise RuntimeError("classification read failed")
    monkeypatch.setattr(ia, "lookup_classification", boom)


def _factory_raises(monkeypatch):
    def boom(code, **kwargs):
        raise reg.UnknownNode(f"no node {code}")
    monkeypatch.setattr(ia, "get_industry_analyst", boom)


@pytest.mark.parametrize("break_it", [_lookup_raises, _factory_raises], ids=["lookup-coverage", "factory"])
def test_routing_failures_degrade_the_memo_instead_of_failing_it(monkeypatch, break_it):
    """The two failure paths no test covered: the gather-stage read raises,
    or the factory cannot build the analyst (a code the registry or the
    knowledge base does not carry). The memo completes, the banner names the
    analyst, and nothing of it reaches the PM or the influence map.

    `factory` is the regression case and fails on the base commit.
    `lookup-coverage` is a
    characterization of behaviour S13 does not change (a raising lookup
    never produced an industry finding, so it passes on the base commit
    too); it pins that path now that routing is on in production."""
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    break_it(monkeypatch)
    seen = _spy_llm(monkeypatch)
    memo = graph.run_stock_memo("JPM")

    assert memo.sector_agent_view.confidence > 0.0
    assert "Industry Group Analyst" in memo.degraded_agents
    (prompt,) = seen["pm"]
    assert _DIGEST_HEAD not in prompt and _JSON_ENTRY not in prompt
    assert "industry_group" not in memo.agent_influence
    if break_it is _factory_raises:
        assert memo.sector_agent_view.data["industry_group"]["error"] == "industry group block unavailable"
        assert memo.extra_agent_views["industry_group"].data.get("degraded") is True


def test_a_digest_that_raises_withholds_the_read_not_the_synthesis(monkeypatch):
    """`_pm_synthesis` runs under `safe_call`; a digest bug must cost the PM
    one read, not ship the "synthesis unavailable" fallback for the memo."""
    import dataclasses

    def boom(finding):
        raise ValueError("digest bug")

    tail = dataclasses.replace(roster.AGENTS[-1], pm_digest=boom)
    monkeypatch.setattr(roster, "AGENTS", (*roster.AGENTS[:-1], tail))
    routed = _routed_finding(monkeypatch)
    sector = AgentFinding(agent="Sector Analyst", headline="Sector read", summary="Neutral.")
    view = graph._pm_view({"sector": sector, "industry_group": routed})
    assert view.digests == [] and list(view.findings) == ["sector"] and list(view.visible) == ["sector"]
    seen = _spy_llm(monkeypatch, pm_reply=_PM_REPLY)
    assert graph._pm_synthesis(_PROFILE, {"sector": sector, "industry_group": routed}, None) == _PM_REPLY
    assert len(seen["pm"]) == 1 and _DIGEST_HEAD not in seen["pm"][0]


def test_industry_analyst_call_has_output_headroom(monkeypatch):
    """The analyst's schema is the largest any specialist is asked for; at
    `chat_json`'s 1,600 default it truncates into a deterministic fallback."""
    kwargs_seen: list[dict] = []

    def spy(prompt, **kwargs):
        kwargs_seen.append(kwargs)
        return dict(_ROUTED_READ)

    monkeypatch.setattr(ia.llm, "chat_json", spy)
    ia.run_industry_group_agent({"ticker": "JPM"}, {}, classification=ic.current_for(["JPM"])["JPM"])
    assert [k.get("max_tokens") for k in kwargs_seen] == [2400]


def test_no_report_prompt_line_trips_the_validator_l1_rule():
    """The weekly report writer sends each group analyst's system prompt
    (and the report rules) to the model, and the validator's L1 rule
    rejects any registry name the model echoes into new prose. The
    Property REITs mandate named its sub-industries ("Office REITs lease
    space ...") in the taxonomy's own capitalisation, so every edition that
    repeated its own mandate was rejected. Every line, every group."""
    from app.agents import industry_report_validator as v
    from app.prompts import load_prompt

    offending: list[tuple[str, list[str], str]] = []
    for group in reg.industry_groups():
        for line in ia.get_industry_analyst(group.code).system_prompt().splitlines():
            if leaks := v.taxonomy_leaks(line, group_code=group.code):
                offending.append((group.code, leaks, line[:120]))
    for line in (load_prompt("industry_report") or "").splitlines():
        if leaks := v.taxonomy_leaks(line):
            offending.append(("rules", leaks, line[:120]))
    assert offending == []
    # The description survives: the Property REITs mandate still says what
    # an office REIT does, in plain words.
    reits = ia.get_industry_analyst("6010").system_prompt()
    assert "office REITs lease space" in reits and "Office REITs" not in reits
