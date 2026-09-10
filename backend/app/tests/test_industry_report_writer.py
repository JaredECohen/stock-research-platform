"""FEAT-003 slice 2 — the report writer's deterministic edition.

Under the test harness there is no LLM, so every edition here is the
deterministic one: thirteen sections in the frozen order, facts built by
the server, an interpretation that passes the validator, drivers ordered
by the eight stages loaded from the knowledge base, `n/a` with a reason
wherever an input is absent, and a `generation` block whose cost comes
from `llm_metrics.estimate_cost_usd` (zero calls, zero dollars).
"""
from __future__ import annotations

from datetime import datetime

import pytest

from app.agents import industry_analysts as ia
from app.agents import industry_report_validator as v
from app.agents import industry_report_writer as w
from app.services import gics_registry as reg
from app.services.industry_group_knowledge import thesis_stages
from app.tests.gating_helpers import seed_demo_universe


@pytest.fixture(scope="module", autouse=True)
def _taxonomy():
    seed_demo_universe()
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None
    yield info


@pytest.fixture()
def analyst():
    ia.clear_cache()
    return ia.get_industry_analyst("4510")


@pytest.fixture(autouse=True)
def _pin_clock(monkeypatch):
    monkeypatch.setattr(w, "_utcnow", lambda: datetime(2026, 9, 6, 6, 45, 0))


def _stats() -> dict:
    return {
        "id": 7, "period_key": "2026-W36", "as_of": datetime(2026, 9, 4, 20, 0), "inputs_hash": "h1",
        "method": {"weighting": ["equal", "market_cap"], "horizons": ["1W", "1M", "QTD", "YTD", "1Y"],
                   "benchmarks": [{"id": "universe_ew"}, {"id": "sector_ew"}, {"id": "KFR.MKT_RF.D"}],
                   "min_sample": 3},
        "sample": {"n_constituents": 4, "n_with_prices": 3, "tickers": ["MSFT", "AAPL", "AMD", "ORCL"],
                   "excluded": [{"ticker": "ORCL", "reason": "no_prices"}]},
        "payload": {
            "status": "ok",
            "returns": {"1W": {"equal_weight": 0.0123, "market_cap_weight": 0.011, "n": 3},
                        "1M": {"equal_weight": -0.031, "market_cap_weight": -0.02, "n": 3},
                        "1Y": {"value": None, "reason": "history_window"}},
            "largest": [{"ticker": "MSFT", "market_cap": 3.1e12}],
            "leaders": [{"ticker": "MSFT", "ret_1m": 0.05}], "laggards": [{"ticker": "AMD", "ret_1m": -0.06}],
            "valuation": {"ev_ebitda": {"median": 26.0, "n": 3}},
        },
        "per_ticker": {"MSFT": {"ret_1m": 0.05}, "AAPL": {"ret_1m": -0.02}, "AMD": {"ret_1m": -0.06}},
    }


def _facts_of(payload: dict) -> dict:
    return {name: s["facts"] for name, s in payload["sections"].items()}


def test_no_llm_writes_thirteen_sections_and_passes_the_validator(analyst):
    res = w.write_report(analyst, _stats(), None, None, [], run_id="run-1")
    payload = res.payload
    assert list(payload["sections"]) == list(v.SECTION_ORDER)
    assert len(payload["sections"]) == 13
    assert payload["section_order"] == list(v.SECTION_ORDER)
    assert payload["analyst_narrative"] == "llm_unavailable"
    assert "analyst_narrative:llm_unavailable" in res.degraded
    assert res.errors == []
    assert v.validate(payload, _facts_of(payload)) == []
    for name in v.INTERPRETED_SECTIONS:
        assert payload["sections"][name]["interpretation"]["text"]
    for name in ("statistics", "sources", "metadata"):
        assert payload["sections"][name]["interpretation"] is None
    assert "research and education" in payload["disclaimer"].lower()


def test_facts_come_from_the_inputs_and_carry_provenance(analyst):
    payload = w.write_report(analyst, _stats(), None, None, [], run_id="run-2").payload
    facts = _facts_of(payload)
    assert facts["overview"]["code"] == "4510" and facts["overview"]["n_constituents"] == 4
    assert [b["code"] for b in facts["overview"]["boundaries"]] == list(analyst.mandate.industry_codes)
    assert facts["performance"]["returns"]["1M"]["equal_weight"] == -0.031
    assert facts["performance"]["returns"]["1Y"] == {"value": None, "reason": "history_window"}
    assert facts["performance"]["benchmarks"][2]["id"] == "KFR.MKT_RF.D"
    assert facts["statistics"]["inputs_hash"] == "h1"
    assert facts["companies"]["constituents"] == ["AAPL", "AMD", "MSFT"]
    assert facts["kpis"]["core_kpis"][0]["industry_codes"]
    assert facts["metadata"]["generated_at"] == "2026-09-06T06:45:00"
    assert facts["metadata"]["run_id"] == "run-2"
    assert facts["metadata"]["generation_mode"] == "llm_unavailable"
    manifest_kinds = {m["kind"] for m in facts["sources"]["manifest"]}
    assert {"knowledge_base", "universe_map", "statistics", "benchmark"} <= manifest_kinds
    assert facts["sources"]["primary_sources"]
    assert facts["sources"]["attribution"] == analyst.mandate.attribution


def test_drivers_follow_the_eight_stages_and_open_on_the_world_change(analyst):
    payload = w.write_report(analyst, _stats(), None, None, [], run_id="run-3").payload
    drivers = payload["sections"]["drivers"]
    expected = [s["id"] for s in thesis_stages()]
    assert [s["id"] for s in drivers["facts"]["stage_order"]] == expected
    assert [s["id"] for s in drivers["interpretation"]["stages"]] == expected
    first = drivers["interpretation"]["stages"][0]
    assert first["id"] == "world_change"
    assert first["text"].startswith("n/a:")  # honest: no dated change asserted
    assert not v.opens_with_kpi_forecast(first["text"])
    # Every mandate mechanism quoted with a causal marker is a registered claim.
    for claim in drivers["interpretation"]["claims"]:
        assert claim["type"] in v.CLAIM_TYPES
    assert not any("drivers" in e for e in v.validate(payload, _facts_of(payload)))


def test_missing_statistics_render_as_na_with_reasons(analyst):
    res = w.write_report(analyst, None, None, None, [], run_id="run-4")
    facts = _facts_of(res.payload)
    assert "statistics:missing" in res.degraded
    assert facts["performance"]["status"] == "unavailable"
    for cell in facts["performance"]["returns"].values():
        assert cell["value"] is None and cell["reason"]
    assert facts["outlook"]["expectations_ledger"]["price_implied"]["value"] is None
    assert facts["what_changed"]["facts_delta"]["value"] is None
    text = res.payload["sections"]["performance"]["interpretation"]["text"]
    assert "unavailable" in text and "No return figure" in text
    assert v.validate(res.payload, facts) == []


def test_insufficient_sample_is_a_labelled_state_not_an_error(analyst):
    stats = _stats()
    stats["payload"] = {"status": "insufficient_sample", "reason": "n=2 below min_sample 3"}
    stats["per_ticker"] = {"MSFT": {"ret_1m": None}, "AAPL": {"ret_1m": None}}
    res = w.write_report(analyst, stats, None, None, [], run_id="run-5")
    facts = _facts_of(res.payload)
    assert facts["performance"]["status"] == "insufficient_sample"
    assert facts["performance"]["reason"] == "n=2 below min_sample 3"
    assert "performance:insufficient_sample" in res.degraded
    assert res.errors == []
    assert v.validate(res.payload, facts) == []


def test_prior_report_produces_a_facts_delta_and_constituent_changes(analyst):
    first = w.write_report(analyst, _stats(), None, None, [], run_id="run-6a")
    prior = {"payload": first.payload, "version": 1, "as_of": datetime(2026, 8, 28, 20, 0),
             "period_key": "2026-W35"}
    stats = _stats()
    stats["payload"]["returns"]["1M"]["equal_weight"] = 0.019
    stats["per_ticker"]["NVDA"] = {"ret_1m": 0.01}
    del stats["per_ticker"]["AMD"]
    second = w.write_report(analyst, stats, None, prior, [], run_id="run-6b")
    wc = _facts_of(second.payload)["what_changed"]
    assert wc["prior_version"] == 1 and wc["prior_period_key"] == "2026-W35"
    delta = wc["facts_delta"]["returns.1M.equal_weight"]
    assert delta == {"from": -0.031, "to": 0.019, "change": 0.05}
    assert wc["constituents"] == {"added": ["NVDA"], "removed": ["AMD"]}
    text = second.payload["sections"]["what_changed"]["interpretation"]["text"]
    assert "Versus edition 1" in text
    assert v.validate(second.payload, _facts_of(second.payload)) == []


def test_cross_industry_facts_label_each_dependency_with_its_source(analyst):
    snapshot = {"id": 3, "as_of": datetime(2026, 9, 4, 21, 0), "period_key": "2026-W36",
                "payload": {"regime": {"macro_regime": "late_cycle"},
                            "spillovers": [{"edge_id": "D01", "origin_code": "4530", "destination_code": "4510",
                                            "transmission": "x", "source": "atlas_dependencies_sheet"},
                                           {"edge_id": "D09", "origin_code": "2010", "destination_code": "1010"}]}}
    payload = w.write_report(analyst, _stats(), snapshot, None, [], run_id="run-7").payload
    ci = payload["sections"]["cross_industry"]["facts"]
    assert [s["edge_id"] for s in ci["spillovers"]] == ["D01"]
    for edge in ci["edges"]:
        assert edge["source"] and "4510" in edge["industry_group_codes"]
    for rel in ci["relationships"]:
        assert rel["source"] and "4510" in rel["industry_group_codes"]
    assert ci["snapshot"] == {"as_of": "2026-09-04T21:00:00", "period_key": "2026-W36", "id": 3}
    assert payload["sections"]["drivers"]["facts"]["sensitivities"]["macro_regime"] == "late_cycle"
    assert v.validate(payload, _facts_of(payload)) == []


def test_deterministic_mode_is_labelled_distinctly_from_llm_unavailable(analyst):
    res = w.write_report(analyst, _stats(), None, None, [], run_id="run-8", deterministic=True)
    assert res.payload["analyst_narrative"] == "deterministic"
    assert "analyst_narrative:deterministic_mode" in res.degraded
    assert res.generation["generation_mode"] == "deterministic"
    assert v.validate(res.payload, _facts_of(res.payload)) == []


def test_generation_block_reports_zero_calls_and_zero_cost_without_an_llm(analyst):
    res = w.write_report(analyst, _stats(), None, None, [], run_id="run-9")
    g = res.generation
    assert g["llm_calls"] == 0 and g["cost_usd"] == 0.0
    assert g["prompt_tokens"] == 0 and g["completion_tokens"] == 0
    assert g["run_id"] == "run-9" and g["max_llm_calls"] >= 1
    assert "cost_usd_run" not in g  # no calls, nothing to aggregate by run_id


def test_llm_sections_are_bounded_and_fall_back_per_section(analyst, monkeypatch):
    """With an LLM configured, at most `industry_report_max_llm_calls` calls
    are made; a section the model omits falls back to the deterministic
    text and is listed in `degraded`; cost is estimated per call."""
    calls: list[tuple[str, ...]] = []

    def fake_chat_json(prompt, **kwargs):
        sections = prompt.split("\n", 1)[0].removeprefix("Sections to interpret now: ").rstrip(".").split(", ")
        calls.append(tuple(sections))
        return {"overview": {"text": "The group has 4 constituents in the sample.",
                             "claims": [{"type": "observed_fact", "text": "4 constituents", "basis": ["overview"]}]}}

    monkeypatch.setattr(w.settings.__class__, "has_llm", property(lambda self: True))
    monkeypatch.setattr(w.llm, "_demo_only", lambda: False)
    monkeypatch.setattr(w.llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(w.llm, "last_usage", lambda: {"provider": "openai", "model": "gpt-4o-mini",
                                                       "input_tokens": 1000, "output_tokens": 500})
    monkeypatch.setattr(w, "cost_per_run", lambda run_id: {"cost_usd_total": 0.001, "n_calls": 2})
    res = w.write_report(analyst, _stats(), None, None, [], run_id="run-10")
    assert len(calls) == min(2, w.settings.industry_report_max_llm_calls)
    assert res.payload["analyst_narrative"] == "llm"
    assert res.payload["sections"]["overview"]["interpretation"]["text"].startswith("The group has 4")
    assert "analyst_narrative:drivers:deterministic" in res.degraded
    assert "analyst_narrative:llm_unavailable" not in res.degraded
    g = res.generation
    assert g["llm_calls"] == len(calls) and g["prompt_tokens"] == 1000 * len(calls)
    assert g["cost_usd"] == pytest.approx(
        w.estimate_cost_usd("openai", "gpt-4o-mini", 1000, 500) * len(calls), abs=1e-6,
    )
    assert g["cost_usd_run"] == 0.001 and g["llm_calls_run"] == 2
    assert v.validate(res.payload, _facts_of(res.payload)) == []
