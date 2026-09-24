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
        # Shaped like a real statistics row: priced names carry `last_close`,
        # and a name the row could not price is KEPT with an exclusion
        # reason rather than dropped. The old fixture had neither, so
        # `n_priced = len(per_ticker)` looked right here while the real
        # producer made it count unpriced names.
        "per_ticker": {
            "MSFT": {"ret_1m": 0.05, "last_close": 410.0},
            "AAPL": {"ret_1m": -0.02, "last_close": 222.0},
            "AMD": {"ret_1m": -0.06, "last_close": 150.0},
            "ORCL": {"ret_1m": None, "last_close": None, "exclusion": "no_prices"},
        },
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
    # Membership is the sample's constituent list, not the priced subset:
    # ORCL has no price series this period but has not left the group.
    assert facts["companies"]["constituents"] == ["AAPL", "AMD", "MSFT", "ORCL"]
    assert facts["companies"]["n_constituents"] == facts["overview"]["n_constituents"] == 4
    assert facts["companies"]["n_priced"] == 3
    assert facts["companies"]["unpriced"] == [{"ticker": "ORCL", "reason": "no_prices"}]
    assert facts["companies"]["membership_source"] == "statistics.sample.tickers"
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


def test_a_declared_horizon_with_no_return_names_why_never_a_bare_na(analyst):
    """REGRESSION: `method.horizons` declares five horizons and the payload
    carried three, so QTD and YTD rendered "QTD n/a (n/a)" — an n/a with no
    reason, which the house rule forbids. Three cases, three distinct
    answers: a horizon the payload never carried says it is absent, one
    that is null-with-a-reason quotes the reason, and one that is null with
    no reason at all says the row recorded none."""
    stats = _stats()
    # A cell that is present, null, and silent about why.
    stats["payload"]["returns"]["QTD"] = {"equal_weight": None, "n": 0}
    res = w.write_report(analyst, stats, None, None, [], run_id="run-12")
    text = res.payload["sections"]["performance"]["interpretation"]["text"]
    assert "n/a (n/a)" not in text
    assert "1Y n/a (history_window)" in text
    assert "YTD n/a (declared in method.horizons, absent from the statistics returns payload)" in text
    assert "QTD n/a (no return and no reason recorded on the statistics row)" in text
    assert v.validate(res.payload, _facts_of(res.payload)) == []


def test_prior_report_produces_a_facts_delta_and_constituent_changes(analyst):
    first = w.write_report(analyst, _stats(), None, None, [], run_id="run-6a")
    prior = {"payload": first.payload, "version": 1, "as_of": datetime(2026, 8, 28, 20, 0),
             "period_key": "2026-W35"}
    stats = _stats()
    stats["payload"]["returns"]["1M"]["equal_weight"] = 0.019
    # A real membership change: NVDA joins the group, AMD leaves it.
    stats["sample"]["tickers"] = ["MSFT", "AAPL", "NVDA", "ORCL"]
    stats["per_ticker"]["NVDA"] = {"ret_1m": 0.01}
    del stats["per_ticker"]["AMD"]
    second = w.write_report(analyst, stats, None, prior, [], run_id="run-6b")
    wc = _facts_of(second.payload)["what_changed"]
    assert wc["prior_version"] == 1 and wc["prior_period_key"] == "2026-W35"
    delta = wc["facts_delta"]["returns.1M.equal_weight"]
    assert delta == {"from": -0.031, "to": 0.019, "change": 0.05}
    assert wc["constituents"] == {"added": ["NVDA"], "removed": ["AMD"], "n_added": 1, "n_removed": 1}
    # The counts the interpretation quotes are facts, or the validator
    # rejects the sentence that quotes them.
    assert wc["n_facts_moved"] == sum(
        1 for v in wc["facts_delta"].values() if isinstance(v, dict) and v.get("change") not in (None, 0)
    )
    text = second.payload["sections"]["what_changed"]["interpretation"]["text"]
    assert "Versus edition 1" in text
    assert v.validate(second.payload, _facts_of(second.payload)) == []


def test_every_deterministic_causal_claim_carries_a_real_falsifier():
    """REGRESSION: `_register_causal` stamped every causal sentence it
    registered with "n/a: ... no dated falsifier in this edition", and the
    validator accepted it because any non-empty string counted as a
    falsifier. The deterministic edition now names the mandate's own tests
    — the observation that would show the quoted mechanism does not apply
    to this group. 2030 is in the list because its mandate is the one whose
    highest-EVI question carries a causal marker."""
    ia.clear_cache()
    seen = 0
    for code in ("2030", "4510", "4530"):
        a = ia.get_industry_analyst(code)
        res = w.write_report(a, _stats(), None, None, [], run_id=f"fals-{code}")
        assert v.validate(res.payload, _facts_of(res.payload)) == [], code
        for section in res.payload["sections"].values():
            interp = section.get("interpretation")
            for claim in (interp or {}).get("claims") or []:
                if claim["type"] == "causal_inference":
                    seen += 1
                    assert v.is_real_falsifier(claim["falsifier"]), (code, claim)
    assert seen, "no causal claim was registered, so the falsifier path went untested"


def test_a_causal_sentence_with_no_falsifier_is_withheld_not_stamped_na():
    """The other half of the rule: when there is no observation to offer,
    the claim is not registered — and the sentence is not printed either,
    because a mechanism the report cannot test does not belong in it. The
    omission is stated, not silent."""
    claims: list[dict] = []
    text = "Capacity returns because the capital cycle turned. Utilization is the KPI that tests it."
    out = w._register_causal(text, claims, ["drivers.value_capture"], "")
    assert claims == []
    assert "because the capital cycle turned" not in out
    assert "Utilization is the KPI that tests it." in out
    assert w.WITHHELD_CAUSAL_NOTE in out
    # The stand-in line must not itself trip the gate it exists to satisfy.
    assert not v._has_causal_marker(w.WITHHELD_CAUSAL_NOTE)
    assert not v.numeric_tokens(w.WITHHELD_CAUSAL_NOTE)
    # With an observation on offer the same sentence is kept and registered.
    claims = []
    kept = w._register_causal(text, claims, ["drivers.value_capture"],
                              "Utilization rises while the capital cycle is said to be turning.")
    assert kept == text
    assert [c["type"] for c in claims] == ["causal_inference"]


def test_scenario_falsifiers_name_an_observation_or_say_why_they_cannot(analyst):
    """REGRESSION: the bull scenario's falsifier list was the bare string
    "n/a" whenever the mandate listed no failure mode, and the other two
    hedged a real observation into nothing with "(n/a in this edition)"."""
    res = w.write_report(analyst, _stats(), None, None, [], run_id="run-scen")
    scenarios = res.payload["sections"]["outlook"]["interpretation"]["scenarios"]
    assert set(scenarios) == {"base", "bull", "bear"}
    for name, sc in scenarios.items():
        assert sc["falsifiers"], name
        for f in sc["falsifiers"]:
            assert f.strip().lower() != "n/a", (name, f)
            # Either it names an observation, or it is an n/a WITH a reason.
            assert v.is_real_falsifier(f) or f.startswith("n/a: "), (name, f)
    # This mandate has both leading indicators and core KPIs, so all three
    # scenarios reach the observation-naming branch.
    for name, sc in scenarios.items():
        assert v.is_real_falsifier(sc["falsifiers"][0]), (name, sc["falsifiers"])
    assert v.validate(res.payload, _facts_of(res.payload)) == []


def test_cross_industry_facts_label_each_dependency_with_its_source(analyst):
    snapshot = {"id": 3, "as_of": datetime(2026, 9, 4, 21, 0), "period_key": "2026-W36",
                "payload": {"regime": {"macro_regime": "late_cycle"},
                            "spillovers": [{"id": "D01", "codes": ["4530", "4510"],
                                            "transmission": "x", "source": "atlas_dependencies_sheet"},
                                           {"id": "D09", "codes": ["2010", "1010"]}]}}
    payload = w.write_report(analyst, _stats(), snapshot, None, [], run_id="run-7").payload
    ci = payload["sections"]["cross_industry"]["facts"]
    # The fixture now uses the shape compute_cross_snapshot really emits
    # (`id` + `codes`), not the origin/destination pair the writer used to
    # filter on: with the imagined shape this assertion passed while the
    # production filter matched nothing.
    assert [s["id"] for s in ci["spillovers"]] == ["D01"]
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


def test_losing_price_coverage_is_not_a_membership_change(analyst):
    """REGRESSION: membership came from `per_ticker` (the priced names), so a
    constituent that merely lost price coverage was reported as having left
    the group — and the report stated two different constituent counts."""
    first = w.write_report(analyst, _stats(), None, None, [], run_id="run-11a")
    prior = {"payload": first.payload, "version": 1, "as_of": datetime(2026, 8, 28, 20, 0),
             "period_key": "2026-W35"}
    stats = _stats()
    del stats["per_ticker"]["AMD"]  # prices dropped out; still a constituent
    stats["sample"]["n_with_prices"] = 2
    stats["sample"]["excluded"] = [{"ticker": "ORCL", "reason": "no_prices"},
                                   {"ticker": "AMD", "reason": "stale_prices"}]
    second = w.write_report(analyst, stats, None, prior, [], run_id="run-11b")
    facts = _facts_of(second.payload)
    assert facts["what_changed"]["constituents"] == {"added": [], "removed": [], "n_added": 0, "n_removed": 0}
    assert facts["companies"]["n_constituents"] == facts["overview"]["n_constituents"] == 4
    assert facts["companies"]["unpriced"] == [{"ticker": "AMD", "reason": "stale_prices"},
                                              {"ticker": "ORCL", "reason": "no_prices"}]
    text = second.payload["sections"]["companies"]["interpretation"]["text"]
    assert "AMD (stale_prices)" in text and "ORCL (no_prices)" in text
    assert "Price coverage: 2 of 4" in text
    assert v.validate(second.payload, facts) == []


def test_membership_falls_back_to_priced_names_and_says_so(analyst, monkeypatch):
    """With no sample ticker list and no classification rows, the priced
    names are the last resort — and the section names that source."""
    monkeypatch.setattr(w.industry_classification, "constituents", lambda code: [])
    stats = _stats()
    stats["sample"] = {"n_with_prices": 3}
    facts = _facts_of(w.write_report(analyst, stats, None, None, [], run_id="run-12").payload)
    assert facts["companies"]["constituents"] == ["AAPL", "AMD", "MSFT"]
    assert "priced names only" in facts["companies"]["membership_source"]
    assert facts["companies"]["unpriced"] == []
    assert facts["overview"]["n_constituents"] == 3


def test_a_budget_of_one_call_reports_the_sections_it_never_reached(analyst, monkeypatch):
    """REGRESSION: with `industry_report_max_llm_calls == 1` the second batch
    (outlook, what_changed) was dropped from the plan, so no
    `analyst_narrative:<section>:deterministic` entry was ever recorded and
    the edition still called itself the analyst's."""
    calls: list[tuple[str, ...]] = []

    def fake_chat_json(prompt, **kwargs):
        sections = prompt.split("\n", 1)[0].removeprefix("Sections to interpret now: ").rstrip(".").split(", ")
        calls.append(tuple(sections))
        out = {name: {"text": f"Analyst text for {name}.", "claims": []} for name in sections}
        if "drivers" in out:  # the drivers section must carry the eight stages
            out["drivers"]["stages"] = [{"id": s["id"], "text": f"Stage {s['id']}: the observed world change, then its consequences."}
                                        for s in thesis_stages()]
        return out

    monkeypatch.setattr(w.settings.__class__, "has_llm", property(lambda self: True))
    monkeypatch.setattr(w.llm, "_demo_only", lambda: False)
    monkeypatch.setattr(w.llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(w.llm, "last_usage", lambda: {})
    monkeypatch.setattr(w.settings, "industry_report_max_llm_calls", 1)
    res = w.write_report(analyst, _stats(), None, None, [], run_id="run-13")
    assert len(calls) == 1 and "outlook" not in calls[0]
    for name in ("outlook", "what_changed"):
        assert f"analyst_narrative:{name}:deterministic" in res.degraded
        assert res.payload["narrative_by_section"][name] == "deterministic"
        # …and the text really is the template, not the analyst's.
        assert not res.payload["sections"][name]["interpretation"]["text"].startswith("Analyst text")
    assert res.payload["narrative_by_section"]["overview"] == "llm"
    assert "outlook" not in (res.generation["llm_planned_sections"] or [])
    assert v.validate(res.payload, _facts_of(res.payload)) == []


# --- registered forecast assumptions (owner decision 1) -----------------------


def _llm_on(monkeypatch, fake_chat_json) -> None:
    monkeypatch.setattr(w.settings.__class__, "has_llm", property(lambda self: True))
    monkeypatch.setattr(w.llm, "_demo_only", lambda: False)
    monkeypatch.setattr(w.llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(w.llm, "last_usage", lambda: {})


def _sections_of(prompt: str) -> tuple[str, ...]:
    return tuple(prompt.split("\n", 1)[0].removeprefix("Sections to interpret now: ").rstrip(".").split(", "))


def _rich_stats() -> dict:
    """A statistics row with every anchor family populated — more numeric
    leaves than the catalogue's cap."""
    stats = _stats()
    horizons = ("1W", "1M", "QTD", "YTD", "1Y")
    p = stats["payload"]
    p["returns"] = {h: {"equal_weight": 0.01 * (i + 1), "median": 0.011 * (i + 1),
                        "market_cap_weight": 0.012 * (i + 1), "n": 3, "n_mcw": 3}
                    for i, h in enumerate(horizons)}
    p["benchmark_relative"] = {b: {h: {"value": 0.002 * (i + 1), "n": 3, "benchmark_n": 30}
                                   for i, h in enumerate(horizons)}
                               for b in ("universe_ew", "sector_ew", "KFR.MKT_RF.D")}
    p["breadth"] = {"1w": {"pct_positive": 0.67, "n": 3},
                    "above_50d_mean": {"share": 0.33, "n": 3, "window_sessions": 50}}
    p["dispersion"] = {"horizon": "1m", "stdev": 0.03, "iqr": 0.02, "range": 0.09, "n": 3}
    p["fundamentals"] = {m: {"median": 0.2, "p25": 0.1, "p75": 0.3, "n": 3}
                         for m in ("revenue_growth_yoy", "op_margin", "fcf_margin")}
    p["valuation"] = {m: {"median": 20.5, "p25": 15.5, "p75": 25.5, "n": 3, "n_excluded_nonpositive": 0}
                      for m in ("ev_ebitda", "pe_ttm", "ev_revenue")}
    return stats


def test_anchor_catalog_is_reproducible_by_server_facts(analyst):
    """The catalogue the model is shown must be exactly what the worker's
    independent rebuild produces (`_server_facts`), or every edition would
    fail the facts-mutation guard. Every entry resolves, at its stated
    value, in those facts; no entry is a count."""
    from app.services import industry_report_worker as jobs

    for stats in (_stats(), _rich_stats()):
        res = w.write_report(analyst, stats, None, None, [], run_id="run-anchor")
        outlook = res.payload["sections"]["outlook"]["facts"]
        rebuilt = jobs._server_facts(analyst, stats, None, None, [], job={"run_id": "run-anchor"},
                                     payload=res.payload)
        assert rebuilt["outlook"] == outlook
        assert outlook["forecast_policy"] == v.FORECAST_POLICY
        anchors = outlook["anchors"]
        assert anchors and len(anchors) <= v.MAX_ANCHORS
        for a in anchors:
            assert v.resolve_fact_path(rebuilt, a["path"]) == a["value"], a
            assert a["path"].startswith(v.ANCHOR_PREFIXES[a["family"]]), a
            assert not v._is_count_leaf(a["path"].rsplit(".", 1)[-1]), a
        assert v.validate(res.payload, rebuilt) == []

    # The small row: its returns and its one valuation multiple, no counts.
    small = w.write_report(analyst, _stats(), None, None, [], run_id="run-anchor-small")
    paths = {a["path"]: a for a in small.payload["sections"]["outlook"]["facts"]["anchors"]}
    assert paths["performance.returns.1M.equal_weight"] == {
        "path": "performance.returns.1M.equal_weight", "value": -0.031, "family": "rate"}
    assert paths["statistics.valuation.ev_ebitda.median"]["family"] == "multiple"
    assert "performance.returns.1M.n" not in paths
    assert small.payload["sections"]["outlook"]["facts"]["anchors_truncated"] == 0

    # The rich row overflows the cap: the drop is counted, and headline
    # leaves go first so the multiples (last in prefix order) survive.
    rich = w.write_report(analyst, _rich_stats(), None, None, [], run_id="run-anchor-rich")
    facts = rich.payload["sections"]["outlook"]["facts"]
    assert len(facts["anchors"]) == v.MAX_ANCHORS and facts["anchors_truncated"] > 0
    assert {a["family"] for a in facts["anchors"]} == {"rate", "multiple"}
    assert "performance.benchmark_relative.KFR.MKT_RF.D.1W.value" in {a["path"] for a in facts["anchors"]}


def test_coerce_keeps_assumption_fields():
    """The writer used to keep only type/text/basis/falsifier of every
    claim, so a model's registration (id, value, horizon, anchor, bounds)
    and a scenario's `assumption_ids` never reached the validator."""
    section = w._coerce_section({
        "text": "Outlook.",
        "claims": [
            {"type": "forecast_assumption", "id": " FA1 ", "text": "t", "value": "21%", "horizon": "next 4 quarters",
             "anchor": "statistics.fundamentals.op_margin.median", "bounds": ["18%", "25%"],
             "basis": ["statistics.fundamentals.op_margin.median"], "falsifier": "f", "extra": "dropped"},
            {"type": "forecast_assumption", "id": "FA2", "text": "t", "value": "21.123456789012345%"},
            {"type": "observed_fact", "text": "o", "id": "FA3", "value": "5%"},
        ],
        "scenarios": {"base": {"text": "b", "falsifiers": ["x"], "assumption_ids": ["FA1", " FA2"]},
                      "bull": {"text": "u"}},
    })
    fa1, fa2, fact = section["claims"]
    assert fa1 == {"type": "forecast_assumption", "text": "t", "basis": ["statistics.fundamentals.op_margin.median"],
                   "falsifier": "f", "id": "FA1", "value": "21%", "horizon": "next 4 quarters",
                   "anchor": "statistics.fundamentals.op_margin.median", "bounds": ["18%", "25%"]}
    # Over the cap: dropped (the validator then names it), never truncated
    # into a different number.
    assert "value" not in fa2 and fa2["id"] == "FA2"
    # Only an assumption carries registration fields.
    assert "id" not in fact and "value" not in fact
    assert section["scenarios"]["base"]["assumption_ids"] == ["FA1", "FA2"]
    assert "assumption_ids" not in section["scenarios"]["bull"]


def test_outlook_batch_uses_raised_max_tokens(analyst, monkeypatch):
    """Each registered assumption costs ~120-150 output tokens; the batch
    that carries the outlook takes its cap from settings, the other keeps
    3200."""
    calls: list[tuple[tuple[str, ...], int]] = []

    def fake_chat_json(prompt, **kwargs):
        calls.append((_sections_of(prompt), kwargs.get("max_tokens")))
        return {}

    _llm_on(monkeypatch, fake_chat_json)
    monkeypatch.setattr(w.settings, "industry_report_outlook_max_tokens", 4321)
    w.write_report(analyst, _stats(), None, None, [], run_id="run-mt")
    caps = dict(calls)
    assert caps[("outlook", "what_changed")] == 4321
    assert [cap for sections, cap in calls if "outlook" not in sections] == [3200]


def test_repair_notes_reach_the_prompt(analyst, monkeypatch):
    prompts: list[str] = []

    def fake_chat_json(prompt, **kwargs):
        prompts.append(prompt)
        return {}

    _llm_on(monkeypatch, fake_chat_json)
    note = "1 validation problem(s): outlook: number '75%' is not in the facts or a registered assumption"
    w.write_report(analyst, _stats(), None, None, [], run_id="run-rn", repair_notes=note)
    assert len(prompts) == 2
    for prompt in prompts:
        assert f"rejected by the validator for: {note}" in prompt
        assert "Fix exactly these problems" in prompt
    prompts.clear()
    w.write_report(analyst, _stats(), None, None, [], run_id="run-rn2")
    assert prompts and not any("rejected by the validator" in p for p in prompts)


def test_template_edition_still_validates():
    """The audit-only template must pass the FULL validator to be stored.
    Its outlook used to register "Scenarios are mandate templates, not
    forecasts." as a `forecast_assumption` with no value or anchor — which
    the registered-assumption contract rejects — and its scenarios quoted
    bracketed industry codes, six-digit numbers the outlook's own facts do
    not carry."""
    ia.clear_cache()
    for code in ("2030", "4510", "4530"):
        a = ia.get_industry_analyst(code)
        for deterministic in (False, True):
            res = w.write_report(a, _stats(), None, None, [], run_id=f"tpl-{code}", deterministic=deterministic)
            assert v.validate(res.payload, _facts_of(res.payload)) == [], code
            outlook = res.payload["sections"]["outlook"]
            assert outlook["facts"]["anchors"], code
            claims = outlook["interpretation"]["claims"]
            assert not [c for c in claims if c["type"] == "forecast_assumption"], code
            assert {"type": "observed_fact", "text": "Scenarios are mandate templates, not forecasts.",
                    "basis": ["outlook.scenario_policy"], "falsifier": ""} in claims
            for sc in outlook["interpretation"]["scenarios"].values():
                for text in [sc["text"], *sc["falsifiers"]]:
                    assert all(v._is_exempt(raw) for raw, _, _ in v.numeric_tokens(text)), (code, text)


def test_stub_analyst_registers_a_valid_assumption(analyst, monkeypatch):
    """The test/fixture stand-in analyst publishes one registered assumption
    anchored on its own observation, through the real coerce + validate
    path, so the published fixture exercises the contract."""
    from app.tests.fixtures import industry_analyst_stub

    industry_analyst_stub.install(monkeypatch)
    res = w.write_report(analyst, _stats(), None, None, [], run_id="run-stub")
    assert res.payload["analyst_narrative"] == "llm"
    assert v.validate(res.payload, _facts_of(res.payload)) == []
    outlook = res.payload["sections"]["outlook"]["interpretation"]
    [fa] = [c for c in outlook["claims"] if c["type"] == "forecast_assumption"]
    first_rate = next(a for a in res.payload["sections"]["outlook"]["facts"]["anchors"] if a["family"] == "rate")
    assert fa["id"] == "FA1" and fa["anchor"] == first_rate["path"]
    assert fa["value"] == f"{first_rate['value'] * 100:.1f}%"
    assert outlook["scenarios"]["base"]["assumption_ids"] == ["FA1"]
