"""FEAT-003 slice 2 — the report validator's rejection rules.

Built around a minimal valid payload so each test flips exactly one thing:
a missing section, mutated facts, a number the facts do not contain, an
advice phrase, an unregistered causal sentence, a drivers section out of
methodology order or opening on a KPI forecast.
"""
from __future__ import annotations

import copy

from app.agents import industry_report_validator as v
from app.agents.prompts import DISCLAIMER
from app.services.industry_group_knowledge import thesis_stages

STAGE_IDS = [s["id"] for s in thesis_stages()]


def _facts() -> dict:
    return {
        "overview": {"n_constituents": 17, "code": "4530"},
        "drivers": {"economic_engine": [{"text": "Wafer starts × ASP", "industry_codes": ["453010"]}]},
        "kpis": {"core_kpis": [{"text": "Utilization"}]},
        "performance": {"returns": {"1M": {"equal_weight": -0.031, "n": 17}}, "status": "ok"},
        "companies": {"constituents": ["A", "B"]},
        "statistics": {"sample": {"n_with_prices": 17}},
        "themes": {"cohort_filing_themes": []},
        "cross_industry": {"edges": []},
        "outlook": {"expectations_ledger": {"reported_consensus": {"value": None, "reason": "none"}}},
        "risks": {"common_failure_modes": []},
        "what_changed": {"facts_delta": {"value": None, "reason": "first edition"}},
        "sources": {"manifest": []},
        "metadata": {"generated_at": "2026-09-06T06:30:00", "run_id": "r1", "report_schema_version": 1},
    }


def _interp(text: str, **extra) -> dict:
    return {"text": text, "claims": [{"type": "observed_fact", "text": text, "basis": ["x"], "falsifier": ""}],
            **extra}


def _payload(facts: dict | None = None) -> dict:
    facts = facts if facts is not None else _facts()
    stages = [{"id": sid, "text": f"n/a: {sid} not asserted"} for sid in STAGE_IDS]
    stages[0]["text"] = "The world change is n/a in this edition; the observed macro regime is unknown."
    interps = {
        "overview": _interp("The group has 17 constituents in the sample."),
        "drivers": _interp("Stage texts quote the mandate.", stages=stages),
        "kpis": _interp("Utilization is the KPI the mandate tests first."),
        "performance": _interp("Observed 1M equal-weight return -3.1% (n=17)."),
        "companies": _interp("Two constituents on file."),
        "themes": _interp("No cohort filing themes on file."),
        "cross_industry": _interp("No edges touch this group."),
        "outlook": _interp("Reported consensus n/a (none).",
                           scenarios={"base": {"text": "Base scenario: unchanged.", "falsifiers": ["n/a"]}}),
        "risks": _interp("No failure modes listed."),
        "what_changed": _interp("What changed: n/a (first edition)."),
    }
    sections = {
        name: {"facts": copy.deepcopy(facts[name]), "interpretation": interps.get(name)}
        for name in v.SECTION_ORDER
    }
    return {"section_order": list(v.SECTION_ORDER), "sections": sections, "disclaimer": DISCLAIMER}


def test_minimal_payload_is_accepted():
    assert v.validate(_payload(), _facts()) == []


def test_missing_section_is_rejected():
    p = _payload()
    del p["sections"]["themes"]
    errs = v.validate(p, _facts())
    assert "section missing: themes" in errs


def test_section_order_is_frozen():
    p = _payload()
    p["section_order"] = list(reversed(v.SECTION_ORDER))
    assert "section_order does not match the frozen order" in v.validate(p, _facts())


def test_mutated_facts_are_rejected():
    p = _payload()
    p["sections"]["overview"]["facts"]["n_constituents"] = 18
    assert "facts mutated: overview" in v.validate(p, _facts())


def test_missing_disclaimer_is_rejected():
    p = _payload()
    p["disclaimer"] = ""
    assert "disclaimer missing" in v.validate(p, _facts())


def test_number_absent_from_facts_is_rejected_and_present_numbers_pass():
    p = _payload()
    p["sections"]["performance"]["interpretation"] = _interp("The group returned +4.2% over one month.")
    errs = v.validate(p, _facts())
    assert any("number '+4.2%'" in e for e in errs), errs
    # The same figure rounded from the facts (-0.031 -> -3.1%) is fine, as is
    # the sample size and a count word like "2 quarters".
    p["sections"]["performance"]["interpretation"] = _interp(
        "Observed -3.1% equal-weight over 1M with n=17 across 2 quarters of data."
    )
    assert v.validate(p, _facts()) == []


def test_rounding_tolerance_matches_displayed_precision():
    p = _payload()
    # -0.031 -> "-3%" at zero decimals is within half a unit.
    p["sections"]["performance"]["interpretation"] = _interp("Observed about -3% over 1M.")
    assert v.validate(p, _facts()) == []
    # "-3.10%" claims two decimals the facts (-3.1) support exactly.
    p["sections"]["performance"]["interpretation"] = _interp("Observed -3.10% over 1M.")
    assert v.validate(p, _facts()) == []
    # "-3.15%" claims precision the facts do not carry.
    p["sections"]["performance"]["interpretation"] = _interp("Observed -3.15% over 1M.")
    assert any("-3.15%" in e for e in v.validate(p, _facts()))


def test_dates_weeks_versions_and_codes_are_not_measurements():
    p = _payload()
    p["sections"]["overview"]["interpretation"] = _interp(
        "As of 2026-09-04 (week 2026-W36, schema v1.0.0), group 4530 with industry 453010."
    )
    assert v.validate(p, _facts()) == []


def test_advice_phrasing_is_rejected():
    p = _payload()
    p["sections"]["outlook"]["interpretation"] = _interp("We recommend buying the leaders now.")
    errs = v.validate(p, _facts())
    assert any("advice phrasing 'we recommend'" in e for e in errs), errs


def test_unregistered_causal_claim_is_rejected_and_a_registered_one_passes():
    p = _payload()
    sentence = "Margins compress because capacity is returning."
    p["sections"]["risks"]["interpretation"] = {"text": sentence, "claims": []}
    errs = v.validate(p, _facts())
    assert any("unsupported causal claim" in e for e in errs), errs
    # Registered as a causal_inference with basis + falsifier -> accepted.
    p["sections"]["risks"]["interpretation"] = {
        "text": sentence,
        "claims": [{"type": "causal_inference", "text": sentence,
                    "basis": ["drivers.economic_engine"], "falsifier": "Utilization rises while margins hold."}],
    }
    assert v.validate(p, _facts()) == []
    # A claim without a falsifier does not count as support.
    p["sections"]["risks"]["interpretation"]["claims"][0]["falsifier"] = ""
    assert any("unsupported causal claim" in e for e in v.validate(p, _facts()))


def test_unknown_claim_type_is_rejected():
    p = _payload()
    p["sections"]["risks"]["interpretation"]["claims"][0]["type"] = "opinion"
    assert any("claim type 'opinion'" in e for e in v.validate(p, _facts()))


def test_drivers_must_follow_the_methodology_order():
    p = _payload()
    stages = p["sections"]["drivers"]["interpretation"]["stages"]
    stages[0], stages[1] = stages[1], stages[0]
    errs = v.validate(p, _facts())
    assert any("not in the methodology order" in e for e in errs), errs


def test_drivers_missing_stages_is_rejected():
    p = _payload()
    del p["sections"]["drivers"]["interpretation"]["stages"]
    assert "drivers: stages missing" in v.validate(p, _facts())


def test_drivers_opening_with_a_kpi_forecast_is_rejected():
    p = _payload()
    stages = p["sections"]["drivers"]["interpretation"]["stages"]
    stages[0]["text"] = "We forecast utilization to reach 92% next year, so the world change is capex."
    errs = v.validate(p, _facts())
    assert "drivers: opens with a KPI forecast before naming the world change" in errs
    # Same anti-pattern in the free text.
    stages[0]["text"] = "The world change: hyperscaler capex is being funded."
    p["sections"]["drivers"]["interpretation"]["text"] = "Revenue will grow to 17 next year."
    errs = v.validate(p, _facts())
    assert "drivers: text opens with a KPI forecast before naming the world change" in errs


def test_drivers_naming_the_world_change_first_passes():
    p = _payload()
    stages = p["sections"]["drivers"]["interpretation"]["stages"]
    stages[0]["text"] = "The world change: hyperscaler capex is being funded on multi-year budgets."
    assert v.validate(p, _facts()) == []


def test_empty_first_stage_is_rejected():
    p = _payload()
    p["sections"]["drivers"]["interpretation"]["stages"][0]["text"] = "  "
    assert "drivers: the first stage (world change) is empty" in v.validate(p, _facts())


def test_interpreted_section_without_interpretation_is_rejected():
    p = _payload()
    p["sections"]["kpis"]["interpretation"] = None
    assert "interpretation missing: kpis" in v.validate(p, _facts())
    # Facts-only sections never need one.
    assert p["sections"]["statistics"]["interpretation"] is None
    assert not any("statistics" in e for e in v.validate(p, _facts()))


def test_errors_are_deduplicated():
    p = _payload()
    p["sections"]["outlook"]["interpretation"] = {
        "text": "Returned +9.9%. Returned +9.9% again.", "claims": [],
        "scenarios": {"bull": {"text": "Bull: +9.9%.", "falsifiers": ["+9.9%"]}},
    }
    errs = v.validate(p, _facts())
    assert errs.count("outlook: number '+9.9%' is not in the facts") == 1
