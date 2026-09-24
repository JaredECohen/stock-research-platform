"""FEAT-003 slice 2 — the report validator's rejection rules.

Built around a minimal valid payload so each test flips exactly one thing:
a missing section, mutated facts, a number the facts do not contain, an
advice phrase, an unregistered causal sentence, a drivers section out of
methodology order or opening on a KPI forecast.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("text, token", [
    ("Margins compressed 8% this quarter.", "8%"),
    ("The group trades at 10x EBITDA.", "10x"),
    ("Spreads widened 12 bps.", "12 bps"),
    ("Returned 2000 bps.", "2000 bps"),
    ("Trades at $5.", "$5"),
    ("Grew +7% year over year.", "+7%"),
])
def test_small_unit_suffixed_figures_are_measurements_not_counts(text, token):
    """The count/year exemption is decided on the raw token: a unit suffix,
    a sign or a currency mark makes a small integer a claim the facts must
    carry (an LLM edition once sailed through with "8%", "10x", "12 bps")."""
    p = _payload()
    p["sections"]["performance"]["interpretation"] = _interp(text)
    errs = v.validate(p, _facts())
    assert errs == [f"performance: number {token!r} is not in the facts"], errs


def test_bare_counts_ordinals_and_years_stay_exempt():
    p = _payload()
    p["sections"]["performance"]["interpretation"] = _interp(
        "Two of 3 industries in 2026 across 12 months at stage 4; 0 excluded, 1 flagged in 1999."
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


@pytest.mark.parametrize("placeholder", [
    "", "   ", "n/a", "N/A.", "n / a", "none", "None", "TBD", "unknown", "not applicable",
    # The exact string the deterministic writer used to stamp on every
    # causal sentence it registered.
    "n/a: mandate-level mechanism quoted from the knowledge base; no dated falsifier in this edition",
    "margins fall",  # too short to name an observation anyone could check
])
def test_a_placeholder_falsifier_does_not_satisfy_the_causal_gate(placeholder):
    """REGRESSION: "every causal link needs a falsifier" was satisfied by ANY
    non-empty string, so the literal "n/a" passed it — and the deterministic
    writer relied on exactly that, which made the gate prove nothing. A
    placeholder now fails twice over: the claim is named as having no usable
    falsifier, and it no longer counts as support for its own sentence."""
    p = _payload()
    sentence = "Margins compress because capacity is returning."
    p["sections"]["risks"]["interpretation"] = {"text": sentence, "claims": [
        {"type": "causal_inference", "text": sentence, "basis": ["risks.common_failure_modes"],
         "falsifier": placeholder},
    ]}
    errs = v.validate(p, _facts())
    assert any("no usable falsifier" in e for e in errs), errs
    assert any("unsupported causal claim" in e for e in errs), errs


def test_a_falsifier_that_names_an_observation_is_accepted():
    assert v.is_real_falsifier("Utilization rises while margins hold.")
    assert v.is_real_falsifier("No quarter in the next year shows bookings above the mandate's floor.")
    # Not a judgement on whether the observation is the RIGHT one — the gate
    # cannot know that, and pretending it could would be a worse lie.
    assert v.is_real_falsifier("The sky is observed to be a different colour entirely.")


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
    # The outlook's own messages (registered-assumption contract): one for
    # the section text, one for the scenario, each once.
    assert errs.count("outlook: number '+9.9%' is not in the facts or a registered assumption") == 1
    assert errs.count(
        "outlook: scenario bull number '+9.9%' is not the value or anchor of an assumption it lists") == 1
    # A backward-looking section repeats the defect three times, reports it once.
    p["sections"]["risks"]["interpretation"] = {
        "text": "Returned +9.9%. Returned +9.9% again.",
        "claims": [{"type": "observed_fact", "text": "Returned +9.9%.", "basis": ["x"], "falsifier": ""}],
    }
    assert v.validate(p, _facts()).count("risks: number '+9.9%' is not in the facts") == 1


@pytest.mark.parametrize("text, token", [
    ("Revenue grew 1700% year over year.", "1700%"),
    ("Margins expanded 1700 bps.", "1700 bps"),
    ("The cohort trades at 1700x earnings.", "1700x"),
])
def test_an_integer_count_does_not_license_its_hundredfold(text, token):
    """REGRESSION: every fact value was also accepted multiplied by 100, so
    the sample size `n_constituents == 17` silently supported "1700%" — a
    fabricated figure published as if observed. Only a ratio scales."""
    p = _payload()
    p["sections"]["performance"]["interpretation"] = _interp(text)
    assert v.validate(p, _facts()) == [f"performance: number {token!r} is not in the facts"]


def test_a_research_priority_of_five_does_not_license_500_bps():
    facts = _facts()
    facts["overview"]["research_priority"] = 5
    p = _payload(facts)
    p["sections"]["overview"]["interpretation"] = _interp("Margins expanded 500 bps on the quarter.")
    assert any("number '500 bps' is not in the facts" in e for e in v.validate(p, facts))


def test_ratios_still_scale_to_percentages_in_both_directions():
    """The percent expansion the fix narrows must still hold for real
    ratios: a decimal return, a ratio of exactly 1, and a single name up
    more than 100% (stored as 1.8)."""
    facts = _facts()
    facts["performance"]["returns"]["1M"]["breadth"] = 1.0
    facts["performance"]["returns"]["1Y"] = {"leader": 1.8, "equal_weight": -0.031}
    p = _payload(facts)
    p["sections"]["performance"]["interpretation"] = _interp(
        "Observed -3.1% over 1M with 100.0% of names positive; the leader is up 180.0% over 1Y."
    )
    assert v.validate(p, facts) == []


def test_a_large_non_integer_fact_does_not_license_its_hundredfold():
    """A median EV/EBITDA of 26.5 is a rendered multiple, not a ratio; it
    must not license "2650 bps"."""
    facts = _facts()
    facts["statistics"]["valuation"] = {"ev_ebitda": {"median": 26.5}}
    p = _payload(facts)
    p["sections"]["performance"]["interpretation"] = _interp("Spreads widened 2650 bps.")
    assert any("number '2650 bps' is not in the facts" in e for e in v.validate(p, facts))


# --- registered forecast assumptions (owner decision 1; rules F1-F10) ---------
#
# The outlook may state a forward number only as a declared assumption,
# anchored to an observed fact of the same unit family. Each rule has a
# passing and a failing case; the failing case names the rule's own message,
# because that message is what the retry's repair hint hands the model.

REPO = Path(__file__).resolve().parents[3]
OP_MARGIN = "statistics.fundamentals.op_margin.median"
EV_EBITDA = "statistics.valuation.ev_ebitda.median"


def _fa_facts() -> dict:
    """`_facts()` plus the observations an assumption can anchor to: a
    median operating margin of 23.4% and a median EV/EBITDA of 26.5x."""
    facts = _facts()
    facts["statistics"]["fundamentals"] = {"op_margin": {"median": 0.234, "p25": 0.18, "n": 17}}
    facts["statistics"]["valuation"] = {"ev_ebitda": {"median": 26.5, "n": 17}}
    return facts


def _fa(**over) -> dict:
    claim = {
        "type": "forecast_assumption", "id": "FA1",
        "text": "Base case assumes group median operating margin of 21% over the next 4 quarters, "
                "against 23.4% observed.",
        "value": "21%", "horizon": "next 4 quarters", "anchor": OP_MARGIN,
        "basis": [OP_MARGIN, "mandate:capital_cycle"],
        "falsifier": "A later weekly statistics row shows the group median operating margin moving away "
                     "from the assumed level.",
    }
    claim.update(over)
    return claim


def _outlook(claims: list[dict], *, text: str = "Reported consensus n/a (none).",
             scenarios: dict | None = None) -> dict:
    return {"text": text, "claims": claims,
            "scenarios": scenarios if scenarios is not None else
            {"base": {"text": "Base scenario: unchanged.", "falsifiers": ["n/a"]}}}


def _check(outlook: dict, facts: dict | None = None, *, section: str = "outlook") -> list[str]:
    facts = facts if facts is not None else _fa_facts()
    p = _payload(facts)
    p["sections"][section]["interpretation"] = outlook
    return v.validate(p, facts)


def test_a_registered_assumption_is_accepted():
    assert _check(_outlook([_fa()])) == []


# F1
def test_forecast_assumption_outside_outlook_is_rejected():
    errs = _check(_outlook([_fa()]), section="risks")
    assert "risks: forecast_assumption claims are accepted only in outlook" in errs
    assert _check(_outlook([_fa()])) == []


# F2
@pytest.mark.parametrize("bad_id", [None, "", "FA0", "FA13", "fa1", "A1", "FA1 "])
def test_assumption_ids_must_be_fa1_to_fa12(bad_id):
    errs = _check(_outlook([_fa(id=bad_id)]))
    assert f"outlook: forecast assumption id {bad_id!r} is missing, malformed or repeated" in errs
    assert _check(_outlook([_fa(id="FA12")])) == []


def test_a_repeated_id_and_more_than_twelve_assumptions_are_rejected():
    errs = _check(_outlook([_fa(), _fa()]))
    assert "outlook: forecast assumption id 'FA1' is missing, malformed or repeated" in errs
    many = [_fa(id=f"FA{i}") for i in range(1, 13)] + [_fa(id="FA1")]
    assert any("13 forecast assumptions declared; at most 12" in e for e in _check(_outlook(many)))
    assert _check(_outlook([_fa(id=f"FA{i}") for i in range(1, 13)])) == []


# F3
@pytest.mark.parametrize("value", ["21", "21% to 22%", "about 21%", "21 m", "", None])
def test_value_must_be_one_rate_or_multiple(value):
    errs = _check(_outlook([_fa(value=value)]))
    assert f"outlook: FA1 value {value!r} is not a single rate or multiple" in errs


def test_rate_and_multiple_values_are_accepted():
    assert _check(_outlook([_fa(value="-240 bps", text=(
        "Base case assumes the median operating margin moves -240 bps over the next 4 quarters, "
        "against 23.4% observed."))])) == []
    assert _check(_outlook([_fa(value="22x", anchor=EV_EBITDA, basis=[EV_EBITDA], text=(
        "Base case assumes the median EV/EBITDA de-rates to 22x over the next 4 quarters, "
        "against 26.5x observed."))])) == []


def test_unanchored_currency_is_rejected():
    """A commodity price has no anchor in the facts: it can be neither an
    assumption's value nor a number anywhere in the outlook."""
    errs = _check(_outlook([_fa(value="$70")]))
    assert "outlook: FA1 value '$70' is not a single rate or multiple" in errs
    errs = _check(_outlook([_fa()], text="Base case assumes oil at $70 over the next 4 quarters."))
    assert "outlook: number '$70' is not in the facts or a registered assumption" in errs
    scenarios = {"bear": {"text": "Bear scenario: oil at $70.", "falsifiers": [], "assumption_ids": ["FA1"]}}
    errs = _check(_outlook([_fa()], scenarios=scenarios))
    assert "outlook: scenario bear number '$70' is not the value or anchor of an assumption it lists" in errs


# F4
@pytest.mark.parametrize("horizon", ["soon", "", None, "over the cycle", "next 4 quarters at 20% growth"])
def test_horizon_must_be_explicit_and_carry_no_measurement(horizon):
    assert "outlook: FA1 has no explicit horizon" in _check(_outlook([_fa(horizon=horizon)]))


@pytest.mark.parametrize("horizon", ["next 4 quarters", "next 12 months", "FY2027", "Q4 2027", "next 18 months"])
def test_explicit_horizons_are_accepted(horizon):
    assert _check(_outlook([_fa(horizon=horizon)])) == []


# F5
@pytest.mark.parametrize("anchor, family", [
    ("overview.n_constituents", "rate"),                    # a count, and not in the catalogue
    ("statistics.fundamentals.op_margin.n", "rate"),        # a count leaf under an anchor prefix
    ("statistics.fundamentals.fcf_margin.median", "rate"),  # does not resolve
    (EV_EBITDA, "rate"),                                    # a multiple cannot anchor a rate
    ("", "rate"),
])
def test_anchor_must_be_an_observed_fact_of_the_same_family(anchor, family):
    errs = _check(_outlook([_fa(anchor=anchor)]))
    assert f"outlook: FA1 anchor {anchor!r} is not an observed {family} in this edition's facts" in errs


def test_anchor_resolves_in_the_server_facts_not_the_payload():
    """The anchor is looked up in the SERVER's facts: a payload that plants
    its own observation is a mutation, and the anchor still fails."""
    facts = _facts()  # no fundamentals on file
    p = _payload(facts)
    p["sections"]["statistics"]["facts"]["fundamentals"] = {"op_margin": {"median": 0.234}}
    p["sections"]["outlook"]["interpretation"] = _outlook([_fa()])
    errs = v.validate(p, facts)
    assert "facts mutated: statistics" in errs
    assert f"outlook: FA1 anchor {OP_MARGIN!r} is not an observed rate in this edition's facts" in errs


# F6
@pytest.mark.parametrize("text", [
    "Base case assumes group median operating margin of 21% over the next 4 quarters.",  # no anchor value
    "Group median operating margin was 23.4% observed; it may ease over the next 4 quarters.",  # no value
    "Base case assumes 21x against 23.4% observed.",  # the value in the wrong unit family
])
def test_text_must_state_value_and_anchor(text):
    assert "outlook: FA1 text must state its value and its anchor's observed value" in _check(
        _outlook([_fa(text=text)]))


def test_assumption_value_matches_prose_at_displayed_precision():
    """The prose may round the declared value to the precision it shows —
    "21.25%" read as "21.3%" or "21%" — and may state it in another unit of
    the same family ("150 bps" is 1.5%); it may not state a different
    number, or claim precision the declaration does not carry."""
    def fa_text(shown: str, value: str = "21.25%") -> list[str]:
        return _check(_outlook([_fa(value=value, text=f"Base case assumes {shown} over the next 4 quarters, "
                                                     "against 23.4% observed.")]))

    assert fa_text("21.25%") == []
    assert fa_text("21.3%") == []
    assert fa_text("21%") == []
    assert any("FA1 text must state its value" in e for e in fa_text("21.4%"))
    assert any("FA1 text must state its value" in e for e in fa_text("21.26%"))
    assert fa_text("150 bps", value="1.5%") == []
    assert any("FA1 text must state its value" in e for e in fa_text("160 bps", value="1.5%"))
    # The same rule for the outlook's other prose (F8).
    fa = _fa(value="21.25%", text="Base case assumes 21.25% over the next 4 quarters, against 23.4% observed.")
    assert _check(_outlook([fa], text="Margins ease to 21.3% in the base case.")) == []
    errs = _check(_outlook([fa], text="Margins ease to 21.4% in the base case."))
    assert "outlook: number '21.4%' is not in the facts or a registered assumption" in errs


# F7
@pytest.mark.parametrize("falsifier", ["", "n/a", "tbd", "margins fall"])
def test_assumption_needs_a_usable_falsifier(falsifier):
    assert "outlook: FA1 has no usable falsifier" in _check(_outlook([_fa(falsifier=falsifier)]))


def test_fa_falsifier_bounds_accepted():
    """Option (a) of the W1 critique: a threshold in a falsifier is a
    declared number — the assumption's `bounds`, in its own unit family."""
    fa = _fa(bounds=["18%", "25%"], falsifier=(
        "Two consecutive weekly statistics rows show the group median operating margin above 25% or below 18%."))
    assert _check(_outlook([fa])) == []
    # Its value and its anchor's observed value may be named there too.
    fa = _fa(falsifier="The group median operating margin holds at 23.4% instead of easing to 21%.")
    assert _check(_outlook([fa])) == []


def test_fa_falsifier_unbounded_number_rejected():
    fa = _fa(falsifier=(
        "Two consecutive weekly statistics rows show the group median operating margin above 25% or below 18%."))
    errs = _check(_outlook([fa]))
    assert "outlook: FA1 falsifier number '25%' is not its bounds, value or anchor value" in errs
    assert "outlook: FA1 falsifier number '18%' is not its bounds, value or anchor value" in errs
    # Bounds must be in the value's family, ordered, and around the value.
    for bounds in (["18x", "25x"], ["22%", "25%"], ["25%", "18%"], ["18%"], "18%-25%"):
        errs = _check(_outlook([dict(fa, bounds=bounds)]))
        assert "outlook: FA1 bounds must be [lo, hi] in its value's unit family, around its value" in errs, bounds


# F8
def test_outlook_numbers_come_from_its_own_facts_or_registered_assumptions():
    """A number that is in ANOTHER section's facts does not support the
    outlook: "17" is the sample size in overview, and the whole pack used
    to license it here."""
    assert _check(_outlook([_fa()], text="Base case: margins ease to 21% from 23.4%.")) == []
    errs = _check(_outlook([_fa()], text="Base case: margins ease to 22% across 17 names."))
    assert "outlook: number '22%' is not in the facts or a registered assumption" in errs
    assert "outlook: number '17' is not in the facts or a registered assumption" in errs
    # The same "17" is still fine where it is an observation of that section.
    p = _payload()
    p["sections"]["overview"]["interpretation"] = _interp("The group has 17 constituents.")
    assert v.validate(p, _facts()) == []
    # A declared value licenses its own unit family only.
    assert "outlook: number '21x' is not in the facts or a registered assumption" in _check(
        _outlook([_fa()], text="Base case: 21x."))


def test_undeclared_scenario_number_coinciding_with_a_fact_is_rejected():
    """REGRESSION (W1 critique, high): the outlook was checked against the
    WHOLE facts pack, so a scenario could publish an undeclared forecast
    whenever its number happened to coincide with any fact. In the captured
    UI fixture "75%" is a breadth reading, "90%" rides on a closing price of
    90.02 and "45x" on the sector code "45" — none is a forecast anyone
    registered, and all three used to pass."""
    wire = json.loads((REPO / "frontend/src/test/fixtures/industry.wire.json").read_text())
    payload = wire["report"]["payload"]
    facts = {name: s["facts"] for name, s in payload["sections"].items()}
    whole_pack = v._numbers(facts)
    for token in ("75%", "90%", "45x"):
        [(raw, value, decimals)] = v.numeric_tokens(token)
        assert v._supported(raw, value, decimals, whole_pack), f"{token} no longer coincides with a fixture fact"
    payload["sections"]["outlook"]["interpretation"]["scenarios"]["bull"] = {
        "text": "Bull scenario: breadth holds at 75%, margins reach 90% and the group re-rates to 45x.",
        "falsifiers": ["Breadth falls below 75% for two weekly rows."],
    }
    outlook_errors = [e for e in v.validate(payload, facts) if e.startswith("outlook:")]
    for token in ("75%", "90%", "45x"):
        assert (f"outlook: scenario bull number {token!r} is not the value or anchor of an assumption it lists"
                in outlook_errors), outlook_errors


def test_a_scenario_may_quote_only_the_assumptions_it_lists():
    scenarios = {"base": {"text": "Base scenario: margins ease to 21% from 23.4%.",
                          "falsifiers": ["Margins hold at 23.4% or higher through the horizon."],
                          "assumption_ids": ["FA1"]}}
    assert _check(_outlook([_fa()], scenarios=scenarios)) == []
    # Declared, but not listed by THIS scenario: rejected.
    scenarios["base"]["assumption_ids"] = []
    errs = _check(_outlook([_fa()], scenarios=scenarios))
    assert "outlook: scenario base number '21%' is not the value or anchor of an assumption it lists" in errs
    # A bound belongs to the assumption's own falsifier, not the scenario's.
    scenarios = {"base": {"text": "Base scenario: margins ease to 21%.",
                          "falsifiers": ["Margins fall below 18%."], "assumption_ids": ["FA1"]}}
    errs = _check(_outlook([_fa(bounds=["18%", "25%"])], scenarios=scenarios))
    assert "outlook: scenario base number '18%' is not the value or anchor of an assumption it lists" in errs


# F9
def test_outlook_causal_basis_must_name_something_checkable():
    sentence = "Margins ease because capacity returns."
    causal = {"type": "causal_inference", "text": sentence, "basis": ["analyst judgement"],
              "falsifier": "Utilization rises while margins hold."}
    errs = _check(_outlook([causal], text=sentence))
    assert ("outlook: causal claim basis names nothing in the facts, the mandate or a registered assumption"
            in errs)
    for basis in (["mandate:capital_cycle"], ["FA1"], ["outlook.expectations_ledger"], [OP_MARGIN]):
        assert _check(_outlook([_fa(), dict(causal, basis=basis)], text=sentence)) == [], basis
    # A bare section name is not a fact path, and an undeclared id is not an assumption.
    for basis in (["outlook"], ["FA2"], ["mandate:"]):
        assert any("causal claim basis names nothing" in e
                   for e in _check(_outlook([_fa(), dict(causal, basis=basis)], text=sentence))), basis
    # Backward-looking sections keep their existing gate.
    p = _payload()
    p["sections"]["risks"]["interpretation"] = {"text": sentence, "claims": [dict(causal, basis=["x"])]}
    assert v.validate(p, _facts()) == []


# F10
def test_scenario_assumption_ids_must_be_declared():
    scenarios = {"bull": {"text": "Bull scenario: unchanged.", "falsifiers": [], "assumption_ids": ["FA2"]}}
    assert "outlook: scenario bull references undeclared assumption FA2" in _check(
        _outlook([_fa()], scenarios=scenarios))
    scenarios["bull"]["assumption_ids"] = ["FA1"]
    assert _check(_outlook([_fa()], scenarios=scenarios)) == []
    scenarios["bull"]["assumption_ids"] = "FA1"
    assert "outlook: scenario bull assumption_ids is not a list" in _check(_outlook([_fa()], scenarios=scenarios))


def test_prompt_example_is_a_valid_registration():
    """The contract's worked example in `prompts/industry_report.md` must
    pass the contract. The design's first draft named 25% and 18% in the
    falsifier with no bounds declared — an example the validator rejects."""
    prompt = (Path(v.__file__).resolve().parent.parent / "prompts" / "industry_report.md").read_text()
    [block] = re.findall(r"```json\n(.*?)\n```", prompt, re.S)
    example = json.loads(block)
    assert any(c.get("bounds") for c in example["claims"])
    assert _check(_outlook(example["claims"], scenarios=example["scenarios"])) == []


def test_resolve_fact_path_handles_dotted_keys_and_list_indices():
    facts = {"performance": {"benchmark_relative": {"KFR.MKT_RF.D": {"1w": {"value": 0.01}}}},
             "companies": {"leaders": [{"ret_1m": 0.05}]}}
    assert v.resolve_fact_path(facts, "performance.benchmark_relative.KFR.MKT_RF.D.1w.value") == 0.01
    assert v.resolve_fact_path(facts, "companies.leaders.0.ret_1m") == 0.05
    assert v.resolve_fact_path(facts, "companies.leaders.3.ret_1m") is v._MISSING
    assert v.resolve_fact_path(facts, "nope") is v._MISSING
