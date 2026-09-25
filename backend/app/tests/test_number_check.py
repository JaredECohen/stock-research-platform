"""W2b 7(a): the unit-aware number check (`agents/number_check.py`).

Pure tests over hand-built registries: extraction (scale, units, ranges,
identifiers, offsets), matching (printed precision, percent vs ratio,
percentage points vs relative changes) and the statuses (traced needs
anchoring; derived facts support only anchored claims; mis_anchored when
the claim names a registered metric whose value is something else).
"""
from __future__ import annotations

import random

import pytest

from app.agents import number_check as nc
from app.agents.source_ledger import FactRegistry, SourceLedger


def registry(*registrations: tuple[str, str, object], **kw) -> FactRegistry:
    ledger = SourceLedger()
    for kind, ref, obj in registrations:
        ledger.register(kind, ref, obj, **kw)
    return ledger.snapshot()


def claims(text: str) -> list[nc.ParsedClaim]:
    return [c for c in nc.extract_claims(text) if c.cls != "exempt"]


def statuses(text: str, reg: FactRegistry) -> list[tuple[str, str]]:
    return [(c.claim.raw, c.status) for c in nc.check_text(text, reg)]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_scale_equivalence():
    reg = registry(("financials", "financials:T", {"income": [{"period": "2025", "revenue": 1_203_000_000}]}))
    text = "Revenue of $1.2B, $1,200 million, 1.2 billion and $1200M."
    got = claims(text)
    assert [c.value for c in got] == [1.2e9] * 4
    assert all(s == "traced" for _, s in statuses(text, reg))


def test_percent_matches_ratio_and_text_not_counts():
    ratio = registry(("financials", "f", {"ratios": {"gross_margin": 0.2603}}))
    text_fact = registry(("filing", "filing:x", {"mda": "Gross margin was 26 percent in the year."}))
    count = registry(("financials", "f", {"ratios": {"gross_margin_count": 26}}))
    assert statuses("Gross margin 26%.", ratio) == [("26%", "traced")]
    assert statuses("Gross margin 26%.", text_fact) == [("26%", "traced")]
    # A structured count of 26 is not 26%: counts never support percentages.
    assert statuses("Gross margin 26%.", count) == [("26%", "untraceable")]


def test_bps_pp_and_ratio_delta():
    got = claims("widened 250 bps, i.e. 2.5 pp, or 2.5 percentage points")
    assert [(c.unit, c.value) for c in got] == [("pp", 2.5)] * 3
    reg = registry(("sector_research", "s", {"trends": {"cohort_op_margin_delta": 0.025}}))
    assert statuses("Cohort op margin widened 250 bps.", reg) == [("250 bps", "traced")]
    assert statuses("Cohort op margin widened 2.5 pp.", reg) == [("2.5 pp", "traced")]


def test_multiple_and_tolerance_at_printed_precision():
    reg = registry(("comps", "comps:T", {"target": {"ev_ebitda": 16.38, "fcf_yield": 0.09398}}))
    assert statuses("EV/EBITDA of 16.4x.", reg) == [("16.4x", "traced")]
    assert statuses("FCF yield 9.4%.", reg) == [("9.4%", "traced")]
    other = registry(("comps", "comps:T", {"target": {"fcf_yield": 0.0952}}))
    assert statuses("FCF yield 9.4%.", other) == [("9.4%", "untraceable")]


def test_ranges_share_units():
    assert [(c.raw, c.unit) for c in claims("fair value $205–220")] == [("$205", "usd"), ("220", "usd")]
    assert [(c.raw, c.unit, c.value) for c in claims("growth of 4-4.5%")] == [
        ("4", "pct", 4.0), ("4.5%", "pct", 4.5)]
    got = claims("a $1.2–1.5B charge")
    assert [(c.unit, c.value) for c in got] == [("usd", 1.2e9), ("usd", 1.5e9)]


def test_identifiers_are_not_claims():
    text = ("FY2025 Q3 2025 2025Q3 2026-2027 2030s 10-K Item 7A S&P 500 0001628280-26-003942 "
            "1st RSI(14) SMA 50/200 52w 5G H100 COVID-19 48-hour 7 consecutive two quarters "
            "scored 25/100 on 2026-05-27")
    assert [c.raw for c in claims(text)] == ["25"]


def test_sign_insensitive():
    reg = registry(("dcf", "dcf:initial", {"base": {"upside_pct": -0.435}}))
    assert statuses("The DCF base case implies -43.5% downside.", reg) == [("-43.5%", "traced")]


def test_offsets_index_the_input():
    rng = random.Random(7)
    pieces = ["$1.2B", "26%", "−4.5%", "250 bps", "16.4x", "$205–220", "4-4.5%", "(-37%)",
              "FY2025", "10-K", "52w", "H100", "3.9x", "6.7 percentage points", "20 percentage-point",
              "1,203,000", "revenue", "margin:", "—", "grew", "2026-05-27", "Q3 2025", "RSI(14) at 68"]
    for _ in range(300):
        text = " ".join(rng.choice(pieces) for _ in range(rng.randint(1, 12)))
        for c in nc.extract_claims(text):
            assert text[c.start:c.end] == c.raw, (text, c)


def test_thresholds_are_counted_not_checked():
    reg = registry(("financials", "f", {"ratios": {"FCF_yield": 0.02}}))
    assert statuses("FCF yield > 4% gives downside support.", reg) == [("4%", "threshold")]
    falsifier = nc.check_text("Revenue growth below 12.3% for two quarters", reg, threshold=True)
    assert [c.status for c in falsifier] == ["threshold"]


# ---------------------------------------------------------------------------
# Statuses
# ---------------------------------------------------------------------------

def test_status_classes():
    reg = registry(("financials", "financials:T", {"ratios": {"operating_margin": 0.41, "ROIC": 0.2063}}))
    # Anchored 2-digit value match: traced.
    assert statuses("Operating margin 41%.", reg) == [("41%", "traced")]
    # 3+ digits no longer trace on value alone: anchoring is required.
    assert statuses("Returns reached 20.6%.", reg) == [("20.6%", "weak")]
    assert statuses("ROIC reached 20.6%.", reg) == [("20.6%", "traced")]
    # A value match next to no metric at all: weak, never traced.
    assert statuses("It reached 41%.", reg) == [("41%", "weak")]
    # No value match: untraceable.
    assert statuses("Operating margin 37%.", reg) == [("37%", "untraceable")]


def test_four_significant_digits_trace_verbatim_facts_only():
    reg = registry(("financials", "financials:T", {"income": [
        {"period": "2024", "revenue": 100.0}, {"period": "2025", "revenue": 112.34}]}))
    assert statuses("A figure of 112.34 appears.", reg) == [("112.34", "traced")]
    # D1 growth 12.34% is derived: without an anchor it does not support.
    assert statuses("A figure of 12.34% appears.", reg) == [("12.34%", "untraceable")]
    assert statuses("Revenue growth was 12.34%.", reg) == [("12.34%", "traced")]


def test_mis_anchored_status():
    """The claim names a registered metric but its value is something else's."""
    reg = registry(("financials", "financials:T", {"ratios": {"operating_margin": 0.41, "ROIC": 0.23}}))
    assert statuses("Operating margin 23%.", reg) == [("23%", "mis_anchored")]
    assert nc.FLAGGED_STATUSES == {"untraceable", "mis_anchored"}


def test_pp_never_matches_relative_change():
    """META v1: a margin 2.6 points above the median is a 6.7% RELATIVE
    increase; "6.7 percentage points" must not trace to it."""
    reg = registry(("comps", "comps:META", {
        "target": {"operating_margin": 0.4144}, "median": {"operating_margin": 0.38825},
        "history": {"current_vs_own_median": {"operating_margin": 0.067}},
    }))
    text = "Its 41.44% operating margin sits 6.7 percentage points above the peer median."
    assert dict(statuses(text, reg))["6.7 percentage points"] in nc.FLAGGED_STATUSES
    # The true gap, in points, traces (D4 target - median).
    assert dict(statuses(text.replace("6.7", "2.6"), reg))["2.6 percentage points"] == "traced"
    # And the relative change still supports a relative claim.
    assert dict(statuses("Operating margin is 6.7% above its own median.", reg))["6.7%"] == "traced"


def test_derived_facts_support_only_anchored_claims():
    reg = registry(("dcf", "dcf:initial", {
        "current_price": 100.0,
        "base": {"implied_share_price": 120.0, "assumptions": {"wacc": 0.10, "revenue_growth": [0.2]}},
        "bull": {"implied_share_price": 150.0, "assumptions": {"wacc": 0.095, "revenue_growth": [0.23]}},
        "bear": {"implied_share_price": 60.0, "assumptions": {"wacc": 0.11, "revenue_growth": [0.17]}},
    }))
    # bull/bear implied-price ratio 2.5x (D3): anchored by "bull/bear ratio".
    assert statuses("A 2.5x bull/bear ratio.", reg) == [("2.5x", "traced")]
    assert statuses("A 2.5x multiple.", reg) == [("2.5x", "untraceable")]
    # Scenario deltas as points.
    assert statuses("Bull case: growth +300bp, WACC -50bp.", reg) == [("+300bp", "traced"), ("-50bp", "traced")]
    # Different DCF scenarios never anchor to one another.
    assert statuses("The bear case implies $150.", reg) == [("$150", "mis_anchored")]


def test_meta_excerpt_regression():
    """Short excerpts in the shape of META v1's prose against a hand-built
    registry: provider-scale figures, multiples and yields trace, and the
    hyphenated "percentage-point" parses as points."""
    reg = registry(
        ("financials", "financials:META", {"income": [{"period": "2025", "revenue": 51_240_000_000,
                                                      "net_income": 3_600_000_000}],
                                          "ratios": {"EV_EBITDA": 16.38, "FCF_yield": 0.0277}}),
    )
    text = ("Revenue reached $51.2B while net income was 3.60B; the stock trades at 16.4x EV/EBITDA "
            "with a 2.77% FCF yield, after a 20 percentage-point acceleration.")
    got = dict(statuses(text, reg))
    assert got["$51.2B"] == got["3.60B"] == got["16.4x"] == got["2.77%"] == "traced"
    (pp,) = [c for c in claims(text) if "percentage" in c.raw]
    assert pp.unit == "pp" and pp.value == 20.0


def test_declared_assumption_status():
    reg = registry(("financials", "financials:T", {"ratios": {"revenue_growth": 0.12}}))
    declared = [{"value": 18.5, "unit": "pct", "basis_ref": "financials:T", "horizon": "FY2027"}]
    text = "We assume revenue growth of 18.5% through FY2027."
    assert [c.status for c in nc.check_text(text, reg)] == ["untraceable"]
    assert [c.status for c in nc.check_text(text, reg, assumptions=declared)] == ["assumption"]


@pytest.mark.parametrize("path", [
    "catalysts[2].title", "extra_agent_views.industry_group.key_points[1]",
    "quality.rating_reconciliation.reason", "bull_case.key_points[0]",
])
def test_resolve_field_paths(path):
    from app.schemas import AgentFinding, BullBearCase, CatalystItem, MemoQuality, RatingReconciliation
    from app.tests.factories import make_memo
    memo = make_memo(
        catalysts=[CatalystItem(title=f"c{i}") for i in range(3)],
        extra_agent_views={"industry_group": AgentFinding(agent="IG", headline="h", summary="s",
                                                          key_points=["a", "b"])},
        quality=MemoQuality(rating_reconciliation=RatingReconciliation(reason="why")),
        bull_case=BullBearCase(headline="b", key_points=["first"]),
    )
    assert nc.resolve_field(memo, path) in ("c2", "b", "why", "first")
