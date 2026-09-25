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


# ---------------------------------------------------------------------------
# Review round: extraction gaps, anchoring of value-first multiples, credit,
# assumption scope and withholding consistency
# ---------------------------------------------------------------------------

def test_ascii_arrow_is_a_change_not_a_threshold():
    """"12% -> 97%" states two values; only a comparison symbol makes a
    figure a threshold, so an invented value after an arrow is checked."""
    got = [(c.raw, c.cls) for c in claims("Revenue growth accelerates 12% -> 97%.")]
    assert got == [("12%", "fact"), ("97%", "fact")]
    assert [(c.raw, c.cls) for c in claims("Gross margin 61%->99.9%.")] == [
        ("61%", "fact"), ("99.9%", "fact")]
    assert [(c.raw, c.cls) for c in claims("Revenue growth 12% => 97%.")][1] == ("97%", "fact")
    # Real comparisons stay thresholds.
    assert [c.cls for c in claims("FCF yield > 4% and leverage >= 2.5x")] == ["threshold", "threshold"]


def test_may_the_verb_is_not_a_date():
    assert [c.raw for c in claims("Revenue may 30% higher next year")] == ["30%"]
    assert [c.raw for c in claims("Margins may reach 45.5% by then")] == ["45.5%"]
    # The month still masks a real date.
    assert claims("Earnings on May 30, 2026 and Mar 12") == []


def test_value_first_multiples_anchor_to_the_multiple():
    """"46.8x earnings" is a P/E, "22.3x sales" a P/S, "34.4x EBITDA" an
    EV/EBITDA: the noun after the x names the multiple, not revenue or
    EBITDA. Without the binding every one read mis_anchored."""
    reg = registry(("financials", "financials:T", {
        "income": [{"period": "2025", "revenue": 130_500_000_000, "ebitda": 86_000_000_000}],
        "ratios": {"PE": 46.79, "PS": 22.31, "EV_Revenue": 22.34, "EV_EBITDA": 34.37, "PFCF": 49.57,
                   "gross_margin": 0.976, "operating_margin": 0.61, "net_debt_to_ebitda": 1.2}}))
    for text in ("Shares trade at 22.3x sales.", "Shares trade at 22.3x revenue.",
                 "The stock trades at 34.4x EBITDA.", "Trading at 46.8x earnings, a premium to peers.",
                 "NVDA trades at 46.8x trailing earnings.",
                 "At 46.8x earnings and 22.3x sales, NVDA is priced for perfection.",
                 "Valuation: 46.8x P/E, 34.4x EV/EBITDA, 49.6x P/FCF.",
                 "P/E 46.8x, EV/EBITDA 34.4x, P/FCF 49.6x.",
                 "NVDA runs a 61% operating margin and 97.6% gross margins.",
                 "Net debt sits at 1.2x EBITDA."):
        assert {s for _, s in statuses(text, reg)} == {"traced"}, (text, statuses(text, reg))
    # The binding is a reading, not a pass: a wrong multiple is still flagged.
    assert statuses("Shares trade at 31.2x earnings.", reg) == [("31.2x", "untraceable")]
    assert statuses("Shares trade at 34.4x earnings.", reg) == [("34.4x", "mis_anchored")]


def test_primary_kind_credit_survives_ref_truncation():
    """The same note block is registered once per agent; a transcript that
    also carries the figure must still earn its primary-kind credit when
    more than eight refs support the claim."""
    ledger = SourceLedger()
    note = {"note": "Data-center revenue grew 94% year over year."}
    for agent in ("comps", "earnings", "filing", "macro", "pm", "sector", "technical", "valuation"):
        ledger.register("research_note", f"notes:{agent}", note)
    ledger.register("transcript", "transcript:2025Q3", {"remarks": note["note"]})
    (c,) = nc.check_text("Data-center revenue grew 94% year over year.", ledger.snapshot())
    assert c.status == "traced"
    assert "transcript" in c.kinds
    assert c.sources[0] == "transcript:2025Q3" and len(c.sources) == 8


def _memo_with(**kw):
    from app.tests.factories import make_memo
    return make_memo(**kw)


def test_declared_assumption_covers_only_pm_fields():
    """A declaration labels the PM's own forward figure; the same value in
    an analyst key point or a risk stays flagged (and withholdable)."""
    from app.schemas import BullBearCase, RiskItem
    reg = registry(("financials", "financials:T", {"ratios": {"revenue_growth": 0.12}}))
    declared = [{"value": 18.5, "unit": "pct", "basis_ref": "financials:T", "horizon": "FY2027"}]
    memo = _memo_with(
        final_pm_view="We assume services growth of 18.5% through FY2027.",
        bull_case=BullBearCase(headline="Bull", key_points=[
            "Services attach reached 18.5% last year", "Tailwind: cloud", "Tailwind: AI", "Tailwind: ads"]),
        key_risks=[RiskItem(title="Mix", detail="Attach fell from 18.5% in a year.", severity="low")],
    )
    result = nc.check_memo(memo, reg, withhold=True, assumptions=declared)
    by_field = {fr.spec.path: [c.status for c in fr.claims] for fr in result.fields if fr.claims}
    assert by_field["final_pm_view"] == ["assumption"]
    assert by_field["bull_case.key_points[0]"] == ["untraceable"]
    assert by_field["key_risks[0].detail"] == ["untraceable"]
    assert result.plan.items == {"bull_case.key_points": [0]}


def _flagged_list_memo(n_bad: int, n: int = 4):
    from app.schemas import BullBearCase
    points = [f"Invented figure {i}: backlog reached {91 + i}.7% of sales" for i in range(n_bad)]
    points += [f"Tailwind: qualitative point {i}" for i in range(n - n_bad)]
    return _memo_with(bull_case=BullBearCase(headline="Bull", key_points=points))


def test_half_list_guard_boundary():
    """Design §4.4: exactly half of a list may be withheld; more than half
    is likelier a registry gap, so the list is flagged and kept."""
    empty = SourceLedger().snapshot()
    two = nc.check_memo(_flagged_list_memo(2), empty, withhold=True)
    assert two.plan.items == {"bull_case.key_points": [0, 1]}
    assert two.plan.lists_not_withheld == []
    three = nc.check_memo(_flagged_list_memo(3), empty, withhold=True)
    assert three.plan.items == {}
    assert three.plan.lists_not_withheld == ["bull_case.key_points"]


def test_withheld_text_is_not_left_in_a_guarded_copy():
    """The memo copies analyst key points into catalysts and bull points.
    When a copy sits in a list the half-list guard keeps, withholding the
    original would leave a withheld record contradicting what the reader
    sees; every copy is kept, flagged, instead."""
    from app.schemas import CatalystItem
    bad1 = "Cloud tailwind: backlog growth hit 81.7% as hyperscalers pre-bought capacity"
    bad2 = "Accelerating demand: pricing power lifted blended ASPs by 64.3% this year"
    view = _memo_with().sector_agent_view.model_copy(update={"key_points": [
        bad1, bad2, "Tailwind: sovereign AI", "Tailwind: networking attach", "Moat: CUDA software",
        "Tailwind: inference demand"]})
    catalysts = [CatalystItem(title=s[:80], detail=s) for s in (bad1, bad2)]
    catalysts.append(CatalystItem(title="Next earnings", detail="Report due next month."))
    memo = _memo_with(sector_agent_view=view, catalysts=catalysts)
    result = nc.check_memo(memo, SourceLedger().snapshot(), withhold=True)
    assert result.plan.items == {}
    assert set(result.plan.lists_not_withheld) == {"catalysts", "sector_agent_view.key_points"}
    # Without the guarded copy, the analyst's two bad points (2 of 6) go.
    alone = nc.check_memo(_memo_with(sector_agent_view=view), SourceLedger().snapshot(), withhold=True)
    assert alone.plan.items == {"sector_agent_view.key_points": [0, 1]}


def test_withheld_copies_go_together():
    """A flagged key point copied into a catalyst list that may withhold it
    is withheld from both, so it survives nowhere outside quality."""
    from app.schemas import CatalystItem, NumberCheck
    bad = "Cloud tailwind: backlog growth hit 81.7% as hyperscalers pre-bought capacity"
    view = _memo_with().sector_agent_view.model_copy(update={"key_points": [
        bad, "Tailwind: sovereign AI", "Tailwind: networking attach", "Moat: CUDA software"]})
    catalysts = [CatalystItem(title=bad[:80], detail=bad),
                 CatalystItem(title="Next earnings", detail="Report due next month."),
                 CatalystItem(title="Product launch", detail="New platform ships.")]
    memo = _memo_with(sector_agent_view=view, catalysts=catalysts)
    result = nc.check_memo(memo, SourceLedger().snapshot(), withhold=True)
    assert result.plan.items == {"catalysts": [0], "sector_agent_view.key_points": [0]}
    check = nc.summarize(result, assumptions=[], notes=[])
    assert isinstance(check, NumberCheck)
    nc.apply_withholding(memo, check, result.plan)
    assert bad not in memo.model_dump_json(exclude={"quality"})


def test_catalyst_withholding_keeps_title_and_detail_offsets():
    """A catalyst withholds as one item: its title and detail are joined
    with " — " and the detail's claims move by len(title) + 3."""
    from app.schemas import CatalystItem, NumberCheck, NumberClaim
    memo = _memo_with(catalysts=[
        CatalystItem(title="Order worth $4.2B", detail="Backlog up 71.3% after the order."),
        CatalystItem(title="Next earnings", detail="Report due next month."),
    ])
    check = NumberCheck(checked=True, claims=[
        NumberClaim(field="catalysts[0].title", start=12, end=17, raw="$4.2B", status="untraceable"),
        NumberClaim(field="catalysts[0].detail", start=11, end=16, raw="71.3%", status="untraceable"),
        NumberClaim(field="catalysts[1].detail", start=0, end=6, raw="Report", status="weak"),
    ])
    assert nc.apply_withholding(memo, check, nc.WithholdPlan(items={"catalysts": [0]})) == [
        "Order worth $4.2B — Backlog up 71.3% after the order."]
    (w,) = check.withheld
    assert (w.field, w.index) == ("catalysts", 0)
    assert [w.text[c.start:c.end] for c in w.claims] == ["$4.2B", "71.3%"]
    assert [c.raw for c in w.claims] == ["$4.2B", "71.3%"]
    assert [(c.field, c.raw) for c in check.claims] == [("catalysts[0].detail", "Report")]
    assert [c.title for c in memo.catalysts] == ["Next earnings"]
