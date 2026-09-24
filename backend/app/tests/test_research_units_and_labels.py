"""FIX-008b — units and labels in deterministic research text.

Pinned against the saved META v1 memo (docs/reviews/2026-09-13-META-v1.json,
local-only):

- Comps own-history: current operating margin 41.44% against an 8y median of
  38.825% is a 2.61 percentage-point gap and a 6.7% RELATIVE change. The
  saved narrative said "6.7 percentage points" because the only number on
  offer, `current_vs_own_median = 0.067`, carried no unit. The builder now
  also emits the gap in points, the deterministic text labels both, and the
  narrative LLM is handed the labelled field (prompt text untouched).
- Sector placement: META's 16.38x EV/EBITDA sat in Q4 of a cohort whose
  maximum was 21.20x, and was labelled "richest in cohort". Quartile labels
  now name the band and the measure ranked — never an extremum, and never
  multiple wording for a yield or a spend ratio.
- Cohort trend deltas are differences of margins, i.e. percentage points.
"""
from __future__ import annotations

import json
from unittest.mock import PropertyMock

import pytest

from app.agents import graph, influence, llm, sector_agents
from app.agents.comps_agent import run_comps_agent
from app.agents.long_form import deterministic_long_form
from app.config import Settings
from app.finance import comps_history as ch
from app.schemas import AgentFinding, CompsHistoryStats, CompsResult, CompsRow
from app.services.sector_research_service import compute_kpi_placements

# META v1, saved values.
META_OP_MARGIN = 0.41437855159579234
META_OWN_MEDIAN = 0.3882502134788519


def _meta_like_history(monkeypatch) -> CompsHistoryStats:
    """Eight annual periods whose operating-margin median is exactly the
    38.825% the saved META payload reports."""
    margins = [0.30, 0.33, 0.37, 0.38, 0.3965, 0.40, 0.41, 0.42]
    long_format: dict[str, list[dict]] = {"revenue": [], "operating_income": []}
    for i, m in enumerate(margins):
        common = {"period": f"FY{2018 + i}", "period_end": f"{2018 + i}-12-31",
                  "fiscal_year": 2018 + i}
        long_format["revenue"].append({**common, "value": 1000.0})
        long_format["operating_income"].append({**common, "value": 1000.0 * m})

    from app.services import history_service, market_data_service
    monkeypatch.setattr(history_service, "get_financial_history", lambda *a, **k: long_format)
    monkeypatch.setattr(market_data_service, "get_price_series", lambda *a, **k: [])
    target = CompsRow(ticker="META", company_name="Meta", operating_margin=META_OP_MARGIN)
    out = ch.build_history_stats("META", target, lookback_quarters=20, min_periods=8)
    assert out is not None
    return out


def test_history_stats_carry_the_gap_in_points_beside_the_relative_change(monkeypatch):
    h = _meta_like_history(monkeypatch)
    assert h.own_median["operating_margin"] == pytest.approx(0.38825)
    assert h.current_vs_own_median["operating_margin"] == 0.067          # relative
    assert h.current_minus_own_median_pp["operating_margin"] == 2.61     # points
    # Multiples are not rate-type: no percentage-point gap is invented for them.
    assert set(h.current_minus_own_median_pp) <= set(ch.PERCENT_POINT_METRICS)


def _meta_comps(history: CompsHistoryStats) -> CompsResult:
    target = CompsRow(ticker="META", company_name="Meta", operating_margin=META_OP_MARGIN,
                      ev_ebitda=16.1)
    median = CompsRow(ticker="MEDIAN", company_name="Peer Median", operating_margin=0.32,
                      ev_ebitda=19.0)
    return CompsResult(target=target, peers=[target], median=median,
                       premium_discount={"ev_ebitda": -0.15}, target_percentiles={},
                       interpretation="peer interpretation", history=history)


def test_comps_key_point_states_points_and_labels_the_relative_change():
    history = CompsHistoryStats(
        lookback_periods=8, lookback_label="8y",
        own_median={"operating_margin": META_OWN_MEDIAN},
        current_percentile={"operating_margin": 0.75},
        current_vs_own_median={"operating_margin": 0.067},  # as saved in META v1
    )
    finding = run_comps_agent({"ticker": "META"}, _meta_comps(history))
    own = [p for p in finding.key_points if "own 8y median" in p]
    assert own == ["Op margin 41.4% vs own 8y median 38.8% — +2.6 percentage points (+6.7% relative)."]
    assert not any("6.7 percentage points" in p or "7% delta" in p for p in finding.key_points)


def test_comps_narrative_llm_is_handed_the_gap_in_points(monkeypatch):
    """The saved sentence came from this LLM call. Its prompt text is an
    owner decision and stays as it is; what changes is that the payload now
    carries the gap under a unit-bearing key."""
    history = _meta_like_history(monkeypatch)
    prompts: list[str] = []

    def capture(prompt, *a, **k):
        prompts.append(prompt)
        return {"narrative": "stub"}

    monkeypatch.setattr(Settings, "has_llm", PropertyMock(return_value=True))
    monkeypatch.setattr(llm, "chat_json", capture)
    run_comps_agent({"ticker": "META"}, _meta_comps(history))
    assert len(prompts) == 1
    # The payload is the prompt's last block; at this size it is not truncated.
    payload = json.loads(prompts[0].rsplit("\n\n", 1)[1])
    assert payload["history"]["current_minus_own_median_pp"]["operating_margin"] == 2.61
    assert payload["history"]["current_vs_own_median"]["operating_margin"] == 0.067


def _comps_long_form(history: dict) -> str:
    f = AgentFinding(agent="Comps Analyst", headline="h", summary="s",
                     key_points=["a"], sources=["peer:X"], data={"history": history})
    return deterministic_long_form(f, ticker="META", agent_name="Comps Analyst")


def test_long_form_self_history_labels_units():
    saved = {  # META v1 history block as saved (no points field yet)
        "lookback_label": "8y",
        "own_median": {"operating_margin": META_OWN_MEDIAN, "ev_ebitda": 14.0},
        "current_percentile": {"operating_margin": 0.75, "ev_ebitda": 0.9},
        "current_vs_own_median": {"operating_margin": 0.067, "ev_ebitda": 0.15},
    }
    md = _comps_long_form(saved)
    om = next(line for line in md.splitlines() if line.startswith("- **operating_margin**"))
    assert "+6.7% relative to own median" in om
    assert "+7% vs own median" not in om
    ev = next(line for line in md.splitlines() if line.startswith("- **ev_ebitda**"))
    assert "+15% vs own median" in ev  # a multiple's relative change is unambiguous

    fresh = {**saved, "current_minus_own_median_pp": {"operating_margin": 2.61}}
    om = next(line for line in _comps_long_form(fresh).splitlines()
              if line.startswith("- **operating_margin**"))
    assert "+2.6 percentage points vs own median (+6.7% relative)" in om


# ---------------------------------------------------------------------------
# Sector quartile placement labels
# ---------------------------------------------------------------------------

_GROUPS = {
    "quality": ["operating_margin"],
    "capital_intensity": ["capex_pct_revenue"],
    "valuation": ["EV_EBITDA", "FCF_yield"],
}


def _placements(target: dict, cohort: dict[str, list[float]]) -> dict:
    n = len(next(iter(cohort.values())))
    rows = [{k: v[i] for k, v in cohort.items()} for i in range(n)]
    return compute_kpi_placements(target, rows, _GROUPS)


def test_meta_top_quartile_multiple_is_not_called_the_cohort_maximum():
    cohort = {
        "EV_EBITDA": [4.37, 5.0, 5.89, 8.0, 10.0, 11.59, 12.0, 13.29, 15.0, 21.20],
        "FCF_yield": [0.0193, 0.0193, 0.025, 0.05, 0.06, 0.0678, 0.09, 0.117, 0.15, 0.215],
        "operating_margin": [0.03, 0.1, 0.14, 0.2, 0.21, 0.213, 0.25, 0.29, 0.30, 0.32],
        "capex_pct_revenue": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.20],
    }
    target = {"EV_EBITDA": 16.38, "FCF_yield": 0.0277, "operating_margin": 0.4144,
              "capex_pct_revenue": 0.15}
    p = _placements(target, cohort)
    assert p["EV_EBITDA"]["distribution"]["max"] == 21.20  # a peer is richer
    assert p["EV_EBITDA"]["quartile"] == 4
    assert p["EV_EBITDA"]["interpretation"] == "top-quartile multiple"
    # A below-median yield is not a "below-median multiple" (it is the rich side).
    assert p["FCF_yield"]["quartile"] == 2
    assert p["FCF_yield"]["interpretation"] == "below-median yield"
    assert p["capex_pct_revenue"]["interpretation"] == "top-quartile intensity"
    assert p["operating_margin"]["interpretation"] == "top quartile"  # unchanged group


@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_no_valuation_label_claims_an_extremum_or_shifts_keyword_signals(q):
    old = {4: "richest in cohort", 3: "above-median multiple",
           2: "below-median multiple", 1: "cheapest in cohort"}[q]
    # One ranked KPI per group, target placed in quartile q of 8 cohort values.
    cohort = [float(i) for i in range(1, 9)]
    target_for_q = {1: 0.5, 2: 2.5, 3: 4.5, 4: 8.5}[q]
    for kpi in ("EV_EBITDA", "FCF_yield", "capex_pct_revenue"):
        p = _placements({kpi: target_for_q}, {kpi: cohort})[kpi]
        assert p["quartile"] == q
        label = p["interpretation"]
        assert not any(w in label for w in ("richest", "cheapest", "in cohort"))
        # ...and names what was ranked: a yield or a spend ratio is not a multiple.
        measure = {"EV_EBITDA": "multiple", "FCF_yield": "yield",
                   "capex_pct_revenue": "intensity"}[kpi]
        assert label.endswith(measure), (kpi, label)
        # Wording only: deterministic bull/bear line selection and influence
        # tone read these lines by keyword, and must score them as before.
        line = f"{kpi}: 1.0 vs cohort median 1.0 — "
        for polarity in ("bull", "bear"):
            new_pick = graph._findings_signal_lines(
                AgentFinding(agent="Sector Analyst", headline="", summary="",
                             key_points=[line + label]), polarity=polarity)
            old_pick = graph._findings_signal_lines(
                AgentFinding(agent="Sector Analyst", headline="", summary="",
                             key_points=[line + old]), polarity=polarity)
            assert bool(new_pick) == bool(old_pick)
        assert influence._tone_score(label) == influence._tone_score(old)


# ---------------------------------------------------------------------------
# Cohort trend deltas
# ---------------------------------------------------------------------------

def test_cohort_margin_and_capex_deltas_are_printed_in_points():
    lines = sector_agents._format_trends({
        "cohort_op_margin_delta": 0.029, "cohort_capex_delta": -0.012,
    })
    assert "Cohort op margin expanding (+2.9pp multi-year)" in lines
    assert "Cohort capex intensity moderating (-1.2pp multi-year)" in lines
    assert not any("% multi-year" in s for s in lines)
