"""FIX-008b — units and labels in deterministic research text.

Pinned against the saved META v1 memo (docs/reviews/2026-09-13-META-v1.json,
local-only):

- Comps own-history: current operating margin 41.44% against an 8y median of
  38.825% is a 2.61 percentage-point gap and a 6.7% RELATIVE change. The
  saved narrative said "6.7 percentage points" because the only number on
  offer, `current_vs_own_median = 0.067`, carried no unit. The builder now
  also emits the gap in points, the deterministic text labels both, and the
  narrative LLM is handed the labelled field (prompt text untouched).
"""
from __future__ import annotations

import json
from unittest.mock import PropertyMock

import pytest

from app.agents import llm
from app.agents.comps_agent import run_comps_agent
from app.agents.long_form import deterministic_long_form
from app.config import Settings
from app.finance import comps_history as ch
from app.schemas import AgentFinding, CompsHistoryStats, CompsResult, CompsRow

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
