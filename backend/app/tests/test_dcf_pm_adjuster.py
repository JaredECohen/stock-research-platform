"""Wave 10 — PM-driven DCF assumption adjuster tests.

Covers:
  - No LLM available → returns (None, [], "") so caller falls through.
  - LLM proposes valid updates → DCF rebuilt + audit trail emitted.
  - LLM proposes update without rationale → silently dropped (discipline).
  - LLM proposes out-of-range value → clamped to ±20% of prior.
  - LLM raises → caller gets the no-op signal, doesn't blow up.

The LLM call (`_propose_adjustments`) is monkey-patched per test. The
`build_dcf` rebuild is also patched so we don't need a full fundamentals
fixture — the adjuster's contract is "given findings + prior DCF, propose
+ rebuild"; we test the propose+clamp+audit path independently of the
DCF compute layer.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from app.agents import dcf_pm_adjuster as adj
from app.schemas import (
    AgentFinding,
    DCFAssumptions,
    DCFResult,
    DCFScenario,
)


def _stub_finding(agent: str, summary: str = "stub") -> AgentFinding:
    return AgentFinding(
        agent=agent, headline=f"{agent} headline",
        summary=summary, key_points=["pt"], confidence=0.7, sources=[],
    )


def _stub_assumptions() -> DCFAssumptions:
    return DCFAssumptions(
        revenue_growth=[0.10, 0.09, 0.08, 0.07, 0.06],
        operating_margin=[0.25, 0.26, 0.27, 0.27, 0.27],
        tax_rate=0.21,
        da_pct_revenue=0.04,
        capex_pct_revenue=0.05,
        nwc_pct_revenue=0.02,
        terminal_growth=0.025,
        exit_ebitda_multiple=15.0,
        wacc=0.085,
        base_revenue=1_000_000.0,
        net_debt=0.0,
        diluted_shares=1_000_000.0,
        current_price=100.0,
    )


def _stub_dcf() -> DCFResult:
    a = _stub_assumptions()
    base = DCFScenario(
        name="base", label="base",
        assumptions=a,
        projections=[],
        pv_explicit=0.0,
        terminal_value_gordon=0.0,
        terminal_value_exit_multiple=0.0,
        pv_terminal_gordon=0.0,
        pv_terminal_exit=0.0,
        enterprise_value_gordon=100_000_000.0,
        enterprise_value_exit=100_000_000.0,
        enterprise_value_blended=100_000_000.0,
        equity_value=100_000_000.0,
        implied_share_price=100.0,
        upside_pct=0.0,
    )
    bull = base.model_copy(update={"name": "bull", "implied_share_price": 130.0, "upside_pct": 0.30})
    bear = base.model_copy(update={"name": "bear", "implied_share_price": 75.0, "upside_pct": -0.25})
    return DCFResult(
        ticker="TEST", current_price=100.0,
        base=base, bull=bull, bear=bear,
        sensitivities=[], summary="stub DCF",
    )


def _stub_findings() -> Dict[str, AgentFinding]:
    return {
        "sector": _stub_finding("Sector Analyst", "cohort growth re-accelerating"),
        "earnings": _stub_finding("Earnings Analyst", "tone constructive"),
        "valuation": _stub_finding("Valuation Analyst"),
    }


def test_no_llm_returns_no_op(monkeypatch):
    monkeypatch.setattr(type(adj.settings), "has_llm", property(lambda self: False))
    out = adj.adjust_dcf_for_pm_view(
        ticker="TEST", initial_dcf=_stub_dcf(),
        findings=_stub_findings(), run_id="r1",
    )
    assert out == (None, [], "")


def test_no_initial_dcf_returns_no_op(monkeypatch):
    monkeypatch.setattr(type(adj.settings), "has_llm", property(lambda self: True))
    out = adj.adjust_dcf_for_pm_view(
        ticker="TEST", initial_dcf=None,
        findings=_stub_findings(), run_id="r2",
    )
    assert out == (None, [], "")


def test_llm_proposal_with_rationale_rebuilds_dcf(monkeypatch):
    """Happy path: PM proposes a margin trim, supplies rationale,
    runtime rebuilds the DCF with adjusted assumptions."""
    monkeypatch.setattr(type(adj.settings), "has_llm", property(lambda self: True))

    # Mock the LLM proposal — trim revenue_growth by ~5% in years 3-5.
    monkeypatch.setattr(adj, "_propose_adjustments", lambda **kw: {
        "updates": {
            "revenue_growth": [0.10, 0.09, 0.075, 0.065, 0.055],
            "wacc": 0.090,
        },
        "rationales": {
            "revenue_growth": "sector agent: cohort decelerating",
            "wacc": "macro agent: rate-up regime",
        },
        "headline": "trimmed growth + raised WACC per sector + macro reads",
    })

    rebuilt_dcf = _stub_dcf()
    rebuilt_dcf = rebuilt_dcf.model_copy(update={"summary": "rebuilt DCF"})

    captured: Dict[str, Any] = {}

    def fake_build_dcf(ticker, *, assumptions=None, force_refresh=False):
        captured["ticker"] = ticker
        captured["assumptions"] = assumptions
        return rebuilt_dcf

    import app.services.valuation_service as vs
    monkeypatch.setattr(vs, "build_dcf", fake_build_dcf)

    out_dcf, audit, headline = adj.adjust_dcf_for_pm_view(
        ticker="TEST", initial_dcf=_stub_dcf(),
        findings=_stub_findings(), run_id="r3",
    )
    assert out_dcf is rebuilt_dcf
    assert "trimmed growth" in headline
    assert captured["ticker"] == "TEST"
    # Both fields proposed showed up in the audit; rationale string preserved.
    fields_changed = {row["field"].split("[")[0] for row in audit}
    assert "revenue_growth" in fields_changed
    assert "wacc" in fields_changed
    # WACC clamp: prior 0.085, proposed 0.090 → within ±20% → applied as-is.
    wacc_row = next(r for r in audit if r["field"] == "wacc")
    assert wacc_row["from"] == pytest.approx(0.085)
    assert wacc_row["to"] == pytest.approx(0.090)
    assert "macro" in wacc_row["rationale"].lower()


def test_proposal_without_rationale_dropped(monkeypatch):
    """Discipline: any field without a rationale is silently discarded.
    The whole call returns no-op when no field had a rationale."""
    monkeypatch.setattr(type(adj.settings), "has_llm", property(lambda self: True))
    monkeypatch.setattr(adj, "_propose_adjustments", lambda **kw: {
        "updates": {"wacc": 0.10},
        "rationales": {"wacc": ""},  # empty — drop
        "headline": "team triangulates",
    })
    out_dcf, audit, headline = adj.adjust_dcf_for_pm_view(
        ticker="TEST", initial_dcf=_stub_dcf(),
        findings=_stub_findings(), run_id="r4",
    )
    assert out_dcf is None
    assert audit == []
    assert "triangulates" in headline


def test_clamp_caps_extreme_proposal(monkeypatch):
    """Runtime clamps changes >±20% of the prior to keep a hallucinating
    LLM from yanking WACC from 8.5% to 25%."""
    monkeypatch.setattr(type(adj.settings), "has_llm", property(lambda self: True))
    monkeypatch.setattr(adj, "_propose_adjustments", lambda **kw: {
        "updates": {"wacc": 0.25},  # absurdly high
        "rationales": {"wacc": "extreme proposal"},
        "headline": "clamp test",
    })

    rebuilt = _stub_dcf()
    import app.services.valuation_service as vs
    monkeypatch.setattr(vs, "build_dcf", lambda ticker, **kw: rebuilt)

    out_dcf, audit, _ = adj.adjust_dcf_for_pm_view(
        ticker="TEST", initial_dcf=_stub_dcf(),
        findings=_stub_findings(), run_id="r5",
    )
    assert out_dcf is rebuilt
    # ±20% of 0.085 = 0.017 → max allowed 0.102.
    wacc_row = next(r for r in audit if r["field"] == "wacc")
    assert wacc_row["to"] == pytest.approx(0.102, rel=1e-3)


def test_propose_returning_none_falls_through(monkeypatch):
    """LLM-call failure → caller gets no-op tuple. Memo build proceeds
    with the initial DCF."""
    monkeypatch.setattr(type(adj.settings), "has_llm", property(lambda self: True))
    monkeypatch.setattr(adj, "_propose_adjustments", lambda **kw: None)
    out = adj.adjust_dcf_for_pm_view(
        ticker="TEST", initial_dcf=_stub_dcf(),
        findings=_stub_findings(), run_id="r6",
    )
    assert out == (None, [], "")


def test_rebuild_failure_returns_no_op(monkeypatch):
    """If `build_dcf` itself raises during the rebuild, the function
    returns no-op rather than a half-rebuilt result."""
    monkeypatch.setattr(type(adj.settings), "has_llm", property(lambda self: True))
    monkeypatch.setattr(adj, "_propose_adjustments", lambda **kw: {
        "updates": {"wacc": 0.090},
        "rationales": {"wacc": "macro tilt"},
        "headline": "test",
    })

    def boom(ticker, **kw):
        raise RuntimeError("simulated rebuild failure")

    import app.services.valuation_service as vs
    monkeypatch.setattr(vs, "build_dcf", boom)

    out = adj.adjust_dcf_for_pm_view(
        ticker="TEST", initial_dcf=_stub_dcf(),
        findings=_stub_findings(), run_id="r7",
    )
    assert out == (None, [], "")
