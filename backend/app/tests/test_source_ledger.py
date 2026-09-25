"""W2b 7(a): the source ledger (`agents/source_ledger.py`).

The ledger is what makes "untraceable" mean something: it must hold every
fact the analysts were given, nothing an LLM wrote, no reference to the
payloads, and it must never be able to hurt the analyst that registers.
"""
from __future__ import annotations

import pytest

from app.agents import earnings_agent, source_ledger
from app.agents.source_ledger import SourceLedger, active_ledger, register_source
from app.agents.valuation_agent import run_valuation_agent


def values(ledger: SourceLedger) -> set[float]:
    return {round(f.value, 6) for f in ledger.snapshot().facts}


def test_register_is_noop_without_active_ledger():
    """Chat, the screener and direct agent calls run with no ledger."""
    assert active_ledger() is None
    assert register_source("financials", "financials:T", {"ratios": {"PE": 20.0}}) is False


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown source kind"):
        register_source("memory", "memory:T", {"x": 1})
    ledger = SourceLedger()
    with ledger.activate(), pytest.raises(ValueError):
        register_source("llm_output", "x", "12%")
    with pytest.raises(ValueError):
        ledger.register("pm_view", "x", "12%")


def test_no_reference_kept():
    ledger = SourceLedger()
    payload = {"ratios": {"PE": 20.5}, "notes": ["margin 41.2%"]}
    ledger.register("financials", "financials:T", payload)
    before = values(ledger)
    payload["ratios"]["PE"] = 99.0
    payload["notes"].append("margin 77.7%")
    assert values(ledger) == before
    assert 99.0 not in before and 77.7 not in before


def test_dedupe_on_refire():
    """A deep-research re-fire registers the same payload again: nothing new."""
    ledger = SourceLedger()
    assert ledger.register("comps", "comps:T", {"target": {"ev_ebitda": 12.3}}) > 0
    n = ledger.fact_count
    assert ledger.register("comps", "comps:T", {"target": {"ev_ebitda": 12.3}}) == 0
    assert ledger.fact_count == n
    # Same ref, different content (a re-fired analyst read new data): added.
    assert ledger.register("comps", "comps:T", {"target": {"ev_ebitda": 14.1}}) > 0


def test_context_var_reset_after_exception():
    outer, inner = SourceLedger(), SourceLedger()
    with pytest.raises(RuntimeError):
        with outer.activate():
            with inner.activate():
                assert active_ledger() is inner
                raise RuntimeError("boom")
    assert active_ledger() is None
    with outer.activate():
        with inner.activate():
            pass
        assert active_ledger() is outer
    assert active_ledger() is None


def test_register_source_contains_walker_errors(monkeypatch, caplog):
    ledger = SourceLedger()

    def boom(*a, **k):
        raise RuntimeError("walker exploded")

    monkeypatch.setattr(SourceLedger, "register", boom)
    with ledger.activate():
        assert register_source("financials", "financials:T", {"x": 1}) is False
    snap = ledger.snapshot()
    assert not snap.complete and snap.incomplete_steps == ("register:financials:financials:T",)
    assert "figures will be reported unchecked" in caplog.text


def test_register_source_error_leaves_finding_unchanged(monkeypatch):
    """A registration failure inside an analyst never changes its finding."""
    profile = {"ticker": "NVDA", "last_price": 118.0}
    ratios = {"PE": 46.8, "EV_EBITDA": 34.4, "FCF_yield": 0.02}
    clean = run_valuation_agent(profile, ratios, None)

    def boom(*a, **k):
        raise RuntimeError("walker exploded")

    monkeypatch.setattr(SourceLedger, "register", boom)
    ledger = SourceLedger()
    with ledger.activate():
        broken = run_valuation_agent(profile, ratios, None)
    assert broken == clean
    assert not ledger.snapshot().complete


def test_llm_outputs_are_not_sources(monkeypatch):
    """A number that exists only in model output — the earnings multi-pass
    addendum, a DCF scenario driver's rationale — is never a source."""
    monkeypatch.setattr(earnings_agent, "_multi_pass_qa_addendum",
                        lambda **kw: {"hard_questions": ["Why did margin reach 87.65%?"]})
    ledger = SourceLedger()
    transcript = {"period": "2025Q3", "prepared_remarks": "Revenue grew 12.5% this quarter.", "qa": ""}
    with ledger.activate():
        earnings_agent.run_earnings_agent({"ticker": "T"}, transcript, {})
        from app.agents.graph import DCF_LLM_KEYS, DCF_TEXT_KEYS
        register_source("dcf", "dcf:initial", {
            "summary": "Base case implies $57.58.",
            "base": {"implied_share_price": 57.58,
                     "drivers": [{"name": "AI ramps", "rationale": "growth +777bp on a 66.6% share",
                                  "assumption_changes": ["revenue_growth"]}]},
            # A live driver NAME is LLM output too (scenario_assumptions
            # copies it from the model's JSON; the memo prints it as "DCF
            # driver — {name}: ..."), so its figure must not trace to itself.
            "bull": {"implied_share_price": 81.0,
                     "drivers": [{"name": "Data-center revenue reaches $412B", "rationale": "",
                                  "assumption_changes": []}]},
            # The engine's own sensitivity-grid name is still read.
            "sensitivities": [{"name": "WACC vs terminal growth at 9.25%", "row_axis": "wacc",
                               "col_axis": "terminal_growth", "rows": [], "cols": [], "cells": []}],
        }, text_keys=DCF_TEXT_KEYS, exclude_keys=DCF_LLM_KEYS)
    got = values(ledger)
    assert 12.5 in got and 57.58 in got and 81.0 in got and 9.25 in got
    assert 87.65 not in got and 66.6 not in got and 7.77 not in got
    assert 412e9 not in got
    reg = ledger.snapshot()
    from app.agents import number_check
    (c,) = number_check.check_text("DCF driver — Data-center revenue reaches $412B.", reg)
    assert c.status == "untraceable"


def test_long_term_memory_is_not_a_source(monkeypatch):
    """Memory is a prior, not evidence (owner decision 11): the sector
    analyst reads it, the ledger never holds it."""
    from app.agents import sector_agents
    from app.config import settings

    class Mem:
        def as_prompt_context(self, **kw):
            return "Last quarter operating margin was 91.23%."

        def as_prompt_context_for(self, *a, **kw):
            return "Sector margin 92.34%."

    monkeypatch.setattr(settings, "enable_long_term_memory", True)
    monkeypatch.setattr(sector_agents.CompanyMemory, "for_ticker", classmethod(lambda cls, t: Mem()))
    monkeypatch.setattr(sector_agents.SectorMemory, "for_sector", classmethod(lambda cls, s: Mem()))
    ledger = SourceLedger()
    with ledger.activate():
        sector_agents.run_sector_agent({"ticker": "NVDA", "sector": "Technology"}, {})
    got = values(ledger)
    assert got, "the sector analyst registered its research"
    assert 91.23 not in got and 92.34 not in got


def test_derived_facts():
    ledger = SourceLedger()
    ledger.register("financials", "financials:T", {"income": [
        {"period": "2024", "revenue": 100.0, "operating_income": 20.0},
        {"period": "2025", "revenue": 125.0, "operating_income": 30.0},
    ], "cash": [{"period": "2025", "cash_from_operations": 40.0, "capex": -15.0}]})
    ledger.register("dcf", "dcf:initial", {
        "current_price": 100.0,
        "base": {"implied_share_price": 120.0, "assumptions": {"wacc": 0.10, "revenue_growth": [0.2, 0.1]}},
        "bull": {"implied_share_price": 150.0, "assumptions": {"wacc": 0.095, "revenue_growth": [0.23, 0.13]}},
        "bear": {"implied_share_price": 60.0, "assumptions": {"wacc": 0.11, "revenue_growth": [0.17, 0.07]}},
    })
    ledger.register("comps", "comps:T", {
        "target": {"operating_margin": 0.41, "ev_ebitda": 20.0},
        "median": {"operating_margin": 0.38, "ev_ebitda": 16.0},
        "history": {"own_median": {"ev_ebitda": 25.0}},
    })
    derived = {(f.derived, f.unit, round(f.value, 4)) for f in ledger.snapshot().facts if f.derived}
    # D1 growth: revenue +25%, operating income +50%.
    assert ("D1", "ratio", 0.25) in derived and ("D1", "ratio", 0.5) in derived
    # D2: operating margin 24% (30/125) and FCF 25 (40 - 15) at a 20% margin.
    assert ("D2", "ratio", 0.24) in derived and ("D2", "ratio", 25.0) in derived
    assert ("D2", "ratio", 0.2) in derived
    # D3: bull/bear ratio 2.5, scenario deltas in points, bull-base spread, growth average.
    assert ("D3", "ratio", 2.5) in derived and ("D3", "pp", 3.0) in derived and ("D3", "pp", 0.5) in derived
    assert ("D3", "ratio", 30.0) in derived and ("D3", "ratio", 0.15) in derived
    # D4: target - median in points (3pp), relative premium (+25%), vs own history (-20%).
    assert ("D4", "pp", 3.0) in derived and ("D4", "ratio", 0.25) in derived and ("D4", "ratio", 0.2) in derived


def test_export_replay_round_trip():
    a = SourceLedger()
    with a.capture() as captured:
        a.register("technical", "technical:T", {"rsi_14": 68.2, "sma_50": 109.79})
    exported = SourceLedger.export(captured)
    b = SourceLedger()
    assert b.replay(exported) == len(captured)
    assert values(b) == values(a)
    assert b.snapshot().resolves("technical:T")
    with pytest.raises(ValueError):
        SourceLedger().replay({"v": 1, "strings": ["ratio", "memory"], "facts": [[1.0, 0, 1, 1, 1, 1, 0]]})


def test_kinds_and_primary_kinds_are_closed():
    assert source_ledger.PRIMARY_KINDS == {"financials", "filing", "transcript"}
    assert "prior_extraction" in source_ledger.SOURCE_KINDS
    assert source_ledger.PRIMARY_KINDS <= source_ledger.SOURCE_KINDS
