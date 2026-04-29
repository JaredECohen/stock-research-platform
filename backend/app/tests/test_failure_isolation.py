"""Phase D — agent failure-isolation tests.

The contract: a thrown exception in any single specialist or critic must NOT
fail the whole memo. The memo is returned with:
  - a typed fallback finding for the failed agent (`confidence=0.0`,
    headline="X unavailable"),
  - the agent name appended to `memo.degraded_agents`,
  - all other agents' contributions intact.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agents import graph
from app.agents.safe_runner import (
    DegradationLog,
    safe_call,
    safe_critic,
    safe_finding,
)
from app.schemas import AgentFinding, CriticReview


def _boom(*args, **kwargs):
    raise RuntimeError("simulated agent failure")


# ---------------------------------------------------------------------------
# Unit tests on the safe_runner helpers
# ---------------------------------------------------------------------------

def test_safe_finding_returns_fallback_on_exception():
    log = DegradationLog()
    result = safe_finding("Sector Analyst", _boom, log_to=log)
    assert isinstance(result, AgentFinding)
    assert result.confidence == 0.0
    assert "unavailable" in result.headline.lower()
    assert log.degraded_agents() == ["Sector Analyst"]


def test_safe_finding_passes_through_normal_result():
    log = DegradationLog()
    happy = AgentFinding(
        agent="Sector Analyst", headline="ok", summary="ok", key_points=[], confidence=0.8,
    )
    result = safe_finding("Sector Analyst", lambda: happy, log_to=log)
    assert result is happy
    assert log.degraded_agents() == []


def test_safe_call_returns_fallback_on_exception():
    log = DegradationLog()
    out = safe_call(_boom, fallback="default", name="Thing", log_to=log)
    assert out == "default"
    assert log.degraded_agents() == ["Thing"]


def test_safe_critic_returns_fallback_review_on_exception():
    log = DegradationLog()
    review = safe_critic(_boom, log_to=log)
    assert isinstance(review, CriticReview)
    assert "unavailable" in review.overall_assessment.lower()


def test_safe_critic_passes_through_none_when_critic_disabled():
    """The critic legitimately returns None when ENABLE_AGENT_CRITIC=false;
    safe_critic must NOT convert that into a fallback review."""
    review = safe_critic(lambda *_: None)
    assert review is None


# ---------------------------------------------------------------------------
# End-to-end: run_stock_memo survives every specialist failing in turn
# ---------------------------------------------------------------------------

def _run_with(monkeypatch, agent_attr: str):
    """Run the NVDA memo with `agent_attr` patched to raise. Assert the memo
    still comes back and `degraded_agents` includes the failed agent."""
    monkeypatch.setattr(graph, agent_attr, _boom)
    memo = graph.run_stock_memo("NVDA")
    return memo


def test_memo_survives_sector_agent_failure(monkeypatch):
    memo = _run_with(monkeypatch, "run_sector_agent")
    assert memo.ticker == "NVDA"
    assert "Sector Analyst" in memo.degraded_agents
    assert memo.sector_agent_view.confidence == 0.0
    # Other findings are still real
    assert memo.earnings_agent_view.confidence > 0.0


def test_memo_survives_earnings_agent_failure(monkeypatch):
    memo = _run_with(monkeypatch, "run_earnings_agent")
    assert "Earnings Analyst" in memo.degraded_agents
    assert memo.earnings_agent_view.confidence == 0.0
    assert memo.sector_agent_view.confidence > 0.0


def test_memo_survives_filing_agent_failure(monkeypatch):
    memo = _run_with(monkeypatch, "run_filing_agent")
    assert "Filing Analyst" in memo.degraded_agents
    assert memo.filing_agent_view.confidence == 0.0


def test_memo_survives_valuation_agent_failure(monkeypatch):
    memo = _run_with(monkeypatch, "run_valuation_agent")
    assert "Valuation Analyst" in memo.degraded_agents
    assert memo.valuation_agent_view.confidence == 0.0


def test_memo_survives_comps_agent_failure(monkeypatch):
    memo = _run_with(monkeypatch, "run_comps_agent")
    assert "Comps Analyst" in memo.degraded_agents
    assert memo.comps_agent_view.confidence == 0.0


def test_memo_survives_macro_agent_failure(monkeypatch):
    memo = _run_with(monkeypatch, "run_macro_agent")
    assert "Macro Analyst" in memo.degraded_agents
    assert memo.macro_sensitivity.confidence == 0.0


def test_memo_survives_risk_agent_failure(monkeypatch):
    memo = _run_with(monkeypatch, "run_risk_agent")
    assert "Risk Analyst" in memo.degraded_agents


def test_memo_survives_critic_failure(monkeypatch):
    memo = _run_with(monkeypatch, "run_critic")
    assert "Risk Committee" in memo.degraded_agents
    # Even with the critic down, the memo carries a typed CriticReview placeholder.
    assert isinstance(memo.risk_committee_challenge, CriticReview)


def test_memo_survives_three_simultaneous_failures(monkeypatch):
    """Worst case: sector + valuation + critic all blow up. The memo must
    still come back populated, with all three names in `degraded_agents`."""
    monkeypatch.setattr(graph, "run_sector_agent", _boom)
    monkeypatch.setattr(graph, "run_valuation_agent", _boom)
    monkeypatch.setattr(graph, "run_critic", _boom)
    memo = graph.run_stock_memo("NVDA")
    assert memo.ticker == "NVDA"
    for name in ("Sector Analyst", "Valuation Analyst", "Risk Committee"):
        assert name in memo.degraded_agents
    # The healthy agents are still real.
    assert memo.earnings_agent_view.confidence > 0.0
    assert memo.macro_sensitivity.confidence > 0.0


def test_memo_with_no_failures_has_empty_degraded_agents():
    memo = graph.run_stock_memo("MSFT")
    assert memo.degraded_agents == []
