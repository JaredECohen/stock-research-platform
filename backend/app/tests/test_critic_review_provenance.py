"""A single prior memo matters; a missing critic is not a passed review."""
from datetime import datetime
from uuid import uuid4

import pytest

from app.agents import critic_agent, graph
from app.agents.intake import IntakeDecision
from app.agents.memo_context import AnalystRound, DCFStage
from app.agents.safe_runner import DegradationLog, safe_critic
from app.config import settings
from app.database import SessionLocal
from app.models import MemoSnapshot
from app.schemas import CriticReview
from app.tests.factories import make_findings, make_inputs


@pytest.fixture
def ticker():
    value = "ZC" + uuid4().hex[:8].upper()
    yield value
    with SessionLocal() as db:
        db.query(MemoSnapshot).filter_by(ticker=value).delete(synchronize_session=False)
        db.commit()


def _snapshot(ticker, version, thesis, *, as_of_date=None):
    with SessionLocal() as db:
        db.add(MemoSnapshot(ticker=ticker, version=version, as_of_date=as_of_date,
                            memo_json={"rating_label": "Neutral", "one_sentence_thesis": thesis}))
        db.commit()


def test_single_existing_live_memo_is_prior_before_current_draft_is_saved(ticker):
    _snapshot(ticker, 1, "Only prior view")
    context = critic_agent._prior_memo_context(ticker)
    assert "PRIOR MEMO (v1" in context
    assert "Only prior view" in context


def test_prior_is_latest_live_version_and_excludes_newer_backtest(ticker):
    _snapshot(ticker, 1, "Old prior")
    _snapshot(ticker, 2, "Latest live view")
    assert "PRIOR MEMO (v2" in critic_agent._prior_memo_context(ticker)
    assert "Latest live view" in critic_agent._prior_memo_context(ticker)
    _snapshot(ticker, 3, "Backtest view", as_of_date=datetime(2020, 1, 1))
    context = critic_agent._prior_memo_context(ticker)
    assert "PRIOR MEMO (v2" in context
    assert "Latest live view" in context
    assert "Old prior" not in context and "Backtest view" not in context


@pytest.mark.parametrize("live_intended", [True, False])
def test_rule_based_fallback_never_implies_passed_research_review(monkeypatch, live_intended):
    monkeypatch.setattr(critic_agent.llm, "chat_json", lambda *a, **kw: None)
    monkeypatch.setattr(settings, "enable_live_data", live_intended)
    monkeypatch.setattr(settings, "use_demo_data", not live_intended)
    log = DegradationLog()
    with log.activate():
        review = critic_agent.run_critic({"rating_label": "Neutral", "key_risks": ["Risk"],
                                          "dcf_summary": {"base": 1}, "sources_used": ["filing:1"]})
    assert review.review_mode == "rule_based"
    assert "Rule-based check only" in review.overall_assessment
    assert "structurally sound" not in review.overall_assessment
    assert "No major issues detected." not in review.challenges
    assert "not assessed" in review.advice_compliance_check
    assert ("Risk Committee" in log.degraded_agents()) is live_intended


def test_live_and_legacy_critic_modes_are_distinct(monkeypatch):
    monkeypatch.setattr(critic_agent.llm, "chat_json", lambda *a, **kw: {"overall_assessment": "Reviewed source conflict."})
    assert critic_agent.run_critic({}).review_mode == "live"
    assert CriticReview(overall_assessment="Legacy stored review.").review_mode == "unknown"
    def boom(*a, **kw):
        raise RuntimeError("offline")
    assert safe_critic(boom, {}).review_mode == "unavailable"


def test_composition_is_pending_and_only_real_review_invokes_critic(monkeypatch):
    from app.services import market_data_service
    monkeypatch.setattr(market_data_service, "get_current_price", lambda ticker: 100)
    monkeypatch.setattr(graph, "_run_reflection_step", lambda *a: ([], []))
    monkeypatch.setattr(graph, "_pm_synthesis", lambda *a, **kw: {
        "rating_label": "Neutral", "confidence_score": 60,
        "final_pm_view": "A complete draft", "one_sentence_thesis": "Unit margins fund reinvestment.",
    })
    calls = []
    def critic(draft):
        calls.append(draft)
        assert draft["ticker"] == "TEST" and draft["one_sentence_thesis"]
        return CriticReview(overall_assessment="Reviewed complete draft.", review_mode="live")
    monkeypatch.setattr(graph, "run_critic", critic)
    inputs = make_inputs(run_id=str(uuid4()))
    analysts = AnalystRound(findings=make_findings(), intake=IntakeDecision())
    memo = graph._compose_memo(inputs, analysts, DCFStage(dcf=None, initial_dcf=None))
    assert calls == []
    assert memo.risk_committee_challenge.review_mode == "pending"
    graph._review_memo(memo, inputs, analysts)
    assert len(calls) == 1
    assert memo.risk_committee_challenge.review_mode == "live"
