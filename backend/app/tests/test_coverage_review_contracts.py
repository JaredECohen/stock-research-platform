"""Consumer regressions for the learning-loop and share-class review."""
from __future__ import annotations

import logging
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import database
from app.models import Company, CronLoopRun, MemoOutcome, MemoPostmortem, MemoSnapshot
from app.monitoring import news_loop, postmortem_loop
from app.schemas import AgentFinding, BullBearCase, CriticReview, StockMemoOut
from app.services import data_service, fundamentals_service, memo_store
from app.services import postmortem_service as pm


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'coverage.db'}")
    database.Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    for module in (database, pm, memo_store):
        monkeypatch.setattr(module, "SessionLocal", sessions)
    yield sessions
    engine.dispose()


def _due(sessions, ticker, *, version=1, rating="Bullish"):
    with sessions() as db:
        snap = MemoSnapshot(
            ticker=ticker, version=version,
            memo_json={"ticker": ticker, "rating_label": rating},
        )
        db.add(snap)
        db.flush()
        db.add(MemoOutcome(
            memo_snapshot_id=snap.id, ticker=ticker, horizon_days=30,
            forward_return=.1, benchmark_return=.05, alpha=.05,
        ))
        db.commit()
        return snap.id


def test_postmortem_budget_names_every_omission_in_persisted_note_and_log(
    isolated_db, caplog,
):
    tickers = [f"CAP{i}" for i in range(8)]
    for ticker in tickers:
        _due(isolated_db, ticker)
    with caplog.at_level(logging.INFO):
        postmortem_loop.run_once(limit_per_horizon=1)
    with isolated_db() as db:
        row = db.query(CronLoopRun).filter_by(loop_name="postmortem_loop").one()
        assert row.success
        assert "deferred=7" in row.note
        for ticker in tickers[1:]:
            assert ticker in row.note
            assert ticker in caplog.text
        assert db.query(MemoPostmortem).count() == 1


def test_postmortem_rate_limit_applies_to_prior_writes_in_same_pass(
    isolated_db, monkeypatch, caplog,
):
    for version, rating in enumerate(("Bullish", "Bearish", "Neutral"), 1):
        _due(isolated_db, "RATE", version=version, rating=rating)
    calls = []
    monkeypatch.setattr(pm, "_llm_postmortem", lambda *args: calls.append(args))
    with caplog.at_level(logging.INFO):
        report = pm.run_postmortems(horizon_days=30, limit=25)
    assert report["written"] == 1
    assert report["deduped"] == 2
    assert report["skipped"] == 0
    assert len(calls) == 1
    assert len(report["deduped_memos"]) == 2
    assert all("recent postmortem" in row["reason"] for row in report["deduped_memos"])
    assert "RATE" in postmortem_loop._summarize(report)
    assert "RATE" in caplog.text
    with isolated_db() as db:
        assert db.query(MemoPostmortem).count() == 1


def test_zero_postmortem_budget_performs_no_work_and_names_all_due(isolated_db):
    _due(isolated_db, "ZERO")
    report = pm.run_postmortems(horizon_days=30, limit=0)
    assert report["written"] == report["due"] == 0
    assert report["deferred"] == 1
    assert report["deferred_memos"][0]["ticker"] == "ZERO"


def test_share_class_profile_keeps_memo_identity_through_real_cache_and_store(
    isolated_db, monkeypatch,
):
    ds = data_service.DataService()
    requests = []

    def response(path, **params):
        requests.append((path, params["symbol"]))
        if path == "/profile" and params["symbol"] == "BRK-B":
            return [{"symbol": "BRK-B", "companyName": "Berkshire Hathaway"}]
        return []

    # Drive the actual FMP normalizer and database cache. The HTTP boundary
    # supplies a provider-native symbol, rather than teaching a fake provider
    # to emit the canonical symbol the caller needs.
    monkeypatch.setattr(ds.fmp, "_get", response)
    monkeypatch.setattr(ds, "_live_chain", lambda capability: [ds.fmp])
    profile = ds.get_company_profile("BRK.B")
    assert ("/profile", "BRK-B") in requests
    count = len(requests)
    warm = ds.get_company_profile("BRK.B")
    assert len(requests) == count
    assert profile["ticker"] == warm["ticker"] == "BRK.B"
    monkeypatch.setattr(data_service, "get_data_service", lambda: ds)
    for method in ("get_financial_statements", "get_ratios", "get_earnings"):
        monkeypatch.setattr(ds, method, lambda ticker: None)
    fin = fundamentals_service._build_full_financials("BRK.B")
    finding = AgentFinding(agent="test", headline="test", summary="test", confidence=.5)
    # Composition chooses profile.ticker; exercise that consumer's persistence
    # contract without invoking the memo-generation pipeline or an LLM.
    memo = StockMemoOut(
        ticker=fin["profile"]["ticker"], company_name="Berkshire Hathaway",
        sector="Financials", final_pm_view="test", rating_label="Neutral",
        confidence_score=50, one_sentence_thesis="test", business_summary="test",
        sector_agent_view=finding, earnings_agent_view=finding,
        filing_agent_view=finding, valuation_agent_view=finding,
        comps_agent_view=finding, macro_sensitivity=finding,
        bull_case=BullBearCase(headline="test", key_points=[]),
        bear_case=BullBearCase(headline="test", key_points=[]),
        catalysts=[], key_risks=[], thesis_breakers=[],
        risk_committee_challenge=CriticReview(overall_assessment="test"),
        final_verdict="test",
    )
    memo_store.save_memo(memo, trigger="first_run")
    assert memo_store.latest_memo("BRK.B") is not None
    assert memo_store.latest_memo("BRK-B") is None


def test_news_agent_outage_is_failure_in_persisted_run(isolated_db, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic outage")

    monkeypatch.setattr(news_loop.news_agent, "run", unavailable)
    monkeypatch.setattr(news_loop, "_last_run_for", lambda ticker: None)
    news_loop.run_once(["OFFLINE"])
    with isolated_db() as db:
        row = db.query(CronLoopRun).filter_by(loop_name="news_loop").one()
        assert not row.success
        assert "1 news agents failed: OFFLINE" in row.note


def test_news_budget_covers_25_and_names_all_five_remaining(isolated_db, monkeypatch, caplog):
    tickers = [f"NEWS{i:02}" for i in range(30)]
    with isolated_db() as db:
        for ticker in tickers:
            db.add(Company(ticker=ticker, company_name=ticker, sector="Technology", industry="Software"))
            db.add(MemoSnapshot(ticker=ticker, version=1, generated_at=datetime.utcnow()))
        db.commit()
    calls = []
    monkeypatch.setattr(news_loop.news_agent, "run", lambda ticker, **kwargs: calls.append(ticker) or [])
    monkeypatch.setattr(news_loop, "_last_run_for", lambda ticker: None)
    monkeypatch.setattr(news_loop, "_record_run_for", lambda ticker: None)
    with caplog.at_level(logging.INFO):
        news_loop.run_once()
    assert len(calls) == len(set(calls)) == 25
    with isolated_db() as db:
        row = db.query(CronLoopRun).filter_by(loop_name="news_loop").one()
        assert row.success
        assert "dropped 5" in row.note
        for ticker in set(tickers) - set(calls):
            assert ticker in row.note
            assert ticker in caplog.text
