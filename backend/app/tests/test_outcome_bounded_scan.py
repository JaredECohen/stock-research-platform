"""Outcome sweeps should not read memo bodies for work they will skip."""
from __future__ import annotations

import weakref
from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import MemoOutcome, MemoSnapshot
from app.services import market_data_service
from app.services import outcome_service as svc

GENERATED = datetime(2026, 1, 1)
TODAY = date(2026, 5, 1)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'outcomes.db'}")
    for model in (MemoSnapshot, MemoOutcome):
        model.__table__.create(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(svc, "SessionLocal", sessions)
    monkeypatch.setattr(settings, "enable_long_term_memory", False)
    for key in ("openai_api_key", "anthropic_api_key", "gemini_api_key"):
        monkeypatch.setattr(settings, key, "")
    yield sessions, engine
    engine.dispose()


def seed(sessions, ticker, *, generated=GENERATED, recorded=False, backtest=False):
    with sessions() as db:
        snap = MemoSnapshot(
            ticker=ticker, version=7, generated_at=generated,
            as_of_date=generated if backtest else None,
            memo_json={"ticker": ticker, "rating_label": "Bullish", "confidence_score": 73.0,
                       "macro_regime_at_memo": "expansion", "unused_body": "x" * (128 * 1024)},
            revision_log=["y" * (128 * 1024)],
        )
        db.add(snap)
        db.flush()
        if recorded:
            db.add(MemoOutcome(memo_snapshot_id=snap.id, ticker=ticker, horizon_days=30,
                               forward_return=.123, benchmark_return=None, alpha=None,
                               note="immutable existing legacy result"))
        db.commit()
        return snap.id


def capture_reads(engine):
    queries = []

    def capture(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("SELECT") and "memo_snapshots" in statement:
            queries.append((statement, params))

    event.listen(engine, "before_cursor_execute", capture)
    return queries, capture


def body_queries(queries):
    return [(statement, params) for statement, params in queries if "memo_snapshots.memo_json" in statement]


def prices(_ticker, _days):
    return [{"date": GENERATED.date().isoformat(), "close": 100.0},
            {"date": (GENERATED.date() + timedelta(days=30)).isoformat(), "close": 110.0},
            {"date": (GENERATED.date() + timedelta(days=90)).isoformat(), "close": 120.0}]


def test_skipped_work_reads_no_bodies_and_existing_outcome_stays_unchanged(isolated, monkeypatch):
    sessions, engine = isolated
    ids = [seed(sessions, f"DONE{i}", recorded=True) for i in range(2)]
    seed(sessions, "FUTURE", generated=datetime(2026, 4, 30))
    seed(sessions, "MISSINGPRICES")
    seed(sessions, "BACKTEST", backtest=True)
    requests = []

    def missing(ticker, days):
        requests.append((ticker, days))
        return []

    monkeypatch.setattr(market_data_service, "get_price_series", missing)
    queries, capture = capture_reads(engine)
    try:
        report = svc.evaluate_all_due(horizons=[30], today=TODAY)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert body_queries(queries) == []
    assert all("revision_log" not in statement for statement, _ in queries)
    assert report["evaluated"] == 4 and report["due"] == 3
    assert report["already_recorded"] == 2 and report["not_due"] == 1
    assert report["data_unavailable"] == report["ticker_prices_unavailable"] == 1
    assert report["written"] == report["errors"] == 0
    assert [ticker for ticker, _ in requests] == ["MISSINGPRICES"]
    with sessions() as db:
        rows = db.execute(select(MemoOutcome).order_by(MemoOutcome.memo_snapshot_id)).scalars().all()
        assert [row.memo_snapshot_id for row in rows] == ids
        assert all(row.note == "immutable existing legacy result" and row.forward_return == .123
                   and row.benchmark_return is None and row.alpha is None for row in rows)


def test_only_scored_snapshot_body_is_loaded_once_across_horizons(isolated, monkeypatch):
    sessions, engine = isolated
    ids = [seed(sessions, f"SCORE{i}") for i in range(4)]
    monkeypatch.setattr(market_data_service, "get_price_series", prices)
    original = svc._evaluate_one
    refs = {}

    def observe(snap, *args, **kwargs):
        refs[snap.id] = weakref.ref(snap)
        result = original(snap, *args, **kwargs)
        assert len(snap.memo_json["unused_body"]) == 128 * 1024
        assert sum(ref() is not None and "memo_json" in ref().__dict__ for ref in refs.values()) <= 1
        return result

    monkeypatch.setattr(svc, "_evaluate_one", observe)
    queries, capture = capture_reads(engine)
    try:
        report = svc.evaluate_all_due(horizons=[30, 90], today=TODAY)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert report["evaluated"] == report["written"] == 8
    assert report["errors"] == 0
    assert len(body_queries(queries)) == len(ids)
    assert all("revision_log" not in statement for statement, _ in queries)
    with sessions() as db:
        rows = db.query(MemoOutcome).all()
        assert len(rows) == 8
        assert all(row.rating_at_memo == "Bullish" and row.confidence_at_memo == 73
                   and row.regime_at_memo == "expansion" for row in rows)


def test_metadata_paging_survives_commits_rollback_and_excludes_new_snapshot(isolated, monkeypatch):
    sessions, engine = isolated
    ids = [seed(sessions, f"PAGE{i:03}") for i in range(205)]
    monkeypatch.setattr(market_data_service, "get_price_series", prices)
    original = svc._evaluate_one
    seen = []
    added = []

    def evaluate(snap, *args, **kwargs):
        seen.append(snap.id)
        db = kwargs["db"]
        if snap.id == ids[100]:
            db.execute(text("SELECT * FROM deliberately_absent_table"))
        out = original(snap, *args, **kwargs)
        if snap.id == ids[0]:
            new = MemoSnapshot(ticker="ARRIVEDDURINGSCAN", version=1, generated_at=GENERATED,
                               memo_json={"rating_label": "Bearish"})
            db.add(new)
            db.flush()
            added.append(new.id)
        return out

    monkeypatch.setattr(svc, "_evaluate_one", evaluate)
    queries, capture = capture_reads(engine)
    try:
        report = svc.evaluate_all_due(horizons=[30], today=TODAY)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert seen == ids
    assert report["evaluated"] == report["due"] == 205
    assert report["written"] == 204 and report["errors"] == 1
    assert report["error_pairs"] == [f"PAGE100:snap={ids[100]}:30d:OperationalError"]
    assert added and added[0] not in seen
    assert len(body_queries(queries)) == 204
    pages = [statement for statement, _ in queries if "memo_snapshots.generated_at" in statement]
    assert len(pages) == 4 and all("LIMIT" in statement for statement in pages)
    with sessions() as db:
        actual = set(db.execute(select(MemoOutcome.memo_snapshot_id)).scalars())
        assert actual == set(ids) - {ids[100]}


def test_all_unavailable_identities_survive_pages_result_log_and_loop_note(isolated, monkeypatch, caplog):
    from app.monitoring import outcome_loop

    sessions, engine = isolated
    expected = []
    reasons = {"MISSING": "ticker_prices_unavailable", "SHORT": "price_history_too_short", "GAP": "price_window_incomplete"}
    for i in range(67):
        for prefix, reason in reasons.items():
            ticker = f"{prefix}{i:03}"
            id_ = seed(sessions, ticker)
            expected.append(f"{ticker}:snap={id_}:30d:{reason}")

    def incomplete(ticker, days):
        if ticker.startswith("MISSING"):
            return []
        if ticker.startswith("SHORT"):
            return [{"date": "2026-02-01", "close": 100.0}]
        if ticker.startswith("GAP"):
            return [{"date": "2026-01-01", "close": 100.0}, {"date": "2026-01-08", "close": 102.0}]
        return prices(ticker, days)

    monkeypatch.setattr(market_data_service, "get_price_series", incomplete)
    report = svc.evaluate_all_due(horizons=[30], today=TODAY)
    assert report["data_unavailable"] == report["due"] == report["evaluated"] == 201
    assert report["unavailable_pairs"] == expected
    assert all(report[key] == 67 for key in reasons.values())
    assert report["written"] == report["errors"] == 0
    assert all(identity in caplog.text for identity in expected)
    monkeypatch.setattr(outcome_loop, "evaluate_all_due", lambda: report)
    recorded = []
    monkeypatch.setattr(outcome_loop, "record_run", lambda *args, **kwargs: recorded.append(kwargs))
    assert outcome_loop.run_once() is report
    assert recorded[0]["success"] is False
    assert all(identity in recorded[0]["note"] for identity in expected)
