"""Postmortem selection must not materialize an entire backlog of memo bodies."""
from __future__ import annotations

import weakref
from datetime import datetime

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import MemoOutcome, MemoPostmortem, MemoSnapshot
from app.services import postmortem_service as pm


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'bounded.db'}")
    for model in (MemoSnapshot, MemoOutcome, MemoPostmortem):
        model.__table__.create(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(pm, "SessionLocal", sessions)
    monkeypatch.setattr(settings, "enable_long_term_memory", False)
    for key in ("openai_api_key", "anthropic_api_key", "gemini_api_key"):
        monkeypatch.setattr(settings, key, "")
    yield sessions, engine
    engine.dispose()


def seed(sessions, ticker, version=1, rating="Bullish", *, outcome=True):
    with sessions() as db:
        snap = MemoSnapshot(
            ticker=ticker, version=version, generated_at=datetime(2026, 1, 1),
            memo_json={"ticker": ticker, "rating_label": rating, "one_sentence_thesis": "actual full memo", "unused_body": "x" * (128 * 1024)},
            revision_log=["y" * (128 * 1024)],
        )
        db.add(snap)
        db.flush()
        if outcome:
            db.add(MemoOutcome(memo_snapshot_id=snap.id, ticker=ticker, horizon_days=90, forward_return=.2, benchmark_return=.05, alpha=.15))
        db.commit()
        return snap.id


def test_scan_never_loads_full_current_or_prior_memos_and_preserves_all_omissions(isolated):
    sessions, engine = isolated
    ids = [seed(sessions, f"BOUND{i:03}") for i in range(130)]
    unchanged = seed(sessions, "BOUND000", version=2)
    with sessions() as db:
        db.add(MemoPostmortem(memo_snapshot_id=ids[1], ticker="BOUND001", horizon_days=90, lesson="z" * (128 * 1024), created_at=datetime.utcnow()))
        db.commit()
    recent = seed(sessions, "BOUND001", version=2, rating="Bearish")
    loaded = []

    def record_load(target, _context):
        loaded.append(type(target).__name__)

    event.listen(MemoSnapshot, "load", record_load)
    event.listen(MemoPostmortem, "load", record_load)
    try:
        scan = pm._scan_due(90, limit=3)
    finally:
        event.remove(MemoSnapshot, "load", record_load)
        event.remove(MemoPostmortem, "load", record_load)
    assert loaded == [], "policy scan loaded memo bodies or prior postmortem lessons"
    assert [item["snapshot"].id for item in scan.items] == [ids[0], ids[2], ids[3]]
    assert [item["memo_snapshot_id"] for item in scan.deferred] == ids[4:]
    assert [item["memo_snapshot_id"] for item in scan.deduped] == [unchanged, recent]
    assert "rating unchanged" in scan.deduped[0]["reason"]
    assert "recent postmortem" in scan.deduped[1]["reason"]
    assert all(not hasattr(item["snapshot"], "memo_json") for item in scan.items)


def test_driver_hydrates_only_current_eligible_memo_and_does_not_retain_selected_bodies(isolated, monkeypatch):
    sessions, engine = isolated
    ids = [seed(sessions, f"PROCESS{i:03}") for i in range(8)]
    loaded = []
    calls = []

    def record_load(target, _context):
        loaded.append((target.id, weakref.ref(target)))

    def inspect_actual_memo(memo, outcome, horizon):
        calls.append(outcome.memo_snapshot_id)
        assert memo["one_sentence_thesis"] == "actual full memo"
        assert len(memo["unused_body"]) == 128 * 1024
        assert sum(ref() is not None for _, ref in loaded) <= 1
        return None

    monkeypatch.setattr(pm, "_llm_postmortem", inspect_actual_memo)
    event.listen(MemoSnapshot, "load", record_load)
    try:
        report = pm.run_postmortems(horizon_days=90, limit=3)
    finally:
        event.remove(MemoSnapshot, "load", record_load)
    assert calls == ids[:3]
    assert [id_ for id_, _ in loaded] == ids[:3]
    assert report["written"] == report["memory_disabled"] == 3
    assert report["skipped"] == 0
    assert [item["memo_snapshot_id"] for item in report["deferred_memos"]] == ids[3:]
    with sessions() as db:
        assert db.query(MemoPostmortem).count() == 3


def test_disappearing_selected_snapshot_reports_its_identity_without_a_model_call(isolated, monkeypatch):
    sessions, _ = isolated
    id_ = seed(sessions, "DISAPPEARED")
    scan = pm._scan_due(90, limit=1)
    with sessions() as db:
        db.query(MemoSnapshot).filter(MemoSnapshot.id == id_).delete()
        db.commit()
    monkeypatch.setattr(pm, "_scan_due", lambda *args, **kwargs: scan)

    def no_llm(*args, **kwargs):
        raise AssertionError("deleted snapshot cannot be postmortemed")

    monkeypatch.setattr(pm, "_llm_postmortem", no_llm)
    report = pm.run_postmortems(horizon_days=90, limit=1)
    assert report["written"] == 0
    assert report["skipped_memos"] == [{"ticker": "DISAPPEARED", "memo_snapshot_id": id_, "reason": "memo_snapshot_missing"}]


@pytest.mark.parametrize("rating", [None, "", 0, False, True, "Neutral", ["unusual"], {"value": "unusual"}])
def test_projected_rating_preserves_defensive_policy_semantics(isolated, rating):
    sessions, _ = isolated
    seed(sessions, "ODDRATING", outcome=False, rating=rating)
    seed(sessions, "ODDRATING", version=2, rating=rating)
    scan = pm._scan_due(90, limit=1)
    expected = str(rating or "").strip()
    assert len(scan.deduped) == bool(expected)
    assert len(scan.items) == (not expected)
