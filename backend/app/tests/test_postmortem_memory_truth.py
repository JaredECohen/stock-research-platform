"""Missing benchmark data and memory failures must survive the full driver."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app import database
from app.agents import llm
from app.config import settings
from app.memory import CompanyMemory, SectorMemory
from app.models import CronLoopRun, MemoOutcome, MemoPostmortem, MemoSnapshot
from app.monitoring import postmortem_loop
from app.services import market_data_service, outcome_service
from app.services import postmortem_service as pm


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'postmortem.db'}")
    database.Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    for module in (database, pm, outcome_service):
        monkeypatch.setattr(module, "SessionLocal", sessions)
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "memory"))
    monkeypatch.setattr(settings, "enable_long_term_memory", True)
    for key in ("openai_api_key", "anthropic_api_key", "gemini_api_key"):
        monkeypatch.setattr(settings, key, "")

    def no_chat(*args, **kwargs):
        raise AssertionError("postmortem must not call an LLM under blank keys")

    monkeypatch.setattr(llm, "chat_json", no_chat)
    yield sessions
    engine.dispose()


def _seed(sessions, ticker="PMMEMORY", *, outcome=True, horizon=90):
    with sessions() as db:
        snap = MemoSnapshot(
            ticker=ticker, version=1, generated_at=datetime(2026, 1, 2),
            memo_json={"ticker": ticker, "sector": "Technology", "rating_label": "Bullish"},
        )
        db.add(snap)
        db.flush()
        if outcome:
            db.add(MemoOutcome(
                memo_snapshot_id=snap.id, ticker=ticker, horizon_days=horizon,
                forward_return=.2, benchmark_return=.06, alpha=.14,
            ))
        db.commit()
        return snap


def test_actual_missing_benchmark_outcome_becomes_pending_postmortem_without_memory(
    isolated, monkeypatch, tmp_path, caplog,
):
    monkeypatch.setattr(settings, "enable_long_term_memory", False)
    snap = _seed(isolated, "PMNOBENCH", outcome=False)
    end_date = snap.generated_at.date() + timedelta(days=90)
    prices = [
        {"date": snap.generated_at.date().isoformat(), "close": 100.0},
        {"date": end_date.isoformat(), "close": 120.0},
    ]
    monkeypatch.setattr(market_data_service, "get_price_series", lambda ticker, *a, **k: prices if ticker == snap.ticker else [])
    with isolated() as db:
        outcome, status = outcome_service._evaluate_one(snap, 90, today=end_date, benchmark="SPY", db=db)
        assert status == "written"
        assert outcome.forward_return == .2
        assert outcome.benchmark_return is outcome.alpha is None
        db.commit()
        outcome_id = outcome.id
        original_note = outcome.note

    with caplog.at_level(logging.INFO):
        report = pm.run_postmortems(horizon_days=90)
    assert report["written"] == report["memory_disabled"] == 1
    assert report["skipped"] == report["memory_failed"] == report["memory_written"] == 0
    assert report["memory_disabled_memos"][0]["memo_snapshot_id"] == snap.id
    assert "PMNOBENCH" in caplog.text and "enable_long_term_memory=false" in caplog.text
    with isolated() as db:
        postmortem = db.query(MemoPostmortem).one()
        assert postmortem.verdict == "pending"
        assert "alpha unavailable" in postmortem.lesson
        assert "benchmark unavailable" in postmortem.lesson
        assert "Realized return 20.0%" in postmortem.lesson
        assert "alpha 0.0%" not in postmortem.lesson
        assert postmortem.written_to_memory is False
        original = db.get(MemoOutcome, outcome_id)
        assert original.benchmark_return is original.alpha is None
        assert original.note == original_note
    assert not (tmp_path / "memory").exists()


def test_disabled_memory_gate_blocks_both_writers_before_opening_files(isolated, monkeypatch):
    monkeypatch.setattr(settings, "enable_long_term_memory", False)

    def no_open(*args, **kwargs):
        raise AssertionError("disabled memory must not open company or sector files")

    monkeypatch.setattr(CompanyMemory, "for_ticker", no_open)
    monkeypatch.setattr(SectorMemory, "for_sector", no_open)
    result = pm._write_lesson_to_memory("DISABLED", "Technology", "lesson", "sector lesson")
    assert result.status == "disabled"
    assert result.written_targets == [] and result.errors == {}


@pytest.mark.parametrize("failed_target", ["company", "sector"])
def test_partial_memory_failure_is_named_and_does_not_mark_row_complete(
    isolated, monkeypatch, caplog, failed_target,
):
    snap = _seed(isolated)
    # A canned parsed result requests both files without any model call.
    monkeypatch.setattr(pm, "_llm_postmortem", lambda *args: {"lesson": "company lesson", "sector_lesson": "sector lesson"})

    def fail_save(self):
        raise OSError("simulated file write failure")

    monkeypatch.setattr(CompanyMemory if failed_target == "company" else SectorMemory, "save", fail_save)
    with caplog.at_level(logging.INFO):
        report = pm.run_postmortems(horizon_days=90)
    assert report["written"] == report["memory_failed"] == 1
    assert report["skipped"] == report["memory_written"] == report["memory_disabled"] == 0
    failure = report["memory_failed_memos"][0]
    assert failure["memo_snapshot_id"] == snap.id
    assert failure["errors"] == {failed_target: "OSError"}
    assert failure["written_targets"] == ["sector" if failed_target == "company" else "company"]
    assert snap.ticker in caplog.text and failed_target in caplog.text
    with isolated() as db:
        assert db.query(MemoPostmortem).one().written_to_memory is False
    # Existing rows are not retried or relabelled by a later run.
    again = pm.run_postmortems(horizon_days=90)
    assert again["written"] == again["memory_failed"] == 0
    with isolated() as db:
        assert db.query(MemoPostmortem).one().written_to_memory is False


def test_successful_memory_write_records_the_real_target_and_completion(isolated):
    snap = _seed(isolated)
    report = pm.run_postmortems(horizon_days=90)
    assert report["memory_written"] == 1
    assert report["memory_written_memos"][0]["memo_snapshot_id"] == snap.id
    assert report["memory_written_memos"][0]["written_targets"] == ["company"]
    with isolated() as db:
        assert db.query(MemoPostmortem).one().written_to_memory is True


@pytest.mark.parametrize("sector, sector_lesson, error", [
    ("Technology", 123, "invalid_lesson_type"),
    (None, "sector lesson", "sector_unavailable"),
])
def test_invalid_sector_request_reports_partial_company_save(
    isolated, monkeypatch, sector, sector_lesson, error,
):
    snap = _seed(isolated)
    with isolated() as db:
        db.get(MemoSnapshot, snap.id).memo_json = {**snap.memo_json, "sector": sector}
        db.commit()
    monkeypatch.setattr(pm, "_llm_postmortem", lambda *args: {
        "lesson": "company lesson", "sector_lesson": sector_lesson,
    })
    report = pm.run_postmortems(horizon_days=90)
    assert report["written"] == report["memory_failed"] == 1
    failure = report["memory_failed_memos"][0]
    assert failure["memo_snapshot_id"] == snap.id
    assert failure["errors"] == {"sector": error}
    assert failure["written_targets"] == ["company"]
    with isolated() as db:
        assert db.query(MemoPostmortem).one().written_to_memory is False


def test_nonstring_company_lesson_returns_error_without_raising(isolated):
    result = pm._write_lesson_to_memory("INVALID", "Technology", 123, "")
    assert result.status == "failed"
    assert result.errors == {"company": "invalid_lesson_type"}
    assert result.written_targets == []


def test_completion_flag_failure_is_reported_even_after_the_file_was_saved(isolated):
    snap = _seed(isolated)

    def fail_completion_flag(session):
        if any(isinstance(row, MemoPostmortem) and row.written_to_memory for row in session.dirty):
            raise OSError("simulated completion flag commit failure")

    event.listen(isolated.class_, "before_commit", fail_completion_flag)
    try:
        report = pm.run_postmortems(horizon_days=90)
    finally:
        event.remove(isolated.class_, "before_commit", fail_completion_flag)
    assert report["memory_failed"] == 1 and report["memory_written"] == 0
    failure = report["memory_failed_memos"][0]
    assert failure["memo_snapshot_id"] == snap.id
    assert failure["written_targets"] == ["company"]
    assert failure["errors"] == {"completion_flag": "OSError"}
    assert CompanyMemory.for_ticker(snap.ticker).path.exists()
    with isolated() as db:
        assert db.query(MemoPostmortem).one().written_to_memory is False


@pytest.mark.parametrize("mode", ["disabled", "failed"])
def test_loop_persists_distinct_memory_outcomes_and_every_identity(isolated, monkeypatch, mode):
    snap = _seed(isolated, "PMLOOP")
    monkeypatch.setattr(settings, "enable_long_term_memory", mode != "disabled")
    if mode == "failed":
        def fail_save(self):
            raise OSError("simulated")
        monkeypatch.setattr(CompanyMemory, "save", fail_save)
    postmortem_loop.run_once()
    with isolated() as db:
        run = db.query(CronLoopRun).filter_by(loop_name="postmortem_loop").one()
        assert run.success is (mode == "disabled")
        assert f"memory_{mode}=1" in run.note
        assert f"PMLOOP#{snap.id}" in run.note


@pytest.mark.parametrize("invalid", ["{broken", "[]"])
def test_parse_failure_names_the_skipped_snapshot(isolated, invalid):
    snap = _seed(isolated, horizon=30)
    with isolated() as db:
        db.get(MemoSnapshot, snap.id).memo_json = invalid
        db.commit()
    report = pm.run_postmortems(horizon_days=30)
    assert report["skipped"] == 1 and report["written"] == 0
    assert report["skipped_memos"][0]["memo_snapshot_id"] == snap.id
    assert report["skipped_memos"][0]["reason"].startswith("memo_parse_error:")
    assert f"{snap.ticker}#{snap.id}" in postmortem_loop._summarize(report)


def test_persist_failure_names_the_skipped_snapshot(isolated, monkeypatch):
    snap = _seed(isolated, horizon=30)
    monkeypatch.setattr(pm, "_persist_postmortem", lambda row: "failed")
    report = pm.run_postmortems(horizon_days=30)
    assert report["skipped"] == 1
    assert report["skipped_memos"] == [{
        "ticker": snap.ticker, "memo_snapshot_id": snap.id, "reason": "postmortem_persist_failed",
    }]
