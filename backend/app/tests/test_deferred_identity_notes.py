"""Bounded work must retain all deferred identities in DB notes and logs."""
from __future__ import annotations

import logging

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import database, monitoring
from app.models import CronLoopRun
from app.monitoring import history_backfill, transcripts_poller


@pytest.mark.parametrize("loop_name", ["transcripts_poller", "history_backfill"])
def test_all_87_deferred_names_survive_actual_persistence_and_log(
    tmp_path, monkeypatch, caplog, loop_name,
):
    # Matches the size of the observed September 13 transcript overflow. No
    # provider calls, ingestion or memo generation is performed by this test.
    engine = create_engine(f"sqlite:///{tmp_path / 'notes.sqlite'}")
    CronLoopRun.__table__.create(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", sessions)
    monkeypatch.setattr(monitoring, "_LAST_RUNS", {})
    tickers = [f"DEFERRED{i:03d}" for i in range(89)]
    cap = 2
    deferred = tickers[cap:]
    processed, seen_writes = [], []
    caplog.set_level(logging.INFO, logger=f"app.monitoring.{loop_name}")

    if loop_name == "transcripts_poller":
        monkeypatch.setattr(transcripts_poller, "MAX_EVENT_TICKERS_PER_PASS", cap)
        monkeypatch.setattr(transcripts_poller, "get_transcripts", lambda t: [{"period": "2026Q2"}])
        monkeypatch.setattr(transcripts_poller, "_seen_periods", lambda t: {"2026Q1"})
        monkeypatch.setattr(transcripts_poller, "_save_seen_periods", lambda t, p: seen_writes.append(t))

        def handle(ticker, **kwargs):
            processed.append(ticker)
            return {"kind": "skipped"}

        monkeypatch.setattr("app.services.update_orchestrator.on_transcript_event", handle)
        result = transcripts_poller.run_once(tickers)
        assert len(result) == cap
        assert seen_writes == tickers[:cap], "Deferred periods must stay unseen"
        label = "ticker"
    else:
        monkeypatch.setattr(history_backfill, "MAX_COLD_TICKERS_PER_PASS", cap)
        monkeypatch.setattr(history_backfill, "_tier1_tickers", lambda: list(tickers))
        monkeypatch.setattr(history_backfill, "backfill_hits_provider", lambda t: True)

        def backfill(ticker, **kwargs):
            processed.append(ticker)
            return {}

        monkeypatch.setattr(history_backfill, "backfill_ticker", backfill)
        result = history_backfill.run_once(day=0)
        assert result["cold_reads"] == cap and result["deferred"] == 87
        label = "cold-read"

    assert processed == tickers[:cap], "Identity reporting must not increase work"
    with sessions() as db:
        row = db.scalars(select(CronLoopRun).where(CronLoopRun.loop_name == loop_name)).one()
        note = row.note
        assert row.success is True, "Budget deferral keeps its existing verdict"
    expected = f"deferred 87 over the {cap}-{label} cap: " + ", ".join(deferred)
    assert expected in note and "+82 more" not in note
    assert len(note) > 1000, "Exercise real long-note persistence"
    assert monitoring.status_snapshot()[loop_name]["note"] == note
    messages = [record.getMessage() for record in caplog.records if record.name == f"app.monitoring.{loop_name}"]
    assert f"{loop_name}: {note}" in messages
    engine.dispose()
