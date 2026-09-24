"""Old schemas must be repaired before queues or HTTP lifespan can start."""
from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import OperationalError

from app import database
from app.models import IndustryReportJob, RegenJob


@pytest.fixture
def old_schema(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'old-schema.sqlite'}")
    RegenJob.__table__.create(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE regen_jobs DROP COLUMN owner_token"))
        conn.execute(text("ALTER TABLE regen_jobs DROP COLUMN lease_expires_at"))
    monkeypatch.setattr(database, "engine", engine)
    yield engine
    engine.dispose()


def test_bootstrap_repairs_old_job_schema_and_is_idempotent(old_schema):
    with old_schema.connect() as conn, pytest.raises(OperationalError):
        conn.execute(select(RegenJob).limit(0))
    database.bootstrap_runtime_schema()
    database.bootstrap_runtime_schema()
    with old_schema.connect() as conn:
        assert conn.execute(select(RegenJob).limit(0)).all() == []


def test_silent_reconciliation_failure_cannot_pass_readiness(old_schema, monkeypatch):
    # The real reconciler intentionally catches failed ALTERs/inspection. Even
    # if init_db returns normally, the actual mapped query must still fail.
    monkeypatch.setattr(database, "reconcile_missing_columns", lambda: [])
    with pytest.raises(OperationalError, match="owner_token"):
        database.bootstrap_runtime_schema()


def test_worker_repairs_schema_before_still_running_async_seed_and_queue(old_schema, monkeypatch):
    from app import worker
    from app.config import settings
    from app.services import memory_probe, outcome_eligibility, regen_worker, scorecard_service, vector_store

    seed_started, release_seed = threading.Event(), threading.Event()
    events = []
    seed_threads = []

    def classify(*, db):
        # W6: the seed thread's FIRST step, DB-only; it must finish before
        # the minutes-long provider seed starts and never block the queue.
        assert threading.current_thread() is not threading.main_thread()
        events.append("eligibility_classified")
        return {}

    def seed():
        seed_threads.append(threading.current_thread())
        # This SELECT fails if the worker leaves migration to this thread.
        with old_schema.connect() as conn:
            conn.execute(select(RegenJob).limit(0))
        seed_started.set()
        assert release_seed.wait(5)
        return {}

    def start_queue():
        assert seed_started.wait(2)
        assert not release_seed.is_set(), "schema readiness waited for provider seed"
        with old_schema.connect() as conn:
            conn.execute(select(RegenJob).limit(0))
        events.append("queue_started_with_seed_pending")
        worker._shutdown.set()
        return True

    monkeypatch.setattr(worker, "_shutdown", threading.Event())
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(settings, "enable_monitoring", False)
    monkeypatch.setattr(settings, "enable_industry_reports", False)
    monkeypatch.setattr("app.seed_universe.run_full_seed", seed)
    monkeypatch.setattr(outcome_eligibility, "classify_pending", classify)
    monkeypatch.setattr(vector_store, "backfill_and_index", lambda: {"skipped": True})
    monkeypatch.setattr(scorecard_service, "ensure_version_registered", lambda: {})
    monkeypatch.setattr(regen_worker, "start_worker", start_queue)
    monkeypatch.setattr(regen_worker, "stop_worker", lambda: None)
    monkeypatch.setattr("app.services.industry_report_worker.stop_worker", lambda: None)
    monkeypatch.setattr(memory_probe, "log_rss", lambda *a: None)
    monkeypatch.setattr(worker, "_heartbeat", lambda: None)
    try:
        assert worker.main() == 0
        assert events == ["eligibility_classified", "queue_started_with_seed_pending"]
    finally:
        release_seed.set()
        for thread in seed_threads:
            thread.join(timeout=2)
            assert not thread.is_alive()


def test_worker_boot_sweep_failure_does_not_stop_the_seed_thread(old_schema, monkeypatch):
    """W6: the boot sweep fails closed (ExclusionSetMismatch when production
    differs from the evidence). The seed thread must log it and carry on to
    the provider seed, the pgvector backfill and scorecard registration, and
    the queue must still start. Without the handler, the exception ended the
    daemon thread before any of them ran."""
    from app import worker
    from app.config import settings
    from app.services import memory_probe, outcome_eligibility, regen_worker, scorecard_service, vector_store

    events: list[str] = []
    done = threading.Event()
    seed_threads = []

    def classify(*, db):
        events.append("eligibility_failed")
        raise outcome_eligibility.ExclusionSetMismatch("x")

    def seed():
        seed_threads.append(threading.current_thread())
        events.append("seed")
        return {}

    def register():
        events.append("scorecard")
        done.set()
        return {}

    def start_queue():
        events.append("queue")
        worker._shutdown.set()
        return True

    monkeypatch.setattr(worker, "_shutdown", threading.Event())
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(settings, "enable_monitoring", False)
    monkeypatch.setattr(settings, "enable_industry_reports", False)
    monkeypatch.setattr("app.seed_universe.run_full_seed", seed)
    monkeypatch.setattr(outcome_eligibility, "classify_pending", classify)
    monkeypatch.setattr(vector_store, "backfill_and_index", lambda: events.append("pgvector") or {"skipped": True})
    monkeypatch.setattr(scorecard_service, "ensure_version_registered", register)
    monkeypatch.setattr(regen_worker, "start_worker", start_queue)
    monkeypatch.setattr(regen_worker, "stop_worker", lambda: None)
    monkeypatch.setattr("app.services.industry_report_worker.stop_worker", lambda: None)
    monkeypatch.setattr(memory_probe, "log_rss", lambda *a: None)
    monkeypatch.setattr(worker, "_heartbeat", lambda: None)
    assert worker.main() == 0
    assert done.wait(5), f"seed thread stopped early: {events}"
    for thread in seed_threads:
        thread.join(timeout=2)
    assert "queue" in events
    seed_steps = [e for e in events if e != "queue"]
    assert seed_steps == ["eligibility_failed", "seed", "pgvector", "scorecard"]


@pytest.mark.parametrize("entrypoint", ["worker", "web"])
@pytest.mark.parametrize("failure", ["init", "missing_columns"])
def test_failed_schema_stops_every_startup_consumer(old_schema, monkeypatch, entrypoint, failure):
    from app import worker
    from app.config import settings
    from app.main import create_app

    events = []
    if failure == "init":
        def failed_init():
            raise RuntimeError("schema setup failed")
        monkeypatch.setattr(database, "init_db", failed_init)
        error = RuntimeError
    else:
        monkeypatch.setattr(database, "reconcile_missing_columns", lambda: [])
        error = OperationalError
    monkeypatch.setattr("app.seed_universe.run_full_seed", lambda: events.append("seed"))
    monkeypatch.setattr("app.services.regen_worker.start_worker", lambda: events.append("queue"))
    monkeypatch.setattr("app.monitoring.register_all", lambda *a: events.append("scheduler"))
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(settings, "enable_monitoring", True)
    with pytest.raises(error):
        if entrypoint == "worker":
            worker.main()
        else:
            with TestClient(create_app()):
                events.append("accepting_requests")
    assert events == []


def test_web_schema_ready_before_seed_and_queue(old_schema, monkeypatch):
    from app.config import settings
    from app.main import create_app

    events = []

    def consumer(name):
        with old_schema.connect() as conn:
            conn.execute(select(RegenJob).limit(0))
        events.append(name)
        return {}

    monkeypatch.setattr(settings, "enable_monitoring", False)
    monkeypatch.setattr("app.seed_universe.run_full_seed", lambda: consumer("seed"))
    monkeypatch.setattr("app.services.regen_worker.start_worker", lambda: consumer("queue"))
    with TestClient(create_app()):
        assert events == ["seed", "queue"]


@pytest.mark.parametrize("column", ["owner_token", "lease_expires_at", "snapshot_id"])
def test_industry_fence_columns_are_required_even_if_reconciliation_swallows_failure(tmp_path, monkeypatch, column):
    engine = create_engine(f"sqlite:///{tmp_path / 'industry-schema.sqlite'}")
    IndustryReportJob.__table__.create(engine)
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE industry_report_jobs DROP COLUMN {column}"))
    monkeypatch.setattr(database, "engine", engine)
    with monkeypatch.context() as patch:
        patch.setattr(database, "reconcile_missing_columns", lambda: [])
        with pytest.raises(OperationalError, match=column):
            database.bootstrap_runtime_schema()
    database.bootstrap_runtime_schema()
    with engine.connect() as conn:
        assert conn.execute(select(IndustryReportJob).limit(0)).all() == []
    engine.dispose()
