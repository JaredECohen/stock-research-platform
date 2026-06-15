"""Theme 5 — DB-backed memo-regen queue.

Cover:
- `enqueue` inserts a queued `RegenJob` and coalesces duplicates per
  ticker (across "restarts" — the row is in the DB, not process memory).
- `process_next_job` runs the claimed job: success records the memo
  version; failure records error_type / message / traceback tail.
- `ticker_status.last_failure` reproduces the old `_REGEN_FAILURES`
  payload shape and clears on the next success.
- `recover_orphans`: a `running` job from a dead process is requeued
  once with the same run_id (checkpoint resume), failed the second
  time (WorkerRestart); stale queued jobs expire (QueueExpired).
- API contract: POST /analyze returns the 202 shape Research.tsx
  expects; GET /analyze/status keeps every legacy field.

`run_stock_memo` is monkeypatched throughout — these are queue-
mechanics tests, not memo tests (the live path is covered by
test_regen_smoke.py in the nightly workflow).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal, init_db
from app.main import app
from app.models import RegenJob
from app.services import regen_worker


@pytest.fixture(autouse=True)
def _clean_regen_jobs():
    """Each test starts and ends with an empty regen queue so leftover
    rows can't leak into other tests (or into a dev server boot that
    shares the sqlite file)."""
    init_db()
    def _purge():
        with SessionLocal() as db:
            db.query(RegenJob).delete()
            db.commit()
    _purge()
    yield
    _purge()


def _fake_memo(ticker: str = "MSFT") -> SimpleNamespace:
    return SimpleNamespace(ticker=ticker, rating_label="Neutral")


@pytest.fixture
def _stub_graph(monkeypatch):
    """Replace the real memo run + snapshot lookup with cheap stubs."""
    calls = []

    def fake_run(ticker, *, scenario="soft_landing", force_refresh=False,
                 run_id=None, **kw):
        calls.append({"ticker": ticker, "scenario": scenario, "run_id": run_id})
        return _fake_memo(ticker)

    monkeypatch.setattr(regen_worker, "run_stock_memo", fake_run)
    import app.services.memo_store as memo_store
    monkeypatch.setattr(
        memo_store, "latest_memo",
        lambda t, **kw: SimpleNamespace(version=7, generated_at=datetime.utcnow()),
    )
    return calls


# ---------------------------------------------------------------------------
# Queue mechanics
# ---------------------------------------------------------------------------

def test_enqueue_creates_and_coalesces():
    job1, created1 = regen_worker.enqueue("msft")
    assert created1 is True
    assert job1["ticker"] == "MSFT"
    assert job1["status"] == "queued"
    assert job1["run_id"]
    job2, created2 = regen_worker.enqueue("MSFT")
    assert created2 is False
    assert job2["id"] == job1["id"]
    # A different ticker gets its own job.
    job3, created3 = regen_worker.enqueue("NVDA")
    assert created3 is True
    assert job3["id"] != job1["id"]


def test_process_next_job_success_records_version(_stub_graph):
    job, _ = regen_worker.enqueue("MSFT")
    done = regen_worker.process_next_job()
    assert done is not None
    assert done["id"] == job["id"]
    assert done["status"] == "succeeded"
    assert done["memo_version"] == 7
    assert done["attempts"] == 1
    assert done["started_at"] and done["finished_at"]
    # The graph was called with the job's run_id so checkpoints + LLM
    # logs attribute to this job.
    assert _stub_graph[0]["run_id"] == job["run_id"]
    steps = [p["step"] for p in done["progress"]]
    assert "worker_claimed" in steps
    assert "job_succeeded" in steps
    # Queue drained.
    assert regen_worker.process_next_job() is None


def test_process_next_job_failure_records_error(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("provider exploded")
    monkeypatch.setattr(regen_worker, "run_stock_memo", boom)
    regen_worker.enqueue("MSFT")
    done = regen_worker.process_next_job()
    assert done["status"] == "failed"
    assert done["error_type"] == "RuntimeError"
    assert "provider exploded" in done["error_message"]
    assert "RuntimeError" in done["traceback_tail"]

    state = regen_worker.ticker_status("MSFT")
    assert state["in_progress"] is False
    failure = state["last_failure"]
    # Exact legacy `_REGEN_FAILURES` key set — the frontend reads these.
    assert set(failure) == {
        "ticker", "error_type", "error_message", "traceback_tail",
        "started_at", "failed_at", "duration_seconds",
    }
    assert failure["error_type"] == "RuntimeError"


def test_failure_cleared_by_next_success(monkeypatch, _stub_graph):
    def boom(*a, **kw):
        raise RuntimeError("first run fails")
    monkeypatch.setattr(regen_worker, "run_stock_memo", boom)
    regen_worker.enqueue("MSFT")
    regen_worker.process_next_job()
    assert regen_worker.ticker_status("MSFT")["last_failure"] is not None

    monkeypatch.setattr(
        regen_worker, "run_stock_memo", lambda t, **kw: _fake_memo(t),
    )
    regen_worker.enqueue("MSFT")
    regen_worker.process_next_job()
    state = regen_worker.ticker_status("MSFT")
    assert state["last_failure"] is None
    assert state["job_status"] == "succeeded"


# ---------------------------------------------------------------------------
# Crash recovery — the part the daemon thread couldn't do
# ---------------------------------------------------------------------------

def test_recover_orphans_requeues_once_then_fails():
    job, _ = regen_worker.enqueue("MSFT")
    # Simulate a process death mid-run: claimed but never finished.
    claimed = regen_worker.claim_next_job()
    assert claimed == job["id"]

    out = regen_worker.recover_orphans()
    assert out["requeued"] == 1
    with SessionLocal() as db:
        row = db.get(RegenJob, job["id"])
        assert row.status == "queued"
        assert row.run_id == job["run_id"]  # checkpoint resume key kept

    # Second orphaning of the same job → permanent failure, no loop.
    regen_worker.claim_next_job()
    out = regen_worker.recover_orphans()
    assert out["failed"] == 1
    state = regen_worker.ticker_status("MSFT")
    assert state["in_progress"] is False
    assert state["last_failure"]["error_type"] == "WorkerRestart"


def test_recover_orphans_expires_stale_queued_jobs():
    job, _ = regen_worker.enqueue("MSFT")
    with SessionLocal() as db:
        row = db.get(RegenJob, job["id"])
        row.enqueued_at = datetime.utcnow() - timedelta(hours=2)
        db.commit()
    out = regen_worker.recover_orphans()
    assert out["expired"] == 1
    state = regen_worker.ticker_status("MSFT")
    assert state["last_failure"]["error_type"] == "QueueExpired"
    # A fresh queued job is untouched by recovery.
    job2, _ = regen_worker.enqueue("MSFT")
    out = regen_worker.recover_orphans()
    assert out == {"requeued": 0, "failed": 0, "expired": 0}


# ---------------------------------------------------------------------------
# API contract (Research.tsx polling loop)
# ---------------------------------------------------------------------------

def _client() -> TestClient:
    from app.tests.fixtures.seed_demo_data import run_full_seed
    run_full_seed()
    return TestClient(app)


def test_analyze_endpoint_enqueues_job_with_202_contract():
    c = _client()
    r = c.post("/api/stocks/MSFT/analyze")
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "started"
    assert body["started_at"]
    assert body["job_id"]
    # Double-fire coalesces into the same job.
    r2 = c.post("/api/stocks/MSFT/analyze")
    assert r2.json()["status"] == "in_progress"
    assert r2.json()["job_id"] == body["job_id"]
    # The job is a durable row, not process state.
    with SessionLocal() as db:
        row = db.get(RegenJob, body["job_id"])
        assert row is not None and row.status == "queued"


def test_status_endpoint_keeps_polling_contract(_stub_graph):
    c = _client()
    r = c.post("/api/stocks/MSFT/analyze")
    started_at = r.json()["started_at"]

    s = c.get("/api/stocks/MSFT/analyze/status").json()
    # Every legacy field Research.tsx / client.ts reads must survive.
    for key in ("ticker", "in_progress", "started_at", "latest_memo_at",
                "latest_version", "last_failure", "last_progress"):
        assert key in s
    assert s["in_progress"] is True
    assert s["started_at"] == started_at
    assert isinstance(s["last_progress"], list)

    # Drain the queue (what the worker thread does in prod) and verify
    # the completion signal the frontend polls for.
    done = regen_worker.process_next_job()
    assert done["status"] == "succeeded"
    s2 = c.get("/api/stocks/MSFT/analyze/status").json()
    assert s2["in_progress"] is False
    assert s2["latest_memo_at"] is not None
    # latest_memo_at advancing past started_at is the frontend's
    # stop-polling condition.
    assert s2["latest_memo_at"] > started_at
