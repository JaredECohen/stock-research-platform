"""Phase 6 (slice C) — the durable scorecard run queue.

Coalescing, the atomic claim, orphan recovery (`WorkerRestart` /
`QueueExpired`), outcome recording and the execute dispatch. Clocks are
injected through `scorecard_queue._utcnow`; nothing here computes a
score (the job bodies are monkeypatched).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app.database import SessionLocal
from app.models import ScorecardRun
from app.services import scorecard_queue as q
from app.tests.scorecard_helpers import REQUESTED_BY, purge, purge_queue

VK = "fs-v1"
AS_OF = date(2026, 3, 31)
NOW = datetime(2026, 4, 1, 3, 45, 0)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    purge_queue()
    purge(requested_by=(REQUESTED_BY,))
    monkeypatch.setattr(q, "_utcnow", lambda: NOW)
    yield
    purge_queue()
    purge(requested_by=(REQUESTED_BY,))


def _enqueue(kind: str = q.KIND_MANUAL, as_of: date = AS_OF, **kw):
    return q.enqueue_run(version_key=VK, as_of=as_of, kind=kind, requested_by=REQUESTED_BY, **kw)


# ---------------------------------------------------------------------------
# enqueue
# ---------------------------------------------------------------------------

def test_enqueue_coalesces_on_version_asof_kind():
    first, created = _enqueue(params={"tickers": ["ZQA"]})
    assert created and first["status"] == q.STATUS_QUEUED and first["params"] == {"tickers": ["ZQA"]}
    again, created_again = _enqueue(params={"tickers": ["ZQB"]})
    assert not created_again and again["id"] == first["id"]
    assert again["params"] == {"tickers": ["ZQA"]}, "a coalesced request must not overwrite the owner's params"
    other_kind, created_other = _enqueue(kind=q.KIND_EVALUATE)
    assert created_other and other_kind["id"] != first["id"]
    other_day, created_day = _enqueue(as_of=AS_OF - timedelta(days=1))
    assert created_day and other_day["id"] != first["id"]


def test_enqueue_refuses_unknown_kinds():
    with pytest.raises(ValueError):
        q.enqueue_run(version_key=VK, as_of=AS_OF, kind="nonsense", requested_by=REQUESTED_BY)
    with pytest.raises(ValueError):
        q.create_running_row(version_key=VK, as_of=AS_OF, kind="nonsense", requested_by=REQUESTED_BY)


def test_finished_rows_do_not_coalesce_new_requests():
    first, _ = _enqueue()
    q.finish_run(first["id"], status=q.STATUS_SUCCEEDED, note="written=1")
    second, created = _enqueue()
    assert created and second["id"] != first["id"]


# ---------------------------------------------------------------------------
# claim
# ---------------------------------------------------------------------------

def test_claim_next_run_claims_exactly_once_in_fifo_order():
    a, _ = _enqueue(as_of=AS_OF - timedelta(days=2))
    b, _ = _enqueue(as_of=AS_OF - timedelta(days=1))
    assert q.claim_next_run() == a["id"]
    assert q.claim_next_run() == b["id"]
    assert q.claim_next_run() is None
    claimed = q.get_run(a["id"])
    assert claimed["status"] == q.STATUS_RUNNING and claimed["attempts"] == 1 and claimed["started_at"] == NOW


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------

def test_recover_orphans_fails_stale_running_rows_without_requeue(monkeypatch):
    stale_id = q.create_running_row(version_key=VK, as_of=AS_OF, kind=q.KIND_MANUAL, requested_by=REQUESTED_BY)
    with SessionLocal() as db:
        row = db.get(ScorecardRun, stale_id)
        row.started_at = NOW - q.RUNNING_GRACE - timedelta(minutes=1)
        db.commit()
    fresh_id = q.create_running_row(
        version_key=VK, as_of=AS_OF - timedelta(days=3), kind=q.KIND_MANUAL, requested_by=REQUESTED_BY,
    )
    out = q.recover_orphans()
    assert out == {"failed": 1, "expired": 0}
    stale = q.get_run(stale_id)
    assert stale["status"] == q.STATUS_FAILED and stale["error_type"] == q.ERROR_WORKER_RESTART
    assert stale["attempts"] == 1, "never requeued"
    assert q.get_run(fresh_id)["status"] == q.STATUS_RUNNING, "a run inside the grace window is left alone"


def test_recover_orphans_expires_queued_rows_older_than_the_max_age():
    old, _ = _enqueue(as_of=AS_OF - timedelta(days=10))
    with SessionLocal() as db:
        row = db.get(ScorecardRun, old["id"])
        row.enqueued_at = NOW - q.QUEUE_MAX_AGE - timedelta(hours=1)
        db.commit()
    recent, _ = _enqueue()
    out = q.recover_orphans()
    assert out == {"failed": 0, "expired": 1}
    expired = q.get_run(old["id"])
    assert expired["status"] == q.STATUS_FAILED and expired["error_type"] == q.ERROR_QUEUE_EXPIRED
    assert q.get_run(recent["id"])["status"] == q.STATUS_QUEUED
    assert q.claim_next_run() == recent["id"], "the expired row is never claimed"


# ---------------------------------------------------------------------------
# outcome recording
# ---------------------------------------------------------------------------

def test_finish_run_records_counts_note_and_merges_params():
    row, _ = _enqueue(params={"tickers": ["ZQA"]})
    q.claim_next_run()
    out = q.finish_run(
        row["id"], status=q.STATUS_SUCCEEDED, note="written=3", universe_size=3, scored_count=2,
        inputs_hash="abc", params_update={"is_month_end": True},
    )
    assert out["status"] == q.STATUS_SUCCEEDED and out["finished_at"] == NOW
    assert out["universe_size"] == 3 and out["scored_count"] == 2 and out["inputs_hash"] == "abc"
    assert out["params"] == {"tickers": ["ZQA"], "is_month_end": True}
    with pytest.raises(ValueError):
        q.finish_run(row["id"], status="queued")


def test_mark_failed_redacts_the_message_and_keeps_the_type():
    row, _ = _enqueue()
    q.claim_next_run()
    out = q.mark_failed(row["id"], RuntimeError("provider said apikey=sk-abcdefghijklmnop"))
    assert out["status"] == q.STATUS_FAILED and out["error_type"] == "RuntimeError"
    assert "sk-abcdefghijklmnop" not in out["error_message"]
    assert "<redacted>" in out["error_message"]


# ---------------------------------------------------------------------------
# execute / drain
# ---------------------------------------------------------------------------

def test_execute_run_dispatches_scoring_kinds_to_the_service(monkeypatch):
    from app.services import scorecard_service

    calls = []

    def fake_run(version_key, as_of, *, run_kind, tickers, run_row_id):
        calls.append((version_key, as_of, run_kind, tickers, run_row_id))
        q.finish_run(run_row_id, status=q.STATUS_SUCCEEDED, note="written=1", params_update={"written": 1})

    monkeypatch.setattr(scorecard_service, "run_scorecard", fake_run)
    row, _ = _enqueue(params={"tickers": ["ZQA"]})
    done = q.drain()
    assert calls == [(VK, AS_OF, q.KIND_MANUAL, ["ZQA"], row["id"])]
    assert [d["status"] for d in done] == [q.STATUS_SUCCEEDED]
    assert q.drain() == []


def test_execute_run_marks_a_raising_body_failed_with_the_exception_type(monkeypatch):
    from app.services import scorecard_service

    def boom(*a, **k):
        raise ValueError("bad inputs")

    monkeypatch.setattr(scorecard_service, "run_scorecard", boom)
    row, _ = _enqueue()
    done = q.drain()
    assert done[0]["status"] == q.STATUS_FAILED and done[0]["error_type"] == "ValueError"
    assert q.get_run(row["id"])["status"] == q.STATUS_FAILED


def test_execute_run_never_leaves_a_body_that_forgot_to_finish_running(monkeypatch):
    from app.services import scorecard_service

    monkeypatch.setattr(scorecard_service, "run_scorecard", lambda *a, **k: None)
    _enqueue()
    done = q.drain()
    assert done[0]["status"] == q.STATUS_FAILED and done[0]["error_type"] == "RunNotFinalized"


def test_execute_run_refuses_a_row_that_is_not_running():
    row, _ = _enqueue()
    with pytest.raises(q.QueueExpired):
        q.execute_run(row["id"])
    assert q.execute_run(10**9) == {}


def test_drain_is_bounded_per_call(monkeypatch):
    from app.services import scorecard_service

    monkeypatch.setattr(
        scorecard_service, "run_scorecard",
        lambda *a, run_row_id, **k: q.finish_run(run_row_id, status=q.STATUS_SKIPPED, note="skipped"),
    )
    for i in range(3):
        _enqueue(as_of=AS_OF - timedelta(days=i))
    assert len(q.drain(max_runs=2)) == 2
    assert len(q.drain(max_runs=2)) == 1


# ---------------------------------------------------------------------------
# readers the loop and routes rely on
# ---------------------------------------------------------------------------

def test_latest_succeeded_run_and_existence_helpers():
    assert q.latest_succeeded_run(VK, as_of=date(1990, 1, 1)) is None
    a, _ = _enqueue(as_of=date(1991, 1, 31))
    b, _ = _enqueue(as_of=date(1991, 2, 28))
    q.finish_run(a["id"], status=q.STATUS_SUCCEEDED)
    q.finish_run(b["id"], status=q.STATUS_FAILED, error_type="X")
    latest = q.latest_succeeded_run(VK, as_of=date(1991, 3, 31))
    assert latest["id"] == a["id"], "failed runs are never resolved"
    assert q.succeeded_run_exists(VK, date(1991, 1, 31))
    assert not q.succeeded_run_exists(VK, date(1991, 2, 28))
    assert q.unfailed_run_exists(VK, date(1991, 1, 31), q.KIND_MANUAL)
    assert not q.unfailed_run_exists(VK, date(1991, 2, 28), q.KIND_MANUAL), "a failed row does not block a retry"
    # A re-run on the same day with new inputs wins over the earlier one.
    c, _ = _enqueue(as_of=date(1991, 1, 31))
    q.finish_run(c["id"], status=q.STATUS_SUCCEEDED)
    assert q.latest_succeeded_run(VK, as_of=date(1991, 1, 31))["id"] == c["id"]
    recent = q.recent_runs(version_key=VK, limit=5)
    assert recent and recent[0]["id"] == c["id"]


def test_run_exists_filters_by_kind_and_excluded_status():
    row, _ = _enqueue(kind=q.KIND_SCHEDULED)
    assert q.run_exists(VK, AS_OF, kinds=(q.KIND_SCHEDULED,))
    assert q.run_exists(VK, AS_OF, kinds=q.SCORING_KINDS, exclude_statuses=(q.STATUS_FAILED,))
    assert not q.run_exists(VK, AS_OF, kinds=(q.KIND_EVALUATE,))
    assert not q.run_exists(VK, AS_OF - timedelta(days=1), kinds=q.SCORING_KINDS)
    q.claim_next_run()
    q.finish_run(row["id"], status=q.STATUS_FAILED, error_type="RuntimeError")
    assert q.run_exists(VK, AS_OF, kinds=(q.KIND_SCHEDULED,)), "the daily gate counts a failed attempt"
    assert not q.run_exists(VK, AS_OF, kinds=q.SCORING_KINDS, exclude_statuses=(q.STATUS_FAILED,))
    assert not q.unfailed_run_exists(VK, AS_OF, q.KIND_SCHEDULED)
