"""Cron-loop liveness must survive crossing a process boundary.

The worker split (#44) moved all 15 monitoring loops into
`marketmosaic-worker` while `/api/admin/cron-health` kept being served by
the web service. `monitoring._LAST_RUNS` is a module-level dict, so the
endpoint went blind and reported:

    {"loops": [], "stale_count": 0}

Verified against production. That is worse than an outage — `stale_count:
0` reads as "every loop is healthy" when it means "I cannot see any
loops", and this endpoint exists specifically to surface silent cron
failures.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

import pytest

from app.database import SessionLocal
from app.models import CronLoopRun

LOOP = "zz_probe_loop"


@pytest.fixture(autouse=True)
def _clean():
    import app.monitoring as monitoring
    monitoring._LAST_RUNS.pop(LOOP, None)
    _purge()
    yield
    monitoring._LAST_RUNS.pop(LOOP, None)
    _purge()


def _purge() -> None:
    with SessionLocal() as db:
        db.query(CronLoopRun).filter(
            CronLoopRun.loop_name.like("zz_probe%")
        ).delete(synchronize_session=False)
        db.commit()


def test_record_run_persists_to_the_database():
    """The DB row is the whole point — it's what another process can read."""
    import app.monitoring as monitoring
    monitoring.record_run(LOOP, success=True, note="probe note")
    with SessionLocal() as db:
        row = db.query(CronLoopRun).filter(CronLoopRun.loop_name == LOOP).one()
    assert row.success is True
    assert row.note == "probe note"
    assert row.reported_by in ("web", "worker")
    assert (datetime.utcnow() - row.last_run_at) < timedelta(minutes=5)


def test_status_snapshot_sees_runs_from_another_process():
    """Simulate the production topology: a row written by the worker, with
    this process's in-memory dict empty — exactly the case that broke."""
    import app.monitoring as monitoring
    with SessionLocal() as db:
        db.add(CronLoopRun(
            loop_name=LOOP, last_run_at=datetime.utcnow(),
            success=True, note="from worker", reported_by="worker",
        ))
        db.commit()
    monitoring._LAST_RUNS.pop(LOOP, None)  # this process never ran it

    snap = monitoring.status_snapshot()
    assert LOOP in snap, "cron status cannot see another process's loop runs"
    assert snap[LOOP]["note"] == "from worker"
    assert snap[LOOP]["reported_by"] == "worker"


def test_record_run_upserts_rather_than_appending():
    """A liveness signal, not an audit log: 15 loops on 30-minute intervals
    would grow unbounded."""
    import app.monitoring as monitoring
    monitoring.record_run(LOOP, success=False, note="first")
    monitoring.record_run(LOOP, success=True, note="second")
    with SessionLocal() as db:
        rows = db.query(CronLoopRun).filter(CronLoopRun.loop_name == LOOP).all()
    assert len(rows) == 1
    assert rows[0].note == "second"
    assert rows[0].success is True


def _other_process(code: str) -> dict:
    """A fresh module state against the same isolated test database."""
    result = subprocess.run(
        [sys.executable, "-c", "from app.tests import netguard; netguard.install(); " + code],
        env={**os.environ, "MM_PROCESS_ROLE": "worker"},
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def test_progress_preserves_failed_completion_across_actual_processes():
    import app.monitoring as monitoring
    from app.api.routes_admin import cron_health_endpoint

    monitoring.record_run(LOOP, success=False, note="prior completed failure: AAPL")
    before = monitoring.status_snapshot()[LOOP]
    child = _other_process(
        "import json; import app.monitoring as m; "
        f"m.record_progress('{LOOP}', success=True, note='new pass has polled 3'); "
        f"print(json.dumps(m.status_snapshot()['{LOOP}']))"
    )
    # A stale _LAST_RUNS entry in the parent must not hide the child's write.
    parent = monitoring.status_snapshot()[LOOP]
    assert parent == child
    assert parent["last_run_at"] == before["last_run_at"]
    assert parent["success"] is False
    assert parent["note"] == before["note"]
    assert parent["progress_success"] is True
    assert parent["progress_note"] == "new pass has polled 3"
    assert parent["progress_at"] is not None
    public = next(row for row in cron_health_endpoint()["loops"] if row["loop"] == LOOP)
    assert public["success"] is False and public["progress_success"] is True
    assert public["progress_at"] == parent["progress_at"]

    # Another fresh process (worker restart) still sees the failure until
    # an actual completion arrives, at which point progress is cleared.
    restarted = _other_process(
        f"import json; import app.monitoring as m; print(json.dumps(m.status_snapshot()['{LOOP}']))"
    )
    assert restarted == parent
    _other_process(
        "import json; import app.monitoring as m; "
        f"m.record_run('{LOOP}', success=True, note='completed recovery'); "
        f"print(json.dumps(m.status_snapshot()['{LOOP}']))"
    )
    completed = monitoring.status_snapshot()[LOOP]
    assert completed["success"] is True
    assert completed["note"] == "completed recovery"
    assert all(completed[key] is None for key in ("progress_at", "progress_note", "progress_success"))


def test_first_progress_is_never_reported_as_a_completed_run():
    import app.monitoring as monitoring
    from app.api.routes_admin import cron_health_endpoint

    monitoring.record_progress(LOOP, note="first pass starting")
    with SessionLocal() as db:
        stored = db.query(CronLoopRun).filter_by(loop_name=LOOP).one()
        assert stored.last_run_at == monitoring.NEVER_COMPLETED_AT
        assert stored.success is False
    child = _other_process(
        f"import json; import app.monitoring as m; print(json.dumps(m.status_snapshot()['{LOOP}']))"
    )
    assert child["last_run_at"] is None and child["success"] is None
    assert child["note"] == "never run"
    assert child["progress_at"] is not None
    assert child["progress_success"] is None
    row = next(row for row in cron_health_endpoint()["loops"] if row["loop"] == LOOP)
    assert row["stale"] is True
    assert row["last_run_at"] is None and row["age_seconds"] is None


def test_interrupted_progress_failure_is_retained_across_restart_until_completion():
    import app.monitoring as monitoring

    monitoring.record_run(LOOP, success=True, note="previous completed pass")
    monitoring.record_progress(LOOP, success=False, note="index errors on 1: AAPL")
    child = _other_process(
        "import json; import app.monitoring as m; "
        f"m.record_progress('{LOOP}', note='new worker starting'); "
        f"m.record_progress('{LOOP}', success=True, note='polled 5 so far'); "
        f"print(json.dumps(m.status_snapshot()['{LOOP}']))"
    )
    assert child["success"] is True  # last completed pass is unchanged
    assert child["progress_success"] is False
    assert "AAPL" in child["progress_note"]
    assert "polled 5 so far" in child["progress_note"]
    assert child["progress_note"].count("Latest activity:") == 1
    monitoring.record_run(LOOP, success=True, note="completed recovery")
    assert monitoring.status_snapshot()[LOOP]["progress_success"] is None


def test_progress_columns_are_nullable_and_reconcile_without_erasing_history(monkeypatch):
    from sqlalchemy import create_engine, text

    from app import database

    isolated = create_engine("sqlite://")
    with isolated.begin() as conn:
        conn.execute(text("CREATE TABLE cron_loop_runs (id INTEGER PRIMARY KEY, loop_name VARCHAR(64), "
                          "last_run_at DATETIME NOT NULL, success BOOLEAN NOT NULL, note TEXT, reported_by VARCHAR(32))"))
        conn.execute(text("INSERT INTO cron_loop_runs VALUES (1, 'old_loop', '2026-09-01 01:00:00', 0, 'failure', 'worker')"))
    monkeypatch.setattr(database, "engine", isolated)
    repaired = database.reconcile_missing_columns()
    assert set(repaired) == {f"cron_loop_runs.{name}" for name in ("progress_at", "progress_note", "progress_success")}
    with isolated.connect() as conn:
        row = conn.execute(text("SELECT * FROM cron_loop_runs")).mappings().one()
    assert row["success"] == 0 and row["note"] == "failure"
    assert row["last_run_at"] == "2026-09-01 01:00:00"
    assert all(row[key] is None for key in ("progress_at", "progress_note", "progress_success"))
    isolated.dispose()


def test_record_run_never_raises_when_the_db_is_unavailable(monkeypatch):
    """A monitoring loop must not fail because its own bookkeeping did."""
    import app.monitoring as monitoring

    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr("app.database.SessionLocal", _boom)
    monitoring.record_run(LOOP, success=True, note="should not raise")
    # In-memory copy still updated, so the endpoint stays useful locally.
    assert monitoring._LAST_RUNS[LOOP]["note"] == "should not raise"


def test_cron_health_endpoint_reports_persisted_loops():
    """End-to-end through the endpoint the operator actually reads."""
    import app.monitoring as monitoring
    from app.api.routes_admin import cron_health_endpoint

    monitoring.record_run(LOOP, success=True, note="fresh")
    out = cron_health_endpoint()
    names = {r["loop"] for r in out["loops"]}
    assert LOOP in names
    row = next(r for r in out["loops"] if r["loop"] == LOOP)
    assert row["stale"] is False, "a just-recorded run must not read as stale"


def test_stale_loop_is_flagged():
    """The endpoint's job is catching loops that stopped reporting."""
    import app.monitoring as monitoring
    from app.api.routes_admin import cron_health_endpoint

    monitoring._LAST_RUNS.pop(LOOP, None)
    with SessionLocal() as db:
        db.add(CronLoopRun(
            loop_name=LOOP,
            last_run_at=datetime.utcnow() - timedelta(days=3),
            success=True, note="ancient", reported_by="worker",
        ))
        db.commit()
    out = cron_health_endpoint()
    row = next(r for r in out["loops"] if r["loop"] == LOOP)
    assert row["stale"] is True
    assert out["stale_count"] >= 1


# ---------------------------------------------------------------------------
# Never-run loops must be visible, not absent
# ---------------------------------------------------------------------------

def test_known_loops_matches_what_register_all_registers():
    """`KNOWN_LOOPS` is hand-maintained; if it drifts from the scheduler
    the endpoint goes back to hiding whichever loop fell off the list."""
    from app.monitoring import KNOWN_LOOPS, register_all

    class FakeScheduler:
        def __init__(self):
            self.ids = []

        def add_job(self, fn, trigger, **kw):
            self.ids.append(kw.get("id") or getattr(fn, "__module__", "?").rsplit(".", 1)[-1])

    sched = FakeScheduler()
    register_all(sched)
    assert set(sched.ids) == set(KNOWN_LOOPS), (
        "KNOWN_LOOPS is out of sync with register_all:\n"
        f"  registered but not listed: {sorted(set(sched.ids) - set(KNOWN_LOOPS))}\n"
        f"  listed but not registered: {sorted(set(KNOWN_LOOPS) - set(sched.ids))}"
    )


def test_a_loop_that_never_ran_is_reported_as_stale_not_omitted():
    """The blind spot that hid `postmortem_loop` dying nightly.

    It raised before reaching `record_run`, so it had no row — and the
    endpoint listed only loops with rows, making a dead loop
    indistinguishable from a healthy one.
    """
    from app.api.routes_admin import cron_health_endpoint
    from app.monitoring import KNOWN_LOOPS

    with SessionLocal() as db:
        db.query(CronLoopRun).filter(
            CronLoopRun.loop_name == "postmortem_loop"
        ).delete(synchronize_session=False)
        db.commit()
    import app.monitoring as monitoring
    monitoring._LAST_RUNS.pop("postmortem_loop", None)

    out = cron_health_endpoint()
    reported = {r["loop"] for r in out["loops"]}
    assert set(KNOWN_LOOPS) <= reported, (
        "registered loops missing from cron-health: "
        f"{sorted(set(KNOWN_LOOPS) - reported)}"
    )
    row = next(r for r in out["loops"] if r["loop"] == "postmortem_loop")
    assert row["last_run_at"] is None
    assert row["stale"] is True, "a loop that never ran must not read as healthy"
    assert out["stale_count"] >= 1


def test_postmortem_loop_records_failure_before_reraising(monkeypatch):
    """An exception inside run_postmortems must reach both health surfaces."""
    from app.monitoring import postmortem_loop

    calls = []

    def fail(**kwargs):
        raise RuntimeError("synthetic postmortem failure")

    monkeypatch.setattr(postmortem_loop, "run_postmortems", fail)
    monkeypatch.setattr(
        postmortem_loop, "record_run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="synthetic postmortem failure"):
        postmortem_loop.run_once()
    assert calls[0][0] == ("postmortem_loop",)
    assert calls[0][1]["success"] is False
    assert "RuntimeError" in calls[0][1]["note"]


def test_outcome_loop_records_failure_on_exception(monkeypatch):
    """W6: an exception inside evaluate_all_due must be persisted as a failed
    run before it re-raises. (A failed eligibility sweep no longer raises; it
    is reported as `classification_error` and turns the run red.)

    Before, `outcome_loop.run_once` had no handler: the exception reached
    APScheduler, `record_run` never ran, and cron-health — served by the
    OTHER process — kept showing the previous night's success. This reads
    the failure back through the database and the endpoint, the path the
    web process actually uses.
    """
    import app.monitoring as monitoring
    from app.api.routes_admin import cron_health_endpoint
    from app.monitoring import outcome_loop

    def fail():
        raise RuntimeError("synthetic eligibility sweep failure")

    monkeypatch.setattr(outcome_loop, "evaluate_all_due", fail)
    try:
        with pytest.raises(RuntimeError, match="synthetic eligibility sweep failure"):
            outcome_loop.run_once()
        monitoring._LAST_RUNS.pop("outcome_loop", None)  # force the DB path
        with SessionLocal() as db:
            row = db.query(CronLoopRun).filter(CronLoopRun.loop_name == "outcome_loop").one()
            assert row.success is False
            assert row.note.startswith("failed: RuntimeError: synthetic eligibility sweep failure")
        public = next(r for r in cron_health_endpoint()["loops"] if r["loop"] == "outcome_loop")
        assert public["success"] is False
    finally:
        monitoring._LAST_RUNS.pop("outcome_loop", None)
        with SessionLocal() as db:
            db.query(CronLoopRun).filter(CronLoopRun.loop_name == "outcome_loop").delete()
            db.commit()


def test_postmortem_loop_marks_skipped_work_as_failed(monkeypatch):
    from app.monitoring import postmortem_loop

    reports = iter([
        {"due": 1, "written": 0, "skipped": 1},
        {"due": 0, "written": 0, "skipped": 0},
    ])
    calls = []
    monkeypatch.setattr(
        postmortem_loop, "run_postmortems", lambda **kwargs: next(reports),
    )
    monkeypatch.setattr(
        postmortem_loop, "record_run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    postmortem_loop.run_once()
    assert calls[0][1]["success"] is False
    assert "skipped=1" in calls[0][1]["note"]
