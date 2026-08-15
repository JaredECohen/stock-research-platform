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
    from app.monitoring import KNOWN_LOOPS
    from app.api.routes_admin import cron_health_endpoint

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
