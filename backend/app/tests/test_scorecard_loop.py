"""Phase 6 (slice C) — `monitoring/scorecard_loop`.

One loop, registered once, recorded every tick. The queue drain is stubbed
(the job bodies have their own suites), the clock is injected, and the
rows the loop enqueues are purged afterwards.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from app.config import settings
from app.database import SessionLocal
from app.models import ScorecardRun, ScorecardScore
from app.monitoring import scorecard_loop as loop
from app.services import scorecard_queue as q
from app.tests.scorecard_helpers import insert_run, purge

LOOP_BY = ("scorecard_loop",)
VK_GC = "fs-looptest"
JULY_2 = datetime(2026, 7, 2, 3, 45, 0)


def _purge_loop_rows() -> None:
    purge(requested_by=LOOP_BY, versions=(VK_GC,))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    _purge_loop_rows()
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(loop, "record_run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(loop, "_utcnow", lambda: JULY_2)
    monkeypatch.setattr(q, "drain", lambda max_runs=200: [])
    yield calls
    _purge_loop_rows()


def _runs(kind: str | None = None, as_of: date | None = None) -> list[ScorecardRun]:
    with SessionLocal() as db:
        qq = db.query(ScorecardRun).filter(ScorecardRun.requested_by == loop.LOOP_NAME)
        if kind:
            qq = qq.filter(ScorecardRun.run_kind == kind)
        if as_of:
            qq = qq.filter(ScorecardRun.as_of == as_of)
        rows = qq.order_by(ScorecardRun.id).all()
        db.expunge_all()
        return rows


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_loop_is_registered_once_under_known_loops():
    from app.monitoring import KNOWN_LOOPS, register_all

    class FakeScheduler:
        def __init__(self):
            self.jobs = []

        def add_job(self, fn, trigger, **kw):
            self.jobs.append((kw.get("id"), trigger, kw))

    sched = FakeScheduler()
    register_all(sched)
    mine = [j for j in sched.jobs if j[0] == loop.LOOP_NAME]
    assert len(mine) == 1 and loop.LOOP_NAME in KNOWN_LOOPS
    _id, trigger, kw = mine[0]
    assert trigger == "cron" and (kw["hour"], kw["minute"]) == (3, 45)
    assert kw["max_instances"] == 1 and kw["coalesce"] is True
    assert (kw["hour"], kw["minute"]) != (4, 30), "must not share mispricing_audit_loop's slot"


# ---------------------------------------------------------------------------
# Tick behaviour
# ---------------------------------------------------------------------------

def test_disabled_loop_records_a_tick_and_does_nothing(_isolate, monkeypatch):
    monkeypatch.setattr(settings, "enable_scorecard_loop", False)
    out = loop.run_once()
    assert out == {"disabled": True}
    assert _isolate[0][0] == (loop.LOOP_NAME,) and "disabled=1" in _isolate[0][1]["note"]
    assert _runs() == []


def test_tick_with_nothing_queued_still_records_written_counts(_isolate):
    out = loop.run_once()
    args, kwargs = _isolate[0]
    assert args == (loop.LOOP_NAME,) and kwargs["success"] is True
    for key in ("written=0", "skipped=0", "failed=0", "as_of=2026-07-01", "gc="):
        assert key in kwargs["note"], kwargs["note"]
    assert out["problems"] == []


def test_tick_enqueues_yesterday_and_the_monthly_pair_exactly_once():
    loop.run_once()
    sched = _runs(q.KIND_SCHEDULED)
    assert [r.as_of for r in sched] == [date(2026, 7, 1)]
    prep = _runs(q.KIND_PIT_PREPARE)
    ev = _runs(q.KIND_EVALUATE)
    assert [r.as_of for r in prep] == [date(2026, 6, 30)] and [r.as_of for r in ev] == [date(2026, 6, 30)]
    assert prep[0].id < ev[0].id, "prices must land before the evaluation joins against them (FIFO)"

    # A second tick the same day — and a tick later in the month — adds nothing.
    loop.run_once()
    assert len(_runs(q.KIND_EVALUATE)) == 1 and len(_runs(q.KIND_PIT_PREPARE)) == 1
    assert len(_runs(q.KIND_SCHEDULED)) == 1


def test_evaluation_is_enqueued_once_per_month_even_after_it_finished(monkeypatch):
    loop.run_once()
    ev = _runs(q.KIND_EVALUATE)[0]
    q.claim_next_run()   # scheduled
    q.claim_next_run()   # pit_prepare
    q.claim_next_run()   # evaluate
    q.finish_run(ev.id, status=q.STATUS_SUCCEEDED, note="written=3")
    for r in _runs():
        if r.status == q.STATUS_RUNNING:
            q.finish_run(r.id, status=q.STATUS_SUCCEEDED)
    monkeypatch.setattr(loop, "_utcnow", lambda: datetime(2026, 7, 20, 3, 45, 0))
    loop.run_once()
    assert len(_runs(q.KIND_EVALUATE, date(2026, 6, 30))) == 1, "a finished evaluation must not be re-enqueued"
    # The month turns: the next month end gets its own pair.
    monkeypatch.setattr(loop, "_utcnow", lambda: datetime(2026, 8, 3, 3, 45, 0))
    loop.run_once()
    assert [r.as_of for r in _runs(q.KIND_EVALUATE)] == [date(2026, 6, 30), date(2026, 7, 31)]


def test_failed_evaluation_is_retried_on_the_next_tick():
    loop.run_once()
    ev = _runs(q.KIND_EVALUATE)[0]
    for _ in range(3):
        q.claim_next_run()
    q.finish_run(ev.id, status=q.STATUS_FAILED, error_type="RuntimeError")
    loop.run_once()
    rows = _runs(q.KIND_EVALUATE, date(2026, 6, 30))
    assert [r.status for r in rows] == [q.STATUS_FAILED, q.STATUS_QUEUED]


def test_tick_counts_drained_outcomes_and_flags_failures(_isolate, monkeypatch):
    drained = [
        {"status": q.STATUS_SUCCEEDED, "params": {"written": 7}, "scored_count": 7},
        {"status": q.STATUS_SKIPPED, "params": {}, "scored_count": 0},
        {"status": q.STATUS_FAILED, "params": {}, "scored_count": None},
    ]
    monkeypatch.setattr(q, "drain", lambda max_runs=200: drained)
    out = loop.run_once()
    kwargs = _isolate[0][1]
    assert kwargs["success"] is False
    assert "written=7" in kwargs["note"] and "skipped=1" in kwargs["note"] and "failed=1" in kwargs["note"]
    assert out["claimed"] == 3


def test_recovery_runs_first_and_a_broken_step_never_hides_the_tick(_isolate, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(q, "recover_orphans", lambda: order.append("recover") or {"failed": 2, "expired": 1})
    monkeypatch.setattr(q, "drain", lambda max_runs=200: order.append("drain") or [])

    def broken_gc(**kw):
        order.append("gc")
        raise RuntimeError("gc broke")

    from app.services import scorecard_service
    monkeypatch.setattr(scorecard_service, "gc_daily_rows", broken_gc)
    out = loop.run_once()
    assert order[0] == "recover" and order.index("drain") < order.index("gc")
    kwargs = _isolate[0][1]
    assert kwargs["success"] is False and "errors=gc:RuntimeError" in kwargs["note"]
    assert "recovered=2" in kwargs["note"] and "expired=1" in kwargs["note"]
    assert out["problems"] == ["gc:RuntimeError"]


def test_gc_step_drops_old_dailies_and_keeps_month_ends():
    old_daily = insert_run(version_key=VK_GC, as_of=date(2026, 4, 15), kind="scheduled", requested_by=loop.LOOP_NAME)
    old_month = insert_run(version_key=VK_GC, as_of=date(2026, 4, 30), kind="scheduled", requested_by=loop.LOOP_NAME)
    with SessionLocal() as db:
        db.add(ScorecardScore(run_id=old_daily, version_key=VK_GC, as_of=date(2026, 4, 15), ticker="ZLP0",
                              is_month_end=False, coverage=1.0))
        db.add(ScorecardScore(run_id=old_month, version_key=VK_GC, as_of=date(2026, 4, 30), ticker="ZLP0",
                              is_month_end=True, coverage=1.0))
        db.commit()
    out = loop.run_once()
    assert out["gc"] >= 1
    with SessionLocal() as db:
        remaining = {(r.as_of, r.is_month_end) for r in db.query(ScorecardScore).filter(ScorecardScore.version_key == VK_GC)}
    assert remaining == {(date(2026, 4, 30), True)}


def test_schedule_helpers():
    assert loop.scheduled_as_of(JULY_2) == date(2026, 7, 1)
    assert loop.last_completed_month_end(date(2026, 7, 1)) == date(2026, 6, 30)
    assert loop.last_completed_month_end(date(2026, 3, 31)) == date(2026, 2, 28)
    assert loop.last_completed_month_end(date(2027, 1, 15)) == date(2026, 12, 31)
