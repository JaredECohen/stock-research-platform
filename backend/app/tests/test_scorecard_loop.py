"""Phase 6 (slice C) — `monitoring/scorecard_loop`.

One loop, registered once (interval trigger, daily step gated to 03:45
UTC inside the tick), recorded on every tick that does something. The
queue drain is stubbed (the job bodies have their own suites), the clock
is injected, and the rows the loop enqueues are purged afterwards.
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
JULY_1 = datetime(2026, 7, 1, 3, 45, 0)     # first daily tick after June ended
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


def _finish_everything(status: str = q.STATUS_SUCCEEDED, *, fail_ids: tuple[int, ...] = ()) -> None:
    """Claim every queued row (FIFO) and finish it, so a later tick sees a
    settled queue the way it would in production."""
    while (row_id := q.claim_next_run()) is not None:
        if row_id in fail_ids:
            q.finish_run(row_id, status=q.STATUS_FAILED, error_type="RuntimeError")
        else:
            q.finish_run(row_id, status=status)


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
    # Interval so admin-enqueued runs drain within minutes; the daily
    # scoring is gated inside the tick, not by the trigger.
    assert trigger == "interval" and kw["minutes"] == loop.INTERVAL_MINUTES == 5
    assert kw["max_instances"] == 1 and kw["coalesce"] is True
    assert (loop.DAILY_HOUR, loop.DAILY_MINUTE) == (3, 45)
    assert (loop.DAILY_HOUR, loop.DAILY_MINUTE) != (4, 30), "must not share mispricing_audit_loop's slot"


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
    for key in ("daily=1", "written=0", "skipped=0", "failed=0", "as_of=2026-07-01", "gc="):
        assert key in kwargs["note"], kwargs["note"]
    assert out["problems"] == [] and out["daily"] is True and out["recorded"] is True


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


def test_first_tick_after_the_month_turns_orders_pit_prepare_before_the_month_end_run(monkeypatch):
    """The reviewer's ordering finding: on Jul 1 the scheduled run IS the
    June month-end cross-section, so the June close must be synced into
    the store before it is scored, and the evaluation must come last."""
    monkeypatch.setattr(loop, "_utcnow", lambda: JULY_1)
    loop.run_once()
    rows = _runs()
    assert [(r.run_kind, r.as_of) for r in rows] == [
        (q.KIND_PIT_PREPARE, date(2026, 6, 30)),
        (q.KIND_SCHEDULED, date(2026, 6, 30)),
        (q.KIND_EVALUATE, date(2026, 6, 30)),
    ], "FIFO by id: pit_prepare < month-end scoring < evaluate"
    assert _runs(q.KIND_MONTH_END) == [], "no catch-up when the scheduled run already covers the month end"
    # FIFO claim order is exactly the id order.
    claimed = [q.claim_next_run() for _ in range(3)]
    assert claimed == [r.id for r in rows]


def test_a_missed_month_end_is_caught_up_with_a_month_end_run_behind_pit_prepare(monkeypatch):
    """The worker was down on Jul 1: the Jul 2 daily tick must still score
    June 30 (month-end rows are the evaluation's sample, kept forever),
    after pit_prepare and before the evaluation."""
    loop.run_once()   # JULY_2, nothing on file for June 30
    rows = _runs()
    assert [(r.run_kind, r.as_of) for r in rows] == [
        (q.KIND_PIT_PREPARE, date(2026, 6, 30)),
        (q.KIND_MONTH_END, date(2026, 6, 30)),
        (q.KIND_SCHEDULED, date(2026, 7, 1)),
        (q.KIND_EVALUATE, date(2026, 6, 30)),
    ]
    # Once a non-failed scoring run for the month end exists, no second catch-up.
    _finish_everything()
    with SessionLocal() as db:
        for r in db.query(ScorecardRun).filter(ScorecardRun.requested_by == loop.LOOP_NAME,
                                               ScorecardRun.run_kind == q.KIND_EVALUATE):
            r.status = q.STATUS_FAILED   # force the monthly block to re-run on the next daily tick
        db.commit()
    # Next daily tick (Jul 3): evaluate is retried, the month end is NOT rescored.
    monkeypatch.setattr(loop, "_utcnow", lambda: datetime(2026, 7, 3, 3, 45, 0))
    loop.run_once()
    assert len(_runs(q.KIND_MONTH_END)) == 1
    assert [r.status for r in _runs(q.KIND_EVALUATE)] == [q.STATUS_FAILED, q.STATUS_QUEUED]


def test_daily_step_fires_once_per_day_and_the_interval_ticks_only_drain(_isolate, monkeypatch):
    loop.run_once()                                   # 03:45 — the daily tick
    assert len(_runs(q.KIND_SCHEDULED)) == 1 and _isolate[-1][1]["note"].startswith("as_of=2026-07-01 daily=1")
    n_recorded = len(_isolate)

    drains: list[int] = []
    monkeypatch.setattr(q, "drain", lambda max_runs=200: drains.append(max_runs) or [])
    for hhmm in ((3, 50), (12, 0), (23, 55)):
        monkeypatch.setattr(loop, "_utcnow", lambda h=hhmm: datetime(2026, 7, 2, *h, 0))
        out = loop.run_once()
        assert out["daily"] is False and out["recorded"] is False
    assert len(_runs(q.KIND_SCHEDULED)) == 1, "the daily run is enqueued once per day, not once per interval"
    assert drains == [loop.MAX_RUNS_PER_TICK] * 3, "every interval tick drains the queue"
    assert len(_isolate) == n_recorded, "a quiet interval tick does not overwrite the last informative record"

    # A tick before 03:45 the next day is not the daily tick either.
    monkeypatch.setattr(loop, "_utcnow", lambda: datetime(2026, 7, 3, 1, 0, 0))
    assert loop.run_once()["daily"] is False
    # 03:45 the next day is.
    monkeypatch.setattr(loop, "_utcnow", lambda: datetime(2026, 7, 3, 3, 45, 0))
    assert loop.run_once()["daily"] is True
    assert [r.as_of for r in _runs(q.KIND_SCHEDULED)] == [date(2026, 7, 1), date(2026, 7, 2)]


def test_a_worker_that_was_down_at_0345_runs_the_daily_step_on_its_first_tick_back(monkeypatch):
    monkeypatch.setattr(loop, "_utcnow", lambda: datetime(2026, 7, 2, 15, 10, 0))
    out = loop.run_once()
    assert out["daily"] is True and [r.as_of for r in _runs(q.KIND_SCHEDULED)] == [date(2026, 7, 1)]


def test_a_failed_daily_run_is_one_attempt_per_day(monkeypatch):
    loop.run_once()
    _finish_everything(fail_ids=tuple(r.id for r in _runs(q.KIND_SCHEDULED)))
    monkeypatch.setattr(loop, "_utcnow", lambda: datetime(2026, 7, 2, 4, 0, 0))
    out = loop.run_once()
    assert out["daily"] is False
    assert [r.status for r in _runs(q.KIND_SCHEDULED)] == [q.STATUS_FAILED], "not retried every interval"


def test_interval_tick_records_when_it_drained_something(_isolate, monkeypatch):
    loop.run_once()
    n_recorded = len(_isolate)
    monkeypatch.setattr(loop, "_utcnow", lambda: datetime(2026, 7, 2, 9, 0, 0))
    monkeypatch.setattr(q, "drain", lambda max_runs=200: [
        {"status": q.STATUS_SUCCEEDED, "params": {"written": 170}, "scored_count": 170},
    ])
    out = loop.run_once()
    assert out["daily"] is False and out["recorded"] is True and out["written"] == 170
    assert len(_isolate) == n_recorded + 1 and "daily=0" in _isolate[-1][1]["note"]
    assert "written=170" in _isolate[-1][1]["note"]


def test_evaluation_is_enqueued_once_per_month_even_after_it_finished(monkeypatch):
    loop.run_once()
    _finish_everything()
    monkeypatch.setattr(loop, "_utcnow", lambda: datetime(2026, 7, 20, 3, 45, 0))
    loop.run_once()
    assert len(_runs(q.KIND_EVALUATE, date(2026, 6, 30))) == 1, "a finished evaluation must not be re-enqueued"
    assert len(_runs(q.KIND_PIT_PREPARE)) == 1 and len(_runs(q.KIND_MONTH_END)) == 1
    # The month turns: the next month end gets its own pair; Jul 31 was
    # not scored on Aug 1 (this tick is Aug 3) so it is caught up too.
    monkeypatch.setattr(loop, "_utcnow", lambda: datetime(2026, 8, 3, 3, 45, 0))
    loop.run_once()
    assert [r.as_of for r in _runs(q.KIND_EVALUATE)] == [date(2026, 6, 30), date(2026, 7, 31)]
    assert [r.as_of for r in _runs(q.KIND_MONTH_END)] == [date(2026, 6, 30), date(2026, 7, 31)]


def test_failed_evaluation_is_retried_on_the_next_daily_tick(monkeypatch):
    loop.run_once()
    ev = _runs(q.KIND_EVALUATE)[0]
    _finish_everything(fail_ids=(ev.id,))
    monkeypatch.setattr(loop, "_utcnow", lambda: datetime(2026, 7, 2, 10, 0, 0))
    loop.run_once()
    assert [r.status for r in _runs(q.KIND_EVALUATE, date(2026, 6, 30))] == [q.STATUS_FAILED], (
        "an interval tick never retries the evaluation — that is the daily step's job"
    )
    monkeypatch.setattr(loop, "_utcnow", lambda: datetime(2026, 7, 3, 3, 45, 0))
    loop.run_once()
    rows = _runs(q.KIND_EVALUATE, date(2026, 6, 30))
    assert [r.status for r in rows] == [q.STATUS_FAILED, q.STATUS_QUEUED]
    # The retry is again behind a pit_prepare for the month end.
    prep = _runs(q.KIND_PIT_PREPARE, date(2026, 6, 30))
    assert prep[-1].id < rows[-1].id


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


def test_a_broken_gate_records_the_error_and_still_drains(_isolate, monkeypatch):
    drains: list[int] = []
    monkeypatch.setattr(q, "drain", lambda max_runs=200: drains.append(max_runs) or [])

    def broken(*a, **k):
        raise RuntimeError("db away")

    monkeypatch.setattr(q, "run_exists", broken)
    out = loop.run_once()
    assert out["daily"] is False and out["problems"] == ["gate:RuntimeError"] and drains == [loop.MAX_RUNS_PER_TICK]
    assert _isolate[-1][1]["success"] is False and "errors=gate:RuntimeError" in _isolate[-1][1]["note"]


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
