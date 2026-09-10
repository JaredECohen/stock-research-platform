"""Phase 6 — the scorecard's one worker loop (daily, 03:45 UTC).

One cron entry, one `KNOWN_LOOPS` name. Each tick, in this order:

1. `recover_orphans` — rows a dead process left `running` become
   `failed/WorkerRestart`, stale `queued` rows `failed/QueueExpired`. Runs
   FIRST so nothing below can coalesce onto a corpse.
2. `ensure_version_registered` — the registry row for the in-code spec
   (idempotent; the web process falls back to the in-code spec anyway).
3. Enqueue the scheduled daily run, `as_of` = yesterday UTC — at 03:45
   UTC the last US close is yesterday's, and a fiscal row becomes
   knowable on its `available_at` date, so "as of yesterday" is the
   latest honest cross-section. When that as-of is a calendar month end
   the run writes month-end rows (kept forever).
4. On the first tick after a month turns: enqueue `pit_prepare` (month-end
   price sync + availability backfill for the universe) and then the
   monthly `evaluate` run for the month that just ended. Exactly once per
   month — a non-failed row for that identity blocks re-enqueue; a failed
   evaluation is retried on the next tick.
5. Drain the queue FIFO (admin refresh / backfill / evaluate requests
   land here too), bounded per tick.
6. GC daily score rows older than the retention window; month-end rows
   are never touched.
7. `record_run("scorecard_loop", …)` with `written=`, `skipped=`,
   `failed=` counts so an empty tick is distinguishable from a dead loop.

03:45 UTC sits after `history_backfill` (03:15), so the daily run sees the
freshest fundamentals, and clear of `mispricing_audit_loop` at 04:30, an
LLM-heavy job on the same 512 MB worker. `ENABLE_SCORECARD_LOOP=false`
records a `disabled` tick and does nothing else — rows stay readable.
"""
from __future__ import annotations

import logging
from calendar import monthrange
from datetime import date, datetime, timedelta

from ..agents.log_safety import safe_exc
from ..config import settings
from ..services import scorecard_queue, scorecard_service
from . import record_run

log = logging.getLogger(__name__)

LOOP_NAME = "scorecard_loop"
CRON_HOUR = 3
CRON_MINUTE = 45
# A 60-month backfill plus its evaluation is ~62 runs; the bound keeps a
# runaway request from owning the worker for a whole day.
MAX_RUNS_PER_TICK = 200


def _utcnow() -> datetime:
    """Clock seam — tests monkeypatch this instead of freezing time."""
    return datetime.utcnow()


def _month_end(d: date) -> date:
    return date(d.year, d.month, monthrange(d.year, d.month)[1])


def last_completed_month_end(today: date) -> date:
    """The last calendar month end strictly before `today`."""
    first = date(today.year, today.month, 1)
    return first - timedelta(days=1)


def scheduled_as_of(now: datetime) -> date:
    return (now - timedelta(days=1)).date()


def run_once() -> dict:
    """One tick. Never raises: every failure lands on a run row or in the
    loop note, because a loop that dies before `record_run` is invisible
    to cron-health (the postmortem_loop lesson)."""
    now = _utcnow()
    if not settings.enable_scorecard_loop:
        record_run(LOOP_NAME, success=True, note="disabled=1 written=0 skipped=0 failed=0")
        return {"disabled": True}

    counts = {"recovered_failed": 0, "recovered_expired": 0, "enqueued": 0, "claimed": 0,
              "written": 0, "skipped": 0, "failed": 0, "gc": 0}
    problems: list[str] = []

    try:
        rec = scorecard_queue.recover_orphans()
        counts["recovered_failed"], counts["recovered_expired"] = rec["failed"], rec["expired"]
    except Exception as exc:
        problems.append(f"recover:{type(exc).__name__}")
        log.warning("scorecard_loop recovery failed: %s", safe_exc(exc))

    version_key = scorecard_service.VERSION_KEY
    try:
        scorecard_service.ensure_version_registered()
    except Exception as exc:
        problems.append(f"registry:{type(exc).__name__}")
        log.warning("scorecard_loop version registry failed: %s", safe_exc(exc))

    try:
        counts["enqueued"] += _enqueue_scheduled(version_key, now)
    except Exception as exc:
        problems.append(f"enqueue:{type(exc).__name__}")
        log.warning("scorecard_loop scheduling failed: %s", safe_exc(exc))

    try:
        for run in scorecard_queue.drain(MAX_RUNS_PER_TICK):
            counts["claimed"] += 1
            status = run.get("status")
            if status == scorecard_queue.STATUS_SUCCEEDED:
                counts["written"] += int((run.get("params") or {}).get("written") or run.get("scored_count") or 0)
            elif status == scorecard_queue.STATUS_SKIPPED:
                counts["skipped"] += 1
            else:
                counts["failed"] += 1
    except Exception as exc:
        problems.append(f"drain:{type(exc).__name__}")
        log.warning("scorecard_loop drain failed: %s", safe_exc(exc))

    try:
        counts["gc"] = scorecard_service.gc_daily_rows(today=now.date())
    except Exception as exc:
        problems.append(f"gc:{type(exc).__name__}")
        log.warning("scorecard_loop gc failed: %s", safe_exc(exc))

    note = (
        f"as_of={scheduled_as_of(now).isoformat()} enqueued={counts['enqueued']} claimed={counts['claimed']} "
        f"written={counts['written']} skipped={counts['skipped']} failed={counts['failed']} "
        f"recovered={counts['recovered_failed']} expired={counts['recovered_expired']} gc={counts['gc']}"
    )
    if problems:
        note += " errors=" + ",".join(problems)
    record_run(LOOP_NAME, success=not problems and counts["failed"] == 0, note=note)
    return {**counts, "problems": problems, "note": note}


def _enqueue_scheduled(version_key: str, now: datetime) -> int:
    """The daily run, plus the monthly pit_prepare + evaluate pair on the
    first tick after a month end. Returns how many rows were created."""
    created = 0
    as_of = scheduled_as_of(now)
    _run, was_created = scorecard_queue.enqueue_run(
        version_key=version_key, as_of=as_of, kind=scorecard_queue.KIND_SCHEDULED, requested_by=LOOP_NAME,
    )
    created += int(was_created)

    month_end = last_completed_month_end(now.date())
    if not scorecard_queue.unfailed_run_exists(version_key, month_end, scorecard_queue.KIND_EVALUATE):
        # Order matters (FIFO by id): the month-end prices must land
        # before the evaluation joins forward returns against them.
        _prep, prep_created = scorecard_queue.enqueue_run(
            version_key=version_key, as_of=month_end, kind=scorecard_queue.KIND_PIT_PREPARE, requested_by=LOOP_NAME,
        )
        created += int(prep_created)
        _ev, ev_created = scorecard_queue.enqueue_run(
            version_key=version_key, as_of=month_end, kind=scorecard_queue.KIND_EVALUATE, requested_by=LOOP_NAME,
        )
        created += int(ev_created)
    return created


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "cron", hour=CRON_HOUR, minute=CRON_MINUTE,
        id=LOOP_NAME, replace_existing=True, max_instances=1, coalesce=True,
    )
