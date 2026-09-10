"""Phase 6 — the scorecard's one worker loop.

One APScheduler job (interval, every `INTERVAL_MINUTES`), one
`KNOWN_LOOPS` name, one durable queue. Every tick, in this order:

1. `recover_orphans` — rows a dead process left `running` become
   `failed/WorkerRestart`, stale `queued` rows `failed/QueueExpired`. Runs
   FIRST so nothing below can coalesce onto a corpse.
2. The DAILY step, gated to the first tick at or after `DAILY_HOUR:DAILY_MINUTE`
   UTC on each day (the gate is the queue itself: "no scheduled row for
   yesterday exists yet", so a worker that was down at 03:45 catches up on
   its first tick back, and a failed daily run is not retried until the
   next day):
   a. `ensure_version_registered` — the registry row for the in-code spec
      (idempotent; the web process falls back to the in-code spec anyway).
   b. On the first daily tick after a month turns (no non-failed
      `evaluate` row for the last completed month end): enqueue
      `pit_prepare` for that month end FIRST. It syncs the month-end close
      into `price_month_ends`, and FIFO order guarantees it runs before
      any scoring of that month end below.
   c. Make sure the month-end cross-section is scored: when yesterday IS
      the month end that is the scheduled run itself; when the loop
      missed that day (worker down on the 1st) and no non-failed scoring
      run for the month end exists, enqueue a `month_end` catch-up. Month-end
      rows are kept forever and are the evaluation's sample — a missed
      tick must not lose a month.
   d. Enqueue the scheduled daily run, `as_of` = yesterday UTC — at 03:45
      UTC the last US close is yesterday's, and a fiscal row becomes
      knowable on its `available_at` date, so "as of yesterday" is the
      latest honest cross-section.
   e. Then the monthly `evaluate` run for the month that just ended.
      Exactly once per month — a non-failed row for that identity blocks
      re-enqueue; a failed evaluation is retried on the next daily tick.
   f. GC daily score rows older than the retention window; month-end
      rows are never touched.
3. Drain the queue FIFO — admin refresh / evaluate / backfill requests
   land here too, so they run within one interval instead of waiting for
   the next 03:45 — bounded per tick.
4. `record_run("scorecard_loop", …)` with `written=`, `skipped=`,
   `failed=` counts. Recorded on the daily tick and on any tick that did
   something (enqueued, claimed, recovered, errored); a quiet interval
   tick leaves the last informative note in place so `cron-health` still
   shows what the loop last did, and the daily record keeps the loop
   inside cron-health's daily staleness class.

03:45 UTC sits after `history_backfill` (03:15), so the daily run sees the
freshest fundamentals, and clear of `mispricing_audit_loop` at 04:30, an
LLM-heavy job on the same 512 MB worker. `ENABLE_SCORECARD_LOOP=false`
records a `disabled` tick and does nothing else — rows stay readable.
"""
from __future__ import annotations

import logging
from calendar import monthrange
from datetime import date, datetime, time, timedelta

from ..agents.log_safety import safe_exc
from ..config import settings
from ..services import scorecard_queue, scorecard_service
from . import record_run

log = logging.getLogger(__name__)

LOOP_NAME = "scorecard_loop"
# The daily step fires on the first tick at/after this UTC time.
DAILY_HOUR = 3
DAILY_MINUTE = 45
DAILY_AT = time(DAILY_HOUR, DAILY_MINUTE)
# Admin-requested runs wait at most one interval. Every tick is a few
# cheap queries when there is nothing to do.
INTERVAL_MINUTES = 5
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


def daily_due(version_key: str, now: datetime) -> bool:
    """True on the first tick at/after `DAILY_AT` UTC whose scheduled run
    (as_of = yesterday) has not been attempted yet, in any status. The
    queue is the gate — cross-process, survives restarts, and a failed
    daily row is one attempt per day rather than one per interval."""
    if now.time() < DAILY_AT:
        return False
    return not scorecard_queue.run_exists(
        version_key, scheduled_as_of(now), kinds=(scorecard_queue.KIND_SCHEDULED,),
    )


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
    version_key = scorecard_service.VERSION_KEY

    try:
        rec = scorecard_queue.recover_orphans()
        counts["recovered_failed"], counts["recovered_expired"] = rec["failed"], rec["expired"]
    except Exception as exc:
        problems.append(f"recover:{type(exc).__name__}")
        log.warning("scorecard_loop recovery failed: %s", safe_exc(exc))

    try:
        daily = daily_due(version_key, now)
    except Exception as exc:
        daily = False
        problems.append(f"gate:{type(exc).__name__}")
        log.warning("scorecard_loop daily gate failed: %s", safe_exc(exc))

    if daily:
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

    if daily:
        try:
            counts["gc"] = scorecard_service.gc_daily_rows(today=now.date())
        except Exception as exc:
            problems.append(f"gc:{type(exc).__name__}")
            log.warning("scorecard_loop gc failed: %s", safe_exc(exc))

    note = (
        f"as_of={scheduled_as_of(now).isoformat()} daily={'1' if daily else '0'} enqueued={counts['enqueued']} "
        f"claimed={counts['claimed']} written={counts['written']} skipped={counts['skipped']} "
        f"failed={counts['failed']} recovered={counts['recovered_failed']} expired={counts['recovered_expired']} "
        f"gc={counts['gc']}"
    )
    if problems:
        note += " errors=" + ",".join(problems)
    active = daily or bool(problems) or any(
        counts[k] for k in ("enqueued", "claimed", "recovered_failed", "recovered_expired")
    )
    if active:
        record_run(LOOP_NAME, success=not problems and counts["failed"] == 0, note=note)
    return {**counts, "daily": daily, "recorded": active, "problems": problems, "note": note}


def _enqueue_scheduled(version_key: str, now: datetime) -> int:
    """The daily step's queue writes, in the order FIFO must run them:
    `pit_prepare` for a freshly ended month, the month-end scoring run
    (scheduled or catch-up), the daily scheduled run, then `evaluate`.
    Returns how many rows were created."""
    created = 0
    as_of = scheduled_as_of(now)
    month_end = last_completed_month_end(now.date())
    month_turned = not scorecard_queue.unfailed_run_exists(version_key, month_end, scorecard_queue.KIND_EVALUATE)

    if month_turned:
        # FIRST: the month-end close must be in `price_month_ends` before
        # the month-end cross-section is scored (it prefers the store only
        # when the store holds the as-of month) and before the evaluation
        # joins forward returns against it.
        _prep, prep_created = scorecard_queue.enqueue_run(
            version_key=version_key, as_of=month_end, kind=scorecard_queue.KIND_PIT_PREPARE, requested_by=LOOP_NAME,
        )
        created += int(prep_created)
        if as_of != month_end and not scorecard_queue.run_exists(
            version_key, month_end, kinds=scorecard_queue.SCORING_KINDS,
            exclude_statuses=(scorecard_queue.STATUS_FAILED,),
        ):
            # The loop was not running on the first of the month: score the
            # month end anyway so the evaluation's sample never loses a month.
            _catch_up, cu_created = scorecard_queue.enqueue_run(
                version_key=version_key, as_of=month_end, kind=scorecard_queue.KIND_MONTH_END, requested_by=LOOP_NAME,
            )
            created += int(cu_created)

    _run, was_created = scorecard_queue.enqueue_run(
        version_key=version_key, as_of=as_of, kind=scorecard_queue.KIND_SCHEDULED, requested_by=LOOP_NAME,
    )
    created += int(was_created)

    if month_turned:
        _ev, ev_created = scorecard_queue.enqueue_run(
            version_key=version_key, as_of=month_end, kind=scorecard_queue.KIND_EVALUATE, requested_by=LOOP_NAME,
        )
        created += int(ev_created)
    return created


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "interval", minutes=INTERVAL_MINUTES,
        id=LOOP_NAME, replace_existing=True, max_instances=1, coalesce=True,
    )
