"""Always-on monitoring loops (Phase 5).

Each loop is a small module exposing a `run_once(ticker_or_None)` function
suitable for unit testing in isolation, plus a `register(scheduler)` hook
that wires up the production cron schedule.

Loops are quiet — they push results into the snapshot cache as `*_hot`
snapshots so other agents can read them through the same interface they use
for warm/cold data.
"""
import contextvars
import functools
from datetime import datetime

# Module-level state used by `/api/admin/monitoring/status`. Defined BEFORE
# we import the per-loop modules so they can call `record_run` during their
# own import-time wiring without a circular import.
_LAST_RUNS: dict = {}

# Existing deployments require last_run_at/success to be NOT NULL. A
# progress-only row uses this sentinel; public snapshots normalize it to
# None rather than inventing a completed run. No real loop ran in 1970.
NEVER_COMPLETED_AT = datetime(1970, 1, 1)

# Every loop `register_all` wires up, by the name it passes to
# `record_run`. Needed because `/api/admin/cron-health` reports what has
# been *recorded* — so a loop that has never completed once has no row
# and was simply absent from the response, rather than flagged.
#
# That blind spot hid a real outage: `postmortem_loop` raised on every
# 03:00 UTC run (a missing `memo_outcomes.regime_at_memo` column) before
# reaching `record_run`, so nightly postmortems were dead and the health
# endpoint showed nothing wrong. `test_cron_health_cross_process`
# asserts this list matches what `register_all` actually registers.
KNOWN_LOOPS: tuple[str, ...] = (
    "billing_loop",
    "catalyst_loop",
    "checkpoint_gc",
    "edgar_poller",
    "history_backfill",
    "industry_classification_loop",
    "industry_weekly_loop",
    "llm_log_gc",
    "macro_loop",
    "mispricing_audit_loop",
    "news_loop",
    "outcome_loop",
    "postmortem_loop",
    "sample_build_loop",
    "scorecard_loop",
    "sector_digest_loop",
    "snapshot_gc",
    "social_loop",
    "theme_exposure_loop",
    "transcripts_poller",
    "weekly_digest_loop",
)


# `_process_role` moved to `app.runtime_role` (the LLM layer labels every
# call row with it and must not import every loop to do so). Re-exported
# here, names unchanged, for `record_run` below and for the callers and
# tests that already use `monitoring._process_role` / `PROCESS_ROLE_ENV`.
from ..runtime_role import (  # noqa: E402,F401
    _ROLES,
    _WORKER_MODULE,
    PROCESS_ROLE_ENV,
    _process_role,
)


def note_names(names) -> str:
    """Retain every identity in the order supplied by the bounded loop.

    Standing rule in this repo: no silent caps. Anything a loop skipped,
    deferred or failed on has to be legible in `/api/admin/cron-health`, and a
    count or "+N more" marker cannot identify the names still waiting. Run
    notes use a Text column and the same complete note is emitted to logs.
    This bounds work at the caller without abbreviating its deferred list.
    """
    return ", ".join(names)


def record_run(loop_name: str, *, success: bool = True, note: str = "") -> None:
    """Record a loop's completion, in memory AND in the database.

    The DB write is what makes `/api/admin/cron-health` work at all now
    that the loops run in `marketmosaic-worker` while the endpoint is
    served by the web service — a module-level dict cannot cross that
    boundary. The in-memory copy is kept because it costs nothing and
    keeps the endpoint honest when the DB write fails.

    Never raises: a monitoring loop must not fail because its own
    bookkeeping failed. A lost record shows up as a stale loop, which is
    the correct thing to report when we don't know.
    """
    _LAST_RUNS[loop_name] = {
        "last_run_at": datetime.utcnow().isoformat(),
        "success": success, "note": note,
        "progress_at": None, "progress_note": None, "progress_success": None,
    }
    try:
        from ..database import SessionLocal
        from ..models import CronLoopRun
        with SessionLocal() as db:
            row = db.query(CronLoopRun).filter(
                CronLoopRun.loop_name == loop_name
            ).one_or_none()
            if row is None:
                row = CronLoopRun(loop_name=loop_name)
                db.add(row)
            row.last_run_at = datetime.utcnow()
            row.success = bool(success)
            row.note = note or ""
            row.reported_by = _process_role()
            row.progress_at = None
            row.progress_note = None
            row.progress_success = None
            db.commit()
    except Exception:  # pragma: no cover — diagnostics must never break a loop
        import logging
        logging.getLogger(__name__).warning(
            "failed to persist cron run for %s", loop_name, exc_info=True,
        )


def record_progress(loop_name: str, *, success: bool | None = None, note: str = "") -> None:
    """Persist in-flight activity without changing the last completed run.

    Progress survives worker death and is cleared only by record_run. The
    nullable verdict describes this pass so far, never overall loop health.
    It is deliberately not held in a process-local fallback dictionary.
    """
    try:
        from ..database import SessionLocal
        from ..models import CronLoopRun
        with SessionLocal() as db:
            row = db.query(CronLoopRun).filter(CronLoopRun.loop_name == loop_name).one_or_none()
            if row is None:
                row = CronLoopRun(
                    loop_name=loop_name, last_run_at=NEVER_COMPLETED_AT,
                    success=False, note="never run", reported_by=_process_role(),
                )
                db.add(row)
            row.progress_at = datetime.utcnow()
            if row.progress_success is False and success is not False:
                # A worker restart must not clear a failure from an
                # interrupted pass just by announcing new activity.
                retained = (row.progress_note or "").split("\nLatest activity: ", 1)[0]
                row.progress_note = f"{retained}\nLatest activity: {note or ''}"
            else:
                row.progress_note = note or ""
                row.progress_success = success
            db.commit()
    except Exception:  # pragma: no cover — diagnostics must never break a loop
        import logging
        logging.getLogger(__name__).warning(
            "failed to persist cron progress for %s", loop_name, exc_info=True,
        )


def status_snapshot() -> dict:
    """Shared DB state is authoritative; local completions are fallback.

    A stale completion cached by this process must not overwrite progress
    or a later completion written by the worker.
    """
    merged: dict = {}
    try:
        from ..database import SessionLocal
        from ..models import CronLoopRun
        with SessionLocal() as db:
            for row in db.query(CronLoopRun).all():
                completed = row.last_run_at != NEVER_COMPLETED_AT
                merged[row.loop_name] = {
                    "last_run_at": row.last_run_at.isoformat() if completed and row.last_run_at else None,
                    "success": row.success if completed else None,
                    "note": row.note or "",
                    "reported_by": row.reported_by or "",
                    "progress_at": row.progress_at.isoformat() if row.progress_at else None,
                    "progress_note": row.progress_note,
                    "progress_success": row.progress_success,
                }
    except Exception:  # pragma: no cover — fall back to in-process state
        import logging
        logging.getLogger(__name__).warning(
            "cron status DB read failed; reporting in-process state only",
            exc_info=True,
        )
    for name, local in _LAST_RUNS.items():
        merged.setdefault(name, local)
    return merged


from . import (  # noqa: E402,F401
    billing_loop,
    catalyst_loop,
    checkpoint_gc,
    edgar_poller,
    history_backfill,
    industry_classification_loop,
    industry_weekly_loop,
    llm_log_gc,
    macro_loop,
    mispricing_audit_loop,
    news_loop,
    outcome_loop,
    postmortem_loop,
    sample_build_loop,
    scorecard_loop,
    sector_digest_loop,
    snapshot_gc,
    social_loop,
    theme_exposure_loop,
    transcripts_poller,
    weekly_digest_loop,
)

__all__ = [
    "billing_loop", "catalyst_loop", "checkpoint_gc", "edgar_poller", "history_backfill",
    "industry_classification_loop", "industry_weekly_loop",
    "llm_log_gc", "macro_loop", "mispricing_audit_loop", "news_loop",
    "outcome_loop", "postmortem_loop", "sample_build_loop", "scorecard_loop",
    "sector_digest_loop", "snapshot_gc", "social_loop",
    "theme_exposure_loop", "transcripts_poller", "weekly_digest_loop",
    "register_all", "record_run", "note_names", "status_snapshot", "KNOWN_LOOPS",
]


def _job_with_origin(func, loop_id: str):
    """`func` run as loop `loop_id`: every LLM call it makes carries
    `origin=loop:<id>` (attribution slice A2a, design §4.8).

    Each run gets a FRESH `contextvars.copy_context()` (attribution critique
    #11): APScheduler's pool threads are reused, so a context variable set
    by one job — the failover-event list, an attempt scope, a context layer
    a loop forgot to close — would otherwise leak into the next job on that
    thread, and on the long-lived worker the failover list would grow
    without bound. An umbrella context names the origin only, never an
    agent: the registry's per-action agents stay correct underneath it.

    `functools.wraps` keeps `__module__`/`__name__`, which the KNOWN_LOOPS
    pin and APScheduler's job repr read.
    """
    origin = f"loop:{loop_id}"

    def _run(*args, **kwargs):
        from ..agents.llm import llm_call_context
        with llm_call_context(origin=origin):
            return func(*args, **kwargs)

    @functools.wraps(func)
    def job(*args, **kwargs):
        return contextvars.copy_context().run(_run, *args, **kwargs)

    return job


class _OriginScheduler:
    """Scheduler proxy for `register_all`: wraps each job so it runs under
    its loop's origin (above). Everything but `add_job` passes through."""

    def __init__(self, scheduler) -> None:
        self._scheduler = scheduler

    def add_job(self, func, *args, **kwargs):
        loop_id = kwargs.get("id") or getattr(func, "__module__", "unknown").rsplit(".", 1)[-1]
        return self._scheduler.add_job(_job_with_origin(func, loop_id), *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._scheduler, name)


def register_all(scheduler) -> None:
    """Register every monitoring loop with an APScheduler instance.

    The loops register against a proxy, so every job's LLM rows and
    `llm_call` lines say which loop started them (`origin=loop:<id>`)."""
    scheduler = _OriginScheduler(scheduler)
    edgar_poller.register(scheduler)
    transcripts_poller.register(scheduler)
    news_loop.register(scheduler)
    social_loop.register(scheduler)
    macro_loop.register(scheduler)
    llm_log_gc.register(scheduler)
    history_backfill.register(scheduler)
    outcome_loop.register(scheduler)
    checkpoint_gc.register(scheduler)
    snapshot_gc.register(scheduler)
    # Wave 10 — postmortem feedback loop, catalyst refresh, theme exposure,
    # weekly filing digest, sector cohort digest, mispricing-audit nightly.
    postmortem_loop.register(scheduler)
    catalyst_loop.register(scheduler)
    theme_exposure_loop.register(scheduler)
    weekly_digest_loop.register(scheduler)
    sector_digest_loop.register(scheduler)
    mispricing_audit_loop.register(scheduler)
    # FEAT-002 — curated public samples for the logged-out site. Polls for
    # admin rebuild requests and builds weekly; see the module docstring.
    sample_build_loop.register(scheduler)
    # FEAT-002 — hourly billing housekeeping: trial-expiry / downgrade
    # funnel events, limiter + analytics GC, stale-reservation settlement.
    # Plan changes never happen here (`plans.resolve_plan` is read-time).
    billing_loop.register(scheduler)
    # Phase 6 — the fundamental scorecard's single daily loop (03:45 UTC):
    # queue recovery, the scheduled scoring run, the monthly evaluation
    # enqueue, the drain and retention GC all happen inside one tick.
    scorecard_loop.register(scheduler)
    # FEAT-003 — daily company → GICS industry-group classification audit
    # (03:40 UTC, DB-only) and the Sunday 06:30 UTC Industry Analysis
    # enqueue. Both registered from day one so cron-health lists them as
    # "never run" rather than not at all; the weekly loop records a
    # `disabled` note until ENABLE_INDUSTRY_REPORTS is on for the worker.
    industry_classification_loop.register(scheduler)
    industry_weekly_loop.register(scheduler)
