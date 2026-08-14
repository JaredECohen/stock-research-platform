"""Always-on monitoring loops (Phase 5).

Each loop is a small module exposing a `run_once(ticker_or_None)` function
suitable for unit testing in isolation, plus a `register(scheduler)` hook
that wires up the production cron schedule.

Loops are quiet — they push results into the snapshot cache as `*_hot`
snapshots so other agents can read them through the same interface they use
for warm/cold data.
"""
from datetime import datetime

# Module-level state used by `/api/admin/monitoring/status`. Defined BEFORE
# we import the per-loop modules so they can call `record_run` during their
# own import-time wiring without a circular import.
_LAST_RUNS: dict = {}


def _process_role() -> str:
    """Best-effort label for which process is reporting: worker or web."""
    import sys
    if any("app.worker" in a for a in sys.argv):
        return "worker"
    return "web"


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
            db.commit()
    except Exception:  # pragma: no cover — diagnostics must never break a loop
        import logging
        logging.getLogger(__name__).warning(
            "failed to persist cron run for %s", loop_name, exc_info=True,
        )


def status_snapshot() -> dict:
    """Merged view of loop runs: DB first, in-process state layered on top.

    DB rows are the cross-process truth. The in-memory dict wins on ties
    only because if this process just ran a loop, its record is at least
    as fresh as anything it could read back.
    """
    merged: dict = {}
    try:
        from ..database import SessionLocal
        from ..models import CronLoopRun
        with SessionLocal() as db:
            for row in db.query(CronLoopRun).all():
                merged[row.loop_name] = {
                    "last_run_at": row.last_run_at.isoformat() if row.last_run_at else None,
                    "success": row.success,
                    "note": row.note or "",
                    "reported_by": row.reported_by or "",
                }
    except Exception:  # pragma: no cover — fall back to in-process state
        import logging
        logging.getLogger(__name__).warning(
            "cron status DB read failed; reporting in-process state only",
            exc_info=True,
        )
    merged.update(_LAST_RUNS)
    return merged


from . import (  # noqa: E402,F401
    catalyst_loop, checkpoint_gc, edgar_poller, history_backfill, llm_log_gc,
    macro_loop, mispricing_audit_loop, news_loop, outcome_loop,
    postmortem_loop, sector_digest_loop, social_loop,
    theme_exposure_loop, transcripts_poller, weekly_digest_loop,
)

__all__ = [
    "catalyst_loop", "checkpoint_gc", "edgar_poller", "history_backfill",
    "llm_log_gc", "macro_loop", "mispricing_audit_loop", "news_loop",
    "outcome_loop", "postmortem_loop", "sector_digest_loop", "social_loop",
    "theme_exposure_loop", "transcripts_poller", "weekly_digest_loop",
    "register_all", "record_run", "status_snapshot",
]


def register_all(scheduler) -> None:
    """Register every monitoring loop with an APScheduler instance."""
    edgar_poller.register(scheduler)
    transcripts_poller.register(scheduler)
    news_loop.register(scheduler)
    social_loop.register(scheduler)
    macro_loop.register(scheduler)
    llm_log_gc.register(scheduler)
    history_backfill.register(scheduler)
    outcome_loop.register(scheduler)
    checkpoint_gc.register(scheduler)
    # Wave 10 — postmortem feedback loop, catalyst refresh, theme exposure,
    # weekly filing digest, sector cohort digest, mispricing-audit nightly.
    postmortem_loop.register(scheduler)
    catalyst_loop.register(scheduler)
    theme_exposure_loop.register(scheduler)
    weekly_digest_loop.register(scheduler)
    sector_digest_loop.register(scheduler)
    mispricing_audit_loop.register(scheduler)
