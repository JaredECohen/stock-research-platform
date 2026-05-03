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


def record_run(loop_name: str, *, success: bool = True, note: str = "") -> None:
    _LAST_RUNS[loop_name] = {
        "last_run_at": datetime.utcnow().isoformat(),
        "success": success, "note": note,
    }


def status_snapshot() -> dict:
    return dict(_LAST_RUNS)


from . import (  # noqa: E402,F401
    edgar_poller, history_backfill, llm_log_gc, macro_loop,
    news_loop, outcome_loop, social_loop,
)

__all__ = [
    "edgar_poller", "history_backfill", "llm_log_gc",
    "macro_loop", "news_loop", "outcome_loop", "social_loop",
    "register_all", "record_run", "status_snapshot",
]


def register_all(scheduler) -> None:
    """Register every monitoring loop with an APScheduler instance."""
    edgar_poller.register(scheduler)
    news_loop.register(scheduler)
    social_loop.register(scheduler)
    macro_loop.register(scheduler)
    llm_log_gc.register(scheduler)
    history_backfill.register(scheduler)
    outcome_loop.register(scheduler)
