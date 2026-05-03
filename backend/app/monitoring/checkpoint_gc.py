"""Wave 6A — daily GC for expired memo-run checkpoints.

Wired only when ENABLE_MONITORING=true. Idempotent — safe to run repeatedly.
"""
from __future__ import annotations

import logging
from typing import Dict

from ..services.checkpoint_store import gc_expired
from . import record_run

log = logging.getLogger(__name__)


def run_once() -> Dict[str, int]:
    n = gc_expired()
    record_run("checkpoint_gc", note=f"deleted {n} expired checkpoints")
    return {"deleted": n}


def register(scheduler) -> None:
    # Daily at 04:00 UTC — after history_backfill (03:15) and outcome_loop (02:30).
    scheduler.add_job(
        run_once, "cron", hour=4, minute=0,
        id="checkpoint_gc", replace_existing=True,
    )
