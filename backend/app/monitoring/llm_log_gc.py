"""Daily GC for LLMCallLog rows older than 90 days.

Wired only when ENABLE_MONITORING=true. Idempotent — safe to run repeatedly.
"""
from __future__ import annotations

import logging
from typing import Dict

from ..services.llm_metrics import gc_old
from . import record_run

log = logging.getLogger(__name__)


def run_once(max_age_days: int = 90) -> Dict[str, int]:
    n = gc_old(max_age_days=max_age_days)
    record_run("llm_log_gc", note=f"deleted {n} rows >{max_age_days}d old")
    return {"deleted": n, "max_age_days": max_age_days}


def register(scheduler) -> None:
    scheduler.add_job(run_once, "interval", days=1, id="llm_log_gc", replace_existing=True)
