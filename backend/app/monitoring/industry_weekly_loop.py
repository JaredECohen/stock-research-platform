"""FEAT-003 — weekly Industry Analysis report enqueue (Sunday 06:30 UTC).

Slice 1 registers the loop so ``KNOWN_LOOPS`` and cron-health know it
exists from day one; slice 4 fills ``run_once`` with the real work:
compute ``period_key`` / ``as_of`` from the prior Friday close, run the
classification, pre-fetch missing price series within
``INDUSTRY_PRICE_WARMUP_BUDGET``, and enqueue one ``industry_report_jobs``
row per active group plus the cross-industry snapshot job.

Until then a tick records an honest note — ``disabled`` when
``ENABLE_INDUSTRY_REPORTS`` is off (the web service and, until the deploy
config flips it, the worker), ``pending: report enqueue not yet wired``
when it is on — so cron-health shows a live loop with nothing hidden
behind it rather than a silent one. The schedule (owner decision 3) is
read from settings, not repeated here.
"""
from __future__ import annotations

import logging

from ..config import settings
from . import record_run

log = logging.getLogger(__name__)

LOOP_NAME = "industry_weekly_loop"


def run_once() -> dict:
    if not settings.enable_industry_reports:
        note = "disabled (ENABLE_INDUSTRY_REPORTS=false)"
        record_run(LOOP_NAME, success=True, note=note)
        return {"enabled": False, "enqueued": 0, "note": note}
    note = "pending: report enqueue not yet wired (FEAT-003 slice 4) enqueued=0"
    record_run(LOOP_NAME, success=True, note=note)
    return {"enabled": True, "enqueued": 0, "note": note}


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "cron",
        day_of_week=settings.industry_reports_cron_dow,
        hour=settings.industry_reports_cron_hour,
        minute=settings.industry_reports_cron_minute,
        id=LOOP_NAME, replace_existing=True, max_instances=1, coalesce=True,
    )
