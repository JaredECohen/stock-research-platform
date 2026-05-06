"""Wave 10 — daily catalyst calendar refresh.

Pulls each curated-universe ticker's next earnings date from FMP into
`catalyst_events`. Scheduled at 06:00 UTC — well after the morning
ingestion loops, before any user activity.

`catalyst_service.refresh_universe()` is idempotent on
`(ticker, event_type, event_date, title)`. Today only the FMP earnings
calendar is wired; the schema also supports FDA / conference /
investor-day events for the Phase F follow-up.
"""
from __future__ import annotations

import logging
from typing import Dict

from ..services.catalyst_service import refresh_universe
from . import record_run

log = logging.getLogger(__name__)


def run_once(*, limit: int | None = None) -> Dict[str, int]:
    res = refresh_universe(limit=limit)
    note = (
        f"tickers={res['tickers_checked']} rows_written={res['rows_written']}"
    )
    record_run("catalyst_loop", note=note)
    return res


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "cron", hour=6, minute=0,
        id="catalyst_loop", replace_existing=True,
    )
