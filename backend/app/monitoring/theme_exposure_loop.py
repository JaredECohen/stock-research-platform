"""Wave 10 — monthly theme exposure recompute.

Theme exposure derives from business descriptions + the trailing few
earnings transcripts. Both are slow-moving, so monthly is enough; the
job only matters for long-horizon screen / comps freshness.

Scheduled on the 1st of the month at 04:00 UTC. APScheduler's `cron`
trigger handles the `day=1` semantics natively; no skew to worry
about.

`theme_exposure_service.refresh_universe()` upserts on
`(ticker, theme)` so re-running mid-month is safe.
"""
from __future__ import annotations

import logging
from typing import Dict

from ..services.theme_exposure_service import refresh_universe
from . import record_run

log = logging.getLogger(__name__)


def run_once(*, limit: int | None = None) -> Dict[str, int]:
    res = refresh_universe(limit=limit)
    note = f"tickers={res['tickers']} rows_written={res['rows_written']}"
    record_run("theme_exposure_loop", note=note)
    return res


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "cron", day=1, hour=4, minute=0,
        id="theme_exposure_loop", replace_existing=True,
    )
