"""Wave 10 — daily postmortem loop.

Runs after `outcome_loop` (which writes the realized returns the
postmortem reads from). Two cadences fire from the same job:

- 30-day "early read" — drift signal; flagged if the call is going
  against us within the first month.
- 90-day "full postmortem" — calibration lesson; verdict + per-agent
  attribution + lesson written back to company / sector / PM memory.

`postmortem_service.run_postmortems` is idempotent on
`(memo_snapshot_id, horizon_days)` and applies the dedupe guard
(rating-change skip + 14-day rate-limit) so high-throughput names
don't spam.

Scheduled at 03:00 UTC, after `outcome_loop` (02:30) but before
`history_backfill` (03:15) so any concurrency stays small.
"""
from __future__ import annotations

import logging
from typing import Dict

from ..services.postmortem_service import run_postmortems
from . import record_run

log = logging.getLogger(__name__)


def run_once(*, limit_per_horizon: int = 25) -> Dict[str, int]:
    early = run_postmortems(horizon_days=30, limit=limit_per_horizon)
    full = run_postmortems(horizon_days=90, limit=limit_per_horizon)
    note = (
        f"30d due={early['due']} written={early['written']} skipped={early['skipped']}; "
        f"90d due={full['due']} written={full['written']} skipped={full['skipped']}"
    )
    success = early.get("skipped", 0) + full.get("skipped", 0) <= (
        early.get("due", 0) + full.get("due", 0)
    )
    record_run("postmortem_loop", success=success, note=note)
    return {
        "early_due": early["due"], "early_written": early["written"],
        "full_due": full["due"], "full_written": full["written"],
    }


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "cron", hour=3, minute=0,
        id="postmortem_loop", replace_existing=True,
    )
