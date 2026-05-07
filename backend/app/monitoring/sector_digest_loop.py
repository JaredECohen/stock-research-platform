"""Wave 10 — weekly sector cohort digest.

Runs Sundays 05:30 UTC, immediately after the per-ticker
`weekly_digest_loop`. Aggregates each sector's filing activity into a
single sector-memory pattern entry so the sector analyst reads "what
happened across the cohort this week" instead of (or alongside) the
per-name digests.

Cost: ~5-15 LLM calls (one per distinct sector in the universe);
each call is ~$0.001-0.002 cheap-tier.
"""
from __future__ import annotations

import logging
from typing import Dict

from ..services.filing_memory import weekly_sector_digest_all
from . import record_run

log = logging.getLogger(__name__)


def run_once(*, days_back: int = 7) -> Dict[str, int]:
    res = weekly_sector_digest_all(days_back=days_back)
    note = (
        f"sectors={res['sectors_checked']} digests={res['digests_written']}"
    )
    record_run("sector_digest_loop", note=note)
    return res


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "cron", day_of_week="sun", hour=5, minute=30,
        id="sector_digest_loop", replace_existing=True,
    )
