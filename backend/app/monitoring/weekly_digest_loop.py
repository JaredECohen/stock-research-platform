"""Wave 10 — weekly filing-memory digest.

Runs Sundays 05:00 UTC. Walks the curated universe, consolidates each
ticker's past-week filing activity into a single memory entry. The
per-filing post_pass entries still fire on ingest; this digest is
additive, summarizing what the human reader saw across the week.

Idempotent: a re-run for the same ISO week appends another entry but
the parser dedupes on (date, body) so duplicates collapse. Cost is
~100 cheap-tier LLM calls (one per ticker that had filings in the
window), capped naturally by how many companies actually file in a
given week (~5-15 typically; up to ~100 during 10-Q season).
"""
from __future__ import annotations

import logging
from typing import Dict

from ..services.filing_memory import weekly_digest_universe
from . import record_run

log = logging.getLogger(__name__)


def run_once(*, days_back: int = 7) -> Dict[str, int]:
    res = weekly_digest_universe(days_back=days_back)
    note = (
        f"tickers={res['tickers_checked']} "
        f"digests={res['digests_written']} "
        f"filings_in_window={res['total_filings_in_window']}"
    )
    record_run("weekly_digest_loop", note=note)
    return res


def register(scheduler) -> None:
    # Sundays 05:00 UTC. Late enough that Friday night's 10-K filings
    # are fully ingested; early enough to be ready before Monday open.
    scheduler.add_job(
        run_once, "cron", day_of_week="sun", hour=5, minute=0,
        id="weekly_digest_loop", replace_existing=True,
    )
