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

The note distinguishes four outcomes per horizon, and the distinction is
load-bearing rather than cosmetic. This loop reported `success=False` every
night with `written=2 skipped=23` while the 23 were memos that already had
a postmortem: the database rejected the duplicate insert, the service
counted the rejection as a skipped postmortem, and `success = skipped == 0`
turned a healthy backlog into a standing red light. A loop that cries wolf
nightly is a loop nobody reads, which is exactly how the dead filings
pipeline stayed dead for the system's whole life.

Scheduled at 03:00 UTC, after `outcome_loop` (02:30) but before
`history_backfill` (03:15) so any concurrency stays small.
"""
from __future__ import annotations

import logging

from ..services.postmortem_service import run_postmortems
from . import record_run

log = logging.getLogger(__name__)


def _summarize(report: dict) -> str:
    """One horizon's counts, with the states kept apart.

    `already_done` and `deduped` are reported even at zero: their absence
    from the note is what let "23 memos we could not postmortem" stand in
    for "23 memos that were already postmortem'd" for months.
    """
    summary = (
        f"due={report.get('due', 0)} written={report.get('written', 0)} "
        f"already_done={report.get('already_done', 0)} "
        f"deduped={report.get('deduped', 0)} skipped={report.get('skipped', 0)} "
        f"deferred={report.get('deferred', 0)}"
    )
    for label in ("deduped", "deferred"):
        memos = report.get(f"{label}_memos", [])
        if memos:
            summary += f"; {label} memos: " + ", ".join(
                f"{m['ticker']}#{m['memo_snapshot_id']} ({m['reason']})"
                for m in memos
            )
    return summary


def run_once(*, limit_per_horizon: int = 25) -> dict[str, int]:
    try:
        early = run_postmortems(horizon_days=30, limit=limit_per_horizon)
        full = run_postmortems(horizon_days=90, limit=limit_per_horizon)
    except Exception as exc:
        # Record before re-raising so cron-health and APScheduler both see
        # the failure.  Previously an exception inside run_postmortems
        # bypassed record_run entirely and made a dead loop look absent.
        record_run(
            "postmortem_loop",
            success=False,
            note=f"failed: {type(exc).__name__}: {exc}"[:1000],
        )
        raise
    note = f"30d {_summarize(early)}; 90d {_summarize(full)}"
    # Only genuinely unwritten work fails the loop. `already_done` and
    # `deduped` are answers, not failures — folding them into `skipped` is
    # what had this loop reporting success=False every single night for a
    # backlog that was entirely fine.
    success = early.get("skipped", 0) + full.get("skipped", 0) == 0
    record_run("postmortem_loop", success=success, note=note)
    return {
        "early_due": early["due"], "early_written": early["written"],
        "early_already_done": early.get("already_done", 0),
        "full_due": full["due"], "full_written": full["written"],
        "full_already_done": full.get("already_done", 0),
    }


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "cron", hour=3, minute=0,
        id="postmortem_loop", replace_existing=True,
    )
