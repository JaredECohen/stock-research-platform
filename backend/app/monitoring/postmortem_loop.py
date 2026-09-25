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

W7: after both horizons, and only when `LEARNING_LEDGER_WRITES` is on, the
same job runs the learning ledger's nightly housekeeping and the capped
cheap-route judge (`_learning_pass`). No new loop: `KNOWN_LOOPS` is pinned.

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
        f"deferred={report.get('deferred', 0)} "
        f"ineligible={report.get('ineligible', 0)} "
        f"memory_written={report.get('memory_written', 0)} "
        f"memory_disabled={report.get('memory_disabled', 0)} "
        f"memory_failed={report.get('memory_failed', 0)} "
        f"memory_not_requested={report.get('memory_not_requested', 0)}"
    )
    summary += (
        f" learning_written={report.get('learning_written', 0)} "
        f"learning_skipped={report.get('learning_skipped', 0)} "
        f"learning_rejected={report.get('learning_rejected', 0)} "
        f"learning_failed={report.get('learning_failed', 0)}"
    )
    if report.get("classification_error"):
        summary += f"; classification_error={report['classification_error']}"
    for label in ("deduped", "deferred", "skipped", "memory_disabled", "memory_failed", "memory_not_requested",
                  "learning_failed"):
        memos = report.get(f"{label}_memos", [])
        if memos:
            summary += f"; {label} memos: " + ", ".join(
                f"{m['ticker']}#{m['memo_snapshot_id']} ({m['reason']})"
                for m in memos
            )
    return summary


def _learning_pass() -> tuple[str, bool]:
    """W7: the ledger's nightly housekeeping, then the capped judge.

    Housekeeping first (epoch, historical backfill, K1 integrity, expiry,
    capacity), so a contaminated lesson is retired before anything judges
    it. The judge is the only spend here: cheap route, at most
    `learning_judge_max_calls_per_night` calls and
    `learning_judge_max_usd_per_night` dollars. Each step is isolated: one
    raising does not skip the other, and either failing turns the loop red.
    Returns (note fragment, ok).
    """
    from ..config import settings
    from ..learning import ledger

    if not settings.learning_ledger_writes:
        return "learning off", True
    ok = True
    parts: list[str] = []
    try:
        n = ledger.nightly()
        parts.append(
            f"learning backfilled={n['backfilled']} backfill_failed={n['backfill_failed']} "
            f"retired={n['retired']} demoted={'yes' if n['demoted'] else 'no'} "
            f"expired={n['expired']} capacity={n['capacity']}"
        )
        if n["backfill_skipped"]:
            parts.append("backfill skipped: " + ", ".join(
                f"{reason}={count}" for reason, count in sorted(n["backfill_skipped"].items())
            ))
        ok = ok and n["backfill_failed"] == 0
    except Exception as exc:
        ok = False
        parts.append(f"learning nightly failed: {type(exc).__name__}: {exc}"[:500])
    try:
        j = ledger.judge_due(
            max_calls=settings.learning_judge_max_calls_per_night,
            max_usd=settings.learning_judge_max_usd_per_night,
        )
        parts.append(
            f"judge status={j['status']} due={j['due']} calls={j['calls']} usd={j['usd']:.4f} "
            f"evidence={j['evidence']} irrelevant={j['irrelevant']} unanswered={j['unanswered']} "
            f"deferred={j['deferred']} "
            f"failed={j['failed']} retired={j['retired']} unavailable={j['unavailable']}"
            + (f" stopped={j['stopped_reason']}" if j["stopped_reason"] else "")
        )
        if j["failed_memos"]:
            parts.append("judge failed memos: " + ", ".join(
                f"{m['ticker']}#{m['memo_snapshot_id']}" for m in j["failed_memos"]
            ))
        ok = ok and j["failed"] == 0
    except Exception as exc:
        ok = False
        parts.append(f"learning judge failed: {type(exc).__name__}: {exc}"[:500])
    return "; ".join(parts), ok


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
    learning_note, learning_ok = _learning_pass()
    note = f"30d {_summarize(early)}; 90d {_summarize(full)}; {learning_note}"
    # Failed postmortem or memory writes fail the loop. `already_done` and
    # `deduped` are answers, not failures — folding them into `skipped` is
    # what had this loop reporting success=False every single night for a
    # backlog that was entirely fine.
    # A failed eligibility sweep is red too: the pass selected only from the
    # ledger rows that already existed. So is a failed learning write, judge
    # call or nightly step: a loop that catches per-item failures and still
    # says success is how zero-output nights went unnoticed for weeks.
    success = (
        sum(report.get(key, 0) for report in (early, full)
            for key in ("skipped", "memory_failed", "learning_failed")) == 0
        and not any(report.get("classification_error") for report in (early, full))
        and learning_ok
    )
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
