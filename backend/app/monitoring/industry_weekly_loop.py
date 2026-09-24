"""FEAT-003 — weekly Industry Analysis report enqueue (Sunday 06:30 UTC).

One tick, in order:

1. resolve the week — ``period_key`` is the ISO week of the prior Friday
   and the as-of is that Friday's close (owner decision 3, read from
   settings, computed in ``industry_report_worker.period_for``);
2. **warm the prices** — ``industry_analytics.warm_up_prices`` pre-fetches
   the missing series within ``INDUSTRY_PRICE_WARMUP_BUDGET`` provider
   calls, lowest-coverage groups first, BEFORE anything is enqueued. Doing
   it here rather than inside the jobs is what makes the first week's
   coverage honest instead of a page of ``no_prices``, and it keeps the
   provider spend inside one rate-budgeted pass instead of scattering it
   across N report jobs;
3. **enqueue the period** — one ``group_report`` job per active group plus
   the period's ``cross_snapshot`` job, coalescing against active jobs and
   skipping groups already generated for this week unless ``force``.

Nothing is computed here. The loop is the scheduler's hand on a durable
queue; the drainer thread on the worker does the work, so a tick that
overruns its window cannot hold the scheduler or lose jobs.

``record_run`` carries the counts — warm-up fetches, groups, enqueued,
coalesced, skipped — because ``success=True enqueued=0`` with no other
numbers is exactly the shape of report that has hidden an outage here
before. ``success=False`` whenever the week could not be enqueued in
full, including when there is no active taxonomy or no classified
constituent to report on.

The note also carries the PREVIOUS week's outcome (``prev_agentic``,
``prev_template``, ``prev_failed``, ``prev_not_updated_rate``,
``prev_healthy`` — ``industry_report_worker.period_outcome``). During the
week the drainer records that outcome as this loop's progress; this tick's
``record_run`` clears the progress, so the note is where last week's
verdict survives. ``success`` stays about the enqueue.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from ..agents.log_safety import safe_exc
from ..config import settings
from . import record_run

log = logging.getLogger(__name__)

LOOP_NAME = "industry_weekly_loop"


def _deps():
    """Import the report stack lazily, inside the tick.

    `monitoring/__init__.py` imports every loop module eagerly so
    `KNOWN_LOOPS` and `register_all` stay in step — which means anything
    that merely wants `record_run` imports this module too, including the
    web service. Pulling the analytics, the queue, the writer and the
    validator in at module scope would therefore load the whole report
    stack into every process on the strength of one monitoring import.
    Behind the flag this loop is a no-op on web, so it must cost nothing
    there. Python caches the modules, so the import is paid once, by the
    process that actually runs a tick.
    """
    from ..services import gics_registry, industry_analytics, industry_classification
    from ..services import industry_report_worker as jobs
    return gics_registry, industry_analytics, industry_classification, jobs


def _note(parts: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in parts.items())


def run_once(
    *, force: bool = False, codes: list[str] | None = None,
    now: datetime | None = None, warm_up: bool = True,
) -> dict[str, Any]:
    """Warm the prices, then enqueue the week. Returns the summary the
    note is built from; never raises past a recorded ``record_run``."""
    if not settings.enable_industry_reports:
        # The web service and any deployment that has not flipped the flag.
        # Recorded as a run so cron-health shows a live, deliberately
        # disabled loop rather than one that looks dead.
        note = "disabled (ENABLE_INDUSTRY_REPORTS=false)"
        record_run(LOOP_NAME, success=True, note=note)
        return {"enabled": False, "enqueued": 0, "note": note}

    gics_registry, industry_analytics, industry_classification, jobs = _deps()
    try:
        info = gics_registry.ensure_taxonomy(activate=True)
        if info is None:
            raise gics_registry.TaxonomyNotImported("no taxonomy could be activated")
        period_key, as_of = jobs.period_for(now)

        # Membership comes from the daily classification loop (03:40 UTC,
        # three hours before this one). On a fresh database that has never
        # classified, every group would report `insufficient_sample` for a
        # whole week, so bootstrap once — and say so, rather than quietly
        # doing a second loop's work every Sunday.
        by_group = industry_classification.constituents_by_group(version=info)
        bootstrapped = False
        if not by_group:
            industry_classification.classify_all(version=info)
            by_group = industry_classification.constituents_by_group(version=info)
            bootstrapped = True

        warm: dict[str, Any] = {"budget": 0, "fetched": 0, "failed": [], "remaining_missing": None,
                                "reason": "skipped by caller"}
        if warm_up:
            warm = industry_analytics.warm_up_prices(
                budget=settings.industry_price_warmup_budget, version=info,
            )

        # Last week's verdict, read BEFORE this tick's `record_run` clears
        # the progress row that carried it: success here stays about the
        # enqueue (unchanged semantics), and the previous week's outcome
        # rides in the note so it outlives the week it described.
        prev_key = industry_analytics.period_key_for(as_of - timedelta(days=7))
        try:
            previous = jobs.period_outcome(prev_key, info)
        except Exception as exc:  # telemetry — never the reason a week is not enqueued
            previous = {"period_key": prev_key, "error": safe_exc(exc)}

        result = jobs.enqueue_period(period_key, codes, source="weekly_cron",
                                     force=force, version=info)
    except Exception as exc:
        record_run(LOOP_NAME, success=False, note=f"error={safe_exc(exc)}")
        raise

    groups_with_members = len(by_group)
    n_groups = result["n_groups"]
    accounted = result["enqueued"] + result["coalesced"] + result["skipped_published"]
    healthy = (
        accounted == n_groups
        and not result["over_budget"]
        and not result["unknown_codes"]
        and groups_with_members > 0
    )
    summary: dict[str, Any] = {
        "enabled": True,
        "period_key": period_key,
        "as_of": as_of.isoformat(),
        "taxonomy_version": info.version_key,
        "groups": n_groups,
        "groups_with_constituents": groups_with_members,
        "groups_without_constituents": max(0, n_groups - groups_with_members),
        "bootstrapped_classification": bootstrapped,
        "enqueued": result["enqueued"],
        "coalesced": result["coalesced"],
        "skipped_published": result["skipped_published"],
        "skipped_withheld": result.get("skipped_withheld", 0),
        "over_budget": result["over_budget"],
        "unknown_codes": result["unknown_codes"],
        "cross_snapshot_job": result["cross_snapshot"].get("job_id"),
        "warm_up": warm,
    }
    note = _note({
        "period": period_key,
        "as_of": as_of.date().isoformat(),
        "taxonomy": info.version_key,
        "groups": n_groups,
        "with_constituents": groups_with_members,
        "enqueued": result["enqueued"],
        "coalesced": result["coalesced"],
        "skipped_published": result["skipped_published"],
        # Of those, weeks generated only as an audit-only template — not
        # on the site, and retried only by an admin `force`.
        "skipped_withheld": result.get("skipped_withheld", 0),
        "over_budget": result["over_budget"],
        "cross_snapshot": result["cross_snapshot"].get("job_id") or "none",
        "warm_fetched": warm.get("fetched", 0),
        "warm_failed": len(warm.get("failed") or []),
        "warm_budget": warm.get("budget", 0),
        "warm_remaining_missing": warm.get("remaining_missing"),
        "bootstrapped_classification": int(bootstrapped),
    })
    summary["previous_period"] = previous
    note = f"{note} " + (
        f"prev_period={previous['period_key']} prev_error={previous['error']}" if "error" in previous
        else jobs.period_outcome_note(previous, prefix="prev_")
    )
    summary["note"] = note
    record_run(LOOP_NAME, success=healthy, note=note)
    log.info("industry weekly loop: %s", note)
    return summary


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "cron",
        day_of_week=settings.industry_reports_cron_dow,
        hour=settings.industry_reports_cron_hour,
        minute=settings.industry_reports_cron_minute,
        id=LOOP_NAME, replace_existing=True, max_instances=1, coalesce=True,
    )
