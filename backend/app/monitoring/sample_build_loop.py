"""FEAT-002 — builds the curated public samples on the worker.

The marketing site renders `public_samples` rows; nothing on the request
path ever generates them. This loop is the only writer. It runs on the
worker (`ENABLE_MONITORING=true`), weekly, and on demand when an operator
calls `POST /api/admin/samples/rebuild` — which lands in a control row in
the same table, because the web and worker processes share nothing but
the database.

Feature gate: the automatic (weekly) build only runs while
`AUTH_ENABLED` is on. With the flags off — render.yaml's default — the
accounts/marketing feature is dark, and "rollback = flip flags off;
tables are inert" has to include this loop: a full build is a
`build_comps` provider fan-out per ticker plus, when the worker has LLM
keys, a commentary call each. An explicit admin rebuild request is still
served either way: an operator asking for a build is not the feature
spending on its own, and the go-live order in the plan runs one right
after the flag flips. While dark, `tick` records an `idle` run so
cron-health shows a healthy loop rather than a dead one.

Scheduling: one APScheduler job (`sample_build_loop`, every
`POLL_MINUTES`) whose `tick` decides whether there is work: a pending
rebuild request, or the weekly build being due. A single job id keeps
`KNOWN_LOOPS` honest; a separate cron entry for Sunday would need a
second id the cron-health endpoint would then expect to see reporting.

The weekly cadence is anchored on the last build that covered the WHOLE
allowlist (`public_samples.last_full_build`, a control row), not on
`cron_loop_runs`: `record_run` is written for every run so cron-health
stays honest about the loop, but an operator's one-ticker rebuild on day
6 must not postpone the others' refresh to day 13, and a full build that
failed outright is retried after `RETRY_HOURS` instead of in a week.

Bounded: sequential over at most `MAX_TICKERS` tickers; each ticker's
kinds are built independently and a failing kind keeps its last good row
(`services/public_samples.build_for_ticker`). The commentary LLM call is
skipped when no LLM is configured. RSS is logged per ticker so the
worker's memory curve for this loop is visible in Render logs.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from ..agents.log_safety import safe_exc
from ..config import settings
from ..database import SessionLocal
from ..services import memory_probe, public_samples
from . import record_run

log = logging.getLogger(__name__)

LOOP_NAME = "sample_build_loop"
POLL_MINUTES = 10
WEEKLY_DAYS = 7
# A full build that could not build a single ticker (provider chain or DB
# down) is retried on this cadence rather than the weekly one. Bounded so
# a lasting outage costs a few cheap failures a day, not one every poll.
RETRY_HOURS = 6
MAX_TICKERS = 5

IDLE_NOTE = "idle: AUTH_ENABLED=false (automatic builds run only while the wall is on; admin rebuilds still served)"


def _targets(tickers: list[str] | None) -> tuple[list[str], list[str], list[str]]:
    """(targets, skipped, full_set): the listed tickers to build, in
    allowlist order and capped; the requested ones that are not listed;
    and the set a weekly run would build — so `full` is decidable."""
    listed = public_samples.allowlist()
    wanted = [t.upper() for t in (tickers if tickers is not None else listed)]
    targets = [t for t in listed if t in set(wanted)][:MAX_TICKERS]
    skipped = [t for t in wanted if t not in listed]
    return targets, skipped, listed[:MAX_TICKERS]


def run_once(tickers: list[str] | None = None, *, now: datetime | None = None) -> dict[str, Any]:
    """Build the samples for `tickers` (default: the whole allowlist),
    capped at `MAX_TICKERS`, and record the run.

    Never lets one ticker's failure stop the rest: a ticker whose build
    raised is counted in `failed` and the loop moves on. Success is
    reported when every ticker that was attempted came back — degraded
    kinds are normal (no DCF stored yet) and are surfaced in the note
    rather than failing the run. A run that could not build a single
    ticker is a failure so cron-health shows it.

    A run that covered the whole allowlist (`full`) moves the weekly
    anchor, successful or not — `weekly_due` turns a failed anchor into
    a `RETRY_HOURS` retry. A subset run leaves the anchor alone.
    """
    now = now or datetime.utcnow()
    targets, skipped, full_set = _targets(tickers)
    full = bool(targets) and targets == full_set

    built = 0
    degraded: list[str] = []
    ok: list[str] = []
    failed: list[str] = []
    for ticker in targets:
        try:
            result = public_samples.build_for_ticker(ticker, now=now)
        except Exception as exc:
            failed.append(ticker)
            degraded.append(f"{ticker}: {safe_exc(exc)}")
            log.warning("sample build failed for %s: %s", ticker, type(exc).__name__, exc_info=True)
            continue
        ok.append(ticker)
        built += len(result["built"])
        degraded.extend(f"{ticker} {n}" for n in result["degraded"])
        memory_probe.log_rss("sample_build", ticker=ticker)

    success = not targets or bool(ok)
    note = (
        f"built={built} degraded={len(degraded)} tickers_ok={','.join(ok) or '-'} "
        f"tickers_failed={','.join(failed) or '-'} scope={'full' if full else 'partial'}"
    )
    if skipped:
        note += f" not_listed={','.join(skipped)}"
    if full:
        try:
            with SessionLocal() as db:
                public_samples.record_full_build(db, now=now, success=success, ok=ok, failed=failed)
        except Exception as exc:  # the build happened; a lost anchor means an early rebuild, not a lost run
            log.warning("could not record the full-build anchor: %s", safe_exc(exc))
            note += " anchor=unrecorded"
    record_run(LOOP_NAME, success=success, note=note)
    return {
        "tickers": targets, "built": built, "degraded": degraded,
        "ok": ok, "failed": failed, "not_listed": skipped, "success": success,
        "full": full,
    }


def weekly_due(*, now: datetime | None = None) -> bool:
    """Whether the automatic full build should run now.

    Never built → due. Last full build succeeded → due `WEEKLY_DAYS`
    later. Last full build failed outright → due `RETRY_HOURS` later. An
    unparseable anchor counts as never built: rebuilding early is the
    cheap mistake, silently never rebuilding is the expensive one.
    """
    now = now or datetime.utcnow()
    with SessionLocal() as db:
        last = public_samples.last_full_build(db)
    if last is None:
        return True
    try:
        completed_at = datetime.fromisoformat(str(last.get("completed_at")))
    except (TypeError, ValueError):
        return True
    wait = timedelta(days=WEEKLY_DAYS) if last.get("success") else timedelta(hours=RETRY_HOURS)
    return now - completed_at >= wait


def tick(*, now: datetime | None = None) -> dict[str, Any] | None:
    """The scheduled entry point: serve a pending admin request; else, if
    the feature is on and the weekly build is due, build; else no-op.

    An exception inside the build is recorded against the loop before it
    propagates (APScheduler logs it; cron-health shows `success=false`),
    the same pattern `postmortem_loop` uses so a crash before `record_run`
    cannot hide.
    """
    now = now or datetime.utcnow()
    with SessionLocal() as db:
        request = public_samples.pending_request(db)
    try:
        if request is not None:
            result = run_once(list(request.get("tickers") or []), now=now)
            with SessionLocal() as db:
                public_samples.clear_request(db, requested_at=request.get("requested_at"))
            return result
        if not settings.auth_enabled:
            record_run(LOOP_NAME, success=True, note=IDLE_NOTE)
            return None
        if weekly_due(now=now):
            return run_once(now=now)
        return None
    except Exception as exc:
        record_run(LOOP_NAME, success=False, note=f"crashed: {safe_exc(exc)}")
        raise


def register(scheduler) -> None:
    scheduler.add_job(
        tick, "interval", minutes=POLL_MINUTES,
        id=LOOP_NAME, replace_existing=True,
    )
