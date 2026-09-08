"""FEAT-002 — builds the curated public samples on the worker.

The marketing site renders `public_samples` rows; nothing on the request
path ever generates them. This loop is the only writer. It runs on the
worker (`ENABLE_MONITORING=true`), weekly, and on demand when an operator
calls `POST /api/admin/samples/rebuild` — which lands in a control row in
the same table, because the web and worker processes share nothing but
the database.

Scheduling: one APScheduler job (`sample_build_loop`, every
`POLL_MINUTES`) whose `tick` decides whether there is work: a pending
rebuild request, or the weekly build being due (no recorded run, or the
last recorded run older than `WEEKLY_DAYS`). A single job id keeps
`KNOWN_LOOPS` honest; a separate cron entry for Sunday would need a
second id the cron-health endpoint would then expect to see reporting.
`record_run` is written only when a build actually ran, so cron-health's
8-day weekly threshold measures the thing that matters — how old the
samples are — not how recently the poll fired.

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
from ..database import SessionLocal
from ..services import memory_probe, public_samples
from . import record_run, status_snapshot

log = logging.getLogger(__name__)

LOOP_NAME = "sample_build_loop"
POLL_MINUTES = 10
WEEKLY_DAYS = 7
MAX_TICKERS = 5


def run_once(tickers: list[str] | None = None, *, now: datetime | None = None) -> dict[str, Any]:
    """Build the samples for `tickers` (default: the whole allowlist),
    capped at `MAX_TICKERS`, and record the run.

    Never lets one ticker's failure stop the rest: a ticker whose build
    raised is counted in `failed` and the loop moves on. Success is
    reported when every ticker that was attempted came back — degraded
    kinds are normal (no DCF stored yet) and are surfaced in the note
    rather than failing the run. A run that could not build a single
    ticker is a failure so cron-health shows it.
    """
    now = now or datetime.utcnow()
    listed = public_samples.allowlist()
    wanted = [t.upper() for t in (tickers if tickers is not None else listed)]
    targets = [t for t in listed if t in set(wanted)][:MAX_TICKERS]
    skipped = [t for t in wanted if t not in listed]

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

    note = (
        f"built={built} degraded={len(degraded)} tickers_ok={','.join(ok) or '-'} "
        f"tickers_failed={','.join(failed) or '-'}"
    )
    if skipped:
        note += f" not_listed={','.join(skipped)}"
    success = not targets or bool(ok)
    record_run(LOOP_NAME, success=success, note=note)
    return {
        "tickers": targets, "built": built, "degraded": degraded,
        "ok": ok, "failed": failed, "not_listed": skipped, "success": success,
    }


def last_run_at() -> datetime | None:
    info = status_snapshot().get(LOOP_NAME) or {}
    raw = info.get("last_run_at") if isinstance(info, dict) else None
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


def weekly_due(*, now: datetime | None = None) -> bool:
    now = now or datetime.utcnow()
    last = last_run_at()
    return last is None or now - last >= timedelta(days=WEEKLY_DAYS)


def tick(*, now: datetime | None = None) -> dict[str, Any] | None:
    """The scheduled entry point: build if asked to or if due, else no-op.

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
