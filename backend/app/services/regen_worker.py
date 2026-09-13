"""Theme 5 — DB-backed background worker for memo regeneration.

Replaces the request-scoped daemon thread in `routes_stocks`. The old
design kept job state in process memory (`_REGEN_JOBS` / `_REGEN_FAILURES`),
so an OOM kill or deploy mid-regen erased all evidence that a job ever
existed — the frontend just spun forever. Here every regen request is a
`RegenJob` row, and a single long-lived worker thread drains the queue:

    POST /analyze  →  enqueue() inserts a `queued` row (coalescing
                      duplicates per ticker)
    worker thread  →  claims oldest queued job (queued→running, atomic
                      conditional UPDATE), runs `run_stock_memo`, marks
                      it succeeded/failed with full error telemetry
    GET /analyze/status → reads the job rows; per-step progress merges
                      the worker's coarse waypoints with the Wave 8A
                      `MemoRunCheckpoint` rows for the job's `run_id`

Crash recovery (the part the daemon thread fundamentally couldn't do):
on worker startup, any job still marked `running` means the previous
process died mid-run (SIGKILL, deploy). It is requeued ONCE with the
same `run_id` — the checkpoint store then skips already-completed steps,
so the retry resumes rather than restarts — and marked `failed` with
`error_type=WorkerRestart` if it orphans a second time, so a ticker
that reliably OOMs the process can't crash-loop the service. Stale
`queued` jobs older than `regen_queue_max_age_minutes` are expired at
startup instead of executed, so a backlog accumulated during downtime
doesn't burn LLM spend on requests nobody is waiting for.

Single-replica by design (same assumption as the rest of the app — see
the rate-limit / regen comments in render.yaml). The claim is still an
atomic conditional UPDATE, so a second replica would degrade safely
(jobs run once) rather than corrupt state.
"""
from __future__ import annotations

import logging
import sys
import threading
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, update

from ..agents.graph import run_stock_memo
from ..agents.llm import llm_call_context
from ..config import settings
from ..database import SessionLocal
from ..models import Company, MemoRunCheckpoint, RegenJob
from ..seed_universe import ensure_company_in_universe
from ..services.history_service import backfill_ticker

log = logging.getLogger(__name__)

# Serializes enqueue's check-then-insert so a double-clicked Analyze
# button can't insert two queued rows for the same ticker.
_ENQUEUE_LOCK = threading.Lock()

# Cap on the per-job waypoint trace, mirroring the old _REGEN_PROGRESS cap.
_MAX_PROGRESS_ENTRIES = 50

_ACTIVE_STATUSES = ("queued", "running")
_FINISHED_STATUSES = ("succeeded", "failed")


def _utcnow() -> datetime:
    return datetime.utcnow()


def _ensure_table(db) -> None:
    RegenJob.__table__.create(bind=db.get_bind(), checkfirst=True)


def _job_dict(job: RegenJob) -> dict[str, Any]:
    """Detached snapshot of a job row, safe to use after the session closes."""
    return {
        "id": job.id,
        "ticker": job.ticker,
        "scenario": job.scenario,
        "run_id": job.run_id,
        "status": job.status,
        "attempts": job.attempts,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "memo_version": job.memo_version,
        "error_type": job.error_type or "",
        "error_message": job.error_message or "",
        "traceback_tail": job.traceback_tail or "",
        "progress": list(job.progress or []),
        "requested_by_user_id": job.requested_by_user_id,
        "usage_event_id": job.usage_event_id,
    }


# ---------------------------------------------------------------------------
# Queue API (called from request handlers)
# ---------------------------------------------------------------------------

def enqueue(
    ticker: str, scenario: str = "soft_landing", *, source: str = "user",
    requested_by_user_id: int | None = None, usage_event_id: int | None = None,
) -> tuple[dict[str, Any], bool]:
    """Queue a regen for `ticker`. Returns `(job, created)`.

    Coalesces: when a queued/running job already exists for the ticker,
    that job is returned with `created=False` instead of inserting a
    duplicate — same frantic-double-click behavior the in-memory
    registry had, but now it also holds across process restarts AND
    across triggers: an EDGAR-driven regen lands on the user's queued
    job instead of running a second memo concurrently.

    `source` tags the job's initial waypoint so telemetry can tell
    user-requested regens from scheduler-driven ones (`filing_event`,
    `transcript_event`, `regime_shift`, `admin_rerun`).

    `requested_by_user_id` / `usage_event_id` (FEAT-002) ride on the row
    only when this call created it: the reservation belongs to the
    request that made it, and the worker commits or releases it by id
    when the job finishes. A coalesced caller gets the existing job back
    untouched and must release its own reservation — it is riding on
    someone else's run.
    """
    t = ticker.upper()
    with _ENQUEUE_LOCK, SessionLocal() as db:
        _ensure_table(db)
        existing = db.execute(
            select(RegenJob)
            .where(RegenJob.ticker == t, RegenJob.status.in_(_ACTIVE_STATUSES))
            .order_by(RegenJob.id.desc())
        ).scalars().first()
        if existing is not None:
            log.info("regen job %d coalesced duplicate request for %s (source=%s)",
                     existing.id, t, source)
            return _job_dict(existing), False
        job = RegenJob(
            ticker=t, scenario=scenario, run_id=str(uuid.uuid4()),
            status="queued", enqueued_at=_utcnow(),
            progress=[{
                "step": "enqueued", "at": _utcnow().isoformat(), "source": source,
            }],
            requested_by_user_id=requested_by_user_id,
            usage_event_id=usage_event_id,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        log.info("regen job %d enqueued for %s (scenario=%s, source=%s)",
                 job.id, t, scenario, source)
        return _job_dict(job), True


def ticker_status(ticker: str) -> dict[str, Any]:
    """Job-queue view of one ticker, shaped for `/analyze/status`.

    `last_failure` reproduces the old `_REGEN_FAILURES` payload (same
    keys) from the most recent *finished* job — present when that job
    failed, None when it succeeded, i.e. "cleared on next success".
    `progress` is the merged waypoint + checkpoint-step trace of the
    most relevant job (the active one if any, else the last finished).
    """
    t = ticker.upper()
    with SessionLocal() as db:
        _ensure_table(db)
        active = db.execute(
            select(RegenJob)
            .where(RegenJob.ticker == t, RegenJob.status.in_(_ACTIVE_STATUSES))
            .order_by(RegenJob.id.desc())
        ).scalars().first()
        last_finished = db.execute(
            select(RegenJob)
            .where(RegenJob.ticker == t, RegenJob.status.in_(_FINISHED_STATUSES))
            .order_by(RegenJob.id.desc())
        ).scalars().first()

        last_failure: dict[str, Any] | None = None
        if last_finished is not None and last_finished.status == "failed":
            duration = None
            if last_finished.started_at and last_finished.finished_at:
                duration = (
                    last_finished.finished_at - last_finished.started_at
                ).total_seconds()
            last_failure = {
                "ticker": t,
                "error_type": last_finished.error_type,
                "error_message": last_finished.error_message,
                "traceback_tail": last_finished.traceback_tail,
                "started_at": (
                    last_finished.started_at.isoformat()
                    if last_finished.started_at else
                    last_finished.enqueued_at.isoformat()
                ),
                "failed_at": (
                    last_finished.finished_at.isoformat()
                    if last_finished.finished_at else None
                ),
                "duration_seconds": duration,
            }

        current = active or last_finished
        progress = _merged_progress(db, current) if current is not None else []
        started_at = None
        if active is not None:
            started_at = (active.started_at or active.enqueued_at).isoformat()
        return {
            "in_progress": active is not None,
            "started_at": started_at,
            "job_id": current.id if current is not None else None,
            "job_status": current.status if current is not None else None,
            "last_failure": last_failure,
            "progress": progress,
        }


def _merged_progress(db, job: RegenJob) -> list[dict[str, str]]:
    """Worker waypoints + per-step checkpoint completions, time-ordered.

    The checkpoint rows are the real per-step telemetry (fundamentals,
    dcf, each specialist, critic) — written by the Wave 8A decorators as
    each step completes under the job's `run_id`. The worker waypoints
    fill in the edges those can't see (claimed, graph entered, persisted,
    exception).
    """
    entries: list[dict[str, str]] = [
        {"step": str(p.get("step", "")), "at": str(p.get("at", ""))}
        for p in (job.progress or [])
        if isinstance(p, dict)
    ]
    try:
        rows = db.execute(
            select(MemoRunCheckpoint.step_name, MemoRunCheckpoint.generated_at)
            .where(MemoRunCheckpoint.run_id == job.run_id)
        ).all()
        entries.extend(
            {"step": f"step_completed {name}", "at": ts.isoformat() if ts else ""}
            for name, ts in rows
        )
    except Exception:  # pragma: no cover — checkpoint table may not exist yet
        pass
    entries.sort(key=lambda e: e["at"])
    return entries


def recent_jobs(ticker: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Newest-first job rows for the admin telemetry endpoint."""
    with SessionLocal() as db:
        _ensure_table(db)
        q = select(RegenJob).order_by(RegenJob.id.desc()).limit(limit)
        if ticker:
            q = q.where(RegenJob.ticker == ticker.upper())
        return [_job_dict(j) for j in db.execute(q).scalars().all()]


# ---------------------------------------------------------------------------
# Worker internals
# ---------------------------------------------------------------------------

def _append_progress(job_id: int, step: str) -> None:
    """Append a waypoint to the job's progress trace. Never raises."""
    try:
        with SessionLocal() as db:
            job = db.get(RegenJob, job_id)
            if job is None:
                return
            steps = list(job.progress or [])
            steps.append({"step": step, "at": _utcnow().isoformat()})
            # Reassign (don't mutate) so the JSON column change is tracked.
            job.progress = steps[-_MAX_PROGRESS_ENTRIES:]
            db.commit()
    except Exception:  # pragma: no cover — telemetry must not break the job
        log.debug("progress append failed for job %d", job_id, exc_info=True)


def _finalize_charge(usage_event_id: int | None, *, commit: bool, job_id: int) -> None:
    """Commit (memo persisted) or release (nothing delivered) the
    `usage_events` reservation a job carries. Both are conditional
    UPDATEs on `status='reserved'`, so calling this twice — or after a
    crash-and-resume — is harmless. Never raises: the job outcome is
    already recorded, and a charge that could not be finalised is what
    `billing_loop` reconciles from the stale-reservation sweep."""
    if usage_event_id is None:
        return
    try:
        from ..auth import usage  # lazy: keeps the ORM-heavy auth package off the worker's import path
        with SessionLocal() as db:
            done = usage.commit(db, usage_event_id) if commit else usage.release(db, usage_event_id)
        log.info("regen job %d %s usage event %d (%s)", job_id,
                 "committed" if commit else "released", usage_event_id,
                 "applied" if done else "already final")
    except Exception as exc:
        log.warning("regen job %d could not %s usage event %d: %s", job_id,
                    "commit" if commit else "release", usage_event_id, type(exc).__name__)


def _introduce_ticker(job_id: int, ticker: str) -> None:
    """Lazy universe resolution, worker-side (FEAT-002).

    `POST /analyze` used to resolve an unknown ticker in the request —
    a live profile lookup plus a 5-year provider backfill — before any
    charge existed. That is provider spend, so with the login wall on
    the route enqueues first and this runs here, inside the job that
    was charged for it. Tickers already in `companies` (the whole
    curated universe, so every scheduler-driven job) cost one SELECT.
    A symbol the provider chain rejects raises, which fails the job and
    releases the charge — the same 404 the route used to produce, now
    visible in `/analyze/status.last_failure`.
    """
    with SessionLocal() as db:
        known = db.execute(select(Company.ticker).where(Company.ticker == ticker)).first()
    if known:
        return
    _append_progress(job_id, "introducing_ticker")
    profile = ensure_company_in_universe(ticker)
    if profile is None:
        raise ValueError(f"{ticker}: provider chain rejected this symbol.")
    # Heavy load (financials + filings + transcripts) so the agent graph
    # has data to work with. Best-effort, as in the route it replaces —
    # a capability that 403s still leaves a memo worth generating.
    try:
        backfill_ticker(ticker)
    except Exception as exc:
        log.warning("backfill for %s failed before regen job %d: %s", ticker, job_id, type(exc).__name__)
    _append_progress(job_id, "ticker_introduced")


def recover_orphans() -> dict[str, int]:
    """Startup pass over jobs the previous process left behind.

    - `running` rows mean the process died mid-regen (OOM kill bypasses
      every except-block — see render.yaml). First orphaning requeues
      the job with the same `run_id` so checkpointed steps are skipped
      on retry; the second marks it failed so a job that kills the
      process every time can't crash-loop the service.
    - `queued` rows older than the max queue age are expired rather
      than executed: nobody is watching a poll loop from hours ago, and
      silently running a backlog after downtime is pure LLM spend.

    A job that ends here as `failed` delivered nothing, so its research-
    run reservation (FEAT-002) is released; a requeued job keeps its
    reservation because the retry may still deliver.
    """
    requeued = failed = expired = 0
    released: list[tuple[int, int]] = []
    with SessionLocal() as db:
        _ensure_table(db)
        now = _utcnow()
        for job in db.execute(
            select(RegenJob).where(RegenJob.status == "running")
        ).scalars().all():
            if job.attempts < 2:
                job.status = "queued"
                job.enqueued_at = now
                job.started_at = None
                job.progress = list(job.progress or []) + [{
                    "step": "requeued_after_process_restart",
                    "at": now.isoformat(),
                }]
                requeued += 1
            else:
                job.status = "failed"
                job.finished_at = now
                job.error_type = "WorkerRestart"
                job.error_message = (
                    "Process died mid-regeneration twice (likely OOM kill or "
                    "deploy). Not retrying automatically — check memory "
                    "headroom and re-trigger manually."
                )
                failed += 1
                if job.usage_event_id is not None:
                    released.append((job.id, job.usage_event_id))
        cutoff = now - timedelta(minutes=settings.regen_queue_max_age_minutes)
        for job in db.execute(
            select(RegenJob).where(
                RegenJob.status == "queued", RegenJob.enqueued_at < cutoff,
            )
        ).scalars().all():
            job.status = "failed"
            job.finished_at = now
            job.error_type = "QueueExpired"
            job.error_message = (
                f"Queued for over {settings.regen_queue_max_age_minutes} "
                "minutes without a worker claiming it; expired at startup "
                "instead of running stale."
            )
            expired += 1
            if job.usage_event_id is not None:
                released.append((job.id, job.usage_event_id))
        db.commit()
    for job_id, event_id in released:
        _finalize_charge(event_id, commit=False, job_id=job_id)
    if requeued or failed or expired:
        log.warning(
            "regen recovery: %d requeued, %d failed (repeat orphan), %d expired",
            requeued, failed, expired,
        )
    return {"requeued": requeued, "failed": failed, "expired": expired}


def claim_next_job() -> int | None:
    """Atomically move the oldest queued job to `running`. Returns its id."""
    with SessionLocal() as db:
        _ensure_table(db)
        row = db.execute(
            select(RegenJob.id, RegenJob.attempts)
            .where(RegenJob.status == "queued")
            .order_by(RegenJob.id)
        ).first()
        if row is None:
            return None
        job_id, attempts = row
        res = db.execute(
            update(RegenJob)
            .where(RegenJob.id == job_id, RegenJob.status == "queued")
            .values(status="running", started_at=_utcnow(), attempts=attempts + 1)
        )
        db.commit()
        return job_id if res.rowcount else None


def execute_job(job_id: int) -> dict[str, Any]:
    """Run one claimed job to completion and record the outcome.

    Catches BaseException (not just Exception) so asyncio cancellation
    and friends land in the failure record rather than vanishing;
    SystemExit / KeyboardInterrupt re-raise after a waypoint so process
    shutdown isn't swallowed.
    """
    with SessionLocal() as db:
        job = db.get(RegenJob, job_id)
        if job is None:
            return {}
        ticker, scenario, run_id = job.ticker, job.scenario, job.run_id
        user_id, usage_event_id = job.requested_by_user_id, job.usage_event_id
    started = _utcnow()
    _append_progress(job_id, "worker_claimed")
    from . import memory_probe
    memory_probe.log_rss("regen_job_start", job=job_id, ticker=ticker)
    log.info("regen job %d STARTING for %s (scenario=%s, run_id=%s)",
             job_id, ticker, scenario, run_id)
    try:
        # A live worker with missing credentials otherwise persists a demo
        # memo and marks the paid/automatic research job successful.
        if settings.app_env.lower() == "production" and not settings.llm_enabled:
            raise RuntimeError("Production memo generation requires a configured LLM and live data")
        _introduce_ticker(job_id, ticker)
        _append_progress(job_id, "calling_run_stock_memo")
        # FEAT-002: every LLM call the run makes is attributed to the
        # customer who asked and to `research_run`, so per-plan margin can
        # be read from `llm_call_logs`; the graph layers its own agent
        # names and run_id on top of this.
        with llm_call_context(user_id=user_id, feature="research_run", run_id=run_id):
            memo = run_stock_memo(
                ticker, scenario=scenario, force_refresh=True, run_id=run_id,
            )
        _append_progress(
            job_id, f"run_stock_memo_returned rating={memo.rating_label}",
        )
        from . import memo_store
        snap = memo_store.latest_memo(ticker)
        with SessionLocal() as db:
            row = db.get(RegenJob, job_id)
            if row is not None:
                row.status = "succeeded"
                row.finished_at = _utcnow()
                row.memo_version = snap.version if snap else None
                row.error_type = ""
                db.commit()
        # The memo is persisted: the charge sticks. Committed after the
        # job row so a crash between the two leaves a `reserved` event on
        # a succeeded job — reconcilable — rather than a paid-for nothing.
        _finalize_charge(usage_event_id, commit=True, job_id=job_id)
        _append_progress(job_id, "job_succeeded")
        log.info("regen job %d SUCCEEDED for %s in %.1fs (version=%s)",
                 job_id, ticker, (_utcnow() - started).total_seconds(),
                 snap.version if snap else None)
    except (SystemExit, KeyboardInterrupt):
        _append_progress(job_id, "process_exit_signal")
        raise
    except BaseException as exc:
        tb = traceback.format_exc()
        _append_progress(
            job_id, f"exception_caught {type(exc).__name__}: {str(exc)[:200]}",
        )
        log.error(
            "regen job %d FAILED for %s after %.1fs: %s: %s\n%s",
            job_id, ticker, (_utcnow() - started).total_seconds(),
            type(exc).__name__, exc, tb,
        )
        with SessionLocal() as db:
            row = db.get(RegenJob, job_id)
            if row is not None:
                row.status = "failed"
                row.finished_at = _utcnow()
                row.error_type = type(exc).__name__
                row.error_message = str(exc)[:500]
                row.traceback_tail = tb[-1500:]
                db.commit()
        # Nothing was delivered: the customer gets the run back.
        _finalize_charge(usage_event_id, commit=False, job_id=job_id)
    # A memo run is the largest allocator in the process — 26+ LLM
    # round-trips, filing bodies, and (pre-2026-08-12) tens of MB of chunk
    # embeddings per specialist. CPython hands those objects back to its
    # own freelists but not to the OS, and Render kills on RSS, so the
    # pages have to be returned explicitly. Runs on both the success and
    # failure paths: a job that died partway through is exactly the case
    # where the most garbage is left behind.
    memory_probe.trim_memory(f"regen_job_{job_id}")
    with SessionLocal() as db:
        job = db.get(RegenJob, job_id)
        return _job_dict(job) if job is not None else {}


def process_next_job() -> dict[str, Any] | None:
    """Claim + execute one queued job synchronously. Returns the finished
    job dict, or None when the queue is empty. This is the worker loop's
    body, exposed directly so tests (incl. the nightly smoke test) can
    drain the queue deterministically without the polling thread."""
    job_id = claim_next_job()
    if job_id is None:
        return None
    return execute_job(job_id)


# ---------------------------------------------------------------------------
# Worker thread lifecycle
# ---------------------------------------------------------------------------

_worker_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _worker_loop() -> None:
    recover_orphans()
    log.info("regen worker started (poll=%.1fs)", settings.regen_worker_poll_seconds)
    while not _stop_event.is_set():
        try:
            if process_next_job() is None:
                _stop_event.wait(settings.regen_worker_poll_seconds)
        except (SystemExit, KeyboardInterrupt):  # pragma: no cover
            raise
        except BaseException:  # pragma: no cover — loop must survive anything
            log.exception("regen worker loop error; continuing")
            _stop_event.wait(settings.regen_worker_poll_seconds)


def start_worker(*, force: bool = False) -> bool:
    """Start the singleton worker thread. Returns True when running.

    No-ops (returning False) when `ENABLE_REGEN_WORKER=false`, or under
    pytest unless `force=True` — tests drive the queue deterministically
    via `process_next_job()` instead of racing a polling thread, and a
    background thread regenerating real memos during a test session is
    exactly the accidental-LLM-spend mode conftest works to prevent.
    """
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return True
    if not settings.enable_regen_worker:
        log.info("regen worker disabled via ENABLE_REGEN_WORKER")
        return False
    if "pytest" in sys.modules and not force:
        log.info("regen worker not started under pytest (use force=True)")
        return False
    _stop_event.clear()
    _worker_thread = threading.Thread(
        target=_worker_loop, name="memo-regen-worker", daemon=True,
    )
    _worker_thread.start()
    return True


def stop_worker(timeout: float = 5.0) -> None:
    """Signal the worker loop to exit. In-flight job finishes on its own
    (daemon thread dies with the process either way)."""
    global _worker_thread
    _stop_event.set()
    if _worker_thread is not None:
        _worker_thread.join(timeout=timeout)
        _worker_thread = None
