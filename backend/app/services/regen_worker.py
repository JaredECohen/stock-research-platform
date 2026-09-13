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

Crash recovery checks expired execution leases, including during idle polling.
A rolling-deploy predecessor keeps renewing while it executes. Per-attempt
ownership fences progress, checkpoints, publication and completion. A snapshot's
exact version is recorded on the job in the publication transaction, allowing
recovery to finish a published job without repeating the graph. Unowned legacy
running rows are reported and deferred until old-process death is verified.

The worker remains single-replica because its other monitoring loops are not
multi-instance safe; regeneration claims tolerate rolling process overlap.
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
from . import regen_lease
from .regen_lease import JobClaim, LeaseLost

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
        "lease_expires_at": job.lease_expires_at.isoformat() if job.lease_expires_at else None,
        "ownership_tracked": job.owner_token is not None,
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
            claim = regen_lease.current_claim()
            if claim is None or claim.job_id != job_id:
                raise LeaseLost(f"No execution ownership for regeneration job {job_id}")
            job = regen_lease.assert_claim(claim, db=db, lock=True)
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


def recover_orphans(*, report_legacy: bool = True) -> dict[str, Any]:
    """Recover only expired owned attempts, including while the worker is idle.

    A live rolling-deploy predecessor retains its renewable lease. Legacy
    unowned rows require explicit process-death evidence and are deferred.
    An atomic publication receipt finishes recovery without repeating a memo.
    """
    requeued = failed = expired = published = 0
    charges: list[tuple[int, int, bool]] = []
    legacy = []
    with SessionLocal() as db:
        _ensure_table(db)
        now = _utcnow()
        candidates = list(db.execute(select(RegenJob).where(RegenJob.status == "running")).scalars())
        for job in candidates:
            if job.owner_token is None or job.lease_expires_at is None:
                legacy.append({"id": job.id, "ticker": job.ticker, "run_id": job.run_id,
                               "started_at": job.started_at.isoformat() if job.started_at else None})
                continue
            if job.lease_expires_at > now:
                continue
            # Compare the exact observed ownership and expiry; a renewal or
            # another recovery that won the race makes this attempt a no-op.
            changed = db.execute(update(RegenJob).where(
                RegenJob.id == job.id, RegenJob.status == "running",
                RegenJob.owner_token == job.owner_token,
                RegenJob.lease_expires_at == job.lease_expires_at,
                RegenJob.lease_expires_at <= now,
            ).values(owner_token=job.owner_token).execution_options(synchronize_session=False)).rowcount
            if not changed:
                continue
            db.refresh(job)
            if job.memo_version is not None:
                job.status = "succeeded"
                job.finished_at = now
                job.error_type = job.error_message = job.traceback_tail = ""
                job.progress = list(job.progress or []) + [{"step": "published_memo_recovered", "at": now.isoformat()}]
                published += 1
                if job.usage_event_id is not None:
                    charges.append((job.id, job.usage_event_id, True))
            elif job.attempts < 2:
                job.status = "queued"
                job.enqueued_at = now
                job.started_at = None
                job.progress = list(job.progress or []) + [{"step": "requeued_after_lease_expired", "at": now.isoformat()}]
                requeued += 1
            else:
                job.status = "failed"
                job.finished_at = now
                job.error_type = "WorkerRestart"
                job.error_message = "Execution lease expired twice without publication; automatic retries exhausted."
                failed += 1
                if job.usage_event_id is not None:
                    charges.append((job.id, job.usage_event_id, False))
            job.owner_token = None
            job.lease_expires_at = None
        cutoff = now - timedelta(minutes=settings.regen_queue_max_age_minutes)
        stale_queued = list(db.execute(select(RegenJob).where(
            RegenJob.status == "queued", RegenJob.enqueued_at < cutoff,
        )).scalars())
        for job in stale_queued:
            changed = db.execute(update(RegenJob).where(
                RegenJob.id == job.id, RegenJob.status == "queued", RegenJob.enqueued_at < cutoff,
            ).values(status="failed", finished_at=now, error_type="QueueExpired",
                error_message=f"Queued for over {settings.regen_queue_max_age_minutes} minutes; expired without execution.")
                .execution_options(synchronize_session=False)).rowcount
            if changed:
                expired += 1
                if job.usage_event_id is not None:
                    charges.append((job.id, job.usage_event_id, False))
        db.commit()
    for job_id, event_id, commit in charges:
        _finalize_charge(event_id, commit=commit, job_id=job_id)
    if requeued or failed or expired or published:
        log.warning("regen recovery: %d requeued, %d failed, %d expired, %d published completions recovered",
                    requeued, failed, expired, published)
    if legacy and report_legacy:
        log.warning("regen recovery deferred %d legacy unowned jobs; verify prior process death before intervention: %s", len(legacy), legacy)
    result: dict[str, Any] = {"requeued": requeued, "failed": failed, "expired": expired}
    if published:
        result["published_recovered"] = published
    if legacy:
        result["legacy_deferred"] = legacy
    return result


def claim_next_job() -> JobClaim | None:
    """Claim the oldest queued job; return this attempt's immutable receipt."""
    with SessionLocal() as db:
        _ensure_table(db)
        row = db.execute(select(RegenJob.id).where(RegenJob.status == "queued").order_by(RegenJob.id)).first()
        if row is None:
            return None
        claim = JobClaim(job_id=row[0], owner_token=str(uuid.uuid4()))
        now = _utcnow()
        changed = db.execute(update(RegenJob).where(
            RegenJob.id == claim.job_id, RegenJob.status == "queued",
        ).values(status="running", started_at=now, attempts=RegenJob.attempts + 1,
                 owner_token=claim.owner_token, lease_expires_at=now + timedelta(seconds=regen_lease.LEASE_SECONDS))).rowcount
        db.commit()
        return claim if changed else None


def _finish_claim(claim: JobClaim, error: BaseException | None = None, tb: str = "") -> dict[str, Any] | None:
    """Only the current owner can finish; a published memo remains success."""
    with SessionLocal() as db:
        try:
            job = regen_lease.assert_claim(claim, db=db, lock=True)
        except LeaseLost:
            return None
        delivered = job.memo_version is not None
        if error is None and not delivered:
            raise RuntimeError("Memo graph returned without an owned publication receipt")
        job.status = "succeeded" if delivered else "failed"
        job.finished_at = _utcnow()
        job.error_type = "" if delivered else type(error).__name__
        job.error_message = "" if delivered else str(error)[:500]
        job.traceback_tail = "" if delivered else tb[-1500:]
        job.progress = (list(job.progress or []) + [{
            "step": "job_succeeded" if delivered else f"exception_caught {type(error).__name__}: {str(error)[:200]}",
            "at": _utcnow().isoformat(),
        }])[-_MAX_PROGRESS_ENTRIES:]
        job.lease_expires_at = None
        db.commit()
        return _job_dict(job)


def execute_job(claim: JobClaim) -> dict[str, Any]:
    """Execute only the captured claim, never adopt a token from a fresh read."""
    if not isinstance(claim, JobClaim):
        raise TypeError("execute_job requires the immutable claim receipt")
    from . import memory_probe
    done = None
    with regen_lease.claim_context(claim), regen_lease.keep_alive(claim):
        try:
            job = regen_lease.assert_claim(claim)
            ticker, scenario, run_id = job.ticker, job.scenario, job.run_id
            started = _utcnow()
            _append_progress(claim.job_id, "worker_claimed")
            memory_probe.log_rss("regen_job_start", job=claim.job_id, ticker=ticker)
            log.info("regen job %d STARTING for %s (scenario=%s, run_id=%s)", claim.job_id, ticker, scenario, run_id)
            if settings.app_env.lower() == "production" and not settings.llm_enabled:
                raise RuntimeError("Production memo generation requires a configured LLM and live data")
            _introduce_ticker(claim.job_id, ticker)
            _append_progress(claim.job_id, "calling_run_stock_memo")
            with llm_call_context(user_id=job.requested_by_user_id, feature="research_run", run_id=run_id):
                memo = run_stock_memo(ticker, scenario=scenario, force_refresh=True, run_id=run_id)
            _append_progress(claim.job_id, f"run_stock_memo_returned rating={memo.rating_label}")
            done = _finish_claim(claim)
            if done:
                log.info("regen job %d SUCCEEDED for %s in %.1fs (version=%s)",
                         claim.job_id, ticker, (_utcnow() - started).total_seconds(), done["memo_version"])
        except LeaseLost as exc:
            log.warning("regen job %d stale attempt stopped: %s", claim.job_id, exc)
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            tb = traceback.format_exc()
            log.error("regen job %d execution failed: %s: %s\n%s", claim.job_id, type(exc).__name__, exc, tb)
            done = _finish_claim(claim, exc, tb)
    if done is not None:
        _finalize_charge(done["usage_event_id"], commit=done["status"] == "succeeded", job_id=claim.job_id)
    memory_probe.trim_memory(f"regen_job_{claim.job_id}")
    if done is not None:
        return done
    # A stale attempt's return must not look like the replacement's success.
    return {"id": claim.job_id, "status": "lease_lost", "attempt_cancelled": True}


def process_next_job() -> dict[str, Any] | None:
    """Claim + execute one queued job synchronously. Returns the finished
    job dict, or None when the queue is empty. This is the worker loop's
    body, exposed directly so tests (incl. the nightly smoke test) can
    drain the queue deterministically without the polling thread."""
    recover_orphans(report_legacy=False)
    claim = claim_next_job()
    if claim is None:
        return None
    return execute_job(claim)


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
