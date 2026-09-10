"""Durable run queue for the Fundamental Factor Scorecard (Phase 6).

`scorecard_runs` is both the queue and the lineage record, the same shape
`regen_worker` gave `regen_jobs`: the web process inserts a `queued` row
(admin refresh / evaluate / backfill), the worker's `scorecard_loop`
claims it with an atomic conditional UPDATE and executes it, and the row
ends `succeeded | skipped | failed` with counts and an error type. Nothing
here lives in a module dict — the two processes share only Postgres.

Differences from the memo queue, on purpose:

* **No requeue on orphaning.** A scorecard run is cheap and atomic (pure
  arithmetic over rows already in the database; the price context is a
  cached read), so a `running` row left behind by a dead process is
  marked `failed/WorkerRestart` and re-requested by hand or by the next
  scheduled tick. The two-strike rule exists for memo regens because they
  are expensive and checkpointed; here a retry would be a fresh run anyway.
* **Recovery runs at the start of every loop tick**, not in the worker's
  seed thread: the seed can take minutes against live providers, and a
  tick that observed a stale `running` row before recovery would coalesce
  a new request onto a corpse.
* **Coalescing identity is `(version_key, as_of, run_kind)`.** Two admin
  refreshes for the same day land on one row; a scheduled run and a manual
  run for the same day are distinct requests and both execute (the second
  ends `skipped` when its inputs hash matches — see `scorecard_service`).
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select, update

from ..agents.log_safety import safe_exc
from ..database import SessionLocal
from ..models import ScorecardRun

log = logging.getLogger(__name__)

# Serialises enqueue's check-then-insert, mirroring `regen_worker`.
_ENQUEUE_LOCK = threading.Lock()

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
ACTIVE_STATUSES: tuple[str, ...] = (STATUS_QUEUED, STATUS_RUNNING)
FINISHED_STATUSES: tuple[str, ...] = (STATUS_SUCCEEDED, STATUS_FAILED, STATUS_SKIPPED)

KIND_SCHEDULED = "scheduled"
KIND_MONTH_END = "month_end"
KIND_BACKFILL = "backfill"
KIND_MANUAL = "manual"
KIND_PIT_PREPARE = "pit_prepare"
KIND_EVALUATE = "evaluate"
SCORING_KINDS: tuple[str, ...] = (KIND_SCHEDULED, KIND_MONTH_END, KIND_BACKFILL, KIND_MANUAL)
RUN_KINDS: tuple[str, ...] = SCORING_KINDS + (KIND_PIT_PREPARE, KIND_EVALUATE)

# Error types written by recovery. `QueueExpired` is a queued row nobody
# claimed for `QUEUE_MAX_AGE`; `WorkerRestart` is a running row whose
# process died. Both are exact strings the admin surface can match on.
ERROR_WORKER_RESTART = "WorkerRestart"
ERROR_QUEUE_EXPIRED = "QueueExpired"
ERROR_UNKNOWN_KIND = "UnknownRunKind"

# The loop drains once a day (03:45 UTC), so a queued row can legitimately
# wait ~24h; a week without a claim means the worker is down and the
# request is stale enough that silently running it later would surprise
# whoever asked. A running row older than the grace period was left by a
# process that no longer exists — the loop is serial (APScheduler
# `max_instances=1`), so nothing of this process can be mid-run at tick
# start; the grace guards a second replica that must never exist.
QUEUE_MAX_AGE = timedelta(days=7)
RUNNING_GRACE = timedelta(minutes=30)


class QueueExpired(RuntimeError):
    """Raised by `execute_run` when asked to run a row recovery already
    expired; kept as an exception so callers that bypass the loop cannot
    revive a stale request by accident."""


def _utcnow() -> datetime:
    """Clock seam — tests monkeypatch this instead of freezing time."""
    return datetime.utcnow()


def _ensure_table(db) -> None:
    ScorecardRun.__table__.create(bind=db.get_bind(), checkfirst=True)


def run_dict(row: ScorecardRun) -> dict[str, Any]:
    """Detached snapshot of a run row, safe to use after the session closes."""
    return {
        "id": row.id,
        "run_id": row.run_id,
        "version_key": row.version_key,
        "as_of": row.as_of,
        "run_kind": row.run_kind,
        "status": row.status,
        "attempts": row.attempts or 0,
        "requested_by": row.requested_by or "",
        "universe_size": row.universe_size,
        "scored_count": row.scored_count,
        "inputs_hash": row.inputs_hash,
        "enqueued_at": row.enqueued_at,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
        "error_type": row.error_type or "",
        "error_message": row.error_message or "",
        "note": row.note or "",
        "params": dict(row.params or {}),
    }


# ---------------------------------------------------------------------------
# Queue API (web and worker)
# ---------------------------------------------------------------------------

def enqueue_run(
    *,
    version_key: str,
    as_of: date,
    kind: str,
    params: dict[str, Any] | None = None,
    requested_by: str = "",
) -> tuple[dict[str, Any], bool]:
    """Queue a run. Returns `(run, created)`.

    Coalesces on an active row with the same `(version_key, as_of, kind)`:
    a double-clicked admin button, or a scheduled tick that already
    enqueued today's run, gets the existing row back with `created=False`
    and its own `params` are dropped (the request that created the row
    owns it). Unknown kinds are refused at the boundary so a typo cannot
    sit in the queue until the worker fails it.
    """
    if kind not in RUN_KINDS:
        raise ValueError(f"unknown scorecard run kind {kind!r}")
    with _ENQUEUE_LOCK, SessionLocal() as db:
        _ensure_table(db)
        existing = db.execute(
            select(ScorecardRun)
            .where(
                ScorecardRun.version_key == version_key,
                ScorecardRun.as_of == as_of,
                ScorecardRun.run_kind == kind,
                ScorecardRun.status.in_(ACTIVE_STATUSES),
            )
            .order_by(ScorecardRun.id.desc())
        ).scalars().first()
        if existing is not None:
            log.info("scorecard run %d coalesced duplicate %s request for %s/%s (by=%s)",
                     existing.id, kind, version_key, as_of, requested_by or "-")
            return run_dict(existing), False
        row = ScorecardRun(
            run_id=str(uuid.uuid4()), version_key=version_key, as_of=as_of,
            run_kind=kind, status=STATUS_QUEUED, attempts=0,
            requested_by=(requested_by or "")[:32], enqueued_at=_utcnow(),
            params=dict(params or {}),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        log.info("scorecard run %d enqueued: %s %s/%s (by=%s)",
                 row.id, kind, version_key, as_of, requested_by or "-")
        return run_dict(row), True


def create_running_row(
    *,
    version_key: str,
    as_of: date,
    kind: str,
    params: dict[str, Any] | None = None,
    requested_by: str = "direct",
) -> int:
    """A run row that starts life `running` — for callers that execute
    synchronously without the queue (tests, scripts). Every score row
    must point at a run row, so a direct `run_scorecard` still leaves
    lineage behind."""
    if kind not in RUN_KINDS:
        raise ValueError(f"unknown scorecard run kind {kind!r}")
    now = _utcnow()
    with SessionLocal() as db:
        _ensure_table(db)
        row = ScorecardRun(
            run_id=str(uuid.uuid4()), version_key=version_key, as_of=as_of,
            run_kind=kind, status=STATUS_RUNNING, attempts=1,
            requested_by=(requested_by or "")[:32], enqueued_at=now, started_at=now,
            params=dict(params or {}),
        )
        db.add(row)
        db.commit()
        return int(row.id)


def get_run(row_id: int) -> dict[str, Any] | None:
    with SessionLocal() as db:
        _ensure_table(db)
        row = db.get(ScorecardRun, row_id)
        return run_dict(row) if row is not None else None


def recent_runs(
    *, limit: int = 50, status: str | None = None, kind: str | None = None,
    version_key: str | None = None,
) -> list[dict[str, Any]]:
    """Newest-first run rows."""
    with SessionLocal() as db:
        _ensure_table(db)
        q = select(ScorecardRun).order_by(ScorecardRun.id.desc()).limit(limit)
        if status:
            q = q.where(ScorecardRun.status == status)
        if kind:
            q = q.where(ScorecardRun.run_kind == kind)
        if version_key:
            q = q.where(ScorecardRun.version_key == version_key)
        return [run_dict(r) for r in db.execute(q).scalars().all()]


def latest_succeeded_run(
    version_key: str, *, as_of: date | None = None, kinds: tuple[str, ...] = SCORING_KINDS,
    db=None,
) -> dict[str, Any] | None:
    """The scoring run readers resolve to: the newest succeeded run for
    `version_key` with `as_of <= as_of` (or the latest when None). Re-runs
    on the same day with changed inputs each succeed; the highest id wins,
    which is the most recently computed cross-section."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        q = (
            select(ScorecardRun)
            .where(
                ScorecardRun.version_key == version_key,
                ScorecardRun.status == STATUS_SUCCEEDED,
                ScorecardRun.run_kind.in_(kinds),
            )
            .order_by(ScorecardRun.as_of.desc(), ScorecardRun.id.desc())
        )
        if as_of is not None:
            q = q.where(ScorecardRun.as_of <= as_of)
        row = db.execute(q.limit(1)).scalars().first()
        return run_dict(row) if row is not None else None
    finally:
        if own:
            db.close()


def succeeded_run_exists(version_key: str, as_of: date, *, kinds: tuple[str, ...] = SCORING_KINDS) -> bool:
    with SessionLocal() as db:
        _ensure_table(db)
        row = db.execute(
            select(ScorecardRun.id).where(
                ScorecardRun.version_key == version_key,
                ScorecardRun.as_of == as_of,
                ScorecardRun.status == STATUS_SUCCEEDED,
                ScorecardRun.run_kind.in_(kinds),
            ).limit(1)
        ).first()
        return row is not None


def unfailed_run_exists(version_key: str, as_of: date, kind: str) -> bool:
    """Any row for the identity that is not `failed` — queued, running,
    succeeded or skipped. The loop's "enqueue once per month" rule reads
    this so a failed evaluation is retried on the next tick while a
    finished or pending one is left alone."""
    with SessionLocal() as db:
        _ensure_table(db)
        row = db.execute(
            select(ScorecardRun.id).where(
                ScorecardRun.version_key == version_key,
                ScorecardRun.as_of == as_of,
                ScorecardRun.run_kind == kind,
                ScorecardRun.status != STATUS_FAILED,
            ).limit(1)
        ).first()
        return row is not None


# ---------------------------------------------------------------------------
# Worker-side state transitions
# ---------------------------------------------------------------------------

def recover_orphans() -> dict[str, int]:
    """Start-of-tick pass over rows a previous process left behind.

    - `running` rows started before `RUNNING_GRACE` ago → `failed /
      WorkerRestart`. Never requeued (see the module docstring).
    - `queued` rows older than `QUEUE_MAX_AGE` → `failed / QueueExpired`.
    """
    failed = expired = 0
    now = _utcnow()
    with SessionLocal() as db:
        _ensure_table(db)
        for row in db.execute(
            select(ScorecardRun).where(ScorecardRun.status == STATUS_RUNNING)
        ).scalars().all():
            started = row.started_at or row.enqueued_at
            if started is not None and now - started < RUNNING_GRACE:
                continue
            row.status = STATUS_FAILED
            row.finished_at = now
            row.error_type = ERROR_WORKER_RESTART
            row.error_message = (
                "Process died mid-run (likely OOM kill or deploy). Scorecard runs are "
                "not retried automatically — re-request it, or wait for the next tick."
            )
            failed += 1
        cutoff = now - QUEUE_MAX_AGE
        for row in db.execute(
            select(ScorecardRun).where(
                ScorecardRun.status == STATUS_QUEUED, ScorecardRun.enqueued_at < cutoff,
            )
        ).scalars().all():
            row.status = STATUS_FAILED
            row.finished_at = now
            row.error_type = ERROR_QUEUE_EXPIRED
            row.error_message = (
                f"Queued for over {QUEUE_MAX_AGE.days} days without a worker claiming it; "
                "expired instead of running stale."
            )
            expired += 1
        db.commit()
    if failed or expired:
        log.warning("scorecard recovery: %d failed (orphaned running), %d expired", failed, expired)
    return {"failed": failed, "expired": expired}


def claim_next_run() -> int | None:
    """Atomically move the oldest queued run to `running`. Returns its id."""
    with SessionLocal() as db:
        _ensure_table(db)
        row = db.execute(
            select(ScorecardRun.id, ScorecardRun.attempts)
            .where(ScorecardRun.status == STATUS_QUEUED)
            .order_by(ScorecardRun.id)
        ).first()
        if row is None:
            return None
        row_id, attempts = row
        res = db.execute(
            update(ScorecardRun)
            .where(ScorecardRun.id == row_id, ScorecardRun.status == STATUS_QUEUED)
            .values(status=STATUS_RUNNING, started_at=_utcnow(), attempts=(attempts or 0) + 1)
        )
        db.commit()
        return int(row_id) if res.rowcount else None


def finish_run(
    row_id: int,
    *,
    status: str,
    note: str = "",
    universe_size: int | None = None,
    scored_count: int | None = None,
    inputs_hash: str | None = None,
    error_type: str = "",
    error_message: str = "",
    params_update: dict[str, Any] | None = None,
    db=None,
) -> dict[str, Any] | None:
    """Record the outcome of a claimed run. `params_update` is merged into
    the row's JSON (reassigned, not mutated, so the change is tracked).

    With `db` given the update joins the caller's transaction and is NOT
    committed here — that is how a scoring run makes "score rows written"
    and "run succeeded" one atomic fact, so a crash between the two can
    never leave a succeeded run with missing rows or orphan rows under a
    running one.
    """
    if status not in FINISHED_STATUSES:
        raise ValueError(f"finish_run: {status!r} is not a finished status")
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        row = db.get(ScorecardRun, row_id)
        if row is None:
            return None
        row.status = status
        row.finished_at = _utcnow()
        row.note = (note or "")[:4000]
        if universe_size is not None:
            row.universe_size = universe_size
        if scored_count is not None:
            row.scored_count = scored_count
        if inputs_hash is not None:
            row.inputs_hash = inputs_hash
        row.error_type = (error_type or "")[:64]
        row.error_message = (error_message or "")[:500]
        if params_update:
            merged = dict(row.params or {})
            merged.update(params_update)
            row.params = merged
        if own:
            db.commit()
        else:
            db.flush()
        return run_dict(row)
    finally:
        if own:
            db.close()


def mark_failed(row_id: int, exc: BaseException, *, note: str = "") -> dict[str, Any] | None:
    """Failure with the exception type as `error_type` and a redacted
    message — provider exceptions can quote request URLs with keys."""
    return finish_run(
        row_id, status=STATUS_FAILED, note=note,
        error_type=type(exc).__name__, error_message=safe_exc(exc),
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_run(row_id: int) -> dict[str, Any]:
    """Run one claimed row to completion by kind and record the outcome.

    The job bodies (`scorecard_service.run_scorecard`, `pit_prepare`,
    `scorecard_evaluation.run_evaluation`) finish their own row with the
    counts they know; this wrapper is the last resort that turns an escape
    into `failed/<ExceptionType>` so a row can never stay `running` with a
    live process. SystemExit / KeyboardInterrupt re-raise so shutdown is
    not swallowed. Imports are lazy to keep the queue free of the service
    (which imports this module).
    """
    run = get_run(row_id)
    if run is None:
        return {}
    if run["status"] != STATUS_RUNNING:
        # Recovery expired it between claim and execute, or a caller passed
        # an unclaimed id; refusing is safer than reviving it.
        raise QueueExpired(f"scorecard run {row_id} is {run['status']!r}, not running")
    kind = run["run_kind"]
    started = _utcnow()
    log.info("scorecard run %d STARTING: %s %s/%s", row_id, kind, run["version_key"], run["as_of"])
    try:
        if kind in SCORING_KINDS:
            from . import scorecard_service
            scorecard_service.run_scorecard(
                run["version_key"], run["as_of"], run_kind=kind,
                tickers=run["params"].get("tickers"), run_row_id=row_id,
            )
        elif kind == KIND_PIT_PREPARE:
            from . import scorecard_service
            scorecard_service.pit_prepare(run_row_id=row_id, tickers=run["params"].get("tickers"))
        elif kind == KIND_EVALUATE:
            from . import scorecard_evaluation
            scorecard_evaluation.run_evaluation(run["version_key"], run_id=run["run_id"], run_row_id=row_id)
        else:  # pragma: no cover — enqueue refuses unknown kinds
            finish_run(row_id, status=STATUS_FAILED, error_type=ERROR_UNKNOWN_KIND,
                       error_message=f"unknown run kind {kind!r}")
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        log.error("scorecard run %d FAILED (%s) after %.1fs: %s", row_id, kind,
                  (_utcnow() - started).total_seconds(), safe_exc(exc))
        current = get_run(row_id)
        if current is not None and current["status"] == STATUS_RUNNING:
            mark_failed(row_id, exc)
    final = get_run(row_id) or {}
    if final.get("status") == STATUS_RUNNING:
        # A job body returned without finishing its row: a bug, but the
        # row must not stay claimed forever.
        finish_run(row_id, status=STATUS_FAILED, error_type="RunNotFinalized",
                   error_message="job body returned without recording an outcome")
        final = get_run(row_id) or {}
    log.info("scorecard run %d %s in %.1fs: %s", row_id, final.get("status", "?").upper(),
             (_utcnow() - started).total_seconds(), (final.get("note") or "")[:200])
    return final


def drain(max_runs: int = 200) -> list[dict[str, Any]]:
    """Claim + execute queued runs in FIFO order, up to `max_runs`, so one
    tick stays bounded even after a large backfill request."""
    done: list[dict[str, Any]] = []
    while len(done) < max_runs:
        row_id = claim_next_run()
        if row_id is None:
            break
        done.append(execute_run(row_id))
    return done


__all__ = [
    "ACTIVE_STATUSES",
    "ERROR_QUEUE_EXPIRED",
    "ERROR_WORKER_RESTART",
    "FINISHED_STATUSES",
    "KIND_BACKFILL",
    "KIND_EVALUATE",
    "KIND_MANUAL",
    "KIND_MONTH_END",
    "KIND_PIT_PREPARE",
    "KIND_SCHEDULED",
    "QUEUE_MAX_AGE",
    "QueueExpired",
    "RUNNING_GRACE",
    "RUN_KINDS",
    "SCORING_KINDS",
    "STATUS_FAILED",
    "STATUS_QUEUED",
    "STATUS_RUNNING",
    "STATUS_SKIPPED",
    "STATUS_SUCCEEDED",
    "claim_next_run",
    "create_running_row",
    "drain",
    "enqueue_run",
    "execute_run",
    "finish_run",
    "get_run",
    "latest_succeeded_run",
    "mark_failed",
    "recent_runs",
    "recover_orphans",
    "run_dict",
    "succeeded_run_exists",
    "unfailed_run_exists",
]
