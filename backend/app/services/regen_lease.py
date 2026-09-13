"""Per-attempt ownership, renewed while a memo runs and fenced at DB writes.

The ContextVar carries an immutable receipt, never replacement ownership read
from the database. Leases are DB state; the keeper thread is only their renewer.
"""
from __future__ import annotations

import contextvars
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select, update

from ..database import SessionLocal
from ..models import RegenJob

log = logging.getLogger(__name__)
LEASE_SECONDS = 120
RENEW_SECONDS = 20


class LeaseLost(BaseException):
    """Cancel stale work; ordinary agent/provider fallback must not swallow it."""


@dataclass(frozen=True)
class JobClaim:
    job_id: int
    owner_token: str


_CLAIM: contextvars.ContextVar[JobClaim | None] = contextvars.ContextVar("regen_claim", default=None)


def utcnow() -> datetime:
    return datetime.utcnow()


def current_claim() -> JobClaim | None:
    return _CLAIM.get()


def owned_predicate(claim: JobClaim):
    return (RegenJob.id == claim.job_id, RegenJob.status == "running",
            RegenJob.owner_token == claim.owner_token, RegenJob.lease_expires_at > utcnow())


def assert_claim(claim: JobClaim, *, db=None, lock: bool = False) -> RegenJob:
    """With lock=True, serialize through commit against recovery/other owners.

    A conditional no-op UPDATE also obtains a write lock on SQLite, whose
    SELECT FOR UPDATE is ignored. Recheck expiry after acquiring that lock.
    """
    own = db is None
    db = db or SessionLocal()
    try:
        if lock:
            changed = db.execute(update(RegenJob).where(*owned_predicate(claim)).values(
                owner_token=claim.owner_token).execution_options(synchronize_session=False)).rowcount
            if not changed:
                raise LeaseLost(f"Regeneration job {claim.job_id} ownership expired or changed")
        row = db.execute(select(RegenJob).where(*owned_predicate(claim))
                         .execution_options(populate_existing=True)).scalar_one_or_none()
        if row is None:
            raise LeaseLost(f"Regeneration job {claim.job_id} ownership expired or changed")
        if own:
            db.expunge(row)
        return row
    except Exception as exc:
        # If ownership cannot be established, stop before more work/spend.
        raise LeaseLost(f"Regeneration job {claim.job_id} ownership check failed: {type(exc).__name__}") from None
    finally:
        if own:
            db.close()


def assert_current(*, db=None, lock: bool = False, run_id: str | None = None) -> RegenJob | None:
    claim = current_claim()
    if claim is None:
        return None
    row = assert_claim(claim, db=db, lock=lock)
    if run_id is not None and row.run_id != run_id:
        raise LeaseLost(f"Regeneration job {claim.job_id} run identity differs")
    return row


def renew(claim: JobClaim) -> bool:
    with SessionLocal() as db:
        count = db.execute(update(RegenJob).where(*owned_predicate(claim)).values(
            lease_expires_at=utcnow() + timedelta(seconds=LEASE_SECONDS))).rowcount
        db.commit()
        return bool(count)


@contextmanager
def claim_context(claim: JobClaim):
    token = _CLAIM.set(claim)
    try:
        yield
    finally:
        _CLAIM.reset(token)


@contextmanager
def keep_alive(claim: JobClaim):
    """Renew only this active execution; shutdown does not surrender live work."""
    stop = threading.Event()

    def run():
        while not stop.wait(RENEW_SECONDS):
            try:
                if renew(claim):
                    continue
                log.warning("regen job %d lease renewal denied; old attempt cannot publish", claim.job_id)
                return
            except Exception as exc:
                log.warning("regen job %d lease renewal failed: %s", claim.job_id, type(exc).__name__)
                # A transient DB error does not surrender a still-valid lease.
                # Retry on the next tick; renew() cannot resurrect an expiry.

    thread = threading.Thread(target=run, name=f"regen-lease-{claim.job_id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)
