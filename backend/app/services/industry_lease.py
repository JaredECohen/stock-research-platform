"""Per-attempt ownership, renewed while an industry job runs and fenced at DB writes.

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
from ..models import IndustryReportJob
from .regen_lease import LeaseLost  # shared cancellation; never normal fallback

log = logging.getLogger(__name__)
LEASE_SECONDS = 120
RENEW_SECONDS = 20




@dataclass(frozen=True)
class IndustryClaim:
    job_id: int
    owner_token: str


_CLAIM: contextvars.ContextVar[IndustryClaim | None] = contextvars.ContextVar("industry_claim", default=None)


def utcnow() -> datetime:
    return datetime.utcnow()


def current_claim() -> IndustryClaim | None:
    return _CLAIM.get()


def owned_predicate(claim: IndustryClaim):
    return (IndustryReportJob.id == claim.job_id, IndustryReportJob.status == "running",
            IndustryReportJob.owner_token == claim.owner_token, IndustryReportJob.lease_expires_at > utcnow())


def assert_claim(claim: IndustryClaim, *, db=None, lock: bool = False) -> IndustryReportJob:
    """With lock=True, serialize through commit against recovery/other owners.

    A conditional no-op UPDATE also obtains a write lock on SQLite, whose
    SELECT FOR UPDATE is ignored. Recheck expiry after acquiring that lock.
    """
    own = db is None
    db = db or SessionLocal()
    try:
        if lock:
            changed = db.execute(update(IndustryReportJob).where(*owned_predicate(claim)).values(
                owner_token=claim.owner_token).execution_options(synchronize_session=False)).rowcount
            if not changed:
                raise LeaseLost(f"Industry job {claim.job_id} ownership expired or changed")
        row = db.execute(select(IndustryReportJob).where(*owned_predicate(claim))
                         .execution_options(populate_existing=True)).scalar_one_or_none()
        if row is None:
            raise LeaseLost(f"Industry job {claim.job_id} ownership expired or changed")
        if own:
            db.expunge(row)
        return row
    except Exception as exc:
        # If ownership cannot be established, stop before more work/spend.
        raise LeaseLost(f"Industry job {claim.job_id} ownership check failed: {type(exc).__name__}") from None
    finally:
        if own:
            db.close()


def assert_current(*, db=None, lock: bool = False, run_id: str | None = None) -> IndustryReportJob | None:
    claim = current_claim()
    if claim is None:
        return None
    row = assert_claim(claim, db=db, lock=lock)
    if run_id is not None and row.run_id != run_id:
        raise LeaseLost(f"Industry job {claim.job_id} run identity differs")
    return row


def renew(claim: IndustryClaim) -> bool:
    with SessionLocal() as db:
        count = db.execute(update(IndustryReportJob).where(*owned_predicate(claim)).values(
            lease_expires_at=utcnow() + timedelta(seconds=LEASE_SECONDS))).rowcount
        db.commit()
        return bool(count)


@contextmanager
def claim_context(claim: IndustryClaim):
    token = _CLAIM.set(claim)
    try:
        yield
    finally:
        _CLAIM.reset(token)


@contextmanager
def keep_alive(claim: IndustryClaim):
    """Renew only this active execution; shutdown does not surrender live work."""
    stop = threading.Event()

    def run():
        while not stop.wait(RENEW_SECONDS):
            try:
                if renew(claim):
                    continue
                log.warning("industry job %d lease renewal denied; old attempt cannot publish", claim.job_id)
                return
            except Exception as exc:
                log.warning("industry job %d lease renewal failed: %s", claim.job_id, type(exc).__name__)
                # A transient DB error does not surrender a still-valid lease.
                # Retry on the next tick; renew() cannot resurrect an expiry.

    thread = threading.Thread(target=run, name=f"industry-lease-{claim.job_id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def assert_identity(job: IndustryReportJob | None, *, kind: str, taxonomy_version_id: int,
                    period_key: str, code: str | None = None) -> None:
    """An owned write must belong to the captured job's exact output identity."""
    if job is not None and (job.kind != kind or job.taxonomy_version_id != taxonomy_version_id
                            or job.period_key != period_key or job.industry_group_code != code):
        raise LeaseLost(f"Industry job {job.id} output identity differs")
