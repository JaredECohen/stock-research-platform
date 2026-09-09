"""Atomic, idempotent usage meters on Postgres and SQLite.

A charge is two rows in one transaction:

  1. a `usage_events` row keyed by the caller's `idempotency_key` — the
     unique constraint means a retried request finds the existing event
     and is NOT charged again;
  2. `UPDATE usage_counters SET used = used + q WHERE … AND used + q <= limit`
     — the row lock makes the check-and-increment atomic, so two requests
     racing for the last unit cannot both win, on either database and
     with no application-side lock.

If the UPDATE affects no row the transaction rolls back (the event row
goes with it) and the caller gets `allowed=False` with the current
numbers. `commit` / `release` are conditional UPDATEs on
`status='reserved'`, so they are idempotent too — the worker can call
them after a crash-and-resume without double counting.

All periods are UTC calendar months (`period_key` = `YYYY-MM`).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import case, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models.accounts import UsageCounter, UsageEvent
from .sanitize import safe_logger

log = safe_logger(__name__)

RESERVED = "reserved"
COMMITTED = "committed"
RELEASED = "released"


def period_key(now: datetime | None = None) -> str:
    now = now or datetime.utcnow()
    return f"{now.year:04d}-{now.month:02d}"


def period_bounds(key: str) -> tuple[datetime, datetime]:
    """(start, end) of the UTC month `key`; `end` is when the meter resets."""
    year, month = (int(p) for p in key.split("-", 1))
    start = datetime(year, month, 1)
    end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    return start, end


def resets_at(now: datetime | None = None) -> datetime:
    return period_bounds(period_key(now))[1]


@dataclass
class Reservation:
    allowed: bool
    event: UsageEvent | None
    used: int
    limit: int | None
    # True when the idempotency key matched an existing event (a retry).
    replayed: bool = False

    @property
    def remaining(self) -> int | None:
        if self.limit is None:
            return None
        return max(0, self.limit - self.used)


def _insert_counter_if_missing(db: Session, user_id: int, feature: str, key: str, now: datetime) -> None:
    """Create the (user, feature, period) counter row at 0, tolerating a
    concurrent creator. Dialect-aware so the common case is one statement."""
    dialect = db.get_bind().dialect.name
    values = dict(user_id=user_id, feature=feature, period_key=key, used=0, updated_at=now)
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        db.execute(pg_insert(UsageCounter).values(**values).on_conflict_do_nothing(
            index_elements=["user_id", "feature", "period_key"]))
        return
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        db.execute(sqlite_insert(UsageCounter).values(**values).on_conflict_do_nothing(
            index_elements=["user_id", "feature", "period_key"]))
        return
    # Other dialects: savepoint so a lost race does not poison the transaction.
    exists = db.execute(select(UsageCounter.id).where(
        UsageCounter.user_id == user_id, UsageCounter.feature == feature, UsageCounter.period_key == key,
    )).first()
    if exists:
        return
    try:
        with db.begin_nested():
            db.add(UsageCounter(**values))
            db.flush()
    except IntegrityError:
        pass


def used(db: Session, user_id: int, feature: str, key: str) -> int:
    row = db.execute(select(UsageCounter.used).where(
        UsageCounter.user_id == user_id, UsageCounter.feature == feature, UsageCounter.period_key == key,
    )).first()
    return int(row[0]) if row else 0


def find_event(db: Session, idempotency_key: str) -> UsageEvent | None:
    return db.execute(select(UsageEvent).where(UsageEvent.idempotency_key == idempotency_key)).scalar_one_or_none()


def reserve(
    db: Session,
    *,
    user_id: int,
    feature: str,
    limit: int | None,
    idempotency_key: str,
    resource_ref: str | None = None,
    plan_at_charge: str = "free",
    quantity: int = 1,
    run_id: str | None = None,
    now: datetime | None = None,
) -> Reservation:
    """Charge `quantity` against this month's allowance, or refuse.

    `limit=None` means unlimited: the counter still increments (Pro usage
    is still measured) but nothing can refuse. Commits the session on
    success; on refusal or a lost idempotency race it rolls back so the
    session is clean for the caller.
    """
    now = now or datetime.utcnow()
    key = period_key(now)
    quantity = max(1, int(quantity))

    existing = find_event(db, idempotency_key)
    if existing is not None and existing.status != RELEASED:
        return Reservation(
            allowed=True, event=existing,
            used=used(db, user_id, feature, key), limit=limit, replayed=True,
        )

    _insert_counter_if_missing(db, user_id, feature, key, now)
    stmt = update(UsageCounter).where(
        UsageCounter.user_id == user_id,
        UsageCounter.feature == feature,
        UsageCounter.period_key == key,
    )
    if limit is not None:
        stmt = stmt.where(UsageCounter.used + quantity <= int(limit))
    result = db.execute(stmt.values(used=UsageCounter.used + quantity, updated_at=now))
    if result.rowcount != 1:
        db.rollback()
        return Reservation(
            allowed=False, event=existing, used=used(db, user_id, feature, key), limit=limit,
        )

    if existing is not None:
        # A released event under this key is a charge that was given back
        # (the handler failed after reserving). Distinct-resource features
        # key on user:feature:month:resource, so refusing here would lock
        # that resource out for the whole month; re-arm the same row as a
        # fresh reservation instead. It is this call's reservation, not a
        # replay, so the caller may release it.
        result = db.execute(update(UsageEvent).where(
            UsageEvent.id == existing.id, UsageEvent.status == RELEASED,
        ).values(status=RESERVED, finalized_at=None, created_at=now,
                 quantity=quantity, run_id=run_id, plan_at_charge=plan_at_charge))
        if result.rowcount != 1:
            db.rollback()
            winner = find_event(db, idempotency_key)
            return Reservation(
                allowed=winner is not None and winner.status != RELEASED, event=winner,
                used=used(db, user_id, feature, key), limit=limit, replayed=True,
            )
        db.commit()
        db.refresh(existing)
        return Reservation(allowed=True, event=existing, used=used(db, user_id, feature, key), limit=limit)

    event = UsageEvent(
        user_id=user_id, feature=feature, period_key=key, quantity=quantity,
        idempotency_key=idempotency_key, status=RESERVED, resource_ref=resource_ref,
        run_id=run_id, plan_at_charge=plan_at_charge, created_at=now,
    )
    db.add(event)
    try:
        db.commit()
    except IntegrityError:
        # Lost a race on the same idempotency key. The rollback also
        # undoes our counter increment, so the winner's charge stands alone.
        db.rollback()
        winner = find_event(db, idempotency_key)
        return Reservation(
            allowed=winner is not None and winner.status != RELEASED, event=winner,
            used=used(db, user_id, feature, key), limit=limit, replayed=True,
        )
    db.refresh(event)
    return Reservation(allowed=True, event=event, used=used(db, user_id, feature, key), limit=limit)


def commit(db: Session, event_id: int | None, *, now: datetime | None = None) -> bool:
    """reserved → committed. True if this call did the transition."""
    if event_id is None:
        return False
    now = now or datetime.utcnow()
    result = db.execute(update(UsageEvent).where(
        UsageEvent.id == event_id, UsageEvent.status == RESERVED,
    ).values(status=COMMITTED, finalized_at=now))
    db.commit()
    return result.rowcount == 1


def release(db: Session, event_id: int | None, *, now: datetime | None = None) -> bool:
    """reserved → released, giving the units back. True if this call did it.

    Only a `reserved` event can be released — a committed charge is a
    charge; refunds are an operator decision (`admin_overrides`), not a
    code path.
    """
    if event_id is None:
        return False
    now = now or datetime.utcnow()
    event = db.get(UsageEvent, event_id)
    if event is None:
        return False
    result = db.execute(update(UsageEvent).where(
        UsageEvent.id == event_id, UsageEvent.status == RESERVED,
    ).values(status=RELEASED, finalized_at=now))
    if result.rowcount != 1:
        db.rollback()
        return False
    q = max(1, int(event.quantity or 1))
    db.execute(update(UsageCounter).where(
        UsageCounter.user_id == event.user_id,
        UsageCounter.feature == event.feature,
        UsageCounter.period_key == event.period_key,
    ).values(
        used=case((UsageCounter.used >= q, UsageCounter.used - q), else_=0),
        updated_at=now,
    ))
    db.commit()
    return True


def distinct_resources(db: Session, user_id: int, feature: str, key: str) -> set[str]:
    """Tickers already charged (reserved or committed) this period — the
    input to the Free memo allowance and the DCF/comps-follows-memo rule."""
    rows = db.execute(select(UsageEvent.resource_ref).where(
        UsageEvent.user_id == user_id,
        UsageEvent.feature == feature,
        UsageEvent.period_key == key,
        UsageEvent.status.in_((RESERVED, COMMITTED)),
        UsageEvent.resource_ref.is_not(None),
    ).distinct()).all()
    return {r[0] for r in rows if r[0]}


def history(db: Session, user_id: int, key: str, *, limit: int = 50) -> list[UsageEvent]:
    return list(db.execute(select(UsageEvent).where(
        UsageEvent.user_id == user_id, UsageEvent.period_key == key,
    ).order_by(UsageEvent.created_at.desc(), UsageEvent.id.desc()).limit(limit)).scalars().all())


def counters_for(db: Session, user_id: int, key: str) -> dict[str, int]:
    rows = db.execute(select(UsageCounter.feature, UsageCounter.used).where(
        UsageCounter.user_id == user_id, UsageCounter.period_key == key,
    )).all()
    return {r[0]: int(r[1]) for r in rows}


def reserved_events_older_than(db: Session, cutoff: datetime) -> list[UsageEvent]:
    """Stale reservations (a process died between reserve and commit); the
    billing loop reconciles these against their job rows."""
    return list(db.execute(select(UsageEvent).where(
        UsageEvent.status == RESERVED, UsageEvent.created_at < cutoff,
    ).order_by(UsageEvent.id)).scalars().all())
