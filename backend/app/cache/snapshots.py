"""ResearchSnapshot ORM model + cache helpers.

A `ResearchSnapshot` is a versioned, lineage-aware piece of research output
(cohort warm cache, company cold cache, news hot cache, etc.). Each snapshot
records the sources it was derived from so we can detect when its inputs
change and invalidate it. Snapshots can declare `parent_snapshot_ids` so
invalidating a parent marks children stale.

This module also exposes a tiny `CacheCostLog` table for measuring how many
LLM tokens we save by re-using cache hits.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    delete,
    select,
    update,
)
from sqlalchemy.orm import Mapped, Session, mapped_column

from ..database import Base, SessionLocal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class ResearchSnapshot(Base):
    """A cached, lineage-aware piece of research output.

    `subject` identifies *what* the snapshot describes (typically a ticker, a
    sector key, or a composite key). `kind` discriminates *how* the subject
    is being described (e.g. "company_cold", "sector_warm", "news_hot").

    `sources_hash` is a deterministic hash of the inputs (filings, transcripts,
    cohort members) used to produce the payload. When a recompute would yield
    a different hash, the cached entry is considered stale.
    """

    __tablename__ = "research_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject: Mapped[str] = mapped_column(String(128), index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    sources_hash: Mapped[str] = mapped_column(String(64), default="")
    sources_used: Mapped[list[str]] = mapped_column(JSON, default=list)
    generated_by: Mapped[str] = mapped_column(String(128), default="")
    cost_tokens: Mapped[int] = mapped_column(Integer, default=0)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    parent_snapshot_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    stale: Mapped[bool] = mapped_column(Boolean, default=False)


Index("ix_snapshot_subject_kind", ResearchSnapshot.subject, ResearchSnapshot.kind)


class CacheCostLog(Base):
    """Append-only ledger of generation costs (and savings).

    Each `cache_put` writes a row with `cost_tokens` for the new computation.
    Hits are also logged with `cost_tokens=0` and `kind` suffixed `:hit` so we
    can compute "tokens saved" = `cost(miss for same key) - 0`.
    """

    __tablename__ = "cache_cost_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject: Mapped[str] = mapped_column(String(128), index=True)
    kind: Mapped[str] = mapped_column(String(96), index=True)
    cost_tokens: Mapped[int] = mapped_column(Integer, default=0)
    note: Mapped[str] = mapped_column(Text, default="")
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sources_fingerprint(sources_used: Iterable[Any]) -> str:
    """Deterministic hash of the source identifiers a snapshot was built on.

    Stable across process restarts. Order-independent (sorted before hashing)
    so re-shuffling parent lookups doesn't invalidate caches.
    """
    items: list[str] = []
    for s in sources_used or []:
        if s is None:
            continue
        if isinstance(s, (dict, list, tuple)):
            try:
                items.append(json.dumps(s, sort_keys=True, default=str))
            except Exception:
                items.append(str(s))
        else:
            items.append(str(s))
    items.sort()
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:32]


def _ensure_table(db: Session) -> None:
    """Create the cache tables if missing (handles in-memory test DBs)."""
    bind = db.get_bind()
    ResearchSnapshot.__table__.create(bind=bind, checkfirst=True)
    CacheCostLog.__table__.create(bind=bind, checkfirst=True)


def _now() -> datetime:
    return datetime.utcnow()


def _as_of_subject(subject: str) -> str:
    """Wave 1C: segregate cache keys by the active as_of date so backtest
    runs don't collide with live data. Lazy import dodges the circular
    `cache → data_service → cache` chain."""
    try:
        from ..services.data_service import current_as_of_date
    except Exception:
        return subject
    as_of = current_as_of_date()
    if as_of is None:
        return subject
    return f"{subject}:asof:{as_of.isoformat()}"


def _is_expired(snap: ResearchSnapshot, max_age_seconds: int | None) -> bool:
    if snap.expires_at and _now() >= snap.expires_at:
        return True
    if max_age_seconds is not None:
        if (_now() - snap.generated_at).total_seconds() > max_age_seconds:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cache_get(
    subject: str,
    kind: str,
    *,
    max_age_seconds: int | None = None,
    db: Session | None = None,
) -> ResearchSnapshot | None:
    """Look up the freshest non-stale, non-invalidated snapshot for the key.

    Returns None if no usable snapshot exists. Callers should treat the return
    as read-only; mutations should go through `cache_put` so we keep history.

    Wave 1C: when an as_of date is active in the calling context the
    `subject` is automatically suffixed `:asof:<YYYY-MM-DD>` so backtest
    cache entries never collide with live data.
    """
    subject = _as_of_subject(subject)
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = (
            select(ResearchSnapshot)
            .where(
                ResearchSnapshot.subject == subject,
                ResearchSnapshot.kind == kind,
                ResearchSnapshot.stale.is_(False),
                ResearchSnapshot.invalidated_at.is_(None),
            )
            .order_by(ResearchSnapshot.generated_at.desc())
            .limit(1)
        )
        row = db.execute(stmt).scalars().first()
        if row is None:
            return None
        if _is_expired(row, max_age_seconds):
            return None
        # Log a hit for cost-saving telemetry
        try:
            db.add(CacheCostLog(
                subject=subject, kind=f"{kind}:hit",
                cost_tokens=0, note=f"hit snapshot id={row.id}",
            ))
            db.commit()
        except Exception:  # pragma: no cover
            db.rollback()
        # Detach so callers can use the object after the session closes; the
        # whole row is already loaded into memory because we just SELECTed it.
        db.refresh(row)
        db.expunge(row)
        # Wave 6D: read-time schema upgrade. Stored payloads at older
        # `schema_version` are walked through the registered migration
        # chain so consumers always see the current shape. The DB row
        # itself is left at its stored version — the upgrade cost is
        # paid once per read and absorbed by the caller's own caching.
        try:
            from .migrations import upgrade_payload
            if isinstance(row.payload, dict):
                row.payload = upgrade_payload(row.kind, row.payload)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("schema upgrade on read failed (id=%s, kind=%s): %s",
                        row.id, row.kind, exc)
        return row
    finally:
        if own:
            db.close()


def cache_put(
    subject: str,
    kind: str,
    payload: dict[str, Any],
    sources_used: list[Any] | None = None,
    generated_by: str = "",
    cost_tokens: int = 0,
    parent_snapshots: list[int] | None = None,
    ttl_seconds: int | None = None,
    schema_version: int = 1,
    db: Session | None = None,
) -> ResearchSnapshot:
    """Store a new snapshot. Returns the persisted row with .id populated.

    Wave 1C: when an as_of date is active the `subject` is suffixed
    `:asof:<YYYY-MM-DD>` so backtest writes don't shadow live entries.
    """
    subject = _as_of_subject(subject)
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        # Make payload JSON-safe
        try:
            json.dumps(payload, default=str)
            safe_payload = payload
        except Exception:
            safe_payload = json.loads(json.dumps(payload, default=str))

        # Ensure schema_version round-trips through the payload too. This lets
        # readers handle older payloads even when the column is dropped.
        if isinstance(safe_payload, dict):
            safe_payload = {**safe_payload, "schema_version": schema_version}

        sources_list = list(sources_used or [])
        snap = ResearchSnapshot(
            subject=subject,
            kind=kind,
            schema_version=schema_version,
            payload=safe_payload,
            sources_hash=sources_fingerprint(sources_list),
            sources_used=sources_list,
            generated_by=generated_by or "",
            cost_tokens=int(cost_tokens or 0),
            generated_at=_now(),
            expires_at=(_now() + timedelta(seconds=ttl_seconds)) if ttl_seconds else None,
            invalidated_at=None,
            parent_snapshot_ids=list(parent_snapshots or []),
            stale=False,
        )
        db.add(snap)
        db.flush()  # populate id
        log_cost(
            subject, kind, cost_tokens,
            note=f"miss snapshot id={snap.id}", db=db,
        )
        db.commit()
        db.refresh(snap)
        db.expunge(snap)
        return snap
    finally:
        if own:
            db.close()


_UPDATE_CHUNK = 500


def _bulk_update(db: Session, ids: list[int], **values: Any) -> int:
    """Apply `values` to `ids` in chunks, without loading a single row.

    Chunked because some drivers cap bound parameters per statement, and a
    cascade over a large lineage can touch far more ids than that cap.
    `synchronize_session=False` is safe here: every caller commits straight
    after, which expires the session's objects anyway.
    """
    if not ids:
        return 0
    for start in range(0, len(ids), _UPDATE_CHUNK):
        db.execute(
            update(ResearchSnapshot)
            .where(ResearchSnapshot.id.in_(ids[start:start + _UPDATE_CHUNK]))
            .values(**values)
            .execution_options(synchronize_session=False)
        )
    return len(ids)


def invalidate(
    subject: str,
    kind: str | None = None,
    *,
    db: Session | None = None,
) -> int:
    """Mark all live snapshots for `subject` (and optional `kind`) invalidated.

    Returns the count of rows touched, and cascades once for the whole set.
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        # Ids only. These rows carry `payload` JSON blobs that this function
        # never reads, and a hot subject accumulates thousands of them.
        stmt = (
            select(ResearchSnapshot.id)
            .where(
                ResearchSnapshot.subject == subject,
                ResearchSnapshot.invalidated_at.is_(None),
            )
        )
        if kind:
            stmt = stmt.where(ResearchSnapshot.kind == kind)
        ids = [row_id for (row_id,) in db.execute(stmt).all()]
        if not ids:
            return 0
        _bulk_update(db, ids, invalidated_at=_now(), stale=True)
        # One cascade for the whole seed set, not one per row: the lineage
        # walk is shared, so N invalidated rows cost one scan, not N.
        mark_stale_descendants(ids, db=db)
        db.commit()
        return len(ids)
    finally:
        if own:
            db.close()


def mark_stale_descendants(
    snapshot_id: int | Iterable[int],
    *,
    db: Session | None = None,
) -> int:
    """Mark every snapshot whose lineage references `snapshot_id` as stale.

    Accepts one id or many. Pass the whole seed set at once: the lineage
    scan is shared, so N seeds cost one scan rather than N.

    The edge lives in a JSON column, which has no portable index, so the
    walk itself happens in Python. What must not happen is loading the rows
    to do it. This function used to run `select(ResearchSnapshot)` — every
    column, `payload` blob included — with no LIMIT, once per frontier
    node, from a caller that re-entered it once per invalidated row. Since
    `research_snapshots` is insert-only and was never reaped, that scan grew
    with the table until it no longer fit in memory: on 2026-09-10 it
    OOM-killed the 512 MB production worker once an hour, every hour, with
    no traceback (SIGKILL leaves none). Keep this function projecting only
    the columns it reads, and keep the query out of the loop.

    Returns the count of newly-stale rows.
    """
    seeds = [snapshot_id] if isinstance(snapshot_id, int) else [int(s) for s in snapshot_id]
    if not seeds:
        return 0
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        # The two columns the walk actually reads, fetched ONCE. Previously
        # this was `select(ResearchSnapshot)` — every column including the
        # `payload` JSON blob — re-run once per frontier node, inside a
        # caller that re-entered it once per invalidated row. See the
        # docstring: that is what killed the worker hourly.
        children: dict[int, list[int]] = {}
        rows = db.execute(
            select(ResearchSnapshot.id, ResearchSnapshot.parent_snapshot_ids)
            .where(ResearchSnapshot.stale.is_(False))
        ).all()
        for row_id, parents in rows:
            for parent_id in (parents or []):
                try:
                    children.setdefault(int(parent_id), []).append(row_id)
                except (TypeError, ValueError):  # hand-written lineage
                    continue

        frontier = list(seeds)
        seen: set[int] = set(seeds)
        descendants: list[int] = []
        while frontier:
            for child_id in children.get(frontier.pop(), ()):
                if child_id in seen:
                    continue
                seen.add(child_id)
                descendants.append(child_id)
                frontier.append(child_id)

        _bulk_update(db, descendants, stale=True)
        db.commit()
        return len(descendants)
    finally:
        if own:
            db.close()


# Retention. `cache_put` is insert-only — every write appends a row and
# leaves the previous one behind, live and un-stale — and until 2026-09-10
# nothing ever deleted from this table. The EDGAR poller alone appends ~170
# bookkeeping rows every 30 minutes (~8k/day), none of which is ever
# invalidated. The table therefore grew without bound, and the unbounded
# lineage scan above eventually could not fit in the worker's memory.
# Bounding the scan stops the crash; this keeps the table from growing back
# into the next one.
SNAPSHOT_RETENTION_DAYS = 14
SNAPSHOT_KEEP_PER_KEY = 1
_GC_MAX_DELETE = 50_000
_GC_STREAM_ROWS = 1_000


def gc_snapshots(
    *,
    retention_days: int = SNAPSHOT_RETENTION_DAYS,
    keep_per_key: int = SNAPSHOT_KEEP_PER_KEY,
    max_delete: int = _GC_MAX_DELETE,
    now: datetime | None = None,
    db: Session | None = None,
) -> dict[str, int]:
    """Delete superseded snapshots. Returns `{scanned, deleted, capped}`.

    The rule is deliberately conservative, because deleting a row that is
    still serving reads turns a cache hit into a recompute (and, for the
    LLM-backed kinds, into real money): for every `(subject, kind)` the
    newest `keep_per_key` rows are kept **whatever their age**. Only rows
    that something newer has already superseded, and that are older than
    `retention_days`, are removed. A key written once and never again keeps
    its single row forever.

    Memory-bounded on purpose — it would be absurd for the reaper that
    exists to prevent an OOM to cause one. It streams four small columns
    (never `payload`), and stops after `max_delete` rows so a first run
    against a very large table is a bounded amount of work; the next run
    continues where it left off.
    """
    cutoff = (now or _now()) - timedelta(days=retention_days)
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = (
            select(
                ResearchSnapshot.id,
                ResearchSnapshot.subject,
                ResearchSnapshot.kind,
                ResearchSnapshot.generated_at,
            )
            .order_by(
                ResearchSnapshot.subject,
                ResearchSnapshot.kind,
                ResearchSnapshot.generated_at.desc(),
                ResearchSnapshot.id.desc(),
            )
            .execution_options(yield_per=_GC_STREAM_ROWS)
        )
        doomed: list[int] = []
        scanned = 0
        capped = 0
        current_key: tuple | None = None
        rank = 0
        for row_id, subject, kind, generated_at in db.execute(stmt):
            scanned += 1
            key = (subject, kind)
            if key != current_key:
                current_key, rank = key, 0
            rank += 1
            if rank <= keep_per_key:
                continue          # newest per key is never touched
            if generated_at is not None and generated_at >= cutoff:
                continue          # superseded, but still inside retention
            doomed.append(row_id)
            if len(doomed) >= max_delete:
                capped = 1
                break

        deleted = 0
        for start in range(0, len(doomed), _UPDATE_CHUNK):
            chunk = doomed[start:start + _UPDATE_CHUNK]
            result = db.execute(delete(ResearchSnapshot).where(ResearchSnapshot.id.in_(chunk)))
            deleted += result.rowcount if result.rowcount is not None else len(chunk)
        db.commit()
        log.info("gc_snapshots: scanned=%d deleted=%d capped=%d", scanned, deleted, capped)
        return {"scanned": scanned, "deleted": deleted, "capped": capped}
    finally:
        if own:
            db.close()


def log_cost(
    subject: str,
    kind: str,
    cost_tokens: int,
    *,
    note: str = "",
    db: Session | None = None,
) -> None:
    """Append a cost row to the cache cost ledger."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        db.add(CacheCostLog(
            subject=subject, kind=kind,
            cost_tokens=int(cost_tokens or 0), note=note or "",
        ))
        if own:
            db.commit()
    finally:
        if own:
            db.close()


def total_token_cost(
    *,
    since: datetime | None = None,
    exclude_hits: bool = True,
    db: Session | None = None,
) -> int:
    """Sum cost_tokens since `since`. Useful for the smoke evaluation gate."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = select(CacheCostLog)
        rows = db.execute(stmt).scalars().all()
        total = 0
        for r in rows:
            if since and r.generated_at < since:
                continue
            if exclude_hits and r.kind.endswith(":hit"):
                continue
            total += int(r.cost_tokens or 0)
        return total
    finally:
        if own:
            db.close()
