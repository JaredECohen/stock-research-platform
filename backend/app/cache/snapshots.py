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
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
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
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    sources_hash: Mapped[str] = mapped_column(String(64), default="")
    sources_used: Mapped[List[str]] = mapped_column(JSON, default=list)
    generated_by: Mapped[str] = mapped_column(String(128), default="")
    cost_tokens: Mapped[int] = mapped_column(Integer, default=0)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    invalidated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    parent_snapshot_ids: Mapped[List[int]] = mapped_column(JSON, default=list)
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
    items: List[str] = []
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


def _is_expired(snap: ResearchSnapshot, max_age_seconds: Optional[int]) -> bool:
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
    max_age_seconds: Optional[int] = None,
    db: Optional[Session] = None,
) -> Optional[ResearchSnapshot]:
    """Look up the freshest non-stale, non-invalidated snapshot for the key.

    Returns None if no usable snapshot exists. Callers should treat the return
    as read-only; mutations should go through `cache_put` so we keep history.
    """
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
        return row
    finally:
        if own:
            db.close()


def cache_put(
    subject: str,
    kind: str,
    payload: Dict[str, Any],
    sources_used: Optional[List[Any]] = None,
    generated_by: str = "",
    cost_tokens: int = 0,
    parent_snapshots: Optional[List[int]] = None,
    ttl_seconds: Optional[int] = None,
    schema_version: int = 1,
    db: Optional[Session] = None,
) -> ResearchSnapshot:
    """Store a new snapshot. Returns the persisted row with .id populated."""
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


def invalidate(
    subject: str,
    kind: Optional[str] = None,
    *,
    db: Optional[Session] = None,
) -> int:
    """Mark all live snapshots for `subject` (and optional `kind`) invalidated.

    Returns the count of rows touched. Cascades by calling
    `mark_stale_descendants` for each invalidated snapshot.
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = (
            select(ResearchSnapshot)
            .where(
                ResearchSnapshot.subject == subject,
                ResearchSnapshot.invalidated_at.is_(None),
            )
        )
        if kind:
            stmt = stmt.where(ResearchSnapshot.kind == kind)
        rows = db.execute(stmt).scalars().all()
        now = _now()
        count = 0
        for row in rows:
            row.invalidated_at = now
            row.stale = True
            count += 1
            mark_stale_descendants(row.id, db=db)
        db.commit()
        return count
    finally:
        if own:
            db.close()


def mark_stale_descendants(snapshot_id: int, *, db: Optional[Session] = None) -> int:
    """Mark every snapshot whose lineage references `snapshot_id` as stale.

    BFS through parent_snapshot_ids since SQLite JSON doesn't support GIN/JSONB
    operators portably. Returns count of newly-stale rows.
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        frontier = [snapshot_id]
        seen: set[int] = set()
        count = 0
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            stmt = select(ResearchSnapshot).where(
                ResearchSnapshot.stale.is_(False),
            )
            rows = db.execute(stmt).scalars().all()
            for row in rows:
                if current in (row.parent_snapshot_ids or []):
                    row.stale = True
                    count += 1
                    frontier.append(row.id)
        db.commit()
        return count
    finally:
        if own:
            db.close()


def log_cost(
    subject: str,
    kind: str,
    cost_tokens: int,
    *,
    note: str = "",
    db: Optional[Session] = None,
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
    since: Optional[datetime] = None,
    exclude_hits: bool = True,
    db: Optional[Session] = None,
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
