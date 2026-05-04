"""Read-through TTL cache for raw provider responses (Wave 9b).

Sits between `data_service` and the live provider chain. Every per-
capability call routes through `cached_call(capability, key, fetcher,
…)` which (a) returns a fresh row if one exists, (b) falls through to
the provider on miss/expiry, (c) writes the new response back, and
(d) serves a stale row when the provider also misses.

Per-capability TTLs are tuned to how frequently the underlying data
actually changes:

    profile     7 days   (description, sector, FY-end, CIK — stable;
                          market cap drifts but doesn't justify daily
                          refetch on every research call)
    prices      1 day    (full daily history; refresh after each close)
    ratios      1 day    (price-dependent metrics)
    estimates   1 day    (sell-side updates frequently but not
                          intra-day for most names)
    earnings    1 day    (calendar / surprises)
    news        1 hour   (time-sensitive)
    macro       1 day    (mostly weekly / monthly publication)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

from sqlalchemy import select

from ..database import SessionLocal
from ..models import ProviderCache

log = logging.getLogger(__name__)


# In-seconds. None = never expires (always reuse cache).
TTL_BY_CAPABILITY: Dict[str, int] = {
    "profile":   7 * 86400,
    "prices":         86400,
    "ratios":         86400,
    "estimates":      86400,
    "earnings":       86400,
    "news":            3600,
    "macro":          86400,
}


def _is_fresh(fetched_at: datetime, ttl_seconds: Optional[int]) -> bool:
    if ttl_seconds is None:
        return True  # never-expire mode
    return datetime.utcnow() - fetched_at < timedelta(seconds=ttl_seconds)


def get(
    capability: str, key: str,
    *, ttl_seconds: Optional[int] = None,
    serve_stale: bool = False,
) -> Optional[Any]:
    """Read the cached payload for `(capability, key)`.

    Returns None when no row exists. When a row exists but is past
    `ttl_seconds`, returns None *unless* `serve_stale=True` (used as
    the last-resort fallback when the provider also missed).
    """
    with SessionLocal() as db:
        row = db.execute(
            select(ProviderCache).where(
                ProviderCache.capability == capability,
                ProviderCache.key == key,
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        if serve_stale or _is_fresh(row.fetched_at, ttl_seconds):
            return row.payload_json
        return None


def put(capability: str, key: str, payload: Any) -> None:
    """Upsert a cache row. No-op when payload is empty (None / [] / {})."""
    if payload in (None, [], {}):
        return
    with SessionLocal() as db:
        existing = db.execute(
            select(ProviderCache).where(
                ProviderCache.capability == capability,
                ProviderCache.key == key,
            )
        ).scalar_one_or_none()
        if existing is None:
            db.add(ProviderCache(
                capability=capability, key=key,
                payload_json=payload, fetched_at=datetime.utcnow(),
            ))
        else:
            existing.payload_json = payload
            existing.fetched_at = datetime.utcnow()
        db.commit()


def invalidate(capability: str, key: Optional[str] = None) -> int:
    """Drop rows. Pass `key=None` to clear every row for `capability`.

    Returns the number of rows deleted.
    """
    with SessionLocal() as db:
        stmt = select(ProviderCache).where(ProviderCache.capability == capability)
        if key is not None:
            stmt = stmt.where(ProviderCache.key == key)
        rows = db.execute(stmt).scalars().all()
        n = len(rows)
        for row in rows:
            db.delete(row)
        db.commit()
        return n


def cached_call(
    capability: str, key: str, fetcher: Callable[[], Any],
    *, ttl_seconds: Optional[int] = None,
    force_refresh: bool = False,
) -> Optional[Any]:
    """Read-through: cache hit → return; miss → call `fetcher`, write,
    return; provider miss → fall back to a stale cached row.

    `ttl_seconds` defaults to `TTL_BY_CAPABILITY[capability]`. Pass an
    explicit value (or `None` to never expire) to override.
    """
    if ttl_seconds is None and not force_refresh:
        ttl_seconds = TTL_BY_CAPABILITY.get(capability)

    if not force_refresh:
        cached = get(capability, key, ttl_seconds=ttl_seconds)
        if cached is not None:
            return cached

    fresh = fetcher()
    if fresh is not None and fresh != [] and fresh != {}:
        put(capability, key, fresh)
        return fresh

    # Provider also missed — better stale than empty.
    return get(capability, key, ttl_seconds=None, serve_stale=True)
