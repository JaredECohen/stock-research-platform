"""Persistent, lineage-aware research snapshot cache.

Re-exports the public surface of `snapshots` so callers can do
`from app.cache import cache_get, cache_put, invalidate, ...`.
"""
from .snapshots import (
    CacheCostLog,
    ResearchSnapshot,
    cache_get,
    cache_put,
    invalidate,
    log_cost,
    mark_stale_descendants,
    sources_fingerprint,
    total_token_cost,
)

__all__ = [
    "CacheCostLog",
    "ResearchSnapshot",
    "cache_get",
    "cache_put",
    "invalidate",
    "log_cost",
    "mark_stale_descendants",
    "sources_fingerprint",
    "total_token_cost",
]
