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


def resolved_cost_tokens(baseline: int = 0) -> int:
    """Return baseline + the most recent provider call's total_tokens (Phase C).

    Demo mode: returns the baseline (no LLM calls were made).
    Live mode: adds the just-completed call's actual tokens, captured via
    `agents.llm.last_usage()`. Consumes the usage value so the same call
    isn't double-counted across two `cache_put` sites in a row.
    """
    try:
        # Lazy import — keeps `app.cache` independent of agents at import time.
        from ..agents.llm import last_usage
    except Exception:  # pragma: no cover
        return baseline
    usage = last_usage()
    if usage and usage.get("total_tokens"):
        return int(baseline or 0) + int(usage["total_tokens"])
    return int(baseline or 0)


__all__ = [
    "CacheCostLog",
    "ResearchSnapshot",
    "cache_get",
    "cache_put",
    "invalidate",
    "log_cost",
    "mark_stale_descendants",
    "resolved_cost_tokens",
    "sources_fingerprint",
    "total_token_cost",
]
