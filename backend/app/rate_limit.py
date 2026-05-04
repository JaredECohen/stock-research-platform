"""Per-IP rate limiting for the expensive endpoints.

Keeps anonymous abuse from racking up real LLM bills. Tiered limits:

  default                          — 60/min  (cheap reads)
  /api/chat                        — 30/min  (LLM-backed reasoning)
  GET  /api/stocks/{t}/memo        — 60/min  (usually cache hits)
  POST /api/stocks/{t}/analyze     —  5/min  (forces fresh agent run)
  POST /api/screener/custom        — 30/min
  POST /api/seed-universe          —  1/min  (admin)
  POST /api/admin/run-backfill     —  1/5min (admin; long-running)

Backed by `slowapi` with the in-memory store by default. Set
`RATE_LIMIT_STORAGE_URL=redis://…` to share state across replicas.
Disable entirely with `RATE_LIMIT_ENABLED=false` (still useful for
tests + local dev).
"""
from __future__ import annotations

from typing import Optional

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request

from .config import settings


def _key_func(request: Request) -> str:
    """Identify the caller. When sitting behind Cloudflare / Render's
    proxy, prefer the forwarded IP from `x-forwarded-for` (slowapi's
    helper handles this, but we strip whitespace defensively)."""
    return get_remote_address(request)


def _build_limiter() -> Limiter:
    """Construct the singleton Limiter.

    `enabled=False` short-circuits all decorated routes (no calls to the
    storage backend, no header overhead). Useful when running under
    pytest's TestClient where every test would otherwise pollute counts.
    """
    storage_uri = getattr(settings, "rate_limit_storage_url", "") or "memory://"
    enabled = getattr(settings, "rate_limit_enabled", True)
    return Limiter(
        key_func=_key_func,
        storage_uri=storage_uri,
        enabled=enabled,
        default_limits=["60/minute"],
        headers_enabled=True,
        strategy="fixed-window",
    )


limiter = _build_limiter()
RateLimitExceeded = RateLimitExceeded  # re-export for callers


# Per-route limit strings. Keep these as module constants so route
# decorators can reference one source of truth.
LIMITS = {
    "chat":            "30/minute",
    "memo_read":       "60/minute",
    "memo_analyze":     "5/minute",
    "custom_screen":   "30/minute",
    "seed_universe":    "1/minute",
    "admin_backfill":  "1/5minute",
}
