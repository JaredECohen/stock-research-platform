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

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

from .config import settings


def client_ip(request: Request) -> str:
    """The caller's address as seen through `TRUSTED_PROXY_HOPS` proxies.

    `request.client.host` is the socket peer, and uvicorn only rewrites it
    from the proxy headers when FORWARDED_ALLOW_IPS is set — which the
    Dockerfile's CMD does not do. Behind Render that made every request
    look like it came from the proxy, so a "3 signups per hour per IP"
    ceiling was really 3 signups per hour for the whole site.

    So the header is read here, and from the RIGHT: each proxy appends the
    address it accepted the connection from, so the last `hops` entries
    are the ones our infrastructure wrote and the caller is the `hops`-th
    from the end. slowapi's `get_remote_address` never reads the header at
    all, and reading the FIRST entry — the obvious mistake — hands every
    client its own bucket for the price of `X-Forwarded-For: <random>`.
    With fewer entries than trusted hops the header was not written by
    our proxies, and the socket peer is the honest answer.

    The value is only ever a bucket key or a salted hash, never trusted
    as an address for anything else, so it is not validated as an IP.
    """
    hops = int(getattr(settings, "trusted_proxy_hops", 1) or 0)
    if hops > 0:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            parts = [p.strip() for p in forwarded.split(",") if p.strip()]
            if len(parts) >= hops:
                return parts[-hops][:128]
    client = request.client
    return client.host if client else "unknown"


def _key_func(request: Request) -> str:
    """Identify the caller for the per-IP limits — through the proxy, so
    the bucket is the browser's address rather than Render's."""
    return client_ip(request)


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
    # FEAT-002. `public_get` is the anonymous ceiling on the marketing
    # reads (`/api/public/*`); `bootstrap` is the signup ceiling on
    # `POST /api/me/bootstrap` — the only per-IP number that bounds how
    # many trials one address can start per hour, so keep it small.
    "public_get":      "60/minute",
    "bootstrap":        "3/hour",
    # FEAT-001. Series/catalog reads are DB-only plus a cached price
    # read, so the default ceiling; commentary is one cheap-route LLM
    # call per request and gets DEVPLAN's lightweight-LLM ceiling.
    "fundamentals_series":     "60/minute",
    "fundamentals_commentary": "10/minute",
    # Phase 6. The export streams a whole run (a few hundred rows) per
    # call; a downstream system polls it, a person does not.
    "scorecard_export":        "30/minute",
    # FEAT-003. Industry Analysis reads are single-row JSON fetches of
    # artifacts the worker already produced (a report payload is tens of
    # KB), so they take the cheap-read ceiling; the ops endpoints import a
    # taxonomy, re-classify the universe or queue a week of report jobs,
    # and a person runs those a handful of times, never in a loop.
    "industry_read":           "60/minute",
    "industry_admin":          "10/minute",
}


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Structured 429 for slowapi's per-IP limits.

    Delegates to `auth.ratelimit.structured_slowapi_handler` so the per-IP
    and per-user limiters emit one shape. Imported lazily: `auth` pulls in
    the ORM, and this module is imported by every route module.
    """
    from .auth.ratelimit import structured_slowapi_handler
    return structured_slowapi_handler(request, exc)

# ---------------------------------------------------------------------------
# slowapi cannot see routes behind an included router — patched here
# ---------------------------------------------------------------------------

def _find_route_handler(routes, scope):
    """Resolve the endpoint for `scope`, descending into nested routers.

    This replaces `slowapi.middleware._find_route_handler`, which scans one
    level deep and takes `hasattr(route, "endpoint")` as its test. Under the
    pinned FastAPI (0.141.1) an `include_router` call leaves a nested router
    object in `app.routes` instead of flattening every `APIRoute` into it —
    `app.routes` holds 22 entries where the older FastAPI held 106. That
    object matches the request FULL but exposes no `endpoint`, so slowapi
    resolved `None`, and `_should_exempt(limiter, None)` is **True**.

    The effect was silent and total: every request looked exempt, so NO
    per-IP limit applied anywhere — not the global 60/minute default, not
    `memo_analyze` (5/minute, the route that spends model budget), not
    `bootstrap` (3/hour, the only per-IP bound on how many trials one
    address can start). It reproduced only against the pinned versions,
    which is what CI and Render install; a developer environment on the
    older FastAPI limited correctly and every test passed.

    Kept faithful to the original in the one respect that matters: the LAST
    full match wins, because that is how slowapi resolves overlapping
    routes and the exemption registry is keyed on the name it returns.
    """
    from starlette.routing import Match

    handler = None
    for route in routes:
        try:
            match, child_scope = route.matches(scope)
        except Exception:       # a route that cannot match is not a handler
            continue
        if match != Match.FULL:
            continue
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None:
            handler = endpoint
            continue
        # `_IncludedRouter` (the pinned FastAPI's wrapper) exposes neither
        # `routes` nor `router`; the included APIRouter is on
        # `original_router`. Try each shape rather than naming one version's,
        # so a future FastAPI that flattens again, or nests differently,
        # still resolves.
        nested = None
        for attr in ("routes", "router", "original_router"):
            candidate = getattr(route, attr, None)
            candidate = getattr(candidate, "routes", candidate)
            if candidate:
                try:
                    nested = list(candidate)
                except TypeError:
                    nested = None
                if nested:
                    break
        if nested:
            inner = _find_route_handler(nested, {**scope, **(child_scope or {})})
            if inner is not None:
                handler = inner
    return handler


def _install_route_resolution() -> None:
    """Point slowapi's middleware at the resolver above. Idempotent."""
    from slowapi import middleware as _slowapi_middleware
    _slowapi_middleware._find_route_handler = _find_route_handler


_install_route_resolution()
