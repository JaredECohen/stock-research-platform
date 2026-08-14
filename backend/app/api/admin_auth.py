"""Bearer-token guard for the admin/ops surface.

`/api/admin/*` was fully unauthenticated: ~30 endpoints that read LLM
traces and memo internals, and mutate real state — `fix-sequences`
rewrites Postgres sequences, `rerun-memos` burns LLM spend,
`run-backfill` hammers rate-limited providers. Anyone with the URL could
call them.

**Middleware, not a per-route `Depends`.** A dependency has to be
remembered on every new route, and this router grows steadily; one
forgotten decorator is a silent hole with no failing test. Matching on
the path prefix means a new admin endpoint is protected the moment it is
mounted, and `test_admin_auth.py` enumerates `app.routes` to prove it.

Deliberately NOT protected:
  - `/health` and the public read APIs (stocks, screener, chat, ...) —
    unchanged behaviour.
  - `/api/admin/ui-log` POST, which the *frontend* calls on every page
    for client-side telemetry. Guarding it would break the UI for real
    users, since the browser has no token. It is append-only and already
    excluded from HTTP logging; the GET side is protected.
"""
from __future__ import annotations

import logging
import secrets
from typing import Iterable

from fastapi import Request
from fastapi.responses import JSONResponse

from ..config import settings

log = logging.getLogger(__name__)

# Prefixes requiring the admin token. `/api/seed-universe` is listed
# explicitly: it is defined in `routes_admin.py` but does NOT sit under
# the `/api/admin` prefix, so prefix matching alone would silently miss
# it — and it is a POST that rewrites the company universe.
PROTECTED_PREFIXES: tuple[str, ...] = (
    "/api/admin",
    "/api/seed-universe",
)

# Endpoints the BROWSER calls, exempt despite matching a protected
# prefix. Each is reached from `frontend/src/` with no Authorization
# header, so guarding them would break user-facing pages rather than
# secure anything.
#
# This list exists because `/api/admin/*` currently conflates two
# unrelated things: real ops endpoints (fix-sequences, rerun-memos,
# run-backfill, llm-breakers) and product features the UI renders
# (track-record, dcf-versions, lopsidedness-audit). Only the former
# should ever want a token. Exempting the latter preserves today's
# behaviour exactly — the site is public and unauthenticated, so these
# are already reachable by any visitor; the guard is not a regression in
# their exposure. The proper fix is to move them to a public namespace so
# the prefix means one thing; tracked as a follow-up.
#
# Matched by (method, path-prefix), because `dcf-versions` is
# parameterised. Scoping by METHOD is load-bearing: `GET
# /api/admin/ui-log` reads the trace table and `DELETE` wipes it — both
# stay protected while the browser's POST does not.
EXEMPT_PREFIXES: tuple[tuple[str, str], ...] = (
    # frontend/src/lib/logger.ts — client telemetry, flushed on a 1.5s tick.
    # Append-only and already returns 200 unconditionally, so the exposure
    # is spam, not data loss.
    ("POST", "/api/admin/ui-log"),
    # frontend/src/pages/TrackRecord.tsx — user-triggered "evaluate now".
    ("POST", "/api/admin/evaluate-outcomes"),
    # frontend/src/api/client.ts — rendered by product pages.
    ("GET", "/api/admin/track-record"),
    ("GET", "/api/admin/dcf-versions/"),
    ("GET", "/api/admin/lopsidedness-audit"),
)


def is_exempt(method: str, path: str) -> bool:
    """Whether this browser-reachable endpoint is deliberately unguarded."""
    m = method.upper()
    return any(m == em and path.startswith(ep) for em, ep in EXEMPT_PREFIXES)


def is_protected(method: str, path: str) -> bool:
    """Whether (method, path) requires the admin token.

    Exposed so tests can assert coverage over `app.routes` without
    re-implementing the matching rules — the check and its test agree by
    construction rather than by hand.
    """
    if is_exempt(method, path):
        return False
    return any(path.startswith(p) for p in PROTECTED_PREFIXES)


def _unauthorized(detail: str, status: int) -> JSONResponse:
    # No WWW-Authenticate challenge: this is a machine-to-machine ops
    # surface, and returning one invites a browser credential prompt.
    return JSONResponse({"detail": detail}, status_code=status)


async def admin_auth_middleware(request: Request, call_next):
    path = request.url.path
    if not is_protected(request.method, path):
        return await call_next(request)

    token = settings.admin_api_token
    if not token:
        if (settings.app_env or "").lower() == "production":
            # Fail CLOSED. An unset token in production is a
            # misconfiguration, and serving the ops surface openly is the
            # exact failure this module exists to prevent. 503 (not 401)
            # says "the server is not set up", which is the truth and is
            # actionable in a way "unauthorized" is not.
            log.error(
                "ADMIN_API_TOKEN is unset in production — refusing %s %s. "
                "Set it in the Render dashboard to restore the admin API.",
                request.method, path,
            )
            return _unauthorized("admin API not configured", 503)
        # Dev/test: no token configured, surface stays open so the suite
        # and local workflows are unchanged.
        return await call_next(request)

    supplied = request.headers.get("authorization") or ""
    expected = f"Bearer {token}"
    # compare_digest, not `==`: a short-circuiting comparison leaks the
    # token a byte at a time to anyone who can time the response.
    # Length is compared first because compare_digest is only
    # constant-time for equal-length inputs.
    if len(supplied) != len(expected) or not secrets.compare_digest(supplied, expected):
        log.warning("rejected unauthenticated admin request: %s %s", request.method, path)
        return _unauthorized("unauthorized", 401)

    return await call_next(request)


def protected_routes(routes: Iterable) -> list[tuple[str, str]]:
    """(method, path) pairs from `app.routes` that the guard covers.

    Helper for tests and for a quick audit of what is actually protected.
    """
    out: list[tuple[str, str]] = []
    for route in routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if not path:
            continue
        for method in methods:
            if is_protected(method, path):
                out.append((method, path))
    return sorted(set(out))
