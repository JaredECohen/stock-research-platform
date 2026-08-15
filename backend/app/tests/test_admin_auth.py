"""The admin/ops surface must require a token — and keep requiring it.

`/api/admin/*` shipped fully unauthenticated: ~30 endpoints reading LLM
traces and memo internals, and mutating real state (`fix-sequences`
rewrites Postgres sequences, `rerun-memos` burns LLM spend,
`run-backfill` hammers rate-limited providers).

The test that matters here is `test_every_admin_route_is_protected`: it
enumerates `app.routes` rather than spot-checking a handful of paths, so
a newly added admin endpoint that somehow escapes the guard fails CI
instead of shipping open. That is the whole reason the guard is
middleware and not a per-route dependency.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import admin_auth
from app.config import settings
from app.main import app

TOKEN = "test-admin-token-abc123"


@pytest.fixture()
def client():
    # Bare TestClient, NOT `with TestClient(app)`, matching
    # test_admin_endpoints_w8c.py. The context-manager form fires the
    # startup event, which runs `run_full_seed()` — a live-provider
    # profile lookup per universe member — on every single test. conftest
    # already creates the tables session-wide, and middleware runs either
    # way, so lifespan buys nothing here and costs minutes.
    return TestClient(app)


@pytest.fixture()
def with_token(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", TOKEN)
    yield TOKEN


# ---------------------------------------------------------------------------
# Coverage — the anti-regression test
# ---------------------------------------------------------------------------

def _admin_routes_from_openapi() -> list[tuple[str, str]]:
    """Ops (method, path) pairs, read from the OpenAPI schema.

    The schema is a documented, version-stable contract. An earlier
    version of this walked `app.routes` and read `.methods` off each
    route object — that passed locally (starlette 1.0) and produced an
    EMPTY list on CI (starlette 1.6), because `requirements.txt` pins
    floors so CI resolves newer versions than a long-lived dev env. The
    test failed loudly rather than silently passing, which is the only
    reason it was caught, but the schema avoids the fragility entirely.
    """
    spec = app.openapi()["paths"]
    return sorted({
        (method.upper(), path)
        for path, operations in spec.items()
        if path.startswith("/api/admin") or path.startswith("/api/seed-universe")
        for method in operations
        if method.upper() not in ("HEAD", "OPTIONS")
    })


def test_openapi_sees_every_admin_route_on_the_router():
    """Close the blind spot in the coverage test below.

    `include_in_schema=False` on an admin route would hide it from the
    OpenAPI schema, so the coverage test would pass while the route
    shipped unguarded. Cross-checked against `routes_admin.router` — the
    APIRouter this repo constructs — rather than `app.routes`.

    That distinction is the bug this test was born from: starlette 1.6
    stopped flattening included routers, so `app.routes` yields
    `_IncludedRouter` wrappers with no `.path` or `.methods`. Walking it
    silently produced an empty set on CI (which resolves newer versions
    than a long-lived dev env, since requirements.txt pins floors) while
    passing locally on starlette 1.0. Our own APIRouter holds real
    `APIRoute` objects on both.
    """
    from app.api import routes_admin

    on_router = {
        (method.upper(), route.path)
        for route in routes_admin.router.routes
        for method in (getattr(route, "methods", None) or set())
        if method.upper() not in ("HEAD", "OPTIONS")
    }
    assert on_router, (
        "no routes found on routes_admin.router — introspection changed again; "
        "verify _admin_routes_from_openapi is still complete before trusting it"
    )
    hidden = on_router - set(_admin_routes_from_openapi())
    assert not hidden, (
        "admin routes missing from the OpenAPI schema (likely "
        "include_in_schema=False) — they would escape the coverage test:\n  "
        + "\n  ".join(f"{m} {p}" for m, p in sorted(hidden))
    )


def test_every_admin_route_is_protected():
    """Every ops route must match the guard.

    Checked against `is_protected` rather than by issuing requests. An
    earlier version called each endpoint and asserted 401, which is a
    genuinely dangerous way to test this: the condition it looks for is
    "this admin route is NOT protected", so on failure it would *execute*
    the handler — and the set includes `run-backfill` (sweeps the
    universe against rate-limited providers), `rerun-memos` (real LLM
    spend) and `seed-universe`. It hung the suite when first run.
    `test_middleware_is_actually_wired_up` covers the HTTP path using a
    read-only endpoint instead.
    """
    admin_routes = _admin_routes_from_openapi()
    assert admin_routes, "no admin routes in the schema — the check is looking in the wrong place"

    unguarded = [
        f"{m} {p}" for m, p in admin_routes
        if not admin_auth.is_protected(m, p) and not admin_auth.is_exempt(m, p)
    ]
    assert not unguarded, (
        "ops routes not covered by the admin guard:\n  " + "\n  ".join(unguarded)
    )


def test_middleware_is_actually_wired_up(with_token, client):
    """Route-table matching is worthless if the middleware isn't
    registered. Uses a read-only endpoint so nothing expensive can run."""
    assert client.get("/api/admin/cron-health").status_code == 401


def test_guard_covers_seed_universe_despite_the_prefix():
    """`/api/seed-universe` lives in routes_admin.py but is NOT under the
    `/api/admin` prefix — pure prefix matching would silently miss it, and
    it is a POST that rewrites the company universe."""
    assert admin_auth.is_protected("POST", "/api/seed-universe")


def test_public_routes_are_untouched(client):
    """The guard must not creep beyond the ops surface."""
    for path in ("/health", "/api/providers/status"):
        assert client.get(path).status_code == 200, path
    assert not admin_auth.is_protected("GET", "/api/stocks/NVDA/memo")
    assert not admin_auth.is_protected("POST", "/api/chat")


# ---------------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------------

def test_correct_token_is_accepted(with_token, client):
    resp = client.get(
        "/api/admin/cron-health", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert resp.status_code == 200


@pytest.mark.parametrize("header", [
    None,
    "",
    "Bearer",
    "Bearer wrong-token-abc123",
    f"Bearer {TOKEN}x",            # correct prefix, extra byte
    f"bearer {TOKEN}",             # scheme is case-sensitive here
    TOKEN,                         # raw token, no scheme
    f"Basic {TOKEN}",
])
def test_bad_credentials_are_rejected(with_token, client, header):
    headers = {} if header is None else {"Authorization": header}
    resp = client.get("/api/admin/cron-health", headers=headers)
    assert resp.status_code == 401, f"{header!r} was accepted"


@pytest.mark.parametrize("method,path", [
    # frontend/src/lib/logger.ts
    ("POST", "/api/admin/ui-log"),
    # frontend/src/pages/TrackRecord.tsx
    ("POST", "/api/admin/evaluate-outcomes"),
    # frontend/src/api/client.ts
    ("GET", "/api/admin/track-record"),
    ("GET", "/api/admin/dcf-versions/NVDA"),
    ("GET", "/api/admin/lopsidedness-audit"),
])
def test_browser_reachable_endpoints_stay_open(with_token, method, path):
    """These are called from `frontend/src/` with no Authorization header.

    Guarding them would break user-facing pages, not secure anything —
    the site is public, so they are already reachable by any visitor.
    Pinned as a contract so a later "tighten the admin surface" pass
    can't silently take the UI down with it. Checked against the matcher
    rather than by issuing requests: `evaluate-outcomes` does real work.
    """
    assert not admin_auth.is_protected(method, path), f"{method} {path} would break the UI"


def test_ui_log_read_and_delete_are_still_protected(with_token, client):
    """The exemption is scoped by METHOD deliberately: GET reads the trace
    table and DELETE wipes it."""
    assert client.get("/api/admin/ui-log").status_code == 401
    assert client.delete("/api/admin/ui-log").status_code == 401


# ---------------------------------------------------------------------------
# Misconfiguration
# ---------------------------------------------------------------------------

def test_unset_token_in_production_fails_closed(monkeypatch, client):
    """Failing open is exactly how this surface became publicly reachable.
    503 (not 401) because the server is misconfigured, not the caller."""
    monkeypatch.setattr(settings, "admin_api_token", "")
    monkeypatch.setattr(settings, "app_env", "production")
    resp = client.get("/api/admin/cron-health")
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"]


def test_unset_token_outside_production_stays_open(monkeypatch, client):
    """Dev and the test suite must keep working without ceremony."""
    monkeypatch.setattr(settings, "admin_api_token", "")
    monkeypatch.setattr(settings, "app_env", "development")
    assert client.get("/api/admin/cron-health").status_code == 200


def test_production_check_is_case_insensitive(monkeypatch, client):
    monkeypatch.setattr(settings, "admin_api_token", "")
    monkeypatch.setattr(settings, "app_env", "PRODUCTION")
    assert client.get("/api/admin/cron-health").status_code == 503
