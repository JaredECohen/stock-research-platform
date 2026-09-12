"""A rate-limited route must not 500 when the limiter is switched on.

This is the guard for a production incident on 2026-09-12, and the shape of
it is worth stating because it will recur otherwise.

Restoring the per-IP limiter (which had been silently enforcing nothing)
re-enabled slowapi's header injection. Its decorator wrapper injects the
`X-RateLimit-*` headers into the handler's `response` parameter, and raises
`parameter 'response' must be an instance of starlette.responses.Response`
when the handler does not declare one. Eleven handlers did not. They had
never needed it, because the limiter had not been running.

The failure only appears on the SUCCESS path — an endpoint that raises an
HTTPException short-circuits before injection — so the industry routes
looked healthy while they were answering 503 "taxonomy not imported", and
started returning 500 the moment the taxonomy imported and they had
something real to return.

Two reasons the existing suite missed it. `conftest` disables the limiter
for the whole run, so the injection path never executed. And the tests that
do enable it covered two routes, both of which happened to declare the
parameter.

So this asserts the property across EVERY rate-limited route, not a sample.
"""
from __future__ import annotations

import inspect

import pytest
from fastapi.testclient import TestClient
from starlette.responses import Response

from app.main import app
from app.rate_limit import limiter


def _rate_limited_endpoints():
    """Every endpoint slowapi will apply a per-route limit to.

    Walks nested routers, because `include_router` hides routes from a
    one-level scan — the same thing that made the limiter inert in the
    first place.
    """
    found, seen = [], set()

    def walk(routes):
        for route in routes:
            endpoint = getattr(route, "endpoint", None)
            if endpoint is not None and id(endpoint) not in seen:
                seen.add(id(endpoint))
                wrapped = getattr(endpoint, "__wrapped__", None) is not None
                name = f"{endpoint.__module__}.{endpoint.__name__}"
                # `@limiter.exempt` wraps too, but an exempt route is never
                # limited and never receives the headers, so it needs no
                # `response` parameter. The billing webhook is the one.
                exempt = name in getattr(limiter, "_exempt_routes", set())
                if wrapped and not exempt:
                    found.append((getattr(route, "path", "?"), endpoint))
            nested = getattr(route, "routes", None)
            if nested is None:
                nested = getattr(getattr(route, "original_router", None), "routes", None)
            if nested:
                walk(list(nested))

    walk(list(app.routes))
    return found


def test_every_rate_limited_handler_can_receive_the_headers():
    """slowapi injects into a `response` parameter and raises without one.

    Asserted structurally because the failure is structural, and because
    exercising all 25 routes for real would need auth and seeded data that
    would make the guard fragile rather than thorough.
    """
    endpoints = _rate_limited_endpoints()
    assert endpoints, "found no rate-limited routes — the walk is broken, not the app"

    offenders = []
    for path, endpoint in endpoints:
        params = inspect.signature(endpoint).parameters
        declares = any(
            p.annotation is Response or getattr(p.annotation, "__name__", "") == "Response"
            or (isinstance(p.annotation, str) and p.annotation.endswith("Response"))
            for p in params.values()
        )
        if not declares:
            offenders.append(f"{endpoint.__module__}.{endpoint.__name__} ({path})")

    assert not offenders, (
        "these rate-limited handlers declare no `response: Response` parameter, so "
        "slowapi raises while injecting its headers and the route answers 500 on its "
        "success path:\n  " + "\n  ".join(offenders)
    )


@pytest.fixture()
def limiting():
    previous = limiter.enabled
    limiter.enabled = True
    limiter.reset()
    try:
        yield
    finally:
        limiter.reset()
        limiter.enabled = previous


def test_a_rate_limited_route_answers_its_success_path_with_the_limiter_on(limiting):
    """The behavioural half, on a route that needs no auth or seeding.

    `/api/public/config` is rate limited and returns a body rather than an
    error, so it exercises exactly the injection path that broke.
    """
    client = TestClient(app)
    resp = client.get("/api/public/config")
    assert resp.status_code == 200, (
        f"a rate-limited route 500s on its success path with the limiter on: {resp.text[:300]}"
    )
    assert "x-ratelimit-limit" in resp.headers, "the headers the injection exists to add are missing"
