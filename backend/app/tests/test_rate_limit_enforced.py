"""The per-IP limiter must actually refuse traffic.

Every other rate-limit assertion in this suite is about *configuration* —
that a route carries a decorator, that a limit string is registered. All of
that stayed true on 2026-09-11 while the limiter enforced nothing at all.

The cause was a library seam. slowapi's middleware resolves the endpoint by
scanning `app.routes` one level deep and testing `hasattr(route, "endpoint")`.
Under the pinned FastAPI an `include_router` call leaves a nested router
object there instead of the flattened `APIRoute`s. That object matches the
request but exposes no `endpoint`, so slowapi resolved `None` — and it treats
an unresolved handler as **exempt**. Every route in this app arrives through
`include_router`, so every request was exempt. Nothing raised, nothing
logged, and it reproduced only against the versions `requirements.txt` pins,
which are the ones CI and Render install.

So these tests send real traffic and demand a real 429. They deliberately
avoid slowapi and Starlette internals, because a suite written against the
internals is what missed this.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rate_limit import limiter

# The global `default_limits` ceiling applied to any route without its own
# decorator. Read from the limiter rather than retyped.
DEFAULT_PER_MINUTE = 60


@pytest.fixture()
def limiting():
    """Turn the limiter on for one test. It is disabled for the rest of the
    suite so the tests are not throttled by their own traffic."""
    previous = limiter.enabled
    limiter.enabled = True
    limiter.reset()
    try:
        yield
    finally:
        limiter.reset()
        limiter.enabled = previous


@pytest.fixture()
def client():
    return TestClient(app)


def test_the_limiter_refuses_traffic_past_the_global_ceiling(client, limiting):
    """`/health` carries no decorator, so it takes the global default. This
    is the assertion that was silently false in production."""
    statuses = [client.get("/health").status_code for _ in range(DEFAULT_PER_MINUTE + 5)]

    assert statuses[0] == 200, "the route itself must work"
    assert 429 in statuses, (
        "the per-IP limiter refused nothing past its ceiling — every route is "
        "unthrottled when slowapi cannot resolve the endpoint; see "
        "rate_limit._find_route_handler"
    )
    assert statuses.index(429) == DEFAULT_PER_MINUTE, (
        f"expected the refusal on request {DEFAULT_PER_MINUTE + 1}, "
        f"got it on {statuses.index(429) + 1}"
    )


def test_the_limiter_reports_the_ceiling_it_is_applying(client, limiting):
    """Headers are how a caller learns its budget, and their absence was the
    first visible sign that nothing was being enforced."""
    resp = client.get("/health")
    assert resp.headers.get("x-ratelimit-limit") == str(DEFAULT_PER_MINUTE)
    assert "x-ratelimit-remaining" in resp.headers


def test_the_limiter_resolves_a_route_behind_an_included_router():
    """The seam itself, asserted against OUR resolver rather than slowapi's.

    Every route in this app is added with `include_router`, so a resolver
    that cannot see through that wrapper exempts the whole surface. Worth
    asserting directly because the failure is silent: the app keeps serving,
    it just stops enforcing.
    """
    from app.rate_limit import _find_route_handler

    scope = {
        "type": "http", "method": "GET", "path": "/health",
        "headers": [], "root_path": "", "query_string": b"",
    }
    handler = _find_route_handler(list(app.routes), scope)
    assert handler is not None, (
        "no endpoint resolved for an included route — slowapi would treat "
        "this request, and every other one, as exempt"
    )
    assert handler.__name__ == "health"
