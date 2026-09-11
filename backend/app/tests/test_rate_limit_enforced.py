"""The per-IP limiter must actually refuse traffic.

Every other rate-limit test in this suite asserts *configuration* — that a
route carries a decorator, that a limit string is registered, that the
exempt list holds the webhook. All of that stayed true on 2026-09-11 while
the limiter enforced nothing at all.

The cause was a library seam. slowapi's middleware resolves the endpoint by
scanning `app.routes` one level deep; under the pinned FastAPI an
`include_router` call leaves a nested router object there instead of the
flattened `APIRoute`s (22 entries where the older FastAPI had 106). That
object matches the request but exposes no `endpoint`, so slowapi resolved
`None` — and it treats an unresolved handler as **exempt**. Nothing raised,
nothing logged: every request simply skipped the limiter. It reproduced only
against the pinned versions, which are what CI and Render install, so a
developer box on the older FastAPI passed the whole suite.

So these tests are deliberately behavioural: send real traffic, demand a
real 429. They must not reference a slowapi or Starlette internal, because a
test written against the internals is exactly what failed to notice.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rate_limit import LIMITS, limiter


@pytest.fixture()
def limiting():
    """Turn the limiter on for one test. `conftest` disables it globally so
    the rest of the suite is not throttled by its own traffic."""
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


def _limit_count(name: str) -> int:
    """The numerator of a configured limit, e.g. '60/minute' -> 60."""
    return int(LIMITS[name].split("/")[0])


def test_a_decorated_route_refuses_traffic_past_its_limit(client, limiting):
    """`/api/public/config` carries the `public_get` ceiling. Past it, the
    limiter must answer 429 — this is the assertion that was silently false
    in production."""
    ceiling = _limit_count("public_get")
    statuses = [client.get("/api/public/config").status_code for _ in range(ceiling + 5)]

    assert statuses[0] == 200, "the route itself must work"
    assert 429 in statuses, (
        "the per-IP limiter did not refuse anything past its ceiling. Every "
        "route is unthrottled when slowapi cannot resolve the endpoint — see "
        "rate_limit._find_route_handler."
    )
    assert statuses.index(429) == ceiling, (
        f"expected the refusal on request {ceiling + 1}, got it on {statuses.index(429) + 1}"
    )


def test_the_limiter_reports_the_ceiling_it_is_applying(client, limiting):
    """Headers are how a caller learns the budget. Their absence was the
    first visible symptom that nothing was being enforced."""
    resp = client.get("/api/public/config")
    assert resp.headers.get("x-ratelimit-limit") == str(_limit_count("public_get"))
    assert "x-ratelimit-remaining" in resp.headers


def test_an_undecorated_route_still_takes_the_global_default(client, limiting):
    """`/health` carries no decorator, so it inherits `default_limits` via
    the middleware. That path resolves the endpoint the same way, and broke
    the same way."""
    statuses = [client.get("/health").status_code for _ in range(65)]
    assert 429 in statuses, "the global default limit is not being applied"


def test_the_limiter_resolves_routes_behind_an_included_router():
    """The specific seam, asserted on OUR resolver rather than on slowapi's.

    Every API route in this app arrives through `include_router`, so a
    resolver that cannot see through that wrapper exempts the entire
    surface. Checked directly because the failure mode is silent: the app
    keeps serving, it simply stops enforcing.
    """
    from app.rate_limit import _find_route_handler

    scope = {
        "type": "http", "method": "GET", "path": "/api/public/config",
        "headers": [], "root_path": "", "query_string": b"",
    }
    handler = _find_route_handler(list(app.routes), scope)
    assert handler is not None, (
        "no endpoint resolved for an included route — slowapi would treat "
        "this request, and every other one, as exempt"
    )
    assert handler.__name__ == "public_config"
