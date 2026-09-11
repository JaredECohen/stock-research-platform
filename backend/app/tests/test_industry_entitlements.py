"""FEAT-003 slice 5 — Industry Analysis under the login wall.

The feature has two gates and they must never disagree:

* `auth/policy.py` classifies each `/api/industries/*` path, and the
  customer middleware refuses before the handler runs;
* `api/entitlements_industry.py` asks the same question inside the
  handler and builds the `access` block every response carries.

Both read their answer from `industry_report_store.access_policy()`, so
the tests below check the two together rather than one at a time: a slice
that moved a tier in one place and not the other would ship a page that
says "Pro" and answers to everyone, or the reverse.

What is pinned here:

* the wall OFF (today's deployment) — every surface answers, and the
  block says `enforced: false` rather than implying entitlement;
* the wall ON — `latest` follows `INDUSTRY_ANALYSIS_ACCESS`, history /
  changes / snapshot are Pro, anonymous is 401 and Free is 402;
* `/taxonomy` answers in every combination, including
  `INDUSTRY_ANALYSIS_ACCESS=pro`, because it is how a signed-out visitor
  learns what the surfaces behind it cost. It reports the tier with
  `route_gated: false` — the price list, not a receipt;
* reading a report is not metered: the LLM was paid for once, when the
  worker generated the edition, and no reader is charged again;
* the ops routes are `admin`, refused without the bearer token even for
  a Pro customer.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.api import entitlements_industry as seam
from app.auth import policy as pol
from app.config import settings
from app.main import app
from app.rate_limit import limiter
from app.services import gics_registry as reg
from app.services import industry_report_store as store
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth
from app.tests.gating_helpers import (
    assert_structured,
    free_user,
    pro_user,
    purge_rate_windows,
    usage_events,
    user_id_for,
)

PRO_SURFACE_PATHS = ("history", "changes")


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture()
def auth_on(monkeypatch, clerk):
    purge_rate_windows()
    yield from enable_auth(monkeypatch, clerk)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def taxonomy():
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None, "the bundled knowledge JSON must import for these tests"
    yield info
    reg.activate_version(info.version_key)


@pytest.fixture(autouse=True)
def no_ip_limiter(monkeypatch):
    """slowapi's per-IP ceiling is keyed on the address, and every test in
    this module calls the same handful of paths from the one TestClient
    address. Switched off here so a 429 cannot masquerade as an
    entitlement refusal; `test_every_industry_route_declares_an_ip_limit`
    is what proves the ceiling still exists."""
    monkeypatch.setattr(limiter, "enabled", False)


@pytest.fixture(scope="module")
def group(taxonomy):
    return reg.industry_groups(version=taxonomy)[0]


def _paths(group) -> dict[str, str]:
    return {
        "taxonomy": "/api/industries/taxonomy",
        "latest": f"/api/industries/{group.code}/report",
        "companies": f"/api/industries/{group.code}/companies",
        "history": f"/api/industries/{group.code}/history",
        "changes": f"/api/industries/{group.code}/changes",
        "snapshot": "/api/industries/snapshot",
    }


# ---------------------------------------------------------------------------
# The wall off — today's deployment
# ---------------------------------------------------------------------------

def test_with_the_wall_off_the_taxonomy_describes_the_policy_without_claiming_it(client):
    body = client.get("/api/industries/taxonomy").json()
    access = body["access"]
    assert access["enforced"] is False, "AUTH_ENABLED is off; nothing is gated"
    assert access["route_gated"] is False, "/taxonomy never applies the tier"
    assert access["tier"] == "public" and access["required_tier"] is None
    # The block is a price list: it names every surface, so a signed-out
    # UI can say "history is Pro" without guessing.
    assert access["surfaces"]["history"] == "pro"
    assert access["surfaces"]["changes"] == "pro"
    assert access["surfaces"]["pm_chat"] == "pro"


def test_with_the_wall_off_a_pro_surface_answers_and_still_names_its_tier(client, group):
    body = client.get(_paths(group)["history"]).json()
    access = body["access"]
    assert access["tier"] == "pro" and access["required_tier"] == "pro"
    assert access["allowed"] is True and access["route_gated"] is True
    assert access["enforced"] is False, "served, but not because the caller was entitled"


# ---------------------------------------------------------------------------
# The wall on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("surface", PRO_SURFACE_PATHS)
def test_anonymous_is_401_on_the_pro_surfaces(auth_on, client, group, surface):
    resp = client.get(_paths(group)[surface])
    assert_structured(resp, code="auth_required", status=401)
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_anonymous_is_401_on_the_cross_industry_snapshot(auth_on, client):
    assert_structured(client.get("/api/industries/snapshot"), code="auth_required", status=401)


def test_anonymous_still_reads_the_taxonomy_and_the_public_surfaces(auth_on, client, group):
    paths = _paths(group)
    body = client.get(paths["taxonomy"]).json()
    assert body["access"]["enforced"] is True, "the wall is up and the block must say so"
    assert body["access"]["tier"] == "public"
    # `latest` is public by default (owner decision 1), so the report and
    # the constituent list answer without a token. 404 is a data state,
    # not a refusal — what matters is that neither is a 401.
    for key in ("latest", "companies"):
        assert client.get(paths[key]).status_code in (200, 404), key


@pytest.mark.parametrize("surface", PRO_SURFACE_PATHS)
def test_free_is_402_on_the_pro_surfaces(auth_on, client, group, surface):
    _sub, tok = free_user(auth_on)
    detail = assert_structured(
        client.get(_paths(group)[surface], headers=bearer(tok)), code="plan_required", status=402)
    assert detail["feature"] == seam.FEATURE
    assert detail["plan"] == "free"
    assert detail["upgrade_url"] == "/pricing"


def test_pro_reads_a_pro_surface_and_the_block_names_the_plan(auth_on, client, group):
    _sub, tok = pro_user(client, auth_on)
    resp = client.get(_paths(group)["history"], headers=bearer(tok))
    assert resp.status_code == 200, resp.text
    access = resp.json()["access"]
    assert access["tier"] == "pro" and access["required_tier"] == "pro"
    assert access["enforced"] is True and access["allowed"] is True
    assert access["route_gated"] is True
    assert access["plan"] == "pro"


def test_reading_industry_analysis_is_not_metered(auth_on, client, group):
    """The edition was paid for once, when the worker generated it. A
    reader is gated by plan and charged nothing — no `usage_events` row,
    so a Pro customer cannot exhaust an allowance by reading."""
    _sub, tok = pro_user(client, auth_on)
    uid = user_id_for(client, tok)
    for key in ("taxonomy", "history"):
        client.get(_paths(group)[key], headers=bearer(tok))
    assert usage_events(uid, seam.FEATURE) == []


# ---------------------------------------------------------------------------
# INDUSTRY_ANALYSIS_ACCESS moves the `latest` surface
# ---------------------------------------------------------------------------

def test_setting_access_to_pro_moves_the_latest_surface_behind_the_wall(
    auth_on, client, group, monkeypatch,
):
    monkeypatch.setattr(settings, "industry_analysis_access", "pro")
    paths = _paths(group)
    for key in ("latest", "companies"):
        assert_structured(client.get(paths[key]), code="auth_required", status=401)
    # …and the taxonomy still answers, because a 401 there would leave the
    # UI with nothing to explain the gate with.
    body = client.get(paths["taxonomy"]).json()
    assert body["access"]["tier"] == "pro"
    assert body["access"]["required_tier"] == "pro"
    assert body["access"]["route_gated"] is False
    assert body["access"]["setting"] == "pro"


def test_an_unreadable_access_setting_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "industry_analysis_access", "nonsense")
    assert seam.tier_for(seam.SURFACE_LATEST) == store.PRO


def test_an_unknown_surface_fails_closed():
    assert seam.tier_for("a-surface-nobody-defined") == store.PRO


# ---------------------------------------------------------------------------
# The two gates agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("setting", ["public", "pro"])
def test_auth_policy_and_the_store_cannot_disagree_about_a_surface(monkeypatch, setting):
    """`auth/policy.py` (the wall) and `entitlements_industry` (the
    handler) resolve the same tier from the same function, under either
    value of the setting."""
    monkeypatch.setattr(settings, "industry_analysis_access", setting)
    for method, template, surface in pol.INDUSTRY_SURFACES:
        path = pol.templated_to_concrete(template)
        policy, explicit = pol.lookup(method, path)
        assert explicit, f"{method} {path} is not named in auth/policy.py"
        if path == pol.INDUSTRY_TAXONOMY_PATH:
            assert policy.level == pol.PUBLIC, "the taxonomy must answer in every configuration"
            continue
        expected = store.surface_tier(surface)
        if expected == store.PUBLIC:
            assert policy.level == pol.PUBLIC, f"{path} should be public while {surface} is"
        else:
            assert policy.level == pol.PRO, f"{path} should be Pro while {surface} is"
            assert policy.feature == seam.FEATURE


def test_the_seam_refuses_with_the_surface_and_the_tier_it_needs(monkeypatch):
    """The handler dependency is the second gate: the wall refuses first
    in a real request, so this calls `enforce` directly to prove the
    refusal it would raise names the surface and the tier — the two
    things a client needs to render "this is a Pro surface"."""
    monkeypatch.setattr(settings, "auth_enabled", True)
    request = Request({
        "type": "http", "method": "GET", "path": "/api/industries/1010/history",
        "headers": [], "query_string": b"",
    })
    with pytest.raises(Exception) as exc_info:
        seam.enforce(request, seam.SURFACE_HISTORY)
    exc = exc_info.value
    assert exc.status_code == 401
    assert exc.detail["code"] == "auth_required"
    assert exc.detail["feature"] == seam.FEATURE
    assert exc.detail["surface"] == seam.SURFACE_HISTORY
    assert exc.detail["required_tier"] == "pro"
    assert exc.headers.get("WWW-Authenticate") == "Bearer"


# ---------------------------------------------------------------------------
# The ops surface
# ---------------------------------------------------------------------------

ADMIN_ROUTES = (
    ("POST", "/api/admin/industries/taxonomy/import"),
    ("POST", "/api/admin/industries/classify"),
    ("POST", "/api/admin/industries/reports/regenerate"),
    ("GET", "/api/admin/industries/jobs"),
)


@pytest.mark.parametrize("method,path", ADMIN_ROUTES)
def test_ops_routes_are_classified_admin(method, path):
    policy, explicit = pol.lookup(method, path)
    assert explicit and policy.level == pol.ADMIN


@pytest.fixture()
def admin_token(monkeypatch):
    """`admin_auth` only guards the prefix once `ADMIN_API_TOKEN` is set —
    unset (the test and local-dev default) it lets everything through and
    logs, so a test that did not set one would prove nothing."""
    token = "admin-token-industry-entitlements-tests"
    monkeypatch.setattr(settings, "admin_api_token", token)
    return token


@pytest.mark.parametrize("method,path", ADMIN_ROUTES)
def test_a_pro_customer_token_does_not_open_an_ops_route(auth_on, admin_token, client, method, path):
    """A customer JWT is not an admin credential, whatever the customer's
    plan: the ops prefix takes the operator's bearer token and nothing
    else. The Pro token is one the customer routes accept, which is what
    makes the refusal here meaningful."""
    _sub, tok = pro_user(client, auth_on)
    resp = client.request(method, path, headers=bearer(tok), json={})
    assert resp.status_code == 401, resp.text
    # …and the operator's own token does open it (any answer but 401
    # proves the guard is the token, not the route being broken).
    ok = client.request(method, path, headers={"Authorization": f"Bearer {admin_token}"}, json={})
    assert ok.status_code != 401, ok.text


# ---------------------------------------------------------------------------
# The IP ceiling this module switches off
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module,handler,limit_key", [
    ("routes_industries", "get_industry_taxonomy", "industry_read"),
    ("routes_industries", "get_industry_report", "industry_read"),
    ("routes_industries", "get_industry_history", "industry_read"),
    ("routes_industries", "get_industry_changes", "industry_read"),
    ("routes_industries", "get_industry_companies", "industry_read"),
    ("routes_industries", "get_industry_snapshot", "industry_read"),
    ("routes_industries_admin", "import_taxonomy_endpoint", "industry_admin"),
    ("routes_industries_admin", "classify_endpoint", "industry_admin"),
    ("routes_industries_admin", "regenerate_reports_endpoint", "industry_admin"),
])
def test_every_industry_route_declares_an_ip_limit(module, handler, limit_key):
    from limits import parse

    from app.rate_limit import LIMITS

    route_limits = limiter._route_limits[f"app.api.{module}.{handler}"]
    assert [str(item.limit) for item in route_limits] == [str(parse(LIMITS[limit_key]))]
