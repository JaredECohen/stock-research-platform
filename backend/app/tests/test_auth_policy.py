"""Route classification and the fail-closed matrix (FEAT-002, S1).

The coverage test is the one that matters: every OpenAPI path must be
named in `auth/policy.py`, so a new route cannot ship without saying
whether the login wall fronts it.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api import admin_auth
from app.auth import policy
from app.auth.middleware import customer_auth_middleware
from app.config import settings
from app.main import app
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth, new_email, new_sub

ADMIN_TOKEN = "admin-token-policy-tests"


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture()
def auth_on(monkeypatch, clerk):
    yield from enable_auth(monkeypatch, clerk)


@pytest.fixture()
def client():
    return TestClient(app)


def _openapi_operations() -> list[tuple[str, str]]:
    spec = app.openapi()["paths"]
    return sorted({
        (m.upper(), path) for path, ops in spec.items() for m in ops if m.upper() not in ("HEAD", "OPTIONS")
    })


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_every_openapi_route_is_explicitly_classified():
    ops = _openapi_operations()
    assert ops, "no routes in the OpenAPI schema — looking in the wrong place"
    unclassified = [
        f"{m} {p}" for m, p in ops
        if not policy.lookup(m, policy.templated_to_concrete(p))[1]
    ]
    assert not unclassified, (
        "routes not named in auth/policy.py (default-deny applies, but say so):\n  "
        + "\n  ".join(unclassified)
    )


def test_unknown_api_route_defaults_to_authenticated():
    pol, explicit = policy.lookup("GET", "/api/some/new/thing")
    assert pol.level == policy.AUTHENTICATED and not explicit
    assert not pol.is_public


def test_non_api_paths_are_public():
    for path in ("/", "/pricing", "/app/research", "/assets/main.js"):
        assert policy.classify("GET", path).is_public, path


def test_admin_paths_belong_to_admin_auth():
    for m, p in _openapi_operations():
        concrete = policy.templated_to_concrete(p)
        if admin_auth.is_protected(m, concrete):
            assert policy.classify(m, concrete).level == policy.ADMIN, f"{m} {p}"
        else:
            assert policy.classify(m, concrete).level != policy.ADMIN, f"{m} {p}"


def test_browser_called_admin_exemptions_are_customer_routes():
    assert policy.classify("POST", "/api/admin/ui-log").is_public
    assert policy.classify("GET", "/api/admin/track-record").level == policy.PRO
    assert policy.classify("POST", "/api/admin/evaluate-outcomes").level == policy.PRO
    assert policy.classify("GET", "/api/admin/dcf-versions/NVDA").level == policy.PRO
    assert policy.classify("GET", "/api/admin/lopsidedness-audit").level == policy.PRO


def test_public_allowlist_is_exactly_what_the_plan_says():
    public = {
        (m, p) for m, p in _openapi_operations()
        if policy.classify(m, policy.templated_to_concrete(p)).is_public
    }
    assert public == {
        ("GET", "/health"),
        ("GET", "/api/providers/status"),
        ("GET", "/api/public/config"),
        ("GET", "/api/public/samples"),
        ("GET", "/api/public/samples/{ticker}"),
        ("POST", "/api/public/events"),
        ("POST", "/api/admin/ui-log"),
    }, sorted(public)


def test_metered_routes_name_a_registered_feature():
    from app.auth import features
    for _m, _template, pol in policy.ROUTES:
        if pol.feature:
            features.get(pol.feature)  # raises on a typo


def test_options_is_always_public():
    assert policy.classify("OPTIONS", "/api/admin/cron-health").is_public
    assert policy.classify("OPTIONS", "/api/me").is_public


# ---------------------------------------------------------------------------
# Fail-closed matrix
# ---------------------------------------------------------------------------

@pytest.fixture()
def auth_on_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "clerk_issuer", "")
    monkeypatch.setattr(settings, "clerk_jwks_url", "")
    yield


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/stocks"),
    ("GET", "/api/me"),
    ("GET", "/api/stocks/NVDA/memo"),
    ("POST", "/api/chat"),
])
def test_unconfigured_wall_refuses_protected_routes_with_503(auth_on_unconfigured, client, method, path):
    resp = client.request(method, path, json={} if method == "POST" else None)
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"]["code"] == "auth_unavailable"


@pytest.mark.parametrize("path", ["/health", "/api/public/config", "/api/providers/status"])
def test_unconfigured_wall_keeps_public_routes_up(auth_on_unconfigured, client, path):
    assert client.get(path).status_code == 200


def test_unconfigured_wall_ignores_any_bearer(auth_on_unconfigured, client):
    """No keys means nothing can be verified, so a token changes nothing."""
    resp = client.get("/api/me", headers=bearer("eyJ.x.y"))
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Configured wall
# ---------------------------------------------------------------------------

def test_signed_in_user_reaches_free_routes(auth_on, client):
    resp = client.get("/api/stocks", headers=bearer(auth_on.token()))
    assert resp.status_code == 200
    # Bare TestClient, no lifespan: the universe may be unseeded, so only
    # the shape is asserted — the point is that the wall let us through.
    assert isinstance(resp.json(), list)


def test_free_user_is_refused_on_pro_routes_before_the_handler(auth_on, client, monkeypatch):
    """No trial (unverified email → bootstrap never starts one), so Free."""
    tok = auth_on.token(verified=False)
    resp = client.get("/api/stocks/NVDA/memos", headers=bearer(tok))
    assert resp.status_code == 402, resp.text
    body = resp.json()["detail"]
    assert body["code"] == "plan_required"
    assert body["feature"] == "memo_history"
    assert body["upgrade_url"] == "/pricing"


def test_trial_user_reaches_pro_routes(auth_on, client):
    tok = auth_on.token(verified=True, email=new_email())
    assert client.post("/api/me/bootstrap", headers=bearer(tok)).json()["plan"]["plan"] == "pro"
    resp = client.get("/api/stocks/NVDA/memos", headers=bearer(tok))
    assert resp.status_code == 200, resp.text


def test_suspended_account_is_403_everywhere_protected(auth_on, client):
    from app.database import SessionLocal
    from app.models import User

    sub = new_sub()
    tok = auth_on.token(sub=sub)
    assert client.get("/api/me", headers=bearer(tok)).status_code == 200
    with SessionLocal() as db:
        u = db.query(User).filter(User.external_id == sub).one()
        u.account_state = "suspended"
        db.commit()
    resp = client.get("/api/stocks", headers=bearer(tok))
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "account_suspended"
    assert client.get("/api/public/config", headers=bearer(tok)).status_code == 200


# ---------------------------------------------------------------------------
# Wall off: behaviour-preserving, principal still attached
# ---------------------------------------------------------------------------

def _mini_app() -> FastAPI:
    mini = FastAPI()
    mini.middleware("http")(customer_auth_middleware)

    @mini.get("/api/probe")
    def probe(request: Request):
        p = request.state.principal
        return {"kind": p.kind, "auth_disabled": p.auth_disabled, "is_user": p.is_user}

    return mini


def test_wall_off_attaches_an_anonymous_principal_and_passes(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    resp = TestClient(_mini_app()).get("/api/probe", headers=bearer("whatever"))
    assert resp.status_code == 200
    assert resp.json() == {"kind": "anon", "auth_disabled": True, "is_user": False}


def test_wall_off_leaves_today_routes_untouched(client, monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    assert client.get("/api/stocks").status_code == 200
    assert client.get("/api/stocks/NVDA/memos").status_code == 200
    resp = client.get("/api/me")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "feature_disabled"


def test_wall_on_attaches_a_user_principal(auth_on):
    resp = TestClient(_mini_app()).get("/api/probe", headers=bearer(auth_on.token()))
    assert resp.status_code == 200
    assert resp.json() == {"kind": "user", "auth_disabled": False, "is_user": True}
