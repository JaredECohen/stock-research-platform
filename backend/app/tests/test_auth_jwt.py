"""Clerk JWT verification and the login-wall middleware (FEAT-002, S1).

Every test runs against a locally generated RSA keypair and a JWKS
document served by `ClerkStub.fetch` — no network, no Clerk tenant.
"""
from __future__ import annotations

import logging
import time

import pytest
from fastapi.testclient import TestClient

from app.auth import jwks, sanitize
from app.config import settings
from app.main import app
from app.tests.auth_helpers import ClerkStub, assert_no_secrets_in_logs, bearer, enable_auth

ADMIN_TOKEN = "test-admin-token-for-auth-separation"


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture()
def auth_on(monkeypatch, clerk):
    yield from enable_auth(monkeypatch, clerk)


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _no_secrets_in_logs(caplog):
    """Every test in this module doubles as a log-leak check."""
    caplog.set_level(logging.DEBUG)
    yield
    assert_no_secrets_in_logs(caplog.records)


# ---------------------------------------------------------------------------
# verify_token
# ---------------------------------------------------------------------------

def test_valid_token_yields_claims(auth_on):
    tok = auth_on.token(sub="user_ok", email="a@example.com", verified=True)
    claims = jwks.verify_token(tok)
    assert claims["sub"] == "user_ok"
    assert claims["email"] == "a@example.com"
    assert claims["email_verified"] is True


def test_expiry_honours_leeway(auth_on):
    now = int(time.time())
    # 30s past exp is inside the 60s leeway; 120s past is not.
    inside = auth_on.token(exp_in=-30, iat=now - 3600)
    outside = auth_on.token(exp_in=-120, iat=now - 3600)
    assert jwks.verify_token(inside)["sub"]
    with pytest.raises(jwks.TokenInvalid) as ei:
        jwks.verify_token(outside)
    assert "expired" in ei.value.message


@pytest.mark.parametrize("kwargs,reason", [
    ({"iss": "https://someone-else.clerk.accounts.dev"}, "issuer"),
    ({"azp": "https://evil.example"}, "party"),
    ({"azp": None}, "party"),
    ({"alg": "none"}, "algorithm"),
    ({"alg": "HS256"}, "algorithm"),
])
def test_bad_tokens_are_refused(auth_on, kwargs, reason):
    with pytest.raises(jwks.TokenInvalid) as ei:
        jwks.verify_token(auth_on.token(**kwargs))
    assert reason in ei.value.message


def test_signature_from_another_key_is_refused(auth_on):
    other = ClerkStub(kid=auth_on.kid)  # same kid, different private key
    with pytest.raises(jwks.TokenInvalid):
        jwks.verify_token(other.token())


@pytest.mark.parametrize("garbage", ["", "abc", "a.b", "a.b.c", "Bearer x.y.z"])
def test_malformed_tokens_are_refused_without_a_fetch(auth_on, garbage):
    with pytest.raises(jwks.TokenInvalid):
        jwks.verify_token(garbage)
    assert auth_on.fetch_calls == 0


def test_unknown_kid_forces_one_throttled_refresh(auth_on):
    assert jwks.verify_token(auth_on.token())["sub"]
    assert auth_on.fetch_calls == 1
    with pytest.raises(jwks.TokenInvalid) as ei:
        jwks.verify_token(auth_on.token(kid="kid-rotated"))
    assert "unknown" in ei.value.message
    assert auth_on.fetch_calls == 2, "an unknown kid must trigger exactly one refresh"
    with pytest.raises(jwks.TokenInvalid):
        jwks.verify_token(auth_on.token(kid="kid-rotated"))
    assert auth_on.fetch_calls == 2, "refreshes on unknown kids are throttled to one per minute"


def test_rotated_key_is_picked_up_by_the_forced_refresh(auth_on):
    assert jwks.verify_token(auth_on.token())["sub"]
    auth_on.kid = "kid-2"  # tenant rotates; the JWKS now serves kid-2
    assert jwks.verify_token(auth_on.token(kid="kid-2"))["sub"]


def test_stale_keys_keep_verifying_while_jwks_is_down(auth_on):
    assert jwks.verify_token(auth_on.token())["sub"]
    auth_on.fail_fetch = True
    jwks._STATE["fetched_at"] -= jwks.LIFESPAN + 10  # due for refresh
    assert jwks.verify_token(auth_on.token())["sub"], "stale-while-error"
    assert auth_on.fetch_calls == 2


def test_no_keys_and_jwks_down_is_unavailable_not_invalid(auth_on):
    auth_on.fail_fetch = True
    with pytest.raises(jwks.AuthUnavailable):
        jwks.verify_token(auth_on.token())


def test_keys_older_than_a_day_stop_verifying(auth_on):
    assert jwks.verify_token(auth_on.token())["sub"]
    auth_on.fail_fetch = True
    jwks._STATE["fetched_at"] -= jwks.STALE_MAX + 10
    jwks._STATE["last_attempt"] = 0.0
    with pytest.raises(jwks.AuthUnavailable):
        jwks.verify_token(auth_on.token())


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

def test_valid_bearer_reaches_a_customer_route(auth_on, client):
    resp = client.get("/api/me", headers=bearer(auth_on.token(sub="user_mw_1")))
    assert resp.status_code == 200, resp.text
    assert resp.json()["user"]["external_id"] == "user_mw_1"


def test_expired_bearer_is_401_invalid(auth_on, client):
    resp = client.get("/api/me", headers=bearer(auth_on.token(exp_in=-600)))
    assert resp.status_code == 401
    assert resp.json()["detail"]["code"] == "auth_invalid"
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_missing_bearer_is_401_required(auth_on, client):
    resp = client.get("/api/stocks")
    assert resp.status_code == 401
    assert resp.json()["detail"]["code"] == "auth_required"


def test_token_in_query_string_is_ignored(auth_on, client):
    tok = auth_on.token()
    assert client.get("/api/me", params={"token": tok}).status_code == 401
    assert client.get("/api/me", params={"authorization": f"Bearer {tok}"}).status_code == 401


def test_query_credentials_never_reach_ui_logs(auth_on, client):
    """The request logger persists query params; credential-shaped keys
    are redacted before the row is written."""
    from app.database import SessionLocal
    from app.models import UILog

    tok = auth_on.token()
    marker = "leak-probe-" + tok[-12:]
    client.get("/api/public/config", params={"token": tok, "probe": marker})
    with SessionLocal() as db:
        row = (
            db.query(UILog).filter(UILog.path == "/api/public/config", UILog.source == "backend")
            .order_by(UILog.id.desc()).first()
        )
    assert row is not None
    assert row.payload["query"]["probe"] == marker
    assert row.payload["query"]["token"] == "<redacted>"
    assert tok not in str(row.payload)


def test_bad_token_on_a_public_route_is_treated_as_anonymous(auth_on, client):
    """A stale token in the browser must not break the landing page."""
    resp = client.get("/api/public/config", headers=bearer("eyJ.garbage.token"))
    assert resp.status_code == 200


def test_cors_preflight_is_not_challenged(auth_on, client):
    resp = client.options("/api/me", headers={
        "Origin": settings.cors_origins_list[0],
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    })
    assert resp.status_code == 200, resp.text
    assert "authorization" in resp.headers.get("access-control-allow-headers", "").lower()


def test_health_and_public_config_need_no_token(auth_on, client):
    assert client.get("/health").status_code == 200
    assert client.get("/api/public/config").status_code == 200


# ---------------------------------------------------------------------------
# Admin token and customer JWT never cross
# ---------------------------------------------------------------------------

def test_admin_token_does_not_satisfy_a_customer_route(auth_on, client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", ADMIN_TOKEN)
    resp = client.get("/api/me", headers=bearer(ADMIN_TOKEN))
    assert resp.status_code == 401
    assert resp.json()["detail"]["code"] == "auth_invalid"


def test_customer_jwt_does_not_satisfy_an_admin_route(auth_on, client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", ADMIN_TOKEN)
    resp = client.get("/api/admin/cron-health", headers=bearer(auth_on.token()))
    assert resp.status_code == 401
    assert resp.json()["detail"] == "unauthorized"  # admin_auth's answer, not ours
    # and the admin token still works there
    assert client.get("/api/admin/cron-health", headers=bearer(ADMIN_TOKEN)).status_code == 200


def test_customer_middleware_never_touches_admin_paths_when_unconfigured(client, monkeypatch):
    """Fail-closed must not swallow the admin surface: with AUTH_ENABLED
    but no Clerk config, the admin token still opens /api/admin/*."""
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "clerk_issuer", "")
    monkeypatch.setattr(settings, "clerk_jwks_url", "")
    monkeypatch.setattr(settings, "admin_api_token", ADMIN_TOKEN)
    assert client.get("/api/admin/cron-health", headers=bearer(ADMIN_TOKEN)).status_code == 200
    assert client.get("/api/admin/cron-health").status_code == 401


# ---------------------------------------------------------------------------
# Secrets never reach the log stream
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("secret", [
    "Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6ImtpZC0xIn0.eyJzdWIiOiJ1c2VyXzEifQ.c2lnbmF0dXJl",
    "sk_live_51H9abcdefghijklmnop",
    "sk_test_51H9abcdefghijklmnop",
    "whsec_9f8e7d6c5b4a3210fedcba",
    "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhYmMifQ.ZmFrZXNpZ25hdHVyZQ",
])
def test_redact_masks_every_credential_shape(secret):
    out = sanitize.redact(f"failed with {secret} in the request")
    for needle in ("sk_", "whsec_", "eyJ"):
        assert needle not in out, out
    assert "Bearer <redacted>" in out or "Bearer" not in out


def test_safe_logger_filter_masks_a_careless_log_line(caplog):
    log = sanitize.safe_logger("app.auth.test_probe")
    tok = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhYmMifQ.ZmFrZXNpZ25hdHVyZQ"
    log.warning("token %s rejected", tok)
    log.warning("header was Bearer %s", tok)
    log.warning("stripe key sk_test_abcdefghijklmnop failed")
    assert caplog.records, "records should be captured"
    # the autouse fixture asserts the forbidden patterns are absent


def test_middleware_logs_carry_no_token(auth_on, client, caplog):
    tok = auth_on.token(exp_in=-600)
    client.get("/api/me", headers=bearer(tok))
    client.get("/api/me", headers=bearer("garbage-not-a-jwt"))
    for rec in caplog.records:
        assert tok not in rec.getMessage()
