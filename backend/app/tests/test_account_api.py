"""`/api/me*` and `/api/public/config` (FEAT-002, S1)."""
from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import User
from app.schemas.accounts import AccountOut, BootstrapOut, PublicConfigOut, UsageOut
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth, new_email, new_sub


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture()
def auth_on(monkeypatch, clerk):
    yield from enable_auth(monkeypatch, clerk)


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/public/config
# ---------------------------------------------------------------------------

def test_public_config_shape_and_cache_header(client, monkeypatch):
    monkeypatch.setattr(settings, "clerk_publishable_key", "pk_test_public_is_fine")
    resp = client.get("/api/public/config")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "public, max-age=60"
    cfg = PublicConfigOut.model_validate(resp.json())
    # Mirrors whatever the environment says (the suite also runs with
    # AUTH_ENABLED=true); the default deployment is all-off.
    assert cfg.auth_enabled is settings.auth_enabled
    assert cfg.billing_enabled is settings.billing_enabled
    assert cfg.usage_limits_enabled is settings.usage_limits_enabled
    assert cfg.sample_tickers == ["NVDA", "COST", "JPM"]
    assert cfg.prices.monthly_cents == 2999 and cfg.prices.annual_cents == 29900
    assert cfg.legal_reviewed is False
    assert cfg.features["memo_view"]["free"] == 3 and cfg.features["memo_view"]["distinct_resources"] is True
    assert cfg.features["pm_chat"]["pro"] == 300
    assert cfg.clerk_publishable_key == "pk_test_public_is_fine"


def test_public_config_never_carries_secrets(client, monkeypatch):
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_should_not_leak")
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_should_not_leak")
    body = client.get("/api/public/config").text
    assert "sk_test" not in body and "whsec_" not in body


def test_public_config_reflects_flags(client, auth_on, monkeypatch):
    monkeypatch.setattr(settings, "legal_reviewed_at", "2026-09-08")
    cfg = client.get("/api/public/config").json()
    assert cfg["auth_enabled"] is True and cfg["usage_limits_enabled"] is True
    assert cfg["clerk_frontend_api"] == settings.clerk_issuer
    assert cfg["legal_reviewed"] is True


# ---------------------------------------------------------------------------
# /api/me
# ---------------------------------------------------------------------------

def test_me_is_404_when_accounts_are_off(client, monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    resp = client.get("/api/me")
    assert resp.status_code == 404 and resp.json()["detail"]["code"] == "feature_disabled"


def test_me_shape_for_a_fresh_user(client, auth_on):
    sub = new_sub()
    resp = client.get("/api/me", headers=bearer(auth_on.token(sub=sub, email=new_email())))
    assert resp.status_code == 200, resp.text
    acct = AccountOut.model_validate(resp.json())
    assert acct.user.external_id == sub and acct.user.email_verified is True
    assert acct.plan.plan == "free" and acct.plan.source == "default"
    assert acct.user.trial_started_at is None, "a GET never starts a trial"
    assert acct.billing.has_subscription is False and acct.billing.portal_available is False
    assert acct.entitlements["memo_view"].limit == 3
    assert acct.period_key == datetime.utcnow().strftime("%Y-%m")
    assert acct.usage_limits_enabled is True
    assert "email" not in resp.json()["user"], "plaintext email is never returned"


# ---------------------------------------------------------------------------
# /api/me/bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_starts_exactly_one_trial(client, auth_on):
    tok = auth_on.token(sub=new_sub(), email=new_email(), verified=True)
    first = client.post("/api/me/bootstrap", headers=bearer(tok))
    assert first.status_code == 200, first.text
    b1 = BootstrapOut.model_validate(first.json())
    assert b1.trial_started_now is True
    assert b1.plan.plan == "pro" and b1.plan.source == "trial"
    assert b1.plan.trial_ends_at is not None
    delta = b1.plan.trial_ends_at - b1.user.trial_started_at
    assert delta.days == settings.trial_days
    assert b1.entitlements["memo_view"].limit is None, "Pro is unlimited"

    second = client.post("/api/me/bootstrap", headers=bearer(tok))
    b2 = BootstrapOut.model_validate(second.json())
    assert b2.trial_started_now is False
    assert b2.plan.trial_ends_at == b1.plan.trial_ends_at, "the end date never moves"


def test_bootstrap_records_abuse_hashes_not_the_ip(client, auth_on):
    sub = new_sub()
    client.post("/api/me/bootstrap", headers={**bearer(auth_on.token(sub=sub, email=new_email())),
                                              "user-agent": "pytest-agent/1.0"})
    with SessionLocal() as db:
        u = db.query(User).filter(User.external_id == sub).one()
        assert u.bootstrap_ip_hash and len(u.bootstrap_ip_hash) == 64
        assert u.bootstrap_ua_hash and len(u.bootstrap_ua_hash) == 64
        assert "testclient" not in u.bootstrap_ip_hash and "pytest" not in u.bootstrap_ua_hash


def test_unverified_email_gets_no_trial(client, auth_on):
    tok = auth_on.token(sub=new_sub(), email=new_email(), verified=False)
    b = BootstrapOut.model_validate(client.post("/api/me/bootstrap", headers=bearer(tok)).json())
    assert b.trial_started_now is False and b.plan.plan == "free"
    assert b.user.email_verified is False


def test_string_true_claim_counts_as_verified(client, auth_on):
    tok = auth_on.token(sub=new_sub(), email=new_email(), verified="true")
    b = BootstrapOut.model_validate(client.post("/api/me/bootstrap", headers=bearer(tok)).json())
    assert b.trial_started_now is True


def test_same_email_hash_gets_one_trial_across_accounts(client, auth_on):
    email = new_email()
    first = client.post("/api/me/bootstrap", headers=bearer(auth_on.token(sub=new_sub(), email=email)))
    assert first.json()["trial_started_now"] is True
    # A second Clerk identity with the same (re-registered) address.
    second = client.post("/api/me/bootstrap", headers=bearer(auth_on.token(sub=new_sub(), email=email.upper())))
    assert second.status_code == 200
    assert second.json()["trial_started_now"] is False
    assert second.json()["plan"]["plan"] == "free"


def test_bootstrap_requires_a_token(client, auth_on):
    assert client.post("/api/me/bootstrap").status_code == 401


def test_bootstrap_ip_hash_is_the_forwarded_client_not_the_proxy(client, auth_on, monkeypatch):
    """Behind Render the socket peer is the proxy; the abuse marker must
    hash the address Render appended to X-Forwarded-For, and a prefix the
    client adds must not change it (or every trial would hash unique)."""
    import hashlib

    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    monkeypatch.setattr(settings, "abuse_hash_salt", "test-salt")
    subs = [new_sub(), new_sub()]
    client.post("/api/me/bootstrap", headers={
        **bearer(auth_on.token(sub=subs[0], email=new_email())), "X-Forwarded-For": "203.0.113.5"})
    client.post("/api/me/bootstrap", headers={
        **bearer(auth_on.token(sub=subs[1], email=new_email())), "X-Forwarded-For": "10.1.1.1, 203.0.113.5"})
    with SessionLocal() as db:
        hashes = {u.bootstrap_ip_hash for u in db.query(User).filter(User.external_id.in_(subs))}
    assert hashes == {hashlib.sha256(b"test-salt|203.0.113.5").hexdigest()}


def test_abuse_report_counts_trials_per_ip_hash_and_429s(client, auth_on, monkeypatch):
    """The numbers behind GET /api/admin/abuse-telemetry (route wired by
    the admin router owner), exercised directly."""
    import hashlib
    import uuid

    from app.auth import analytics
    from app.auth.principal import Principal

    salt = uuid.uuid4().hex  # unique hash per run on the shared sqlite file
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    monkeypatch.setattr(settings, "abuse_hash_salt", salt)
    for _ in range(2):
        resp = client.post("/api/me/bootstrap", headers={
            **bearer(auth_on.token(sub=new_sub(), email=new_email())), "X-Forwarded-For": "203.0.113.77"})
        assert resp.json()["trial_started_now"] is True, resp.text
    route = "/api/me/bootstrap-" + salt[:8]
    with SessionLocal() as db:
        assert analytics.track("rate_limit_hit", db=db, principal=Principal.anonymous(),
                               props={"scope": "ip:bootstrap", "kind": "ip", "route": route, "method": "POST"})
        report = analytics.abuse_report(db, hours=24)
    expected = hashlib.sha256(f"{salt}|203.0.113.77".encode()).hexdigest()
    row = next(r for r in report["trials_per_ip_hash"] if r["ip_hash"] == expected)
    assert row["trials"] == 2
    assert report["trials_started"] >= 2 and report["window_hours"] == 24
    assert report["rate_limit_hits"]["by_scope"].get("ip:bootstrap", 0) >= 1
    assert report["rate_limit_hits"]["by_route"].get(f"POST {route}") == 1
    assert report["rate_limit_hits"]["by_plan"].get("anon", 0) >= 1
    assert set(report) >= {"quota_hits", "api_requests_non_public", "api_requests_429", "share_429"}
    assert 0.0 <= report["share_429"] <= 1.0
    assert "203.0.113.77" not in str(report), "the report carries hashes, never addresses"


# ---------------------------------------------------------------------------
# /api/me/usage
# ---------------------------------------------------------------------------

def test_usage_reports_the_current_period(client, auth_on):
    tok = auth_on.token(sub=new_sub(), email=new_email())
    client.post("/api/me/bootstrap", headers=bearer(tok))
    resp = client.get("/api/me/usage", headers=bearer(tok))
    assert resp.status_code == 200, resp.text
    u = UsageOut.model_validate(resp.json())
    assert u.period_key == datetime.utcnow().strftime("%Y-%m")
    assert set(u.features) >= {"memo_view", "research_run", "pm_chat"}
    assert u.history == []


def test_usage_accepts_a_past_period_and_rejects_garbage(client, auth_on):
    tok = auth_on.token(sub=new_sub(), email=new_email())
    resp = client.get("/api/me/usage", headers=bearer(tok), params={"period": "2026-01"})
    assert resp.status_code == 200 and resp.json()["period_key"] == "2026-01"
    assert client.get("/api/me/usage", headers=bearer(tok), params={"period": "jan"}).status_code == 422


def test_usage_shows_charges_made_through_authorize(client, auth_on):
    from starlette.requests import Request

    from app.auth.entitlements import authorize
    from app.auth.principal import Principal

    sub = new_sub()
    tok = auth_on.token(sub=sub, email=new_email())
    client.get("/api/me", headers=bearer(tok))
    with SessionLocal() as db:
        uid = db.query(User).filter(User.external_id == sub).one().id
    req = Request({"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b"",
                   "client": ("127.0.0.1", 1), "state": {}})
    req.state.principal = Principal(kind="user", user_id=uid, external_id=sub, email_verified=True)
    authorize(req, "memo_view", resource="NVDA").commit()
    u = UsageOut.model_validate(client.get("/api/me/usage", headers=bearer(tok)).json())
    assert u.features["memo_view"].used == 1 and u.features["memo_view"].remaining == 2
    assert u.history and u.history[0].feature == "memo_view" and u.history[0].resource_ref == "NVDA"
    assert u.history[0].status == "committed"
