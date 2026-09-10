"""`POST /api/billing/checkout`, `/portal`, `/reconcile` and the admin
billing routes (FEAT-002, S4).

Stripe is a `FakeStripe` standing in for `stripe_client.request`, so
every assertion about what would be sent — the idempotency key, the
`trial_end` pass-through, the form shape — reads the recorded call.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import AdminOverride, User
from app.services import billing_service as bs
from app.services import stripe_client
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth, new_email, new_sub
from app.tests.billing_helpers import (
    BASE_URL,
    FakeStripe,
    enable_billing,
    events_for,
    load_event,
    make_user,
    plan_for,
    reload_user,
    subscription_rows,
    uid,
)
from app.tests.gating_helpers import assert_structured, free_user, pro_user, user_id_for

ADMIN_TOKEN = "admin-token-billing-tests"


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture()
def auth_on(monkeypatch, clerk):
    yield from enable_auth(monkeypatch, clerk)


@pytest.fixture()
def stripe(monkeypatch, auth_on):
    enable_billing(monkeypatch)
    return FakeStripe().install(monkeypatch)


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def admin(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", ADMIN_TOKEN)
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def _checkout(client, token, interval="month"):
    return client.post("/api/billing/checkout", json={"interval": interval}, headers=bearer(token))


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def test_wall_off_means_the_route_is_inert(client, monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "billing_enabled", True)
    resp = client.post("/api/billing/checkout", json={"interval": "month"})
    assert_structured(resp, code="feature_disabled", status=404)
    assert client.post("/api/billing/portal").status_code == 404
    assert client.post("/api/billing/reconcile").status_code == 404


def test_billing_disabled_is_404_feature_disabled(client, auth_on, monkeypatch):
    monkeypatch.setattr(settings, "billing_enabled", False)
    _, tok = pro_user(client, auth_on)
    assert_structured(_checkout(client, tok), code="feature_disabled", status=404)
    assert_structured(client.post("/api/billing/portal", headers=bearer(tok)), code="feature_disabled", status=404)
    assert_structured(client.post("/api/billing/reconcile", headers=bearer(tok)), code="feature_disabled", status=404)


def test_anonymous_is_401(client, stripe):
    assert client.post("/api/billing/checkout", json={"interval": "month"}).status_code == 401


def test_unverified_email_is_403(client, stripe, auth_on):
    tok = auth_on.token(sub=new_sub(), email=new_email(), verified=False)
    resp = _checkout(client, tok)
    assert_structured(resp, code="email_unverified", status=403)
    assert stripe.calls == [], "nothing reaches Stripe before the checks pass"


def test_stripe_unavailable_is_503_and_nothing_changes(client, auth_on, monkeypatch):
    enable_billing(monkeypatch, key=False)
    FakeStripe().install(monkeypatch)
    _, tok = pro_user(client, auth_on)
    resp = _checkout(client, tok)
    assert_structured(resp, code="billing_unavailable", status=503)
    assert client.get("/api/me", headers=bearer(tok)).json()["plan"]["plan"] == "pro"


def test_stripe_unreachable_is_503(client, stripe, auth_on):
    _, tok = pro_user(client, auth_on)
    stripe.fail = stripe_client.BillingUnavailable("timeout")
    assert_structured(_checkout(client, tok), code="billing_unavailable", status=503)


def test_stripe_refusal_is_502_with_the_code_only(client, stripe, auth_on):
    _, tok = pro_user(client, auth_on)
    stripe.fail = stripe_client.StripeError(402, code="card_declined", error_type="card_error",
                                            message="Your card jane@example.com was declined")
    resp = _checkout(client, tok)
    detail = assert_structured(resp, code="billing_error", status=502)
    assert detail["extra"]["stripe_code"] == "card_declined"
    assert "jane@example.com" not in resp.text


def test_existing_active_subscription_is_409(client, stripe, auth_on):
    sub, tok = pro_user(client, auth_on)
    uid_ = user_id_for(client, tok)
    with SessionLocal() as db:
        user = db.get(User, uid_)
        user.stripe_customer_id = f"cus_{uid()}"
        db.commit()
        obj = load_event("customer_subscription_created", customer=user.stripe_customer_id, sub=f"sub_{uid()}",
                         user_id=uid_)["data"]["object"]
        bs.apply_subscription_object(db, user, obj, event_created=1)
        db.commit()
    assert_structured(_checkout(client, tok), code="already_subscribed", status=409)
    assert stripe.calls == []


def test_unpaid_or_canceled_subscription_allows_a_fresh_checkout(client, stripe, auth_on):
    _, tok = pro_user(client, auth_on)
    uid_ = user_id_for(client, tok)
    with SessionLocal() as db:
        user = db.get(User, uid_)
        user.stripe_customer_id = f"cus_{uid()}"
        db.commit()
        obj = load_event("customer_subscription_deleted", customer=user.stripe_customer_id, sub=f"sub_{uid()}",
                         user_id=uid_)["data"]["object"]
        bs.apply_subscription_object(db, user, obj, event_created=1)
        db.commit()
    assert _checkout(client, tok).status_code == 200


# ---------------------------------------------------------------------------
# The happy path and what it sends
# ---------------------------------------------------------------------------

def test_checkout_creates_the_customer_once_and_returns_the_url(client, stripe, auth_on):
    sub, tok = pro_user(client, auth_on)
    resp = _checkout(client, tok)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["url"].startswith("https://checkout.stripe.com/")
    uid_ = user_id_for(client, tok)
    assert reload_user(uid_).stripe_customer_id == stripe.customer_id

    (method, path, data, key, _params) = stripe.calls_to("POST", "/customers")[0]
    assert key == f"customer:{uid_}"
    assert data["metadata"]["user_id"] == str(uid_) and data["email"].endswith("@example.com")

    resp2 = _checkout(client, tok)
    assert resp2.status_code == 200
    assert len(stripe.calls_to("POST", "/customers")) == 1, "the second checkout reuses the customer"
    assert len(stripe.calls_to("POST", "/checkout/sessions")) == 2


def test_customer_idempotency_key_is_reused_after_a_lost_save(client, stripe, auth_on):
    """If the customer was created but the `users` write was lost, the
    retry sends the same key — Stripe replays the same customer."""
    _, tok = pro_user(client, auth_on)
    uid_ = user_id_for(client, tok)
    assert _checkout(client, tok).status_code == 200
    with SessionLocal() as db:  # simulate the lost save
        db.get(User, uid_).stripe_customer_id = None
        db.commit()
    assert _checkout(client, tok).status_code == 200
    keys = [c[3] for c in stripe.calls_to("POST", "/customers")]
    assert keys == [f"customer:{uid_}", f"customer:{uid_}"]


def test_checkout_session_shape(client, stripe, auth_on):
    _, tok = pro_user(client, auth_on)
    uid_ = user_id_for(client, tok)
    assert _checkout(client, tok, "year").status_code == 200
    (_m, _p, data, key, _) = stripe.calls_to("POST", "/checkout/sessions")[0]
    assert data["mode"] == "subscription" and data["customer"] == stripe.customer_id
    assert data["line_items"] == [{"price": "price_test_annual", "quantity": 1}]
    assert data["client_reference_id"] == str(uid_)
    assert data["subscription_data"]["metadata"]["user_id"] == str(uid_)
    assert data["success_url"] == f"{BASE_URL}/app/billing/success?session_id={{CHECKOUT_SESSION_ID}}"
    assert data["cancel_url"] == f"{BASE_URL}/app/billing/canceled"
    assert key.startswith(f"checkout:{uid_}:year:")
    flat = stripe_client.flatten(data)
    assert flat["line_items[0][price]"] == "price_test_annual" and flat["allow_promotion_codes"] == "true"


def test_mid_trial_checkout_carries_the_remaining_trial_to_stripe(client, stripe, auth_on):
    _, tok = pro_user(client, auth_on)  # 7-day trial just started
    uid_ = user_id_for(client, tok)
    resp = _checkout(client, tok)
    assert resp.status_code == 200
    body = resp.json()
    trial_ends_at = reload_user(uid_).trial_ends_at
    assert body["billing_starts"] == "at_trial_end"
    assert datetime.fromisoformat(body["trial_end"]) == trial_ends_at
    data = stripe.calls_to("POST", "/checkout/sessions")[0][2]
    assert data["subscription_data"]["trial_end"] == bs.unix(trial_ends_at)
    assert events_for(uid_, "checkout_started")[0].props["source"] == "at_trial_end"


def test_under_48h_of_trial_bills_now_and_says_so(client, stripe, auth_on):
    _, tok = pro_user(client, auth_on)
    uid_ = user_id_for(client, tok)
    with SessionLocal() as db:
        db.get(User, uid_).trial_ends_at = datetime.utcnow() + timedelta(hours=47)
        db.commit()
    body = _checkout(client, tok).json()
    assert body["billing_starts"] == "now" and body["trial_end"] is None
    assert "trial_end" not in stripe.calls_to("POST", "/checkout/sessions")[0][2]["subscription_data"]


def test_free_user_with_no_trial_bills_now(client, stripe, auth_on):
    _, tok = free_user(auth_on)
    body = _checkout(client, tok).json()
    assert body["billing_starts"] == "now"


def test_checkout_redirect_is_not_proof_of_payment(client, stripe, auth_on):
    _, tok = free_user(auth_on)
    assert _checkout(client, tok).status_code == 200
    me = client.get("/api/me", headers=bearer(tok)).json()
    assert me["plan"]["plan"] == "free" and me["billing"]["has_subscription"] is False
    assert subscription_rows(user_id_for(client, tok)) == []


def test_flatten_encodes_nested_and_drops_none():
    assert stripe_client.flatten({"a": {"b": 1, "c": None}, "d": [{"e": True}, "x"], "f": False}) == {
        "a[b]": "1", "d[0][e]": "true", "d[1]": "x", "f": "false",
    }


def test_request_raises_unavailable_when_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "stripe_secret_key", "")
    with pytest.raises(stripe_client.BillingUnavailable):
        stripe_client.request("GET", "/subscriptions/sub_x")
    with pytest.raises(stripe_client.BillingUnavailable):
        stripe_client.retrieve_subscription("sub_x")


# ---------------------------------------------------------------------------
# Portal
# ---------------------------------------------------------------------------

def test_portal_needs_a_billing_account(client, stripe, auth_on):
    _, tok = free_user(auth_on)
    assert_structured(client.post("/api/billing/portal", headers=bearer(tok)), code="no_billing_account", status=409)


def test_portal_returns_a_url(client, stripe, auth_on, monkeypatch):
    monkeypatch.setattr(settings, "stripe_portal_configuration_id", "bpc_test")
    _, tok = pro_user(client, auth_on)
    assert _checkout(client, tok).status_code == 200
    resp = client.post("/api/billing/portal", headers=bearer(tok))
    assert resp.status_code == 200 and resp.json()["url"].startswith("https://billing.stripe.com/")
    data = stripe.calls_to("POST", "/billing_portal/sessions")[0][2]
    assert data == {"customer": stripe.customer_id, "return_url": f"{BASE_URL}/app/account", "configuration": "bpc_test"}
    me = client.get("/api/me", headers=bearer(tok)).json()
    assert me["billing"]["portal_available"] is True


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------

def test_reconcile_applies_the_fetched_object(client, stripe, auth_on):
    _, tok = free_user(auth_on)
    uid_ = user_id_for(client, tok)
    cust = f"cus_{uid()}"
    with SessionLocal() as db:
        db.get(User, uid_).stripe_customer_id = cust
        db.commit()
    stripe.subscriptions = [load_event("customer_subscription_created", customer=cust, sub=f"sub_{uid()}",
                                       user_id=uid_)["data"]["object"]]
    resp = client.post("/api/billing/reconcile", headers=bearer(tok))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reconcile"] == {"ok": True, "fetched": 1, "applied": 1}
    assert body["plan"]["plan"] == "pro" and body["billing"]["has_subscription"] is True
    assert stripe.calls_to("GET", "/subscriptions")[0][4] == {"customer": cust, "status": "all", "limit": 10}


def test_reconcile_never_downgrades_on_a_fetch_error(client, stripe, auth_on):
    _, tok = free_user(auth_on)
    uid_ = user_id_for(client, tok)
    cust = f"cus_{uid()}"
    with SessionLocal() as db:
        user = db.get(User, uid_)
        user.stripe_customer_id = cust
        db.commit()
        obj = load_event("customer_subscription_created", customer=cust, sub=f"sub_{uid()}", user_id=uid_)["data"]["object"]
        bs.apply_subscription_object(db, user, obj, event_created=1)
        db.commit()
    assert plan_for(uid_) == "pro"
    for exc in (stripe_client.BillingUnavailable("down"), stripe_client.StripeError(500, code="api_error")):
        stripe.fail = exc
        resp = client.post("/api/billing/reconcile", headers=bearer(tok))
        assert resp.status_code == 200, resp.text
        assert resp.json()["reconcile"]["ok"] is False and resp.json()["plan"]["plan"] == "pro"
    assert plan_for(uid_) == "pro"


def test_reconcile_with_no_customer_is_a_noop(client, stripe, auth_on):
    _, tok = free_user(auth_on)
    resp = client.post("/api/billing/reconcile", headers=bearer(tok))
    assert resp.status_code == 200 and resp.json()["reconcile"] == {"ok": True, "fetched": 0, "applied": 0}
    assert stripe.calls == []


def test_reconcile_without_a_key_is_503(client, auth_on, monkeypatch):
    enable_billing(monkeypatch, key=False)
    _, tok = free_user(auth_on)
    assert_structured(client.post("/api/billing/reconcile", headers=bearer(tok)), code="billing_unavailable", status=503)


def test_reconcile_ends_a_row_stripe_canceled(client, stripe, auth_on):
    _, tok = free_user(auth_on)
    uid_ = user_id_for(client, tok)
    cust = f"cus_{uid()}"
    sub = f"sub_{uid()}"
    with SessionLocal() as db:
        user = db.get(User, uid_)
        user.stripe_customer_id = cust
        db.commit()
        obj = load_event("customer_subscription_created", customer=cust, sub=sub, user_id=uid_)["data"]["object"]
        bs.apply_subscription_object(db, user, obj, event_created=1)
        db.commit()
    stripe.subscriptions = [load_event("customer_subscription_deleted", customer=cust, sub=sub, user_id=uid_)["data"]["object"]]
    body = client.post("/api/billing/reconcile", headers=bearer(tok)).json()
    assert body["plan"]["plan"] == "free" and body["reconcile"]["applied"] == 1


# ---------------------------------------------------------------------------
# Admin billing routes — admin token only, never a customer JWT
# ---------------------------------------------------------------------------

def test_admin_billing_routes_need_the_admin_token(client, auth_on, admin):
    _, tok = pro_user(client, auth_on)
    sub = client.get("/api/me", headers=bearer(tok)).json()["user"]["external_id"]
    assert client.get(f"/api/admin/billing/users/{sub}").status_code == 401
    assert client.get(f"/api/admin/billing/users/{sub}", headers=bearer(tok)).status_code == 401, \
        "a customer JWT never opens an admin route"
    assert client.get(f"/api/admin/billing/users/{sub}", headers=admin).status_code == 200


def test_admin_token_never_opens_a_customer_route(client, stripe, admin):
    assert client.post("/api/billing/checkout", json={"interval": "month"}, headers=admin).status_code == 401


def test_admin_user_view_shape(client, stripe, auth_on, admin):
    _, tok = pro_user(client, auth_on)
    uid_ = user_id_for(client, tok)
    sub_ext = client.get("/api/me", headers=bearer(tok)).json()["user"]["external_id"]
    assert _checkout(client, tok).status_code == 200
    with SessionLocal() as db:
        user = db.get(User, uid_)
        obj = load_event("customer_subscription_created", customer=user.stripe_customer_id, sub=f"sub_{uid()}",
                         user_id=uid_)["data"]["object"]
        bs.apply_subscription_object(db, user, obj, event_created=1)
        db.commit()
    resp = client.get(f"/api/admin/billing/users/{sub_ext}", headers=admin)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user"]["external_id"] == sub_ext and body["user"]["stripe_customer_id"] == stripe.customer_id
    assert "email" not in body["user"]
    assert body["plan"]["plan"] == "pro" and body["plan"]["source"] == "subscription"
    assert body["subscription"]["stripe_status"] == "active"
    assert body["overrides"] == [] and isinstance(body["usage_events"], list)
    assert client.get("/api/admin/billing/users/user_nope", headers=admin).status_code == 404


def test_admin_override_plan_and_suspend(client, auth_on, admin):
    _, tok = free_user(auth_on)
    sub_ext = client.get("/api/me", headers=bearer(tok)).json()["user"]["external_id"]
    assert client.get("/api/me", headers=bearer(tok)).json()["plan"]["plan"] == "free"

    resp = client.post("/api/admin/billing/overrides", headers=admin, json={
        "external_id": sub_ext, "kind": "plan", "value": "pro", "reason": "comp for beta feedback",
        "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat(),
    })
    assert resp.status_code == 201, resp.text
    assert resp.json()["kind"] == "plan" and resp.json()["value"] == "pro"
    me = client.get("/api/me", headers=bearer(tok)).json()
    assert me["plan"]["plan"] == "pro" and me["plan"]["source"] == "override"

    resp = client.post("/api/admin/billing/overrides", headers=admin, json={
        "external_id": sub_ext, "kind": "suspend", "reason": "chargeback",
    })
    assert resp.status_code == 201
    assert client.get("/api/me", headers=bearer(tok)).status_code == 403
    view = client.get(f"/api/admin/billing/users/{sub_ext}", headers=admin).json()
    assert [o["kind"] for o in view["overrides"]] == ["suspend", "plan"]
    assert view["plan"]["suspended"] is True


def test_admin_override_validation(client, auth_on, admin):
    _, tok = free_user(auth_on)
    sub_ext = client.get("/api/me", headers=bearer(tok)).json()["user"]["external_id"]
    bad = [
        {"external_id": sub_ext, "kind": "plan", "value": "enterprise", "reason": "nope"},
        {"external_id": sub_ext, "kind": "quota", "value": "lots", "feature": "pm_chat", "reason": "nope"},
        {"external_id": sub_ext, "kind": "quota", "value": "5", "reason": "no feature"},
        {"external_id": sub_ext, "kind": "plan", "value": "pro", "reason": "past",
         "expires_at": (datetime.utcnow() - timedelta(days=1)).isoformat()},
    ]
    for body in bad:
        assert client.post("/api/admin/billing/overrides", headers=admin, json=body).status_code == 422, body
    assert client.post("/api/admin/billing/overrides", headers=admin,
                       json={"external_id": sub_ext, "kind": "plan", "value": "pro", "reason": "x"}).status_code == 422
    assert client.post("/api/admin/billing/overrides", headers=admin,
                       json={"external_id": "user_missing", "kind": "plan", "value": "pro", "reason": "nobody"}).status_code == 404
    with SessionLocal() as db:
        uid_ = user_id_for(client, tok)
        assert db.query(AdminOverride).filter(AdminOverride.user_id == uid_).count() == 0


def test_admin_trial_reset_rewrites_the_trial_and_is_audited(client, auth_on, admin):
    _, tok = free_user(auth_on)
    uid_ = user_id_for(client, tok)
    sub_ext = client.get("/api/me", headers=bearer(tok)).json()["user"]["external_id"]
    with SessionLocal() as db:  # an old, expired trial
        u = db.get(User, uid_)
        u.trial_started_at = datetime.utcnow() - timedelta(days=30)
        u.trial_ends_at = datetime.utcnow() - timedelta(days=23)
        db.commit()
    assert client.get("/api/me", headers=bearer(tok)).json()["plan"]["plan"] == "free"
    resp = client.post("/api/admin/billing/overrides", headers=admin, json={
        "external_id": sub_ext, "kind": "trial_reset", "reason": "support ticket 42", "created_by": "jared",
    })
    assert resp.status_code == 201 and resp.json()["value"] == str(settings.trial_days)
    user = reload_user(uid_)
    assert user.trial_source == "admin" and user.trial_ends_at > datetime.utcnow() + timedelta(days=settings.trial_days - 1)
    me = client.get("/api/me", headers=bearer(tok)).json()
    assert me["plan"]["plan"] == "pro" and me["plan"]["source"] == "trial"
    view = client.get(f"/api/admin/billing/users/{sub_ext}", headers=admin).json()
    assert view["overrides"][0]["created_by"] == "jared" and view["user"]["trial_source"] == "admin"


def test_ensure_customer_is_used_by_checkout_not_by_a_plain_get():
    u = make_user()
    with SessionLocal() as db:
        assert bs.current_subscription(db, u.id) is None
    assert reload_user(u.id).stripe_customer_id is None
