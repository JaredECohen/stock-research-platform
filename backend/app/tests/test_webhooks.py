"""`POST /api/billing/webhook` (FEAT-002, S4).

Every request here carries a signature computed at test time over the
exact bytes sent, with a throwaway secret — the verification path is
the real one. Runs with the login wall off and on: the route is
classified `public` because the signature, not a session, authenticates
it, and that must hold in both modes.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import BillingWebhookEvent, User
from app.rate_limit import limiter
from app.services import billing_service, stripe_client
from app.tests.auth_helpers import ClerkStub, assert_no_secrets_in_logs, enable_auth
from app.tests.billing_helpers import (
    FakeStripe,
    enable_billing,
    encode,
    events_for,
    load_event,
    make_user,
    plan_for,
    post_event,
    reload_user,
    signed_headers,
    subscription_rows,
    uid,
    webhook_row,
)


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture(params=["wall_off", "wall_on"])
def wall(request, monkeypatch, clerk):
    if request.param == "wall_on":
        yield from enable_auth(monkeypatch, clerk)
    else:
        monkeypatch.setattr(settings, "auth_enabled", False)
        yield None


@pytest.fixture()
def billing(monkeypatch, wall):
    enable_billing(monkeypatch)
    return FakeStripe().install(monkeypatch)


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def customer():
    """A user with a Stripe customer id, as `ensure_customer` leaves them."""
    return make_user(customer=f"cus_{uid()}")


# ---------------------------------------------------------------------------
# Signature and body
# ---------------------------------------------------------------------------

def test_bad_signature_is_400(client, billing, customer):
    ev = load_event("customer_subscription_created", customer=customer.stripe_customer_id, sub=f"sub_{uid()}", user_id=customer.id)
    resp = post_event(client, ev, secret="whsec_wrong_" + "1" * 20)
    assert resp.status_code == 400 and resp.json()["detail"]["code"] == "invalid_signature"
    assert webhook_row(ev["id"]) is None, "an unverified event leaves no trace"
    assert plan_for(customer.id) == "free"


def test_missing_header_is_400(client, billing, customer):
    ev = load_event("unhandled_charge_succeeded")
    resp = client.post("/api/billing/webhook", content=encode(ev), headers={"Content-Type": "application/json"})
    assert resp.status_code == 400


def test_stale_timestamp_is_400(client, billing):
    ev = load_event("unhandled_charge_succeeded")
    resp = post_event(client, ev, timestamp=int(time.time()) - 301)
    assert resp.status_code == 400
    resp = post_event(client, ev, timestamp=int(time.time()) - 299)
    assert resp.status_code == 200, "inside the 300s tolerance"


def test_tampered_body_is_400(client, billing):
    ev = load_event("unhandled_charge_succeeded")
    body = encode(ev)
    headers = signed_headers(body)
    resp = client.post("/api/billing/webhook", content=body.replace(b"2999", b"1"), headers=headers)
    assert resp.status_code == 400


def test_no_secret_configured_is_503_so_stripe_retries(client, billing, monkeypatch):
    monkeypatch.setattr(settings, "stripe_webhook_secret", "")
    resp = post_event(client, load_event("unhandled_charge_succeeded"))
    assert resp.status_code == 503 and resp.json()["detail"]["code"] == "billing_unavailable"


def test_oversized_body_is_413(client, billing):
    ev = load_event("unhandled_charge_succeeded", patch={"description": "x" * (256 * 1024 + 1)})
    resp = post_event(client, ev)
    assert resp.status_code == 413 and resp.json()["detail"]["code"] == "payload_too_large"


def test_non_json_and_non_event_bodies_are_400(client, billing):
    body = b"not json"
    resp = client.post("/api/billing/webhook", content=body, headers=signed_headers(body))
    assert resp.status_code == 400
    body = b'{"object": "event"}'
    resp = client.post("/api/billing/webhook", content=body, headers=signed_headers(body))
    assert resp.status_code == 400
    body = b"[1,2]"
    resp = client.post("/api/billing/webhook", content=body, headers=signed_headers(body))
    assert resp.status_code == 400


def test_verify_signature_rotation_and_shape():
    body = b'{"id":"evt_x"}'
    ts = 1_700_000_000
    good = stripe_client.compute_signature(body, "s1", ts)
    stripe_client.verify_signature(body, f"t={ts},v1=deadbeef,v1={good}", "s1", now=ts + 10)
    with pytest.raises(stripe_client.SignatureError):
        stripe_client.verify_signature(body, f"t=abc,v1={good}", "s1", now=ts)
    with pytest.raises(stripe_client.SignatureError):
        stripe_client.verify_signature(body, f"t={ts}", "s1", now=ts)
    with pytest.raises(stripe_client.SignatureError):
        stripe_client.verify_signature(body, f"t={ts},v1={good}", "", now=ts)


# ---------------------------------------------------------------------------
# Idempotency and ordering
# ---------------------------------------------------------------------------

def test_subscription_created_grants_pro_and_records_the_event(client, billing, customer):
    sub = f"sub_{uid()}"
    ev = load_event("customer_subscription_created", customer=customer.stripe_customer_id, sub=sub, user_id=customer.id)
    resp = post_event(client, ev)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"received": True, "outcome": "applied"}
    row = webhook_row(ev["id"])
    assert row.outcome == "applied" and row.processed_at is not None and row.object_id == sub
    assert row.event_type == "customer.subscription.created"
    assert plan_for(customer.id) == "pro"


def test_duplicate_delivery_is_a_noop(client, billing, customer):
    sub = f"sub_{uid()}"
    ev = load_event("customer_subscription_created", customer=customer.stripe_customer_id, sub=sub, user_id=customer.id)
    assert post_event(client, ev).json()["outcome"] == "applied"
    before = subscription_rows(customer.id)[0]
    # The redelivery even claims a different status; the id wins.
    ev["data"]["object"]["status"] = "canceled"
    resp = post_event(client, ev)
    assert resp.status_code == 200 and resp.json()["outcome"] == "duplicate"
    after = subscription_rows(customer.id)[0]
    assert after.stripe_status == "active" and after.updated_at == before.updated_at
    assert after.snapshot == before.snapshot
    with SessionLocal() as db:
        assert db.query(BillingWebhookEvent).filter(BillingWebhookEvent.stripe_event_id == ev["id"]).count() == 1


def test_out_of_order_updated_after_deleted_is_ignored_stale(client, billing, customer):
    sub = f"sub_{uid()}"
    kw = dict(customer=customer.stripe_customer_id, sub=sub, user_id=customer.id)
    assert post_event(client, load_event("customer_subscription_created", created=1000, **kw)).json()["outcome"] == "applied"
    assert post_event(client, load_event("customer_subscription_deleted", created=3000, **kw)).json()["outcome"] == "applied"
    assert plan_for(customer.id) == "free"
    resp = post_event(client, load_event("customer_subscription_updated_active", created=2000, **kw))
    assert resp.status_code == 200 and resp.json()["outcome"] == "ignored_stale"
    assert plan_for(customer.id) == "free", "a delayed `updated` must not resurrect access"
    assert subscription_rows(customer.id)[0].stripe_status == "canceled"


def test_full_lifecycle_through_the_route(client, billing, customer):
    sub = f"sub_{uid()}"
    kw = dict(customer=customer.stripe_customer_id, sub=sub, user_id=customer.id)
    post_event(client, load_event("customer_subscription_created", **kw))
    assert plan_for(customer.id) == "pro"
    post_event(client, load_event("customer_subscription_updated_past_due", **kw))
    row = subscription_rows(customer.id)[0]
    assert row.stripe_status == "past_due" and row.grace_until is not None
    post_event(client, load_event("customer_subscription_updated_active", **kw))
    row = subscription_rows(customer.id)[0]
    assert row.stripe_status == "active" and row.grace_until is None
    post_event(client, load_event("customer_subscription_updated_cancel_at_period_end", **kw))
    row = subscription_rows(customer.id)[0]
    assert row.cancel_at_period_end is True
    post_event(client, load_event("customer_subscription_deleted", **kw))
    row = subscription_rows(customer.id)[0]
    assert row.stripe_status == "canceled" and row.ended_at is not None
    assert plan_for(customer.id) == "free"
    assert [e.props["reason"] for e in events_for(customer.id, "subscription_canceled")] == \
        ["cancel_at_period_end", "canceled"]


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------

def test_checkout_completed_with_mismatched_client_reference_is_error_and_no_grant(client, billing, customer):
    other = make_user(customer=f"cus_{uid()}")
    billing.subscription = load_event("customer_subscription_created", customer=customer.stripe_customer_id,
                                      sub=f"sub_{uid()}", user_id=customer.id)["data"]["object"]
    ev = load_event("checkout_session_completed", customer=customer.stripe_customer_id, sub=billing.subscription["id"],
                    user_id=other.id)
    resp = post_event(client, ev)
    assert resp.status_code == 200 and resp.json()["outcome"] == "error"
    assert "ownership" in webhook_row(ev["id"]).error
    assert plan_for(customer.id) == "free" and plan_for(other.id) == "free"
    assert subscription_rows(customer.id) == [] and subscription_rows(other.id) == []
    assert billing.calls_to("GET", f"/subscriptions/{billing.subscription['id']}") == [], "no fetch before ownership"


def test_checkout_completed_with_mismatched_customer_is_error(client, billing, customer):
    ev = load_event("checkout_session_completed", customer=f"cus_{uid()}", sub=f"sub_{uid()}", user_id=customer.id)
    resp = post_event(client, ev)
    assert resp.json()["outcome"] == "error" and plan_for(customer.id) == "free"


def test_checkout_completed_unknown_reference_is_error(client, billing, customer):
    ev = load_event("checkout_session_completed", customer=customer.stripe_customer_id, sub=f"sub_{uid()}",
                    user_id="not-a-number")
    assert post_event(client, ev).json()["outcome"] == "error"


def test_checkout_completed_fetches_and_applies_the_subscription(client, billing, customer):
    sub = f"sub_{uid()}"
    billing.subscription = load_event("customer_subscription_created_trialing", customer=customer.stripe_customer_id,
                                      sub=sub, user_id=customer.id)["data"]["object"]
    ev = load_event("checkout_session_completed", customer=customer.stripe_customer_id, sub=sub, user_id=customer.id)
    resp = post_event(client, ev)
    assert resp.json()["outcome"] == "applied"
    assert len(billing.calls_to("GET", f"/subscriptions/{sub}")) == 1
    row = subscription_rows(customer.id)[0]
    assert row.stripe_status == "trialing" and row.billing_interval == "year"
    assert len(events_for(customer.id, "checkout_completed")) == 1


def test_checkout_completed_when_stripe_is_down_still_records_and_waits_for_the_subscription_event(
        client, billing, customer):
    sub = f"sub_{uid()}"
    billing.fail = stripe_client.BillingUnavailable("down")
    ev = load_event("checkout_session_completed", customer=customer.stripe_customer_id, sub=sub, user_id=customer.id)
    resp = post_event(client, ev)
    assert resp.status_code == 200 and resp.json()["outcome"] == "applied"
    assert "fetch skipped" in webhook_row(ev["id"]).error
    assert plan_for(customer.id) == "free", "no grant without a subscription object"
    billing.fail = None
    post_event(client, load_event("customer_subscription_created", customer=customer.stripe_customer_id, sub=sub,
                                  user_id=customer.id))
    assert plan_for(customer.id) == "pro"


def test_subscription_for_unknown_customer_is_error(client, billing):
    ev = load_event("customer_subscription_created", customer=f"cus_{uid()}", sub=f"sub_{uid()}", user_id=0)
    resp = post_event(client, ev)
    assert resp.status_code == 200 and resp.json()["outcome"] == "error"
    assert "unknown customer" in webhook_row(ev["id"]).error


def test_metadata_may_link_a_customer_we_failed_to_save_but_never_reassign_one(client, billing):
    lost = make_user(customer=None)  # created the customer, DB save was lost
    cust = f"cus_{uid()}"
    ev = load_event("customer_subscription_created", customer=cust, sub=f"sub_{uid()}", user_id=lost.id)
    assert post_event(client, ev).json()["outcome"] == "applied"
    assert reload_user(lost.id).stripe_customer_id == cust and plan_for(lost.id) == "pro"

    other = make_user(customer=f"cus_{uid()}")
    ev = load_event("customer_subscription_created", customer=cust, sub=f"sub_{uid()}", user_id=other.id)
    assert post_event(client, ev).json()["outcome"] == "error"
    assert plan_for(other.id) == "free"


# ---------------------------------------------------------------------------
# Other event types
# ---------------------------------------------------------------------------

def test_unhandled_type_is_200_ignored(client, billing):
    ev = load_event("unhandled_charge_succeeded")
    resp = post_event(client, ev)
    assert resp.status_code == 200 and resp.json()["outcome"] == "ignored_unhandled"
    assert webhook_row(ev["id"]).outcome == "ignored_unhandled"


def test_invoice_paid_emits_renewal_and_refreshes_the_subscription(client, billing, customer):
    sub = f"sub_{uid()}"
    kw = dict(customer=customer.stripe_customer_id, sub=sub, user_id=customer.id)
    post_event(client, load_event("customer_subscription_created", **kw))
    post_event(client, load_event("customer_subscription_updated_past_due", **kw))
    assert subscription_rows(customer.id)[0].stripe_status == "past_due"
    billing.subscription = load_event("customer_subscription_updated_active", **kw)["data"]["object"]
    # The invoice fixture is timestamped BEFORE the past_due update (as a
    # real renewal invoice is); the fetched object must still apply.
    resp = post_event(client, load_event("invoice_paid", **kw))
    assert resp.json()["outcome"] == "applied"
    row = subscription_rows(customer.id)[0]
    assert row.stripe_status == "active" and row.grace_until is None
    assert row.latest_event_created > 1_760_000_000, "applied as of the fetch, not the invoice"
    assert len(events_for(customer.id, "subscription_renewed")) == 1


def test_invoice_events_for_unknown_subscription_are_ignored(client, billing, customer):
    resp = post_event(client, load_event("invoice_payment_failed", customer=customer.stripe_customer_id,
                                         sub=f"sub_{uid()}", user_id=customer.id))
    assert resp.json()["outcome"] == "ignored_unhandled"


def test_invoice_subscription_id_is_read_from_the_newer_parent_shape(client, billing, customer):
    sub = f"sub_{uid()}"
    kw = dict(customer=customer.stripe_customer_id, sub=sub, user_id=customer.id)
    post_event(client, load_event("customer_subscription_created", **kw))
    ev = load_event("invoice_paid", **kw)
    ev["data"]["object"].pop("subscription")
    ev["data"]["object"]["parent"] = {"subscription_details": {"subscription": sub}}
    billing.subscription = load_event("customer_subscription_updated_active", **kw)["data"]["object"]
    assert post_event(client, ev).json()["outcome"] == "applied"


def test_customer_deleted_ends_open_rows_and_unlinks_the_customer(client, billing, customer):
    sub = f"sub_{uid()}"
    kw = dict(customer=customer.stripe_customer_id, sub=sub, user_id=customer.id)
    post_event(client, load_event("customer_subscription_created", **kw))
    resp = post_event(client, load_event("customer_deleted", **kw))
    assert resp.json()["outcome"] == "applied"
    row = subscription_rows(customer.id)[0]
    assert row.stripe_status == "canceled" and row.ended_at is not None
    assert reload_user(customer.id).stripe_customer_id is None
    assert plan_for(customer.id) == "free"


def test_customer_deleted_for_unknown_customer_is_ignored(client, billing):
    assert post_event(client, load_event("customer_deleted", customer=f"cus_{uid()}")).json()["outcome"] == "ignored_unhandled"


# ---------------------------------------------------------------------------
# Never writes trial columns; never logs secrets
# ---------------------------------------------------------------------------

def test_webhooks_never_write_trial_columns(client, billing, caplog):
    user = make_user(customer=f"cus_{uid()}", trial_ends_in=timedelta(days=5))
    before = (user.trial_started_at, user.trial_ends_at, user.trial_source)
    kw = dict(customer=user.stripe_customer_id, sub=f"sub_{uid()}", user_id=user.id)
    far = 1_900_000_000
    with caplog.at_level(logging.DEBUG):
        for name in ("customer_subscription_created_trialing", "customer_subscription_updated_active",
                     "customer_subscription_deleted", "customer_deleted"):
            ev = load_event(name, **kw)
            if ev["type"].startswith("customer.subscription"):
                ev["data"]["object"]["trial_end"] = far
                ev["data"]["object"]["trial_start"] = far - 86400
            assert post_event(client, ev).status_code == 200
    after = reload_user(user.id)
    assert (after.trial_started_at, after.trial_ends_at, after.trial_source) == before
    assert_no_secrets_in_logs(caplog.records)
    assert "should-never-be-stored" not in caplog.text


# ---------------------------------------------------------------------------
# Storage failure → 500 (Stripe retries), and the retry then processes
# ---------------------------------------------------------------------------

def test_db_failure_is_500_and_the_retry_processes(client, billing, customer, monkeypatch):
    sub = f"sub_{uid()}"
    ev = load_event("customer_subscription_created", customer=customer.stripe_customer_id, sub=sub, user_id=customer.id)
    real = billing_service._dispatch
    calls = {"n": 0}

    def flaky(db, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OperationalError("UPDATE subscriptions", {}, Exception("database is locked"))
        return real(db, *a, **k)

    monkeypatch.setattr(billing_service, "_dispatch", flaky)
    resp = post_event(client, ev)
    assert resp.status_code == 500 and resp.json()["detail"]["code"] == "webhook_storage_failed"
    row = webhook_row(ev["id"])
    assert row is not None and row.processed_at is None, "claimed but not finished — the retry must process it"
    assert plan_for(customer.id) == "free"

    resp = post_event(client, ev)
    assert resp.status_code == 200 and resp.json()["outcome"] == "applied"
    assert plan_for(customer.id) == "pro"


def test_claim_failure_is_500(client, billing, monkeypatch):
    def boom(db, event, obj):
        raise OperationalError("INSERT billing_webhook_events", {}, Exception("disk full"))
    monkeypatch.setattr(billing_service, "_claim_event", boom)
    assert post_event(client, load_event("unhandled_charge_succeeded")).status_code == 500


# ---------------------------------------------------------------------------
# The route bypasses slowapi and the customer policy
# ---------------------------------------------------------------------------

@pytest.fixture()
def auth_on(monkeypatch, clerk):
    yield from enable_auth(monkeypatch, clerk)


def test_webhook_needs_no_session_when_the_wall_is_on(client, auth_on, monkeypatch):
    """Explicit wall-on case with no bearer: the signature is the auth."""
    enable_billing(monkeypatch)
    FakeStripe().install(monkeypatch)
    resp = post_event(client, load_event("unhandled_charge_succeeded"))
    assert resp.status_code == 200, resp.text
    assert client.get("/api/me").status_code == 401, "the wall is really on"


def test_webhook_is_exempt_from_slowapi_default_limits(client, billing, monkeypatch):
    """`@limiter.exempt`: with the per-IP limiter switched on, 65 events
    in a row all land, while an undecorated route trips the 60/minute
    default on the same client address."""
    monkeypatch.setattr(limiter, "enabled", True)
    limiter.reset()
    try:
        for _ in range(65):
            resp = post_event(client, load_event("unhandled_charge_succeeded"))
            assert resp.status_code == 200, resp.text
        statuses = [client.get("/health").status_code for _ in range(65)]
        assert 429 in statuses, "the limiter was not actually enforcing — the exemption proves nothing"
    finally:
        limiter.reset()


def test_route_is_classified_public_and_exempt():
    from app.api import routes_billing
    from app.auth import policy
    assert policy.classify("POST", "/api/billing/webhook").is_public
    assert f"{routes_billing.webhook.__module__}.{routes_billing.webhook.__name__}" in limiter._exempt_routes


def test_apply_event_rejects_non_events():
    with SessionLocal() as db:
        with pytest.raises(ValueError):
            billing_service.apply_event(db, {"id": "evt_x"})
        with pytest.raises(ValueError):
            billing_service.apply_event(db, {"data": {"object": {}}})


def test_stale_event_row_from_a_crashed_attempt_is_reprocessed(client, billing, customer):
    """A row with no `processed_at` is a previous attempt that died —
    the redelivery should finish the job, not be called a duplicate."""
    sub = f"sub_{uid()}"
    ev = load_event("customer_subscription_created", customer=customer.stripe_customer_id, sub=sub, user_id=customer.id)
    with SessionLocal() as db:
        db.add(BillingWebhookEvent(stripe_event_id=ev["id"], event_type=ev["type"], event_created=ev["created"],
                                   received_at=datetime.utcnow() - timedelta(minutes=5)))
        db.commit()
    assert post_event(client, ev).json()["outcome"] == "applied"
    assert plan_for(customer.id) == "pro"
    with SessionLocal() as db:
        assert db.get(User, customer.id).stripe_customer_id == customer.stripe_customer_id
