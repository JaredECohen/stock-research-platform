"""Shared harness for the S4 billing tests (FEAT-002).

Three things every billing test needs and none should re-invent:

  - **Signed synthetic events.** `load_event` reads a fixture from
    `fixtures/stripe_events/`, swaps the placeholder ids for this test's
    own, and `post_event` signs the exact bytes with the throwaway
    webhook secret the test installed — so the signature path the route
    runs in CI is the real one, with no Stripe account anywhere.
  - **A fake Stripe.** `FakeStripe` replaces `stripe_client.request`,
    the module's single network entry point, records every call (method,
    path, form data, idempotency key) and answers from canned objects.
    Nothing in the suite can reach api.stripe.com.
  - **Users made directly.** State-machine and loop tests need `users`
    rows without a Clerk token; `make_user` inserts one. Route tests use
    the ClerkStub from `auth_helpers` instead.

Tests import these by name and define their own `clerk` / `auth_on` /
`client` fixtures (see `auth_helpers` for why).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import settings
from app.database import SessionLocal
from app.models import AnalyticsEvent, BillingWebhookEvent, Subscription, User
from app.services import stripe_client

FIXTURES = Path(__file__).parent / "fixtures" / "stripe_events"
WEBHOOK_SECRET = "whsec_test_" + "0" * 24
SECRET_KEY = "sk_test_" + "0" * 24
BASE_URL = "https://app.marketmosaic.test"


def uid() -> str:
    return uuid.uuid4().hex[:12]


def new_external_id() -> str:
    return "user_" + uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def enable_billing(monkeypatch, *, key: bool = True) -> None:
    """BILLING_ENABLED with a complete (fake) Stripe configuration. With
    `key=False` the flag is on but no secret key is set — the
    'unavailable' shape."""
    monkeypatch.setattr(settings, "billing_enabled", True)
    monkeypatch.setattr(settings, "stripe_secret_key", SECRET_KEY if key else "")
    monkeypatch.setattr(settings, "stripe_webhook_secret", WEBHOOK_SECRET)
    monkeypatch.setattr(settings, "stripe_price_pro_monthly", "price_test_monthly")
    monkeypatch.setattr(settings, "stripe_price_pro_annual", "price_test_annual")
    monkeypatch.setattr(settings, "stripe_portal_configuration_id", "")
    monkeypatch.setattr(settings, "public_base_url", BASE_URL)


# ---------------------------------------------------------------------------
# Users / rows
# ---------------------------------------------------------------------------

def make_user(
    *,
    customer: str | None = None,
    verified: bool = True,
    trial_ends_in: timedelta | None = None,
    now: datetime | None = None,
    external_id: str | None = None,
) -> User:
    """Insert a `users` row and return it detached (id populated)."""
    now = now or datetime.utcnow()
    user = User(
        external_id=external_id or new_external_id(), auth_provider="clerk",
        email_hash=uuid.uuid4().hex + uuid.uuid4().hex, email_verified_at=now if verified else None,
        created_at=now, last_seen_at=now, stripe_customer_id=customer,
    )
    if trial_ends_in is not None:
        user.trial_started_at = now + trial_ends_in - timedelta(days=settings.trial_days)
        user.trial_ends_at = now + trial_ends_in
        user.trial_source = "signup"
    with SessionLocal() as db:
        db.add(user)
        db.commit()
        db.refresh(user)
        db.expunge(user)
    return user


def reload_user(user_id: int) -> User:
    with SessionLocal() as db:
        u = db.get(User, user_id)
        assert u is not None
        db.expunge(u)
        return u


def subscription_rows(user_id: int) -> list[Subscription]:
    with SessionLocal() as db:
        rows = db.query(Subscription).filter(Subscription.user_id == user_id).order_by(Subscription.id).all()
        db.expunge_all()
        return rows


def webhook_row(event_id: str) -> BillingWebhookEvent | None:
    with SessionLocal() as db:
        row = db.query(BillingWebhookEvent).filter(BillingWebhookEvent.stripe_event_id == event_id).one_or_none()
        if row is not None:
            db.expunge(row)
        return row


def events_for(user_id: int, name: str | None = None) -> list[AnalyticsEvent]:
    with SessionLocal() as db:
        q = db.query(AnalyticsEvent).filter(AnalyticsEvent.user_id == user_id)
        if name:
            q = q.filter(AnalyticsEvent.event_name == name)
        rows = q.order_by(AnalyticsEvent.id).all()
        db.expunge_all()
        return rows


def plan_for(user_id: int) -> str:
    from app.auth.entitlements import resolve_for_user
    with SessionLocal() as db:
        return resolve_for_user(db, db.get(User, user_id)).plan


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def load_event(
    name: str,
    *,
    customer: str = "cus_none",
    sub: str = "sub_none",
    user_id: int | str = 0,
    event_id: str | None = None,
    created: int | None = None,
    patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A fixture with this test's ids. `patch` updates `data.object`."""
    text = (FIXTURES / f"{name}.json").read_text()
    text = (text.replace("cus_TEST", customer).replace("sub_TEST", sub)
            .replace("USER_TEST", str(user_id)).replace("in_TEST", f"in_{uid()}"))
    event = json.loads(text)
    event["id"] = event_id or f"evt_{uid()}"
    if created is not None:
        event["created"] = int(created)
    if patch:
        event["data"]["object"].update(patch)
    return event


def encode(event: dict[str, Any]) -> bytes:
    return json.dumps(event, separators=(",", ":")).encode("utf-8")


def signed_headers(body: bytes, *, secret: str = WEBHOOK_SECRET, timestamp: int | None = None) -> dict[str, str]:
    return {
        "Stripe-Signature": stripe_client.sign_payload(body, secret, timestamp=timestamp),
        "Content-Type": "application/json",
    }


def post_event(client, event: dict[str, Any], *, secret: str = WEBHOOK_SECRET, timestamp: int | None = None,
               headers: dict[str, str] | None = None):
    body = encode(event)
    return client.post("/api/billing/webhook", content=body,
                       headers={**signed_headers(body, secret=secret, timestamp=timestamp), **(headers or {})})


def subscription_object(**overrides: Any) -> dict[str, Any]:
    """A bare Stripe subscription object for the state-machine tests."""
    obj = load_event("customer_subscription_created")["data"]["object"]
    obj.update(overrides)
    return obj


# ---------------------------------------------------------------------------
# Fake Stripe
# ---------------------------------------------------------------------------

class FakeStripe:
    """Stands in for `stripe_client.request`. Each recorded call is
    `(method, path, data, idempotency_key, params)`; answers come from
    `responses` (a list consumed in order per (method, path prefix)) or
    from the defaults below. Set `fail` to an exception to raise it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None, str | None, dict[str, Any] | None]] = []
        self.responses: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.fail: Exception | None = None
        self.customer_id = f"cus_{uid()}"
        self.subscription: dict[str, Any] | None = None
        self.subscriptions: list[dict[str, Any]] | None = None

    def queue(self, method: str, path: str, response: dict[str, Any]) -> None:
        self.responses.setdefault((method.upper(), path), []).append(response)

    def install(self, monkeypatch) -> FakeStripe:
        monkeypatch.setattr(stripe_client, "request", self)
        return self

    def __call__(self, method, path, data=None, *, idempotency_key=None, params=None):
        if not stripe_client.configured():
            raise stripe_client.BillingUnavailable("STRIPE_SECRET_KEY is not set")
        self.calls.append((method.upper(), path, data, idempotency_key, params))
        if self.fail is not None:
            raise self.fail
        for (m, prefix), queued in self.responses.items():
            if m == method.upper() and path.startswith(prefix) and queued:
                return queued.pop(0)
        if method.upper() == "POST" and path == "/customers":
            return {"id": self.customer_id, "object": "customer"}
        if method.upper() == "POST" and path == "/checkout/sessions":
            return {"id": f"cs_test_{uid()}", "object": "checkout.session",
                    "url": f"https://checkout.stripe.com/c/pay/cs_test_{uid()}"}
        if method.upper() == "POST" and path == "/billing_portal/sessions":
            return {"id": f"bps_{uid()}", "object": "billing_portal.session",
                    "url": f"https://billing.stripe.com/p/session/{uid()}"}
        if method.upper() == "GET" and path.startswith("/subscriptions/"):
            if self.subscription is None:
                raise stripe_client.StripeError(404, code="resource_missing")
            return self.subscription
        if method.upper() == "GET" and path == "/subscriptions":
            return {"object": "list", "data": list(self.subscriptions or [])}
        raise AssertionError(f"FakeStripe has no answer for {method} {path}")

    def calls_to(self, method: str, path: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == method.upper() and c[1] == path]
