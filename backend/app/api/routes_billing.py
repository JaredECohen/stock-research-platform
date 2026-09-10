"""Stripe billing routes (FEAT-002, S4).

Customer side — `POST /api/billing/checkout`, `/portal`, `/reconcile` —
sit behind the login wall (`auth/policy.py` classifies them
`authenticated`; the middleware has already refused anonymous callers)
and answer 404 `feature_disabled` while `BILLING_ENABLED` is off, so the
default deployment exposes nothing that talks to Stripe.

`POST /api/billing/webhook` is the one route Stripe calls. It is
classified `public` because the `Stripe-Signature` header, verified
locally against `STRIPE_WEBHOOK_SECRET`, is its authentication; it is
`@limiter.exempt` because a burst of legitimate events (a monthly
renewal cycle) must not be 429'd into Stripe's retry queue. The body is
read with a hard 256 KB cap, verified before it is parsed, and handed to
`billing_service.apply_event`, which owns idempotency and ordering.
Status codes are chosen for Stripe's retry logic: 2xx = done (including
duplicate / stale / unhandled — retrying cannot change the answer), 400
= never retry (it will not verify next time either), 500 = retry (the
database refused a write; the unique event row makes that safe).

Admin side — `/api/admin/billing/*` — is covered by `admin_auth` by
prefix (bearer `ADMIN_API_TOKEN`); a customer JWT is never consulted.

Nothing here logs a token, a key, a signature or a webhook body.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import features, usage
from ..auth.entitlements import QUOTA_UNLIMITED, EntitlementError, current_subscription, resolve_for_user
from ..auth.sanitize import safe_logger
from ..config import settings
from ..database import get_db
from ..models.accounts import AdminOverride, Subscription, UsageEvent, User
from ..rate_limit import limiter
from ..schemas.accounts import AccountOut
from ..services import billing_service, stripe_client
from .gating import enforce_scope, feature_disabled
from .routes_account import _account, _load_user, _require_user

log = safe_logger(__name__)
router = APIRouter()

MAX_WEBHOOK_BYTES = 256 * 1024


# ---------------------------------------------------------------------------
# Schemas (route-local: nothing else consumes them)
# ---------------------------------------------------------------------------

class CheckoutIn(BaseModel):
    interval: Literal["month", "year"] = "month"


class CheckoutOut(BaseModel):
    url: str
    # "at_trial_end": the remaining local trial was carried to Stripe and
    # the first charge lands at `trial_end`; "now": billing starts at
    # checkout (no trial, or under 48h left — Stripe's floor).
    billing_starts: Literal["now", "at_trial_end"]
    trial_end: datetime | None = None


class PortalOut(BaseModel):
    url: str


class ReconcileOut(AccountOut):
    reconcile: dict[str, Any] = Field(default_factory=dict)


class WebhookOut(BaseModel):
    received: bool = True
    outcome: str


class OverrideIn(BaseModel):
    external_id: str = Field(min_length=1, max_length=128)
    kind: Literal["plan", "quota", "trial_reset", "suspend"]
    feature: str | None = Field(default=None, max_length=32)
    value: str | None = Field(default=None, max_length=64)
    expires_at: datetime | None = None
    reason: str = Field(min_length=3, max_length=2000)
    created_by: str = Field(default="admin", max_length=64)


class OverrideOut(BaseModel):
    id: int
    user_id: int
    kind: str
    feature: str | None = None
    value: str | None = None
    starts_at: datetime
    expires_at: datetime | None = None
    reason: str
    created_by: str
    created_at: datetime
    revoked_at: datetime | None = None


class AdminSubscriptionOut(BaseModel):
    id: int
    stripe_subscription_id: str
    stripe_customer_id: str | None = None
    stripe_price_id: str | None = None
    billing_interval: str | None = None
    stripe_status: str
    current_period_start: datetime | None = None
    current_period_end: datetime | None = None
    cancel_at_period_end: bool = False
    canceled_at: datetime | None = None
    ended_at: datetime | None = None
    past_due_since: datetime | None = None
    grace_until: datetime | None = None
    latest_event_created: int = 0
    updated_at: datetime


class AdminUsageEventOut(BaseModel):
    id: int
    feature: str
    period_key: str
    quantity: int
    status: str
    resource_ref: str | None = None
    plan_at_charge: str
    created_at: datetime
    finalized_at: datetime | None = None


class AdminUserOut(BaseModel):
    id: int
    external_id: str
    account_state: str
    email_verified: bool
    created_at: datetime
    last_seen_at: datetime | None = None
    trial_started_at: datetime | None = None
    trial_ends_at: datetime | None = None
    trial_source: str | None = None
    stripe_customer_id: str | None = None
    first_value_at: datetime | None = None


class AdminBillingUserOut(BaseModel):
    user: AdminUserOut
    plan: dict[str, Any]
    subscription: AdminSubscriptionOut | None = None
    overrides: list[OverrideOut] = Field(default_factory=list)
    usage_events: list[AdminUsageEventOut] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Customer routes
# ---------------------------------------------------------------------------

def _billing_on() -> None:
    if not settings.billing_enabled:
        raise feature_disabled("billing is not enabled on this deployment")


def _unavailable(exc: stripe_client.BillingUnavailable) -> EntitlementError:
    return EntitlementError(503, "billing_unavailable", "billing is temporarily unavailable; your plan is unchanged")


def _stripe_error(exc: stripe_client.StripeError) -> EntitlementError:
    # Stripe's own message can echo request params; the code is enough
    # for the UI ("card_declined" etc.) and safe to return.
    return EntitlementError(502, "billing_error", "Stripe refused the request",
                            extra={"stripe_code": exc.code, "stripe_type": exc.error_type})


@router.post("/api/billing/checkout", response_model=CheckoutOut)
def checkout(body: CheckoutIn, request: Request, db: Session = Depends(get_db)) -> CheckoutOut:
    """A Stripe-hosted Checkout URL. The redirect back is not proof of
    anything — Pro appears on `/api/me` only once the webhook (or a
    reconcile) has written the subscription row."""
    principal = _require_user(request)
    _billing_on()
    enforce_scope(request, db, "checkout")
    if not principal.email_verified:
        raise EntitlementError(403, "email_unverified", "verify your email address before subscribing")
    user = _load_user(db, principal)
    try:
        result = billing_service.create_checkout(db, user, email=principal.email, interval=body.interval)
    except billing_service.AlreadySubscribed:
        raise EntitlementError(409, "already_subscribed",
                               "this account already has an active subscription; use Manage billing instead") from None
    except stripe_client.BillingUnavailable as exc:
        raise _unavailable(exc) from None
    except stripe_client.StripeError as exc:
        raise _stripe_error(exc) from None
    return CheckoutOut(url=result.url, billing_starts=result.billing_starts, trial_end=result.trial_end)


@router.post("/api/billing/portal", response_model=PortalOut)
def portal(request: Request, db: Session = Depends(get_db)) -> PortalOut:
    principal = _require_user(request)
    _billing_on()
    enforce_scope(request, db, "checkout")
    user = _load_user(db, principal)
    try:
        url = billing_service.create_portal(db, user)
    except billing_service.NoBillingAccount:
        raise EntitlementError(409, "no_billing_account", "no billing account yet — subscribe first") from None
    except stripe_client.BillingUnavailable as exc:
        raise _unavailable(exc) from None
    except stripe_client.StripeError as exc:
        raise _stripe_error(exc) from None
    return PortalOut(url=url)


@router.post("/api/billing/reconcile", response_model=ReconcileOut)
def reconcile(request: Request, db: Session = Depends(get_db)) -> ReconcileOut:
    """Pull the customer's subscriptions from Stripe and apply them — the
    'my payment went through but I am still Free' button. Additive: a
    fetch failure leaves the plan exactly as it was."""
    principal = _require_user(request)
    _billing_on()
    enforce_scope(request, db, "reconcile")
    user = _load_user(db, principal)
    try:
        outcome = billing_service.reconcile_user(db, user)
    except stripe_client.BillingUnavailable as exc:
        raise _unavailable(exc) from None
    principal.plan_state = None  # re-resolve after the write
    account = _account(db, user, principal)
    return ReconcileOut(**account.model_dump(), reconcile=outcome)


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

def _reply(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"detail": {"code": code, "message": message}}, status_code=status)


@router.post("/api/billing/webhook", response_model=WebhookOut)
@limiter.exempt
async def webhook(request: Request):
    """Stripe → us. See the module docstring for the status-code contract."""
    declared = request.headers.get("content-length")
    try:
        if declared is not None and int(declared) > MAX_WEBHOOK_BYTES:
            return _reply(413, "payload_too_large", f"webhook body exceeds {MAX_WEBHOOK_BYTES} bytes")
    except ValueError:
        return _reply(400, "bad_request", "invalid Content-Length")

    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_WEBHOOK_BYTES:
            return _reply(413, "payload_too_large", f"webhook body exceeds {MAX_WEBHOOK_BYTES} bytes")
        chunks.append(chunk)
    body = b"".join(chunks)

    secret = settings.stripe_webhook_secret
    if not secret:
        # Cannot verify → cannot accept. 503 makes Stripe retry, which is
        # right: the secret is on its way, the events must not be lost.
        log.error("webhook received but STRIPE_WEBHOOK_SECRET is not set — refusing")
        return _reply(503, "billing_unavailable", "webhook secret not configured")
    try:
        stripe_client.verify_signature(body, request.headers.get("stripe-signature", ""), secret)
    except stripe_client.SignatureError as exc:
        log.warning("webhook signature rejected: %s", exc)
        return _reply(400, "invalid_signature", "Stripe-Signature did not verify")

    try:
        event = json.loads(body)
    except ValueError:
        return _reply(400, "bad_request", "body is not JSON")
    if not isinstance(event, dict):
        return _reply(400, "bad_request", "body is not a Stripe event")

    from starlette.concurrency import run_in_threadpool

    def _process() -> str:
        from ..database import SessionLocal
        with SessionLocal() as db:
            return billing_service.apply_event(db, event)

    try:
        outcome = await run_in_threadpool(_process)
    except ValueError:
        return _reply(400, "bad_request", "body is not a Stripe event")
    except billing_service.WebhookStorageError as exc:
        log.error("webhook storage failed (%s); asking Stripe to retry", exc)
        return _reply(500, "webhook_storage_failed", "could not record the event; retry")
    return WebhookOut(received=True, outcome=outcome)


# ---------------------------------------------------------------------------
# Admin routes (`admin_auth` guards the prefix)
# ---------------------------------------------------------------------------

def _admin_user(db: Session, external_id: str) -> User:
    user = db.execute(select(User).where(User.external_id == external_id)).scalar_one_or_none()
    if user is None:
        raise EntitlementError(404, "not_found", "no user with that external id")
    return user


def _override_out(o: AdminOverride) -> OverrideOut:
    return OverrideOut(
        id=o.id, user_id=o.user_id, kind=o.kind, feature=o.feature, value=o.value, starts_at=o.starts_at,
        expires_at=o.expires_at, reason=o.reason or "", created_by=o.created_by or "admin",
        created_at=o.created_at, revoked_at=o.revoked_at,
    )


def _subscription_out(s: Subscription | None) -> AdminSubscriptionOut | None:
    if s is None:
        return None
    return AdminSubscriptionOut(
        id=s.id, stripe_subscription_id=s.stripe_subscription_id, stripe_customer_id=s.stripe_customer_id,
        stripe_price_id=s.stripe_price_id, billing_interval=s.billing_interval, stripe_status=s.stripe_status or "",
        current_period_start=s.current_period_start, current_period_end=s.current_period_end,
        cancel_at_period_end=bool(s.cancel_at_period_end), canceled_at=s.canceled_at, ended_at=s.ended_at,
        past_due_since=s.past_due_since, grace_until=s.grace_until,
        latest_event_created=int(s.latest_event_created or 0), updated_at=s.updated_at,
    )


@router.get("/api/admin/billing/users/{external_id}", response_model=AdminBillingUserOut)
def admin_billing_user(external_id: str, db: Session = Depends(get_db)) -> AdminBillingUserOut:
    """Support view: the user (no email — none is stored), the current
    subscription row, every override (revoked ones included, for the
    audit trail) and the last 20 usage events."""
    user = _admin_user(db, external_id)
    now = datetime.utcnow()
    state = resolve_for_user(db, user, now)
    overrides = db.execute(select(AdminOverride).where(AdminOverride.user_id == user.id)
                           .order_by(AdminOverride.id.desc())).scalars().all()
    events = db.execute(select(UsageEvent).where(UsageEvent.user_id == user.id)
                        .order_by(UsageEvent.id.desc()).limit(20)).scalars().all()
    return AdminBillingUserOut(
        user=AdminUserOut(
            id=user.id, external_id=user.external_id, account_state=user.account_state or "active",
            email_verified=user.email_verified_at is not None, created_at=user.created_at,
            last_seen_at=user.last_seen_at, trial_started_at=user.trial_started_at, trial_ends_at=user.trial_ends_at,
            trial_source=user.trial_source, stripe_customer_id=user.stripe_customer_id,
            first_value_at=user.first_value_at,
        ),
        plan={
            "plan": state.plan, "source": state.source, "ends_at": state.ends_at,
            "warning": state.warning, "suspended": state.suspended, "period_key": usage.period_key(now),
        },
        subscription=_subscription_out(current_subscription(db, user.id)),
        overrides=[_override_out(o) for o in overrides],
        usage_events=[AdminUsageEventOut(
            id=e.id, feature=e.feature, period_key=e.period_key, quantity=e.quantity, status=e.status,
            resource_ref=e.resource_ref, plan_at_charge=e.plan_at_charge, created_at=e.created_at,
            finalized_at=e.finalized_at,
        ) for e in events],
    )


@router.post("/api/admin/billing/overrides", response_model=OverrideOut, status_code=201)
def admin_create_override(body: OverrideIn, db: Session = Depends(get_db)) -> OverrideOut:
    """Operator escape hatch. `plan` (value `pro`) and `suspend` are read
    by `resolve_plan` at request time; `quota` (a registered feature +
    `unlimited` or an integer) replaces that feature's monthly limit in
    `entitlements.authorize` for as long as the row is active, for a
    feature the plan already allows; `trial_reset` also rewrites the
    user's trial columns — the one writer of those besides bootstrap,
    and an operator decision rather than a Stripe event — so it takes
    effect immediately and is auditable through the override row."""
    user = _admin_user(db, body.external_id)
    now = datetime.utcnow()
    value = (body.value or "").strip() or None
    if body.expires_at is not None and body.expires_at <= now:
        raise EntitlementError(422, "invalid_override", "expires_at is in the past")
    if body.kind == "plan":
        if (value or "").lower() != "pro":
            raise EntitlementError(422, "invalid_override", "a plan override's value must be 'pro'")
        value = "pro"
    elif body.kind == "quota":
        if not body.feature:
            raise EntitlementError(422, "invalid_override", "a quota override names a feature")
        if body.feature not in features.FEATURES or not features.get(body.feature).metered:
            # An override nothing reads would be a 201 that changes nothing.
            raise EntitlementError(422, "invalid_override",
                                   f"quota overrides apply to a metered feature: "
                                   f"{', '.join(n for n, f in features.FEATURES.items() if f.metered)}")
        value = (value or "").lower()
        if value != QUOTA_UNLIMITED and not value.isdigit():
            raise EntitlementError(422, "invalid_override", "a quota override's value is 'unlimited' or an integer")
    elif body.kind == "trial_reset":
        days = max(0, int(settings.trial_days))
        user.trial_started_at = now
        user.trial_ends_at = now + timedelta(days=days)
        user.trial_source = "admin"
        value = str(days)

    row = AdminOverride(
        user_id=user.id, kind=body.kind, feature=body.feature, value=value, starts_at=now,
        expires_at=body.expires_at, reason=body.reason.strip(), created_by=body.created_by.strip() or "admin",
        created_at=now,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    log.info("admin override %s created: kind=%s user=%s by=%s", row.id, row.kind, user.id, row.created_by)
    return _override_out(row)
