"""Stripe state machine + the operations around it (FEAT-002, S4).

The one rule everything here serves: **a checkout redirect is never
proof of payment.** Pro is granted from a `subscriptions` row, and the
only writers of that row are a signature-verified webhook
(`apply_event`) and an authenticated reconcile that applies an object
fetched from Stripe with the account's own key (`reconcile_user`,
`reconcile_stale_subscriptions`). Both go through
`apply_subscription_object`, so there is exactly one place where a
Stripe status becomes local state.

What `apply_subscription_object` guarantees:

  - **Ordering.** `subscriptions.latest_event_created` is the unix time
    the row's state is "as of". A webhook older than that is
    `ignored_stale` — an `updated` that was delayed past the `deleted`
    it preceded cannot resurrect access. A reconcile fetch is applied as
    of *now*, because a freshly fetched object is newer than any event
    already delivered.
  - **Transitions** (plan §5.2, as `auth/plans.py` reads them):
    `past_due` sets `past_due_since` (once) and `grace_until = since +
    GRACE_DAYS`; `active` / `trialing` clear both; `canceled` sets
    `ended_at`. `trialing` is an EXPECTED Pro state: Checkout during a
    local trial carries the remaining trial time to Stripe.
  - **Trial columns are never written.** A Stripe event must not be able
    to extend or reset `users.trial_started_at` / `trial_ends_at`; the
    only writers are `/api/me/bootstrap` and an operator override.
  - **Analytics never touch the caller's transaction.** Funnel events
    derived from a transition (`trial_converted`, `subscription_canceled`,
    …) are queued on the session and written in their own session only
    after the caller's commit succeeds; a rollback drops them. The
    alternative — `analytics.track(db=db)` mid-apply — commits (and on a
    failure rolls back) the caller's session, which once discarded a
    flushed subscription row while the event was still recorded as
    `applied`, so the redelivery was answered `duplicate` and the
    customer never became Pro.
  - **Ownership.** A subscription is applied to the user whose
    `stripe_customer_id` matches the object's customer; a
    `checkout.session.completed` must also carry that user's id in
    `client_reference_id`. Mismatch → outcome `error`, no grant, 200 (so
    Stripe does not retry a payload that will never verify).
  - **Snapshot is whitelisted.** Ids, status, period, price, flags. No
    card, address, or email data is ever copied into the DB.

`apply_event` adds idempotency (`billing_webhook_events.stripe_event_id`
is unique: a redelivery is `duplicate`, no state change) and turns a
storage failure into `WebhookStorageError`, which the route maps to 500
so Stripe retries; the unique row makes that retry safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import event as sa_event
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from ..agents.log_safety import safe_exc
from ..auth import analytics, usage
from ..auth.entitlements import current_subscription
from ..auth.sanitize import safe_logger
from ..config import settings
from ..models.accounts import BillingWebhookEvent, Subscription, UsageEvent, User
from ..models.jobs import RegenJob
from . import stripe_client
from .stripe_client import BillingUnavailable, StripeError

log = safe_logger(__name__)

_EPOCH = datetime(1970, 1, 1)

# Stripe statuses under which a customer already has a subscription that
# a second Checkout would duplicate. `unpaid` (Stripe gave up collecting)
# and `incomplete*` (the first payment never went through) are not in
# the set: a fresh Checkout is the way out of both.
BLOCKING_STATUSES = frozenset({"active", "trialing", "past_due"})
# Rows worth re-fetching in the periodic reconcile.
LIVE_STATUSES = frozenset({"active", "trialing", "past_due", "unpaid"})

OUTCOME_APPLIED = "applied"
OUTCOME_STALE = "ignored_stale"
OUTCOME_UNHANDLED = "ignored_unhandled"
OUTCOME_DUPLICATE = "duplicate"
OUTCOME_ERROR = "error"

HANDLED_EVENT_TYPES = (
    "checkout.session.completed",
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
    "invoice.paid",
    "invoice.payment_failed",
    "customer.deleted",
)

# Stripe's minimum for `subscription_data[trial_end]`.
MIN_TRIAL_CARRYOVER = timedelta(hours=48)


class AlreadySubscribed(Exception):
    status_code = 409
    code = "already_subscribed"


class NoBillingAccount(Exception):
    """Portal asked for before any Checkout created a Stripe customer."""

    status_code = 409
    code = "no_billing_account"


class OwnershipMismatch(Exception):
    """The event's user/customer references do not agree with our rows.
    Recorded as outcome `error`; never applied."""


class WebhookStorageError(Exception):
    """The database refused a write while processing a webhook. The
    route answers 500 so Stripe retries; the unique event row makes the
    retry a no-op for anything already applied."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_PENDING_ANALYTICS = "billing_pending_analytics"


def _queue_analytics(db: Session, event_name: str, **kwargs: Any) -> None:
    """Record a funnel event to be written AFTER the caller's commit.

    The queue lives in `Session.info`; the first call on a session
    attaches `after_commit` (write the queue, each event in its own
    session — `analytics.track` never raises) and `after_rollback`
    (drop it: the transition it describes did not happen). Nothing here
    commits, flushes or rolls back the caller's session.
    """
    info = db.info
    if _PENDING_ANALYTICS not in info:
        info[_PENDING_ANALYTICS] = []
        sa_event.listen(db, "after_commit", _flush_pending_analytics)
        sa_event.listen(db, "after_rollback", _discard_pending_analytics)
    info[_PENDING_ANALYTICS].append((event_name, kwargs))


def _flush_pending_analytics(db: Session) -> None:
    pending = db.info.get(_PENDING_ANALYTICS) or []
    db.info[_PENDING_ANALYTICS] = []
    for name, kwargs in pending:
        analytics.track(name, **kwargs)  # own session; False on failure, never raises


def _discard_pending_analytics(db: Session) -> None:
    if _PENDING_ANALYTICS in db.info:
        db.info[_PENDING_ANALYTICS] = []


def _take_pending_analytics(db: Session) -> list[tuple[str, dict[str, Any]]]:
    pending = list(db.info.get(_PENDING_ANALYTICS) or [])
    _discard_pending_analytics(db)
    return pending


def _ts(value: Any) -> datetime | None:
    """Stripe unix seconds → naive UTC datetime (the codebase convention)."""
    if value is None or value == "":
        return None
    try:
        return _EPOCH + timedelta(seconds=int(value))
    except (TypeError, ValueError, OverflowError):
        return None


def unix(dt: datetime) -> int:
    return int((dt - _EPOCH).total_seconds())


def _id(value: Any) -> str | None:
    """Stripe fields are either an id string or an expanded object."""
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        v = value.get("id")
        return str(v) if v else None
    return None


def _first_item(obj: dict[str, Any]) -> dict[str, Any]:
    items = obj.get("items")
    data = items.get("data") if isinstance(items, dict) else None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return {}


def _period(obj: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    """`current_period_*` lives on the subscription up to API 2024-xx and
    on the first item from 2025-03 on; read whichever is present."""
    start = obj.get("current_period_start")
    end = obj.get("current_period_end")
    if start is None and end is None:
        item = _first_item(obj)
        start, end = item.get("current_period_start"), item.get("current_period_end")
    return _ts(start), _ts(end)


def extract_price(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    """(price id, billing interval) from the first subscription item."""
    price = _first_item(obj).get("price")
    if not isinstance(price, dict):
        return (_id(price), None)
    recurring = price.get("recurring") if isinstance(price.get("recurring"), dict) else {}
    interval = recurring.get("interval")
    return (_id(price), str(interval) if interval else None)


def snapshot_of(obj: dict[str, Any]) -> dict[str, Any]:
    """The subset of a Stripe subscription worth keeping. Everything else
    (default payment method, addresses, invoice settings) stays in Stripe."""
    price_id, interval = extract_price(obj)
    start, end = _period(obj)

    def iso(value: Any) -> str | None:
        dt = _ts(value)
        return dt.isoformat() if dt else None

    return {
        "id": _id(obj.get("id")),
        "status": obj.get("status"),
        "customer": _id(obj.get("customer")),
        "price_id": price_id,
        "interval": interval,
        "current_period_start": start.isoformat() if start else None,
        "current_period_end": end.isoformat() if end else None,
        "cancel_at_period_end": bool(obj.get("cancel_at_period_end")),
        "canceled_at": iso(obj.get("canceled_at")),
        "ended_at": iso(obj.get("ended_at")),
        "trial_end": iso(obj.get("trial_end")),
        "livemode": bool(obj.get("livemode")),
        "created": obj.get("created"),
    }


def _user_by_customer(db: Session, customer_id: str | None) -> User | None:
    if not customer_id:
        return None
    return db.execute(select(User).where(User.stripe_customer_id == customer_id)).scalar_one_or_none()


def _user_by_id(db: Session, value: Any) -> User | None:
    try:
        return db.get(User, int(str(value)))
    except (TypeError, ValueError):
        return None


def _metadata_user_id(obj: dict[str, Any]) -> str | None:
    meta = obj.get("metadata")
    if isinstance(meta, dict) and meta.get("user_id") not in (None, ""):
        return str(meta["user_id"])
    return None


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------

def apply_subscription_object(
    db: Session,
    user: User,
    obj: dict[str, Any],
    *,
    event_created: int,
    now: datetime | None = None,
    source: str = "webhook",
) -> tuple[Subscription | None, str]:
    """Upsert the `subscriptions` row for one Stripe subscription object
    belonging to `user`. Returns `(row, outcome)` where outcome is
    `applied` or `ignored_stale`. Flushes; the caller commits, so an
    event's state and its `billing_webhook_events` outcome land together.
    Never commits or rolls back — the funnel events it derives are queued
    (`_queue_analytics`) and written only once the caller's commit lands.

    Never touches `users.trial_*`. The only `users` write is filling an
    empty `stripe_customer_id` from the object — recovery for a Checkout
    whose customer save was lost.
    """
    now = now or datetime.utcnow()
    sub_id = _id(obj.get("id"))
    if not sub_id:
        raise OwnershipMismatch("subscription object has no id")
    customer_id = _id(obj.get("customer"))
    if user.stripe_customer_id and customer_id and user.stripe_customer_id != customer_id:
        raise OwnershipMismatch("subscription customer does not match the user's customer")

    row = db.execute(select(Subscription).where(
        Subscription.stripe_subscription_id == sub_id,
    )).scalar_one_or_none()
    created_now = row is None
    if row is not None:
        if row.user_id != user.id:
            raise OwnershipMismatch("subscription belongs to a different user")
        if int(event_created) < int(row.latest_event_created or 0):
            log.info("subscription %s: %s event at %s is older than applied state at %s — ignored",
                     sub_id, source, event_created, row.latest_event_created)
            return row, OUTCOME_STALE
    else:
        row = Subscription(user_id=user.id, stripe_subscription_id=sub_id, plan="pro", latest_event_created=0)
        db.add(row)

    prev_status = (row.stripe_status or "") if not created_now else ""
    prev_cancel = bool(row.cancel_at_period_end) if not created_now else False
    status = str(obj.get("status") or "").lower()
    event_time = _ts(event_created) or now
    price_id, interval = extract_price(obj)
    start, end = _period(obj)

    row.stripe_customer_id = customer_id or row.stripe_customer_id
    row.stripe_price_id = price_id or row.stripe_price_id
    row.billing_interval = interval or row.billing_interval
    row.stripe_status = status
    row.current_period_start = start
    row.current_period_end = end
    row.cancel_at_period_end = bool(obj.get("cancel_at_period_end"))
    row.canceled_at = _ts(obj.get("canceled_at"))
    row.ended_at = _ts(obj.get("ended_at"))

    if status == "canceled" and row.ended_at is None:
        # Stripe always sets `ended_at` on cancellation; if a hand-made
        # object lacks it, the cancellation moment is the honest value.
        row.ended_at = row.canceled_at or event_time
    if status in ("past_due", "unpaid"):
        if row.past_due_since is None:
            row.past_due_since = event_time
            row.grace_until = row.past_due_since + timedelta(days=max(0, int(settings.grace_days)))
    elif status in ("active", "trialing"):
        row.past_due_since = None
        row.grace_until = None
    # canceled / incomplete / paused keep whatever grace was recorded —
    # `resolve_plan` does not consult it for those statuses.

    row.latest_event_created = max(int(row.latest_event_created or 0), int(event_created))
    row.updated_at = now
    row.snapshot = snapshot_of(obj)
    if not user.stripe_customer_id and customer_id:
        user.stripe_customer_id = customer_id
    db.flush()

    _track_transitions(db, user, row, created_now=created_now, prev_status=prev_status,
                       prev_cancel=prev_cancel, now=now, source=source)
    return row, OUTCOME_APPLIED


def _track_transitions(db: Session, user: User, row: Subscription, *, created_now: bool,
                       prev_status: str, prev_cancel: bool, now: datetime, source: str) -> None:
    """Funnel events derived from the transition just applied. Queued,
    not written: they go out after the caller commits and are dropped on
    rollback, so a lost event is never a lost subscription and a lost
    subscription never has an event claiming otherwise."""
    status = row.stripe_status or ""
    props = {"interval": row.billing_interval, "source": source, "status": status}
    if created_now and status in ("active", "trialing"):
        if user.trial_ends_at is not None and user.trial_ends_at > now:
            _queue_analytics(db, "trial_converted", user_id=user.id, plan="pro", props=props, now=now)
    if status == "canceled" and prev_status != "canceled":
        _queue_analytics(db, "subscription_canceled", user_id=user.id, plan="pro",
                         props={**props, "reason": "canceled"}, now=now)
    elif row.cancel_at_period_end and not prev_cancel and status != "canceled":
        _queue_analytics(db, "subscription_canceled", user_id=user.id, plan="pro",
                         props={**props, "reason": "cancel_at_period_end"}, now=now)


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------

def _claim_event(db: Session, event: dict[str, Any], obj: dict[str, Any]) -> BillingWebhookEvent | None:
    """Insert the idempotency row. None when the event was already fully
    processed (a redelivery). An existing row with no `processed_at` is a
    previous attempt that died before recording an outcome — the retry
    Stripe sent is exactly what should process it, so it is returned."""
    event_id = str(event["id"])
    row = BillingWebhookEvent(
        stripe_event_id=event_id[:64], event_type=str(event.get("type") or "")[:64],
        event_created=int(event.get("created") or 0), object_id=(_id(obj.get("id")) or "")[:64] or None,
        received_at=datetime.utcnow(),
    )
    db.add(row)
    try:
        db.commit()
        return row
    except IntegrityError:
        db.rollback()
    existing = db.execute(select(BillingWebhookEvent).where(
        BillingWebhookEvent.stripe_event_id == event_id[:64],
    )).scalar_one_or_none()
    if existing is None:  # pragma: no cover — the unique violation says it exists
        raise WebhookStorageError("event row vanished after a unique violation")
    if existing.processed_at is not None:
        return None
    return existing


def apply_event(db: Session, event: dict[str, Any], *, now: datetime | None = None) -> str:
    """Process one verified Stripe event. Returns the outcome recorded in
    `billing_webhook_events.outcome`. Raises `ValueError` for a payload
    that is not an event (route → 400) and `WebhookStorageError` when the
    database refuses a write (route → 500 so Stripe retries)."""
    now = now or datetime.utcnow()
    event_id = event.get("id") if isinstance(event, dict) else None
    data = event.get("data") if isinstance(event, dict) else None
    obj = data.get("object") if isinstance(data, dict) else None
    if not isinstance(event_id, str) or not event_id or not isinstance(obj, dict):
        raise ValueError("not a Stripe event")
    event_type = str(event.get("type") or "")
    try:
        created = int(event.get("created") or 0)
    except (TypeError, ValueError):
        created = 0

    try:
        row = _claim_event(db, event, obj)
    except SQLAlchemyError as exc:
        db.rollback()
        raise WebhookStorageError(type(exc).__name__) from None
    if row is None:
        log.info("webhook %s (%s) already processed — duplicate", event_id, event_type)
        return OUTCOME_DUPLICATE
    row_id = row.id

    error = ""
    try:
        outcome = _dispatch(db, event_type, obj, created=created, now=now)
    except SQLAlchemyError as exc:
        db.rollback()
        raise WebhookStorageError(type(exc).__name__) from None
    except OwnershipMismatch as exc:
        db.rollback()
        outcome, error = OUTCOME_ERROR, f"ownership: {exc}"
        log.warning("webhook %s (%s) refused: %s", event_id, event_type, exc)
    except (BillingUnavailable, StripeError) as exc:
        # A follow-up fetch failed; the event itself was fine. Recorded
        # so an operator can see it; not retried (the subscription
        # events carry the same state). The funnel events queued before
        # the fetch (`checkout_completed`, `subscription_renewed`)
        # describe the event, not the fetch, so they survive the rollback.
        pending = _take_pending_analytics(db)
        db.rollback()
        for name, kwargs in pending:
            _queue_analytics(db, name, **kwargs)
        outcome, error = OUTCOME_APPLIED, f"fetch skipped: {type(exc).__name__}"
        log.warning("webhook %s (%s): follow-up Stripe fetch failed: %s", event_id, event_type, type(exc).__name__)
    except Exception as exc:
        db.rollback()
        outcome, error = OUTCOME_ERROR, safe_exc(exc)
        log.warning("webhook %s (%s) failed: %s", event_id, event_type, type(exc).__name__)

    # One commit carries the state change (flushed, uncommitted) and the
    # outcome row; the queued analytics are written by `after_commit`.
    # If it fails nothing is `applied`, the row keeps `processed_at`
    # NULL, and the retry Stripe sends re-processes the event.
    try:
        stored = db.get(BillingWebhookEvent, row_id)
        if stored is not None:
            stored.outcome = outcome[:24]
            stored.error = error[:2000]
            stored.processed_at = now
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise WebhookStorageError(type(exc).__name__) from None
    return outcome


def _dispatch(db: Session, event_type: str, obj: dict[str, Any], *, created: int, now: datetime) -> str:
    if event_type == "checkout.session.completed":
        return _on_checkout_completed(db, obj, created=created, now=now)
    if event_type in ("customer.subscription.created", "customer.subscription.updated",
                      "customer.subscription.deleted"):
        return _on_subscription_event(db, obj, created=created, now=now)
    if event_type in ("invoice.paid", "invoice.payment_failed"):
        return _on_invoice_event(db, event_type, obj, created=created, now=now)
    if event_type == "customer.deleted":
        return _on_customer_deleted(db, obj, created=created, now=now)
    return OUTCOME_UNHANDLED


def _on_checkout_completed(db: Session, session: dict[str, Any], *, created: int, now: datetime) -> str:
    if session.get("mode") != "subscription":
        return OUTCOME_UNHANDLED
    user = _user_by_id(db, session.get("client_reference_id"))
    if user is None:
        raise OwnershipMismatch("client_reference_id does not name a user")
    customer_id = _id(session.get("customer"))
    if not customer_id or user.stripe_customer_id != customer_id:
        raise OwnershipMismatch("session customer does not match the user's customer")
    _queue_analytics(db, "checkout_completed", user_id=user.id, plan="pro", props={"source": "webhook"}, now=now)

    sub_id = _id(session.get("subscription"))
    if not sub_id:
        return OUTCOME_APPLIED
    existing = db.execute(select(Subscription).where(
        Subscription.stripe_subscription_id == sub_id,
    )).scalar_one_or_none()
    if existing is not None:
        # `customer.subscription.created` got here first; nothing to add.
        return OUTCOME_APPLIED
    # The session does not embed the subscription. Fetch it so the grant
    # does not wait on the (usually simultaneous) subscription event; if
    # Stripe is unreachable that event will grant when it lands. A fetched
    # object is state as of NOW, not as of the session event, so it is
    # applied like a reconcile: newer than any webhook already delivered.
    obj = stripe_client.retrieve_subscription(sub_id)
    _, outcome = apply_subscription_object(db, user, obj, event_created=unix(now), now=now, source="fetch")
    return outcome


def _resolve_owner(db: Session, obj: dict[str, Any]) -> User:
    """Which user a subscription object belongs to. By customer id first;
    a metadata `user_id` may only *confirm* that, or fill in a customer
    id we created but failed to save — never override a different owner."""
    customer_id = _id(obj.get("customer"))
    user = _user_by_customer(db, customer_id)
    meta_uid = _metadata_user_id(obj)
    if user is not None:
        if meta_uid is not None and meta_uid != str(user.id):
            raise OwnershipMismatch("metadata user_id disagrees with the customer's user")
        return user
    if meta_uid is not None:
        candidate = _user_by_id(db, meta_uid)
        if candidate is not None and not candidate.stripe_customer_id:
            return candidate
    raise OwnershipMismatch("unknown customer")


def _on_subscription_event(db: Session, obj: dict[str, Any], *, created: int, now: datetime) -> str:
    user = _resolve_owner(db, obj)
    _, outcome = apply_subscription_object(db, user, obj, event_created=created, now=now)
    return outcome


def _invoice_subscription_id(invoice: dict[str, Any]) -> str | None:
    sub = _id(invoice.get("subscription"))
    if sub:
        return sub
    parent = invoice.get("parent")
    details = parent.get("subscription_details") if isinstance(parent, dict) else None
    return _id(details.get("subscription")) if isinstance(details, dict) else None


def _on_invoice_event(db: Session, event_type: str, invoice: dict[str, Any], *, created: int, now: datetime) -> str:
    """Invoices carry money facts, not subscription state. They emit the
    renewal analytics and, when Stripe is configured, trigger a fetch of
    the subscription so a `past_due → active` flip lands promptly even if
    the `customer.subscription.updated` event is delayed. The invoice's
    own timestamp never reaches the ordering guard: the fetched object is
    applied as of now (an invoice is often *older* than the subscription
    update it caused, and would otherwise read as stale)."""
    sub_id = _invoice_subscription_id(invoice)
    if not sub_id:
        return OUTCOME_UNHANDLED
    row = db.execute(select(Subscription).where(Subscription.stripe_subscription_id == sub_id)).scalar_one_or_none()
    if row is None:
        return OUTCOME_UNHANDLED
    user = db.get(User, row.user_id)
    if user is None:
        raise OwnershipMismatch("subscription row has no user")
    if event_type == "invoice.paid" and invoice.get("billing_reason") == "subscription_cycle":
        _queue_analytics(db, "subscription_renewed", user_id=user.id, plan="pro",
                         props={"interval": row.billing_interval}, now=now)
    if stripe_client.configured():
        obj = stripe_client.retrieve_subscription(sub_id)
        _, outcome = apply_subscription_object(db, user, obj, event_created=unix(now), now=now, source="fetch")
        return outcome
    return OUTCOME_APPLIED


def _on_customer_deleted(db: Session, customer: dict[str, Any], *, created: int, now: datetime) -> str:
    customer_id = _id(customer.get("id"))
    user = _user_by_customer(db, customer_id)
    if user is None:
        return OUTCOME_UNHANDLED
    event_time = _ts(created) or now
    rows = db.execute(select(Subscription).where(
        Subscription.user_id == user.id, Subscription.ended_at.is_(None),
    )).scalars().all()
    for row in rows:
        if int(created) < int(row.latest_event_created or 0):
            continue
        row.stripe_status = "canceled"
        row.ended_at = event_time
        row.canceled_at = row.canceled_at or event_time
        row.latest_event_created = int(created)
        row.updated_at = now
    user.stripe_customer_id = None
    db.flush()
    return OUTCOME_APPLIED


# ---------------------------------------------------------------------------
# Checkout / portal
# ---------------------------------------------------------------------------

@dataclass
class CheckoutResult:
    url: str
    billing_starts: str  # "now" | "at_trial_end"
    trial_end: datetime | None
    session_id: str | None = None


def _price_for(interval: str) -> str:
    if interval == "year":
        return settings.stripe_price_pro_annual
    return settings.stripe_price_pro_monthly


def ensure_customer(db: Session, user: User, *, email: str | None) -> str:
    """The user's Stripe customer id, creating the customer on first use.

    The idempotency key is `customer:<user_id>`, so a retry after a lost
    response — or two racing first checkouts — gets Stripe's stored
    reply for the same customer rather than a second one.
    """
    if user.stripe_customer_id:
        return user.stripe_customer_id
    customer = stripe_client.create_customer(
        email=email, user_id=user.id, external_id=user.external_id, idempotency_key=f"customer:{user.id}",
    )
    customer_id = _id(customer.get("id"))
    if not customer_id:
        raise BillingUnavailable("Stripe returned a customer without an id")
    user.stripe_customer_id = customer_id
    try:
        db.commit()
    except IntegrityError:
        # Another request saved (the same) id first.
        db.rollback()
        db.refresh(user)
        if not user.stripe_customer_id:
            raise
    return user.stripe_customer_id


def create_checkout(
    db: Session,
    user: User,
    *,
    email: str | None,
    interval: str,
    now: datetime | None = None,
) -> CheckoutResult:
    """A Stripe-hosted Checkout URL for Pro.

    Trial hand-off: while the local card-less trial has at least 48h
    left, Checkout is created with `subscription_data[trial_end]` equal
    to `users.trial_ends_at`, so Stripe's subscription starts `trialing`
    and the first charge lands when the trial the customer was promised
    ends. With less than 48h left (Stripe's floor) billing starts at
    checkout and the response says so (`billing_starts="now"`).
    """
    now = now or datetime.utcnow()
    interval = "year" if interval == "year" else "month"
    price_id = _price_for(interval)
    if not (settings.billing_configured and price_id and settings.public_base_url):
        raise BillingUnavailable("billing is not fully configured (key, webhook secret, price ids, PUBLIC_BASE_URL)")

    current = current_subscription(db, user.id)
    if current is not None and current.ended_at is None and (current.stripe_status or "") in BLOCKING_STATUSES:
        raise AlreadySubscribed()

    customer_id = ensure_customer(db, user, email=email)

    trial_end: datetime | None = None
    billing_starts = "now"
    if user.trial_ends_at is not None and user.trial_ends_at - now >= MIN_TRIAL_CARRYOVER:
        trial_end = user.trial_ends_at
        billing_starts = "at_trial_end"

    base = settings.public_base_url.rstrip("/")
    session = stripe_client.create_checkout_session(
        customer_id=customer_id, price_id=price_id,
        success_url=f"{base}/app/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base}/app/billing/canceled",
        client_reference_id=str(user.id), user_id=user.id, external_id=user.external_id,
        trial_end=unix(trial_end) if trial_end else None,
        # One session per user/interval/minute: a double click or a retry
        # after a timeout reuses it instead of littering Stripe.
        idempotency_key=f"checkout:{user.id}:{interval}:{unix(now) // 60}",
    )
    url = session.get("url")
    if not isinstance(url, str) or not url:
        raise BillingUnavailable("Stripe returned a Checkout session without a url")
    analytics.track("checkout_started", db=db, user_id=user.id, plan=("pro" if trial_end else "free"),
                    props={"interval": interval, "source": billing_starts}, now=now)
    return CheckoutResult(url=url, billing_starts=billing_starts, trial_end=trial_end,
                          session_id=_id(session.get("id")))


def create_portal(db: Session, user: User, *, now: datetime | None = None) -> str:
    """A Stripe-hosted billing-portal URL for the user's customer.

    Every Stripe write carries an idempotency key; a portal session's is
    `portal:<user>:<second>`, so a double click (or an httpx retry
    after a timeout) within the same second gets the one link back
    instead of minting another, while a later click gets a fresh
    session — portal URLs are short-lived and single-use."""
    now = now or datetime.utcnow()
    if not (settings.billing_configured and settings.public_base_url):
        raise BillingUnavailable("billing is not fully configured")
    if not user.stripe_customer_id:
        raise NoBillingAccount()
    session = stripe_client.create_portal_session(
        customer_id=user.stripe_customer_id,
        return_url=f"{settings.public_base_url.rstrip('/')}/app/account",
        configuration_id=settings.stripe_portal_configuration_id or None,
        idempotency_key=f"portal:{user.id}:{unix(now)}",
    )
    url = session.get("url")
    if not isinstance(url, str) or not url:
        raise BillingUnavailable("Stripe returned a portal session without a url")
    return url


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def reconcile_user(db: Session, user: User, *, now: datetime | None = None) -> dict[str, Any]:
    """Fetch the customer's subscriptions with our key and apply them.

    Additive by construction: a fetch failure returns `ok=False` and
    changes nothing, so a Stripe outage can never revoke Pro; only a
    successfully fetched object goes through the state machine, as of
    `now` (newer than any webhook already delivered). The fetched objects
    are applied as one transaction: an ownership refusal on any of them
    rolls back all of them (and drops their queued analytics), so
    "nothing changed" is literally true.
    """
    now = now or datetime.utcnow()
    if not stripe_client.configured():
        raise BillingUnavailable("STRIPE_SECRET_KEY is not set")
    if not user.stripe_customer_id:
        return {"ok": True, "fetched": 0, "applied": 0}
    try:
        objects = stripe_client.list_subscriptions(user.stripe_customer_id)
    except (BillingUnavailable, StripeError) as exc:
        log.warning("reconcile for user %s: Stripe fetch failed (%s); nothing changed", user.id, type(exc).__name__)
        return {"ok": False, "fetched": 0, "applied": 0, "error": type(exc).__name__}
    applied = 0
    try:
        for obj in objects:
            _, outcome = apply_subscription_object(db, user, obj, event_created=unix(now), now=now, source="reconcile")
            applied += outcome == OUTCOME_APPLIED
        db.commit()
    except OwnershipMismatch as exc:
        db.rollback()
        log.warning("reconcile for user %s refused: %s", user.id, exc)
        return {"ok": False, "fetched": len(objects), "applied": 0, "error": "ownership"}
    return {"ok": True, "fetched": len(objects), "applied": applied}


def reconcile_stale_subscriptions(db: Session, *, now: datetime | None = None, limit: int = 50,
                                  max_age: timedelta = timedelta(hours=24)) -> dict[str, int]:
    """Billing-loop duty: re-fetch up to `limit` live rows not updated in
    `max_age` and apply what Stripe says. Per-row failures are counted
    and skipped — a row is never changed on a fetch error."""
    now = now or datetime.utcnow()
    if not stripe_client.configured():
        return {"checked": 0, "applied": 0, "errors": 0}
    rows = db.execute(select(Subscription).where(
        Subscription.ended_at.is_(None), Subscription.updated_at < now - max_age,
        Subscription.stripe_status.in_(sorted(LIVE_STATUSES)),
    ).order_by(Subscription.updated_at.asc()).limit(limit)).scalars().all()
    applied = errors = 0
    for row in rows:
        user = db.get(User, row.user_id)
        if user is None:
            errors += 1
            continue
        try:
            obj = stripe_client.retrieve_subscription(row.stripe_subscription_id)
            _, outcome = apply_subscription_object(db, user, obj, event_created=unix(now), now=now, source="reconcile")
            db.commit()
            applied += outcome == OUTCOME_APPLIED
        except (BillingUnavailable, StripeError, OwnershipMismatch) as exc:
            db.rollback()
            errors += 1
            log.warning("stale-subscription reconcile skipped %s: %s", row.stripe_subscription_id, type(exc).__name__)
    return {"checked": len(rows), "applied": applied, "errors": errors}


def reconcile_stale_reservations(db: Session, *, cutoff: datetime, now: datetime | None = None) -> dict[str, int]:
    """Finalise `usage_events` still `reserved` after `cutoff`.

    A reservation is committed or released by the request (or the worker
    job) that made it; one that is still reserved hours later belongs to
    a process that died in between. For a research run the job row says
    what happened: succeeded → commit (the memo was delivered), failed →
    release, still queued/running → leave it (orphan recovery owns it).
    A reservation with no job, or for an in-request feature, delivered
    nothing the customer kept, so it is released.
    """
    now = now or datetime.utcnow()
    committed = released = left = 0
    for event in usage.reserved_events_older_than(db, cutoff):
        job = _job_for_event(db, event)
        if job is not None and job.status not in ("succeeded", "failed"):
            left += 1
            continue
        if job is not None and job.status == "succeeded":
            committed += usage.commit(db, event.id, now=now)
        else:
            released += usage.release(db, event.id, now=now)
    return {"committed": committed, "released": released, "left": left}


def _job_for_event(db: Session, event: UsageEvent) -> RegenJob | None:
    return db.execute(select(RegenJob).where(RegenJob.usage_event_id == event.id)
                      .order_by(RegenJob.id.desc()).limit(1)).scalar_one_or_none()
