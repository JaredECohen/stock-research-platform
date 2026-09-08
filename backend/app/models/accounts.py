"""FEAT-002 — customer accounts, subscriptions, meters, limits.

Everything a login wall and a paid plan need to remember, and nothing
that a login wall could leak. No plaintext email is stored: `users`
carries a salted-free sha256 of the normalised address for the
one-trial-per-person check and support lookup, and the account page
shows the address from the Clerk client session instead. Stripe rows
keep ids, statuses and period bounds — never card data.

Two design rules, both learned the hard way elsewhere in this repo:

  - **Every table here is cross-process state.** Web serves the API,
    the worker runs the memo graph; they share only Postgres. A usage
    reservation made on web is committed or released by the worker, so
    it has to be a row, not a dict.
  - **Every column added to an existing table is nullable or defaulted**
    (see `RegenJob` / `LLMCallLog`), and every new table is created by
    `create_all`, so `database.init_db()` migrates a long-lived DB with
    no manual step. Rollback is a flag flip; the tables are inert.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class User(Base):
    """One row per Clerk identity. `external_id` is the Clerk `user_…` id
    and is immutable; the integer `id` is what every other table joins on.

    Trial columns are written exactly once, by `POST /api/me/bootstrap`,
    and never by webhook processing — a Stripe event must not be able to
    extend a trial. `bootstrap_ip_hash` / `bootstrap_ua_hash` (salted
    sha256) exist so repeated trial creation from one source is visible in
    the abuse telemetry without storing the IP itself. `first_value_at`
    is the once-per-user marker behind the `first_value` analytics event;
    a column rather than a query so two processes agree on "once".
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    auth_provider: Mapped[str] = mapped_column(String(16), default="clerk")
    email_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    account_state: Mapped[str] = mapped_column(String(16), default="active")  # active / suspended / deleted
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    trial_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    trial_ends_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    trial_source: Mapped[str | None] = mapped_column(String(24), nullable=True)  # signup / admin
    stripe_customer_id: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    marketing_consent: Mapped[bool] = mapped_column(Boolean, default=False)
    # Reserved: reminder emails are deferred to a follow-up (no email
    # provider in the repo). Kept so the column exists when they land.
    trial_reminder_sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    bootstrap_ip_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    bootstrap_ua_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    first_value_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Subscription(Base):
    """Server-side mirror of a Stripe subscription — the only thing that
    ever grants Pro from a purchase. A checkout redirect is not proof of
    payment; a webhook (or an authenticated reconcile) writing this row is.

    `latest_event_created` is the unix timestamp of the last applied
    webhook so an out-of-order delivery (an `updated` arriving after the
    `deleted` it preceded) is ignored rather than resurrecting access.
    `snapshot` holds a whitelisted subset of the Stripe object (ids,
    status, period, price) — never card or address data. Historical rows
    are kept; "current" is the highest id with `ended_at IS NULL`, else
    the most recent.
    """
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    stripe_subscription_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    stripe_customer_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    stripe_price_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    plan: Mapped[str] = mapped_column(String(16), default="pro")
    billing_interval: Mapped[str | None] = mapped_column(String(8), nullable=True)  # month / year
    stripe_status: Mapped[str] = mapped_column(String(32), default="")
    current_period_start: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, default=False)
    canceled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    past_due_since: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    grace_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    latest_event_created: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    snapshot: Mapped[dict] = mapped_column(JSON, default=dict)


Index("ix_subscriptions_user_status", Subscription.user_id, Subscription.stripe_status)


class UsageCounter(Base):
    """Fast path for meters: one row per (user, feature, UTC month).

    `used` only moves through the conditional UPDATE in `auth/usage.py`
    (`used + q <= limit`), which is what makes a reservation atomic on
    both Postgres and SQLite without an application-side lock.
    """
    __tablename__ = "usage_counters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    feature: Mapped[str] = mapped_column(String(32))
    period_key: Mapped[str] = mapped_column(String(16))  # YYYY-MM, UTC calendar month
    used: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "feature", "period_key", name="uq_usage_counter_user_feature_period"),
    )


class UsageEvent(Base):
    """Audit + idempotency for every charge.

    `idempotency_key` is unique, so a retried request returns the existing
    event instead of charging twice. `status` walks reserved → committed
    (the work succeeded) or reserved → released (it failed, the counter
    was decremented). `resource_ref` (a ticker) is what "distinct tickers
    this month" — the Free memo allowance and the DCF/comps-follows-memo
    rule — is computed from. Never GC'd: this is the customer's bill.
    """
    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    feature: Mapped[str] = mapped_column(String(32))
    period_key: Mapped[str] = mapped_column(String(16))
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(12), default="reserved")  # reserved / committed / released
    resource_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    plan_at_charge: Mapped[str] = mapped_column(String(16), default="free")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


Index(
    "ix_usage_events_user_feature_period_resource",
    UsageEvent.user_id, UsageEvent.feature, UsageEvent.period_key, UsageEvent.resource_ref,
)


class AdminOverride(Base):
    """Operator escape hatch: grant Pro, lift a quota, reset a trial, or
    suspend. `kind` ∈ plan / quota / trial_reset / suspend; `value` is
    `pro`, `unlimited`, or an integer, depending on `kind`. `reason` is
    mandatory in spirit — an override with no reason is a support ticket
    nobody can close.
    """
    __tablename__ = "admin_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    kind: Mapped[str] = mapped_column(String(24))
    feature: Mapped[str | None] = mapped_column(String(32), nullable=True)
    value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    starts_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String(64), default="admin")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class BillingWebhookEvent(Base):
    """One row per Stripe event id, inserted before processing — the
    unique constraint is what makes a redelivered webhook a no-op. The
    raw payload is deliberately not stored (it carries customer detail);
    `outcome` ∈ applied / ignored_stale / ignored_unhandled / duplicate /
    error is enough to audit what happened. Kept forever.
    """
    __tablename__ = "billing_webhook_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stripe_event_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True, default="")
    event_created: Mapped[int] = mapped_column(Integer, default=0)
    object_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    outcome: Mapped[str] = mapped_column(String(24), default="")
    error: Mapped[str] = mapped_column(Text, default="")


class RateLimitWindow(Base):
    """Fixed-window counters for the identity-aware limiter.

    `key` is `{scope}:{identity}:{window_start_epoch}`; one upsert per
    request on the primary key. Lives in the DB rather than slowapi's
    `memory://` store because the limit is per *user*, and a user's
    requests can land on any process. Expired rows are GC'd by the
    billing loop; a stale row is harmless (its window has passed).
    """
    __tablename__ = "rate_limit_windows"

    key: Mapped[str] = mapped_column(String(160), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class ActiveAction(Base):
    """Concurrency leases: "this user has N chat turns in flight".

    A row is a lease with a TTL; the check is a count of unexpired rows
    per (user, feature). Released in `finally`, and a crashed process
    simply lets its leases expire — that is the whole reason for the TTL.
    """
    __tablename__ = "active_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    feature: Mapped[str] = mapped_column(String(32))
    resource_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_token: Mapped[str] = mapped_column(String(64), unique=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)


Index("ix_active_actions_user_feature", ActiveAction.user_id, ActiveAction.feature)
