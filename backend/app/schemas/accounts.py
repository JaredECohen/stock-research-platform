"""Account / billing schemas (FEAT-002). Additive — nothing here is
referenced by the memo pipeline, so this module sits beside `chat.py` at
the leaf of the schema DAG and imports only `common`-level things.

`StructuredError` is the shape every entitlement, quota and rate-limit
refusal puts inside FastAPI's `{"detail": ...}` envelope. The frontend
switches on `code`; everything else is there so the UpgradePrompt /
RateLimitNotice can say something specific instead of "an error occurred".
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ErrorCode = Literal[
    "auth_required",        # 401 — no token
    "auth_invalid",         # 401 — token present but not verifiable
    "auth_unavailable",     # 503 — login wall on, Clerk unreachable/unconfigured
    "email_unverified",     # 403
    "account_suspended",    # 403
    "plan_required",        # 402
    "quota_exceeded",       # 402
    "rate_limited",         # 429
    "concurrency_limited",  # 429
    "feature_disabled",     # 404
    "no_memo",              # 409 — GET /memo under auth with nothing stored
    "already_subscribed",   # 409
    "billing_unavailable",  # 503
]


class StructuredError(BaseModel):
    code: str
    message: str = ""
    feature: str | None = None
    plan: str | None = None
    used: int | None = None
    limit: int | None = None
    remaining: int | None = None
    resets_at: datetime | None = None
    upgrade_url: str | None = None
    # rate limits
    scope: str | None = None
    retry_after: int | None = None
    window_seconds: int | None = None
    # route-specific extras (e.g. `analyze_path` on `no_memo`)
    extra: dict[str, Any] = Field(default_factory=dict)


class EntitlementOut(BaseModel):
    feature: str
    allowed: bool
    limit: int | None = None       # None = unlimited (when allowed)
    used: int = 0
    remaining: int | None = None
    resets_at: datetime | None = None
    # Free users may open DCF/comps only for tickers already counted as a
    # memo view this month; the UI explains the rule with this flag.
    follows_memo: bool = False
    metered: bool = False


class PlanStateOut(BaseModel):
    plan: Literal["free", "pro", "none"]
    source: Literal["trial", "subscription", "override", "grace", "default", "suspended"]
    trial_ends_at: datetime | None = None
    period_end: datetime | None = None
    cancel_at_period_end: bool = False
    grace_until: datetime | None = None
    ends_at: datetime | None = None
    warning: str | None = None


class UserOut(BaseModel):
    id: int
    external_id: str
    email_verified: bool
    created_at: datetime
    account_state: str
    trial_started_at: datetime | None = None
    trial_ends_at: datetime | None = None


class BillingOut(BaseModel):
    has_subscription: bool = False
    stripe_status: str | None = None
    interval: str | None = None
    portal_available: bool = False
    billing_enabled: bool = False


class AccountOut(BaseModel):
    user: UserOut
    plan: PlanStateOut
    entitlements: dict[str, EntitlementOut]
    billing: BillingOut
    period_key: str
    usage_limits_enabled: bool


class BootstrapOut(AccountOut):
    trial_started_now: bool = False


class UsageHistoryItem(BaseModel):
    feature: str
    resource_ref: str | None = None
    created_at: datetime
    status: str
    quantity: int = 1


class UsageOut(BaseModel):
    period_key: str
    features: dict[str, EntitlementOut]
    history: list[UsageHistoryItem] = Field(default_factory=list)


class PublicPrices(BaseModel):
    monthly_cents: int = 2999
    annual_cents: int = 29900
    currency: str = "usd"


class PublicConfigOut(BaseModel):
    auth_enabled: bool
    billing_enabled: bool
    usage_limits_enabled: bool
    clerk_publishable_key: str | None = None
    clerk_frontend_api: str | None = None
    sample_tickers: list[str] = Field(default_factory=list)
    prices: PublicPrices = Field(default_factory=PublicPrices)
    legal_reviewed: bool = False
    app_env: str = "development"
    trial_days: int = 7
    # Allowance table for the pricing page, straight from the registry so
    # the copy and the enforcement cannot drift apart.
    features: dict[str, dict[str, Any]] = Field(default_factory=dict)
