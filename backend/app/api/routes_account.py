"""Account endpoints (FEAT-002): `/api/me`, `/api/me/bootstrap`,
`/api/me/usage`, and the public `/api/public/config`.

Everything a signed-in user can learn about their own account and
nothing about anyone else's: the principal comes from the middleware,
and every query is keyed on `principal.user_id`. The address itself is
not returned (the backend does not store it); the account page reads it
from the Clerk client session.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import analytics, features, usage
from ..auth.entitlements import EntitlementError, entitlement_snapshot, resolve_for_user
from ..auth.principal import Principal, current_principal
from ..auth.ratelimit import client_ip
from ..auth.sanitize import safe_logger
from ..config import settings
from ..database import get_db
from ..models.accounts import Subscription, User
from ..rate_limit import LIMITS, limiter
from ..schemas.accounts import (
    AccountOut,
    BillingOut,
    BootstrapOut,
    EntitlementOut,
    PlanStateOut,
    PublicConfigOut,
    PublicPrices,
    UsageHistoryItem,
    UsageOut,
    UserOut,
)

log = safe_logger(__name__)
router = APIRouter()

_PERIOD_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


# ---------------------------------------------------------------------------
# Public config
# ---------------------------------------------------------------------------

@router.get("/api/public/config", response_model=PublicConfigOut)
def public_config(response: Response) -> PublicConfigOut:
    """What the frontend needs before anyone signs in: flags, the Clerk
    publishable key (public by definition), sample tickers, prices and
    the entitlement matrix. Cacheable for a minute; nothing here is
    per-user."""
    response.headers["Cache-Control"] = "public, max-age=60"
    return PublicConfigOut(
        auth_enabled=settings.auth_enabled,
        billing_enabled=settings.billing_enabled,
        usage_limits_enabled=settings.usage_limits_enabled,
        clerk_publishable_key=settings.clerk_publishable_key or None,
        clerk_frontend_api=settings.clerk_issuer or None,
        sample_tickers=settings.sample_tickers_list,
        prices=PublicPrices(),
        legal_reviewed=settings.legal_reviewed,
        app_env=settings.app_env,
        trial_days=settings.trial_days,
        features=features.registry_for_config(),
    )


# ---------------------------------------------------------------------------
# /api/me
# ---------------------------------------------------------------------------

def _require_user(request: Request) -> Principal:
    """The middleware already refused anonymous callers when the wall is
    on; this covers the wall-off case (the endpoint has no meaning) and
    direct handler calls."""
    if not settings.auth_enabled:
        raise EntitlementError(404, "feature_disabled", "accounts are not enabled on this deployment")
    principal = current_principal(request)
    if principal.is_anon:
        raise EntitlementError(401, "auth_required", "sign in to continue", headers={"WWW-Authenticate": "Bearer"})
    return principal


def _load_user(db: Session, principal: Principal) -> User:
    user = db.get(User, principal.user_id)
    if user is None:
        raise EntitlementError(401, "auth_required", "account not found; sign in again")
    return user


def _account(db: Session, user: User, principal: Principal, *, now: datetime | None = None) -> AccountOut:
    now = now or datetime.utcnow()
    state = resolve_for_user(db, user, now)
    principal.plan_state = state
    sub = db.execute(select(Subscription).where(
        Subscription.user_id == user.id,
    ).order_by(Subscription.id.desc()).limit(1)).scalar_one_or_none()
    return AccountOut(
        user=UserOut(
            id=user.id, external_id=user.external_id,
            email_verified=user.email_verified_at is not None or principal.email_verified,
            created_at=user.created_at, account_state=user.account_state or "active",
            trial_started_at=user.trial_started_at, trial_ends_at=user.trial_ends_at,
        ),
        plan=PlanStateOut(
            plan=state.plan, source=state.source, trial_ends_at=state.trial_ends_at,
            period_end=state.period_end, cancel_at_period_end=state.cancel_at_period_end,
            grace_until=state.grace_until, ends_at=state.ends_at, warning=state.warning,
        ),
        entitlements=entitlement_snapshot(db, user, state, now=now),
        billing=BillingOut(
            has_subscription=sub is not None,
            stripe_status=sub.stripe_status if sub is not None else None,
            interval=sub.billing_interval if sub is not None else None,
            portal_available=bool(settings.billing_enabled and settings.billing_configured and user.stripe_customer_id),
            billing_enabled=settings.billing_enabled,
        ),
        period_key=usage.period_key(now),
        usage_limits_enabled=settings.usage_limits_enabled,
    )


@router.get("/api/me", response_model=AccountOut)
def get_me(request: Request, db: Session = Depends(get_db)) -> AccountOut:
    principal = _require_user(request)
    user = _load_user(db, principal)
    return _account(db, user, principal)


def _abuse_hash(value: str) -> str:
    return hashlib.sha256(f"{settings.abuse_hash_salt}|{value}".encode()).hexdigest()


@router.post("/api/me/bootstrap", response_model=BootstrapOut)
@limiter.limit(LIMITS["bootstrap"])
def bootstrap(request: Request, response: Response, db: Session = Depends(get_db)) -> BootstrapOut:
    """Idempotent first call after sign-in.

    Starts the card-less Pro trial exactly once: only when the email is
    verified, this user has never had one, and no other user with the
    same `email_hash` has had one (a re-registered address does not get a
    second trial). Records the salted IP / user-agent hashes on the first
    call so repeated trial creation from one source is visible later.
    Never resets a trial — that is an `admin_overrides(kind=trial_reset)`.
    """
    principal = _require_user(request)
    user = _load_user(db, principal)
    now = datetime.utcnow()
    started_now = False

    if user.bootstrap_ip_hash is None:
        user.bootstrap_ip_hash = _abuse_hash(client_ip(request))
        user.bootstrap_ua_hash = _abuse_hash(request.headers.get("user-agent", ""))
        db.commit()
        analytics.track("signup_completed", db=db, principal=principal, props={"source": "clerk"})

    if user.trial_started_at is None and principal.email_verified:
        already = None
        if user.email_hash:
            already = db.execute(select(User.id).where(
                User.email_hash == user.email_hash, User.id != user.id, User.trial_started_at.is_not(None),
            ).limit(1)).first()
        if already is None:
            user.trial_started_at = now
            user.trial_ends_at = now + timedelta(days=max(0, int(settings.trial_days)))
            user.trial_source = "signup"
            db.commit()
            started_now = True
            analytics.track(
                "trial_activated", db=db, principal=principal,
                props={"trial_source": "signup", "plan": "pro"},
            )
        else:
            log.info("trial not started for user %s: another account with the same email already had one", user.id)

    account = _account(db, user, principal, now=now)
    return BootstrapOut(**account.model_dump(), trial_started_now=started_now)


@router.get("/api/me/usage", response_model=UsageOut)
def get_usage(
    request: Request,
    period: str | None = Query(default=None, description="UTC month, YYYY-MM; defaults to the current month"),
    db: Session = Depends(get_db),
) -> UsageOut:
    principal = _require_user(request)
    user = _load_user(db, principal)
    now = datetime.utcnow()
    key = usage.period_key(now)
    if period:
        if not _PERIOD_RE.match(period):
            raise EntitlementError(422, "invalid_period", "period must look like YYYY-MM")
        key = period
    state = resolve_for_user(db, user, now)
    # Entitlement snapshot is for the CURRENT month; for a past month we
    # report the counters as they stand with no limits attached.
    if key == usage.period_key(now):
        feats = entitlement_snapshot(db, user, state, now=now)
    else:
        counters = usage.counters_for(db, user.id, key)
        _start, end = usage.period_bounds(key)
        feats = {
            name: EntitlementOut(feature=name, allowed=True, limit=None, used=counters.get(name, 0),
                                 remaining=None, resets_at=end, metered=features.get(name).metered)
            for name in features.FEATURES
        }
    hist = [
        UsageHistoryItem(feature=e.feature, resource_ref=e.resource_ref, created_at=e.created_at,
                         status=e.status, quantity=e.quantity)
        for e in usage.history(db, user.id, key, limit=50)
    ]
    return UsageOut(period_key=key, features=feats, history=hist)
