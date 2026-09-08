"""`authorize()` — the one function every protected route goes through.

Order of checks, each producing a structured error the frontend can
render as something other than "an error occurred":

  AUTH_ENABLED off        → allow, uncharged (today's behaviour)
  anonymous               → 401 auth_required
  suspended               → 403 account_suspended
  unverified + cost-bearing → 403 email_unverified
  plan lacks the feature  → 402 plan_required (incl. Free without a memo
                            for DCF/comps — "follows memo")
  concurrency lease full  → 429 concurrency_limited
  meter full              → 402 quota_exceeded with used/limit/resets_at

A successful call returns a `Grant`. Metered grants hold a reserved
`usage_events` id; the route (or the worker, for research runs) commits
it on success and releases it on failure, so a generation that blows up
costs the customer nothing. `require_feature(name)` is the FastAPI
dependency form that does that commit/release automatically.

Frontend gating is UX only. This is the authorization.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.requests import Request

from ..config import settings
from ..database import SessionLocal, get_db
from ..models.accounts import AdminOverride, Subscription, User
from ..schemas.accounts import EntitlementOut, StructuredError
from . import analytics, features, ratelimit, usage
from .plans import PlanState, resolve_plan
from .principal import Principal, current_principal
from .sanitize import safe_logger

log = safe_logger(__name__)

UPGRADE_URL = "/pricing"


class EntitlementError(HTTPException):
    """HTTPException whose `detail` is a `StructuredError` dict."""

    def __init__(self, status_code: int, code: str, message: str, *, headers: dict[str, str] | None = None, **fields: Any) -> None:
        err = StructuredError(code=code, message=message, **fields)
        super().__init__(
            status_code=status_code,
            detail=err.model_dump(mode="json", exclude_none=True),
            headers=headers,
        )
        self.code = code


# --- loaders ---------------------------------------------------------------

def current_subscription(db: Session, user_id: int) -> Subscription | None:
    """Highest id with `ended_at IS NULL`, else the most recent row."""
    live = db.execute(select(Subscription).where(
        Subscription.user_id == user_id, Subscription.ended_at.is_(None),
    ).order_by(Subscription.id.desc()).limit(1)).scalar_one_or_none()
    if live is not None:
        return live
    return db.execute(select(Subscription).where(
        Subscription.user_id == user_id,
    ).order_by(Subscription.id.desc()).limit(1)).scalar_one_or_none()


def active_overrides(db: Session, user_id: int, now: datetime | None = None) -> list[AdminOverride]:
    now = now or datetime.utcnow()
    rows = db.execute(select(AdminOverride).where(
        AdminOverride.user_id == user_id, AdminOverride.revoked_at.is_(None),
    )).scalars().all()
    return [o for o in rows if (o.starts_at is None or o.starts_at <= now)
            and (o.expires_at is None or o.expires_at > now)]


def resolve_for_user(db: Session, user: User, now: datetime | None = None) -> PlanState:
    now = now or datetime.utcnow()
    return resolve_plan(user, current_subscription(db, user.id), active_overrides(db, user.id, now), now)


def _plan_state(db: Session, principal: Principal, now: datetime) -> PlanState:
    if principal.plan_state is not None:
        return principal.plan_state
    user = db.get(User, principal.user_id)
    if user is None:
        raise EntitlementError(401, "auth_required", "account not found; sign in again")
    state = resolve_for_user(db, user, now)
    principal.plan_state = state
    return state


# --- grants ----------------------------------------------------------------

@dataclass
class Grant:
    feature: str
    plan: str
    charged: bool = False
    usage_event_id: int | None = None
    lease_token: str | None = None
    used: int = 0
    limit: int | None = None
    remaining: int | None = None
    resets_at: datetime | None = None
    user_id: int | None = None
    replayed: bool = False
    _finalized: bool = field(default=False, repr=False)

    def commit(self, db: Session | None = None) -> None:
        """Make the charge stick and drop the lease. Idempotent."""
        if self._finalized:
            return
        self._finalized = True
        self._finish(db, commit=True)

    def release(self, db: Session | None = None) -> None:
        """Give the units back (the work failed) and drop the lease. Idempotent."""
        if self._finalized:
            return
        self._finalized = True
        self._finish(db, commit=False)

    def _finish(self, db: Session | None, *, commit: bool) -> None:
        own = db is None
        session = db or SessionLocal()
        try:
            if self.usage_event_id is not None:
                if commit:
                    usage.commit(session, self.usage_event_id)
                else:
                    usage.release(session, self.usage_event_id)
            if self.lease_token:
                ratelimit.release_lease(session, self.lease_token)
            if commit and self.charged and self.user_id is not None and self.feature in FIRST_VALUE_FEATURES:
                user = session.get(User, self.user_id)
                if user is not None:
                    analytics.mark_first_value(session, user, feature=self.feature)
        except Exception as exc:  # a failed finalisation must not fail the response
            log.warning("grant finalisation failed for %s: %s", self.feature, type(exc).__name__)
            try:
                session.rollback()
            except Exception:  # pragma: no cover
                pass
        finally:
            if own:
                session.close()


FIRST_VALUE_FEATURES = frozenset({"memo_view", "research_run", "dcf", "pm_chat"})


def _uncharged(feature: str, plan: str, *, user_id: int | None = None, used: int = 0, limit: int | None = None, now: datetime | None = None) -> Grant:
    remaining = None if limit is None else max(0, limit - used)
    return Grant(feature=feature, plan=plan, charged=False, used=used, limit=limit,
                 remaining=remaining, resets_at=usage.resets_at(now), user_id=user_id)


def authorize(
    request: Request,
    feature: str,
    *,
    resource: str | None = None,
    idempotency_key: str | None = None,
    db: Session | None = None,
    quantity: int = 1,
    now: datetime | None = None,
) -> Grant:
    """See the module docstring for the check order. `resource` is the
    ticker for memo/dcf/comps; `idempotency_key` makes a retried request
    reuse its reservation (callers with a natural key pass it; a random
    one is generated otherwise, which means "charge each call")."""
    feat = features.get(feature)
    now = now or datetime.utcnow()
    if not settings.auth_enabled:
        return Grant(feature=feature, plan="unrestricted", charged=False)

    principal = current_principal(request)
    if principal.is_anon:
        raise EntitlementError(401, "auth_required", "sign in to continue", feature=feature,
                               headers={"WWW-Authenticate": "Bearer"})

    own_session = db is None
    session = db or SessionLocal()
    try:
        state = _plan_state(session, principal, now)
        if state.suspended or principal.account_state != "active":
            raise EntitlementError(403, "account_suspended", "this account is suspended", feature=feature)
        if feat.cost_bearing and not principal.email_verified:
            raise EntitlementError(403, "email_unverified",
                                   "verify your email address to run research", feature=feature, plan=state.plan)

        plan = state.plan
        allowance = features.allowance(feature, plan)
        user_id = principal.user_id
        assert user_id is not None
        pk = usage.period_key(now)
        resource = (resource or "").strip().upper() or None

        if not allowance.allowed:
            raise EntitlementError(
                402, "plan_required", f"{feat.description} is a Pro feature",
                feature=feature, plan=plan, upgrade_url=UPGRADE_URL,
            )

        if not settings.usage_limits_enabled:
            # Plan gating applies; meters, leases and follows-memo do not.
            return _uncharged(feature, plan, user_id=user_id, now=now)

        if allowance.follows_memo:
            if not resource:
                raise EntitlementError(402, "plan_required", f"{feat.description} needs a ticker on Free",
                                       feature=feature, plan=plan, upgrade_url=UPGRADE_URL)
            if resource not in usage.distinct_resources(session, user_id, "memo_view", pk):
                raise EntitlementError(
                    402, "plan_required",
                    f"On Free, {feat.description.lower()} is available for tickers whose memo you have opened "
                    f"this month. Open the {resource} memo first, or upgrade to Pro.",
                    feature=feature, plan=plan, upgrade_url=UPGRADE_URL, extra={"ticker": resource},
                )
            return _uncharged(feature, plan, user_id=user_id, now=now)

        lease_token: str | None = None
        if feat.max_concurrent:
            lease_token = ratelimit.lease(session, user_id=user_id, feature=feature,
                                          max_concurrent=feat.max_concurrent, resource_ref=resource, now=now)
            if lease_token is None:
                raise EntitlementError(
                    429, "concurrency_limited",
                    f"You already have {feat.max_concurrent} {feat.description.lower()} requests in flight; "
                    "wait for one to finish.",
                    feature=feature, plan=plan, scope=f"concurrency:{feature}", retry_after=5, window_seconds=120,
                    headers={"Retry-After": "5"},
                )

        if not allowance.metered:
            grant = _uncharged(feature, plan, user_id=user_id, now=now)
            grant.lease_token = lease_token
            return grant

        limit = allowance.limit
        if feat.distinct_resources and resource:
            seen = usage.distinct_resources(session, user_id, feature, pk)
            if resource in seen:
                grant = _uncharged(feature, plan, user_id=user_id, used=len(seen), limit=limit, now=now)
                grant.lease_token = lease_token
                return grant
            idempotency_key = idempotency_key or f"{user_id}:{feature}:{pk}:{resource}"
        key = idempotency_key or f"{user_id}:{feature}:{pk}:{uuid.uuid4().hex}"

        res = usage.reserve(
            session, user_id=user_id, feature=feature, limit=limit, idempotency_key=key,
            resource_ref=resource, plan_at_charge=plan, quantity=quantity, now=now,
        )
        if not res.allowed:
            ratelimit.release_lease(session, lease_token)
            analytics.track("quota_hit", db=session, principal=principal,
                            props={"feature": feature, "plan": plan, "used": res.used, "limit": res.limit})
            raise EntitlementError(
                402, "quota_exceeded", _quota_message(feat, plan, res.limit),
                feature=feature, plan=plan, used=res.used, limit=res.limit, remaining=0,
                resets_at=usage.resets_at(now), upgrade_url=UPGRADE_URL,
            )
        return Grant(
            feature=feature, plan=plan, charged=res.event is not None,
            usage_event_id=res.event.id if res.event is not None else None, lease_token=lease_token,
            used=res.used, limit=res.limit, remaining=res.remaining, resets_at=usage.resets_at(now),
            user_id=user_id, replayed=res.replayed,
        )
    finally:
        if own_session:
            session.close()


def _quota_message(feat: features.Feature, plan: str, limit: int | None) -> str:
    label = "Free Explorer" if plan == "free" else "Pro"
    what = feat.description.lower()
    if feat.distinct_resources:
        return f"{label} includes {limit} distinct tickers for {what} per calendar month (UTC)."
    return f"{label} includes {limit} × {what} per calendar month (UTC)."


def require_feature(name: str, *, resource_param: str | None = "ticker") -> Callable[..., Iterator[Grant]]:
    """FastAPI dependency: `Depends(require_feature("memo_view"))`.

    Authorizes before the handler runs, commits the reservation when the
    handler returns, releases it when the handler raises. `resource_param`
    names the path parameter carrying the ticker (None for features that
    have no resource, e.g. chat)."""
    features.get(name)  # fail at import time on a typo, not at first request

    def dependency(request: Request, db: Session = Depends(get_db)) -> Iterator[Grant]:
        resource = None
        if resource_param:
            raw = request.path_params.get(resource_param)
            resource = str(raw).upper() if raw else None
        grant = authorize(request, name, resource=resource, db=db)
        request.state.grant = grant
        try:
            yield grant
        except Exception:
            grant.release(db)
            raise
        else:
            grant.commit(db)

    dependency.__name__ = f"require_{name}"
    return dependency


# --- account snapshot (for /api/me) ------------------------------------------

def entitlement_snapshot(db: Session, user: User, state: PlanState, *, now: datetime | None = None) -> dict[str, EntitlementOut]:
    now = now or datetime.utcnow()
    pk = usage.period_key(now)
    counters = usage.counters_for(db, user.id, pk)
    reset = usage.resets_at(now)
    out: dict[str, EntitlementOut] = {}
    for name, feat in features.FEATURES.items():
        a = features.allowance(name, state.plan)
        used_n = counters.get(name, 0)
        limit = a.limit if (a.metered and settings.usage_limits_enabled) else None
        allowed = a.allowed and not state.suspended
        remaining = None if limit is None else max(0, limit - used_n)
        out[name] = EntitlementOut(
            feature=name, allowed=allowed, limit=limit, used=used_n, remaining=remaining,
            resets_at=reset, follows_memo=a.follows_memo, metered=feat.metered,
        )
    return out
