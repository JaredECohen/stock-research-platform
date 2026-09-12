"""Plan resolution: rows in → effective plan out. Pure, so the whole
state table is exhaustively testable without a database.

The effective plan is computed at READ time from `users`, the current
`subscriptions` row and active `admin_overrides`. No cron flips anyone
to Free when a trial or a grace period ends — the boundary is a
comparison against `now`, which means two processes never disagree and
there is no job that can silently fail to downgrade.

Stripe status → plan (the §5.2 table):

  trialing                       pro until current_period_end (EXPECTED:
                                 Checkout during a local trial carries
                                 the remaining trial time to Stripe)
  active, !cancel_at_period_end  pro, rolling
  active,  cancel_at_period_end  pro until current_period_end, then free
  past_due                       pro until grace_until, then free
  unpaid                         free (after grace_until if still ahead)
  canceled                       free from ended_at
  incomplete/_expired, paused    free
  no subscription, trial ahead   pro (trial) until trial_ends_at
  override plan=pro              pro until expires_at
  override suspend / suspended   none — every route 403s

Resolution is "max": any source that says Pro wins, and among Pro
sources the one that lasts longest (a rolling subscription beats a
dated trial). A Stripe read failure never reaches this function — the
DB row is authoritative and reconcile only ever applies a successfully
fetched object — so an outage cannot revoke Pro.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..config import settings

PLAN_FREE = "free"
PLAN_PRO = "pro"
PLAN_NONE = "none"


@dataclass
class PlanState:
    plan: str = PLAN_FREE
    source: str = "default"  # trial / subscription / override / grace / default / suspended
    trial_ends_at: datetime | None = None
    period_end: datetime | None = None
    cancel_at_period_end: bool = False
    grace_until: datetime | None = None
    # When Pro stops unless something renews it; None = rolling.
    ends_at: datetime | None = None
    warning: str | None = None
    suspended: bool = False

    @property
    def is_pro(self) -> bool:
        return self.plan == PLAN_PRO


def _fmt(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d") if dt else "?"


def _override_active(o: Any, now: datetime) -> bool:
    if getattr(o, "revoked_at", None) is not None:
        return False
    starts = getattr(o, "starts_at", None)
    if starts is not None and starts > now:
        return False
    expires = getattr(o, "expires_at", None)
    if expires is not None and expires <= now:
        return False
    return True


def _from_subscription(sub: Any, now: datetime, grace_days: int) -> PlanState | None:
    """Plan implied by the current subscription row, or None when there
    is no row. Free results carry the warning the UI should show."""
    if sub is None:
        return None
    status = (getattr(sub, "stripe_status", "") or "").lower()
    period_end = getattr(sub, "current_period_end", None)
    cancel = bool(getattr(sub, "cancel_at_period_end", False))
    base = dict(period_end=period_end, cancel_at_period_end=cancel)

    if status == "trialing":
        if period_end is None or period_end > now:
            return PlanState(PLAN_PRO, "subscription", ends_at=period_end, **base)
        return PlanState(PLAN_FREE, "default", **base)

    if status == "active":
        if not cancel:
            return PlanState(PLAN_PRO, "subscription", ends_at=None, **base)
        if period_end is not None and period_end > now:
            return PlanState(
                PLAN_PRO, "subscription", ends_at=period_end,
                warning=f"Pro ends on {_fmt(period_end)}", **base,
            )
        return PlanState(PLAN_FREE, "default", **base)

    if status == "past_due":
        grace_until = getattr(sub, "grace_until", None)
        if grace_until is None:
            since = getattr(sub, "past_due_since", None) or getattr(sub, "updated_at", None) or now
            grace_until = since + timedelta(days=grace_days)
        if grace_until > now:
            return PlanState(
                PLAN_PRO, "grace", ends_at=grace_until, grace_until=grace_until,
                warning="Payment failed — update your card to keep Pro", **base,
            )
        return PlanState(
            PLAN_FREE, "default", grace_until=grace_until,
            warning="Payment failed — Pro is paused until the card is updated", **base,
        )

    if status == "unpaid":
        grace_until = getattr(sub, "grace_until", None)
        if grace_until is not None and grace_until > now:
            return PlanState(
                PLAN_PRO, "grace", ends_at=grace_until, grace_until=grace_until,
                warning="Subscription unpaid — update your card to keep Pro", **base,
            )
        return PlanState(PLAN_FREE, "default", grace_until=grace_until, warning="Subscription unpaid", **base)

    if status == "canceled":
        ended_at = getattr(sub, "ended_at", None)
        if ended_at is not None and ended_at > now:
            return PlanState(
                PLAN_PRO, "subscription", ends_at=ended_at,
                warning=f"Pro ends on {_fmt(ended_at)}", **base,
            )
        if ended_at is None and cancel and period_end is not None and period_end > now:
            return PlanState(
                PLAN_PRO, "subscription", ends_at=period_end,
                warning=f"Pro ends on {_fmt(period_end)}", **base,
            )
        return PlanState(PLAN_FREE, "default", **base)

    if status in ("incomplete", "incomplete_expired"):
        return PlanState(PLAN_FREE, "default", warning="Checkout not completed", **base)
    if status == "paused":
        return PlanState(PLAN_FREE, "default", warning="Subscription paused", **base)
    # Unknown status: never grant on something we do not understand.
    return PlanState(PLAN_FREE, "default", **base)


def resolve_plan(
    user: Any,
    subscription: Any,
    overrides: list[Any] | tuple[Any, ...] | None,
    now: datetime | None = None,
    *,
    grace_days: int | None = None,
) -> PlanState:
    """Effective plan for `user` right now. Naive-UTC datetimes throughout,
    matching every persisted column in this codebase."""
    now = now or datetime.utcnow()
    grace_days = settings.grace_days if grace_days is None else grace_days
    overrides = [o for o in (overrides or []) if _override_active(o, now)]
    trial_ends_at = getattr(user, "trial_ends_at", None)

    # Suspension beats everything, including a paid subscription: a
    # suspended account is an operator decision, not a billing state.
    if (getattr(user, "account_state", "active") or "active") != "active" or any(
        getattr(o, "kind", "") == "suspend" for o in overrides
    ):
        return PlanState(PLAN_NONE, "suspended", trial_ends_at=trial_ends_at, suspended=True)

    candidates: list[PlanState] = []
    sub_state = _from_subscription(subscription, now, grace_days)
    if sub_state is not None:
        candidates.append(sub_state)
    if trial_ends_at is not None and trial_ends_at > now:
        candidates.append(PlanState(
            PLAN_PRO, "trial", ends_at=trial_ends_at, trial_ends_at=trial_ends_at,
            warning=f"Trial ends {_fmt(trial_ends_at)}",
        ))
    for o in overrides:
        if getattr(o, "kind", "") == "plan" and (getattr(o, "value", "") or "").lower() == PLAN_PRO:
            candidates.append(PlanState(PLAN_PRO, "override", ends_at=getattr(o, "expires_at", None)))

    pro = [c for c in candidates if c.is_pro]
    if pro:
        # Rolling (ends_at None) beats any dated grant; else the latest end.
        rolling = [c for c in pro if c.ends_at is None]
        chosen = rolling[0] if rolling else max(pro, key=lambda c: c.ends_at)
    elif sub_state is not None:
        chosen = sub_state
    else:
        chosen = PlanState(PLAN_FREE, "default")
    chosen.trial_ends_at = trial_ends_at
    if sub_state is not None and chosen is not sub_state:
        # Keep the billing facts visible even when a trial/override wins.
        chosen.period_end = chosen.period_end or sub_state.period_end
        chosen.cancel_at_period_end = chosen.cancel_at_period_end or sub_state.cancel_at_period_end
    return chosen
