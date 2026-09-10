"""FEAT-002 — hourly billing housekeeping on the worker.

No plan changes here. `auth/plans.resolve_plan` computes the effective
plan at read time, so a trial or a grace period ending needs no cron to
downgrade anyone — the boundary is a comparison against `now`, and the
two processes cannot disagree. What does need a periodic pass:

  1. **Funnel events** for boundaries nobody's request crossed:
     `trial_expired` (a trial ended and the user is not Pro by any other
     source) and `downgraded` (a subscription ended, was canceled at
     period end, or its grace ran out, and the user is now Free). Once
     per user: a candidate is skipped when an event of that name already
     exists for them since the boundary. Candidates are drawn from a
     `LOOKBACK_DAYS` window that is shorter than the analytics retention
     (90d), so GC can never re-open a boundary the loop already noted.
  2. **GC** of expired `rate_limit_windows` / `active_actions` and of
     `analytics_events` past retention (`auth/analytics.RETENTION_DAYS`).
     `usage_events` and `billing_webhook_events` are audit and are never
     deleted.
  3. **Stale reservations**: a `usage_events` row still `reserved` six
     hours on belongs to a process that died between reserve and commit;
     `billing_service.reconcile_stale_reservations` settles it against
     the job row (succeeded → commit, else release).
  4. **Stale subscriptions**, only when `STRIPE_SECRET_KEY` is set on
     this process: up to 50 live rows not updated in 24h are re-fetched
     and applied. Per the deploy decision the worker carries no Stripe
     key, so in production this step is a no-op and webhooks +
     `POST /api/billing/reconcile` are the reconciliation path; the
     branch exists for a single-process deployment. A fetch error never
     changes a row.

Runs whether or not `AUTH_ENABLED` is on: every step is a no-op on
empty tables, and GC of the limiter tables is wanted regardless.
`record_run` is written on every path, including a crash, so
cron-health flags the loop rather than losing it (the `postmortem_loop`
lesson).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import or_, select

from ..agents.log_safety import safe_exc
from ..auth import analytics, ratelimit
from ..auth.entitlements import resolve_for_user
from ..database import SessionLocal
from ..models.accounts import Subscription, User
from ..models.public import AnalyticsEvent
from ..services import billing_service
from . import record_run

log = logging.getLogger(__name__)

LOOP_NAME = "billing_loop"
INTERVAL_HOURS = 1
BATCH = 200
LOOKBACK_DAYS = 30
STALE_RESERVATION_HOURS = 6
RECONCILE_LIMIT = 50

assert LOOKBACK_DAYS < analytics.RETENTION_DAYS, "the once-per-user check must outlive the candidate window"


def _already_noted(db, event_name: str, user_ids: list[int], since: datetime) -> set[int]:
    if not user_ids:
        return set()
    rows = db.execute(select(AnalyticsEvent.user_id).where(
        AnalyticsEvent.event_name == event_name, AnalyticsEvent.user_id.in_(user_ids), AnalyticsEvent.ts >= since,
    ).distinct()).all()
    return {int(r[0]) for r in rows if r[0] is not None}


def emit_trial_expired(db, *, now: datetime | None = None) -> int:
    """`trial_expired` for every user whose trial ended in the lookback
    window and who is not Pro now (a converted user is not "expired").
    Returns how many events this call wrote."""
    now = now or datetime.utcnow()
    since = now - timedelta(days=LOOKBACK_DAYS)
    written = 0
    last_id = 0
    while True:
        users = db.execute(select(User).where(
            User.trial_ends_at.is_not(None), User.trial_ends_at <= now, User.trial_ends_at > since,
            User.id > last_id,
        ).order_by(User.id).limit(BATCH)).scalars().all()
        if not users:
            break
        last_id = users[-1].id
        noted = _already_noted(db, "trial_expired", [u.id for u in users], since)
        for user in users:
            if user.id in noted:
                continue
            state = resolve_for_user(db, user, now)
            if state.is_pro:
                continue
            if analytics.track("trial_expired", db=db, user_id=user.id, plan=state.plan,
                               props={"trial_source": user.trial_source or "signup", "source": "billing_loop"},
                               now=now):
                written += 1
        if len(users) < BATCH:
            break
    return written


def _lapse_time(sub: Subscription, now: datetime) -> datetime | None:
    """When this row stopped granting Pro, or None if it has not."""
    status = (sub.stripe_status or "").lower()
    if sub.ended_at is not None and sub.ended_at <= now:
        return sub.ended_at
    if status in ("active", "canceled") and sub.cancel_at_period_end and sub.current_period_end is not None \
            and sub.current_period_end <= now:
        return sub.current_period_end
    if status in ("past_due", "unpaid") and sub.grace_until is not None and sub.grace_until <= now:
        return sub.grace_until
    return None


def emit_downgraded(db, *, now: datetime | None = None) -> int:
    """`downgraded` for every user whose subscription lapsed in the
    lookback window and who is Free now. Once per user per lapse: a
    `downgraded` event since the lapse time counts as noted."""
    now = now or datetime.utcnow()
    since = now - timedelta(days=LOOKBACK_DAYS)
    written = 0
    last_id = 0
    while True:
        subs = db.execute(select(Subscription).where(
            Subscription.id > last_id,
            or_(
                Subscription.ended_at > since,
                Subscription.current_period_end > since,
                Subscription.grace_until > since,
            ),
        ).order_by(Subscription.id).limit(BATCH)).scalars().all()
        if not subs:
            break
        last_id = subs[-1].id
        by_user: dict[int, datetime] = {}
        for sub in subs:
            lapsed = _lapse_time(sub, now)
            if lapsed is None or lapsed <= since:
                continue
            by_user[sub.user_id] = max(lapsed, by_user.get(sub.user_id, lapsed))
        for user_id, lapsed in by_user.items():
            if _already_noted(db, "downgraded", [user_id], lapsed):
                continue
            user = db.get(User, user_id)
            if user is None:
                continue
            state = resolve_for_user(db, user, now)
            if state.is_pro or state.suspended:
                continue
            if analytics.track("downgraded", db=db, user_id=user.id, plan=state.plan,
                               props={"reason": state.warning or "subscription_ended", "source": "billing_loop"},
                               now=now):
                written += 1
        if len(subs) < BATCH:
            break
    return written


def run_once(*, now: datetime | None = None) -> dict[str, Any]:
    """One pass; see the module docstring. Each step is isolated so a
    failure in one (a Stripe outage, say) does not stop the GC."""
    now = now or datetime.utcnow()
    result: dict[str, Any] = {
        "expired": 0, "downgraded": 0, "gc_limits": 0, "gc_analytics": 0,
        "reservations": {"committed": 0, "released": 0, "left": 0},
        "reconciled": {"checked": 0, "applied": 0, "errors": 0},
        "step_errors": [],
    }
    steps = (
        ("expired", lambda db: emit_trial_expired(db, now=now)),
        ("downgraded", lambda db: emit_downgraded(db, now=now)),
        ("gc_limits", lambda db: ratelimit.gc_expired(db, now=now)),
        ("gc_analytics", lambda db: analytics.gc_old(db, now=now)),
        ("reservations", lambda db: billing_service.reconcile_stale_reservations(
            db, cutoff=now - timedelta(hours=STALE_RESERVATION_HOURS), now=now)),
        ("reconciled", lambda db: billing_service.reconcile_stale_subscriptions(
            db, now=now, limit=RECONCILE_LIMIT)),
    )
    for name, step in steps:
        try:
            with SessionLocal() as db:
                result[name] = step(db)
        except Exception as exc:
            result["step_errors"].append(f"{name}: {safe_exc(exc)}")
            log.warning("billing_loop step %s failed: %s", name, type(exc).__name__, exc_info=True)

    res = result["reservations"]
    rec = result["reconciled"]
    note = (
        f"expired={result['expired']} downgraded={result['downgraded']} "
        f"gc={result['gc_limits']}+{result['gc_analytics']} "
        f"reservations={res['committed']}c/{res['released']}r/{res['left']}l "
        f"reconciled={rec['applied']}/{rec['checked']} errors={rec['errors']}"
    )
    if result["step_errors"]:
        note += " failed=" + ";".join(result["step_errors"])
    record_run(LOOP_NAME, success=not result["step_errors"], note=note[:1000])
    result["note"] = note
    return result


def tick() -> dict[str, Any]:
    """Scheduler entry point. A crash before `run_once` could record
    anything still reaches cron-health."""
    try:
        return run_once()
    except Exception as exc:
        record_run(LOOP_NAME, success=False, note=f"crashed: {safe_exc(exc)}")
        raise


def register(scheduler) -> None:
    scheduler.add_job(tick, "interval", hours=INTERVAL_HOURS, id=LOOP_NAME, replace_existing=True)
