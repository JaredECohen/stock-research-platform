"""Server-side product analytics + abuse telemetry writer.

Every event lands in `analytics_events` with an allowlisted name and
allowlisted, length-bounded props. No third-party script, no
fingerprinting: `anon_id` is a random uuid the browser keeps in
localStorage. `track()` never raises — a funnel write must not fail a
request — and opens its own session when the caller has none.

`first_value` is the one event with once-per-user semantics; it is
gated on `users.first_value_at` (a column, so two processes agree).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from ..models.accounts import User
from ..models.public import AnalyticsEvent
from .principal import Principal
from .sanitize import safe_logger

log = safe_logger(__name__)

ANALYTICS_EVENT_ALLOWLIST: frozenset[str] = frozenset({
    "landing_view", "sample_view", "sample_interact",
    "signup_started", "signup_completed", "trial_activated", "first_value",
    "pricing_view", "checkout_started", "checkout_completed", "trial_converted",
    "trial_expired", "subscription_renewed", "subscription_canceled", "downgraded",
    "quota_hit", "rate_limit_hit",
})

PROP_KEY_ALLOWLIST: frozenset[str] = frozenset({
    "ticker", "feature", "plan", "scope", "kind", "route", "method", "status",
    "interval", "source", "reason", "page", "path", "limit", "used", "window_seconds",
    "trial_source",
})
MAX_PROP_CHARS = 200
MAX_PROPS = 16
RETENTION_DAYS = 90


def clean_props(props: dict[str, Any] | None) -> dict[str, Any]:
    """Keep allowlisted keys with scalar values; strings bounded."""
    out: dict[str, Any] = {}
    if not isinstance(props, dict):
        return out
    for k, v in props.items():
        if len(out) >= MAX_PROPS:
            break
        if not isinstance(k, str) or k not in PROP_KEY_ALLOWLIST:
            continue
        if isinstance(v, bool | int | float) or v is None:
            out[k] = v
        elif isinstance(v, str):
            out[k] = v[:MAX_PROP_CHARS]
        # anything else (lists, dicts) is dropped — no free-form payloads
    return out


def track(
    event_name: str,
    *,
    db: Session | None = None,
    principal: Principal | None = None,
    user_id: int | None = None,
    plan: str | None = None,
    props: dict[str, Any] | None = None,
    source: str = "be",
    session_id: str | None = None,
    anon_id: str | None = None,
    now: datetime | None = None,
) -> bool:
    """Write one event. False (never an exception) when refused or failed."""
    if event_name not in ANALYTICS_EVENT_ALLOWLIST:
        log.debug("analytics: refusing unknown event %r", event_name)
        return False
    if principal is not None:
        if user_id is None and principal.is_user:
            user_id = principal.user_id
        if plan is None and principal.plan_state is not None:
            plan = principal.plan_state.plan
    row = AnalyticsEvent(
        ts=now or datetime.utcnow(), event_name=event_name, user_id=user_id,
        anon_id=(anon_id or None) and str(anon_id)[:64],
        session_id=(session_id or None) and str(session_id)[:64],
        plan=(plan or None) and str(plan)[:16], source=(source or "be")[:8],
        props=clean_props(props),
    )
    try:
        if db is not None:
            db.add(row)
            db.commit()
            return True
        from ..database import SessionLocal
        with SessionLocal() as own:
            own.add(row)
            own.commit()
        return True
    except Exception as exc:
        try:
            if db is not None:
                db.rollback()
        except Exception:  # pragma: no cover
            pass
        log.debug("analytics write failed: %s", type(exc).__name__)
        return False


def mark_first_value(db: Session, user: User, *, feature: str, now: datetime | None = None) -> bool:
    """Set `users.first_value_at` once and emit `first_value`. True only on
    the call that did it."""
    if user.first_value_at is not None:
        return False
    now = now or datetime.utcnow()
    user.first_value_at = now
    db.commit()
    track("first_value", db=db, user_id=user.id, props={"feature": feature}, now=now)
    return True


def abuse_report(db: Session, *, hours: int = 24, now: datetime | None = None) -> dict[str, Any]:
    """The numbers behind `GET /api/admin/abuse-telemetry` (wired by the
    admin router): 429s by scope / plan / route, trial creations per
    bootstrap IP hash, and the share of API requests that were refused
    with 429 — all over the trailing `hours`.

    The 429 share is computed from `ui_logs` rows on `/api/` paths.
    `ui_logs` has no user column (deliberately — no partial PII), so
    "authenticated requests" is approximated by "requests to non-public
    routes" via the policy table. Rate-limit refusals from the customer
    middleware itself never reach the logger (it runs outermost), so this
    is a floor, not a ceiling; the analytics rows above are the full count.
    """
    from sqlalchemy import func, select

    from ..models.telemetry import UILog
    from .policy import classify

    now = now or datetime.utcnow()
    since = now - timedelta(hours=hours)

    by_scope: dict[str, int] = {}
    by_plan: dict[str, int] = {}
    by_route: dict[str, int] = {}
    quota_by_feature: dict[str, int] = {}
    events = db.execute(select(AnalyticsEvent).where(
        AnalyticsEvent.ts >= since, AnalyticsEvent.event_name.in_(("rate_limit_hit", "quota_hit")),
    )).scalars().all()
    for e in events:
        props = e.props or {}
        if e.event_name == "rate_limit_hit":
            by_scope[str(props.get("scope") or "?")] = by_scope.get(str(props.get("scope") or "?"), 0) + 1
            by_plan[str(e.plan or "anon")] = by_plan.get(str(e.plan or "anon"), 0) + 1
            route = f"{props.get('method') or '?'} {props.get('route') or '?'}"
            by_route[route] = by_route.get(route, 0) + 1
        else:
            feat = str(props.get("feature") or "?")
            quota_by_feature[feat] = quota_by_feature.get(feat, 0) + 1

    trials = db.execute(select(User.bootstrap_ip_hash, func.count(User.id)).where(
        User.trial_started_at >= since, User.bootstrap_ip_hash.is_not(None),
    ).group_by(User.bootstrap_ip_hash)).all()
    trials_per_ip = sorted(
        ({"ip_hash": h, "trials": int(n)} for h, n in trials), key=lambda r: -r["trials"],
    )

    rows = db.execute(select(UILog.method, UILog.path, UILog.status_code).where(
        UILog.ts >= since, UILog.source == "backend", UILog.path.like("/api/%"),
    )).all()
    total = 0
    refused = 0
    for method, path, status in rows:
        if classify(method or "GET", path or "").is_public:
            continue
        total += 1
        if status == 429:
            refused += 1
    return {
        "window_hours": hours,
        "rate_limit_hits": {"total": len([e for e in events if e.event_name == "rate_limit_hit"]),
                            "by_scope": by_scope, "by_plan": by_plan, "by_route": by_route},
        "quota_hits": {"total": sum(quota_by_feature.values()), "by_feature": quota_by_feature},
        "trials_per_ip_hash": trials_per_ip[:50],
        "trials_started": int(sum(r["trials"] for r in trials_per_ip)),
        "api_requests_non_public": total,
        "api_requests_429": refused,
        "share_429": (refused / total) if total else 0.0,
    }


def gc_old(db: Session, *, older_than_days: int = RETENTION_DAYS, now: datetime | None = None) -> int:
    """Delete events past retention. Billing-loop duty; anonymous traffic
    must not grow this table without bound."""
    cutoff = (now or datetime.utcnow()) - timedelta(days=older_than_days)
    n = db.execute(delete(AnalyticsEvent).where(AnalyticsEvent.ts < cutoff)).rowcount
    db.commit()
    return int(n or 0)
