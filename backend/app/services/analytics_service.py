"""Browser-side analytics intake (FEAT-002, S3).

`POST /api/public/events` is the one write endpoint anonymous traffic can
reach, so everything about it is bounded: the body size, the number of
events per batch, the event names, the prop keys and the string lengths.
The allowlists themselves live in `auth/analytics.py` (S1) — this module
only applies them to an untrusted batch and writes the survivors in one
transaction.

The endpoint always answers 200. A rejected event is a client bug or a
probe, and neither deserves a status code that tells it what to change;
the response carries `accepted` / `rejected` counts so the frontend's own
telemetry can notice a broken build.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from ..auth.analytics import ANALYTICS_EVENT_ALLOWLIST, clean_props
from ..auth.principal import Principal
from ..auth.sanitize import safe_logger
from ..models.public import AnalyticsEvent

log = safe_logger(__name__)

MAX_EVENTS_PER_BATCH = 50
MAX_BODY_BYTES = 64 * 1024
MAX_ID_CHARS = 64
# A client clock can be wrong by hours; a timestamp outside this window
# is replaced by the server's, not rejected — the event still happened.
MAX_CLOCK_SKEW = timedelta(days=7)
MAX_FUTURE_SKEW = timedelta(minutes=5)


def _parse_ts(raw: Any, now: datetime) -> datetime:
    """Client `ts` (ISO-8601 or epoch seconds/ms), clamped to a sane window
    around `now`. Naive UTC is the persisted convention."""
    if raw is None:
        return now
    ts: datetime | None = None
    try:
        if isinstance(raw, bool):
            return now
        if isinstance(raw, int | float):
            seconds = float(raw)
            if seconds > 1e11:  # milliseconds
                seconds /= 1000.0
            ts = datetime.fromtimestamp(seconds, tz=UTC).replace(tzinfo=None)
        elif isinstance(raw, str):
            s = raw.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            parsed = datetime.fromisoformat(s)
            if parsed.tzinfo is not None:
                parsed = (parsed - parsed.utcoffset()).replace(tzinfo=None)
            ts = parsed
    except (ValueError, OverflowError, OSError, TypeError):
        ts = None
    if ts is None:
        return now
    if ts > now + MAX_FUTURE_SKEW or ts < now - MAX_CLOCK_SKEW:
        return now
    return ts


def _bounded_id(raw: str | None) -> str | None:
    if not raw:
        return None
    value = str(raw).strip()[:MAX_ID_CHARS]
    return value or None


def ingest_batch(
    body: Any,
    *,
    db: Session,
    principal: Principal | None = None,
    anon_id: str | None = None,
    session_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, int]:
    """Validate an untrusted `{events: [...]}` body and write what survives.

    Returns `{accepted, rejected}`; never raises. Attribution rules:

      - `user_id` is set only from a principal the middleware verified
        (a real bearer token). A client cannot assert who it is.
      - `anon_id` / `session_id` are opaque client ids, length-bounded.
      - `plan` is copied from the principal so funnel queries can split
        by plan without a join.

    Events beyond `MAX_EVENTS_PER_BATCH` are dropped and counted as
    rejected: a batch that large is a bug (the client flushes every few
    seconds) or a flood, and either way the first fifty are the useful ones.
    """
    now = now or datetime.utcnow()
    events = body.get("events") if isinstance(body, dict) else None
    if not isinstance(events, list):
        return {"accepted": 0, "rejected": 0}

    user_id: int | None = None
    plan: str | None = None
    if principal is not None and principal.is_user:
        user_id = principal.user_id
        plan = principal.plan

    anon = _bounded_id(anon_id)
    session = _bounded_id(session_id)

    rows: list[AnalyticsEvent] = []
    rejected = 0
    for i, item in enumerate(events):
        if i >= MAX_EVENTS_PER_BATCH:
            rejected += len(events) - MAX_EVENTS_PER_BATCH
            break
        if not isinstance(item, dict):
            rejected += 1
            continue
        name = item.get("name")
        if not isinstance(name, str) or name not in ANALYTICS_EVENT_ALLOWLIST:
            rejected += 1
            continue
        rows.append(AnalyticsEvent(
            ts=_parse_ts(item.get("ts"), now),
            event_name=name,
            user_id=user_id,
            anon_id=anon,
            session_id=session,
            plan=plan,
            source="fe",
            props=clean_props(item.get("props")),
        ))

    if not rows:
        return {"accepted": 0, "rejected": rejected}
    try:
        db.add_all(rows)
        db.commit()
    except Exception as exc:
        try:
            db.rollback()
        except Exception:  # pragma: no cover — rollback of a dead session
            pass
        log.debug("analytics batch write failed: %s", type(exc).__name__)
        return {"accepted": 0, "rejected": rejected + len(rows)}
    return {"accepted": len(rows), "rejected": rejected}
