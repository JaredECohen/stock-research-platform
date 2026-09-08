"""Identity-aware rate limits and concurrency leases, in the database.

slowapi stays for the anonymous per-IP ceilings (its `memory://` store is
fine for that: a single web replica, and the worst case of losing it is
being briefly generous). Per-USER limits cannot live there — a user's
requests can arrive at any process, and the abuse the limits exist to
stop is exactly the kind that would spread them out. So they are
fixed-window counters in `rate_limit_windows`, one upsert per request on
the primary key.

The upsert is `INSERT … ON CONFLICT (key) DO UPDATE SET count = count + 1
RETURNING count` on Postgres and on SQLite ≥ 3.35 (the first release with
RETURNING; checked at import). Older SQLite falls back to insert-or-
ignore + UPDATE + SELECT inside one transaction, which is still correct
because the UPDATE takes the database write lock.

Every refusal is a structured 429: `code`, `scope`, `retry_after`,
`window_seconds`, `message` in `detail`, plus a `Retry-After` header.
`structured_slowapi_handler` makes slowapi's per-IP refusals look the
same, so the frontend has one shape to handle.
"""
from __future__ import annotations

import json
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.orm import Session
from starlette.requests import Request

from ..config import settings
from ..models.accounts import ActiveAction, RateLimitWindow

# One definition of "the caller's address" for slowapi's per-IP limits,
# the `ip:` buckets below and the bootstrap IP hash — proxy-aware, and
# reading X-Forwarded-For from the right so a client cannot pick its own
# bucket. It lives in `rate_limit.py` (no ORM import) and is re-exported
# here for the callers that think of it as part of the limiter.
from ..rate_limit import client_ip
from .sanitize import safe_logger

log = safe_logger(__name__)

_EPOCH = datetime(1970, 1, 1)
SQLITE_RETURNING = sqlite3.sqlite_version_info >= (3, 35, 0)

# Scope → "N/window". DEVPLAN numbers; RATE_LIMIT_OVERRIDES_JSON can
# change any of them per environment ({"research": "5/hour"}).
DEFAULT_SCOPES: dict[str, str] = {
    "data":              "120/minute",   # provider-cached reads
    "series":            "60/minute",    # prices / macro series
    "llm_light":         "10/minute",    # chat, dcf, comps
    "research":          "3/hour",       # POST /analyze
    "checkout":          "10/hour",
    "reconcile":         "5/hour",
    "bootstrap":         "3/hour",       # per IP; the trial-creation ceiling
    "evaluate_outcomes": "1/10minute",   # global, keyed "global"
}
# The secondary per-IP ceiling on authenticated scopes is this many times
# the per-user limit: several users behind one NAT must not starve each
# other, but one address running many accounts must still hit a wall.
IP_MULTIPLIER = 3

_SPEC = re.compile(r"^\s*(\d+)\s*/\s*(\d*)\s*(second|minute|hour|day)s?\s*$", re.I)
_UNITS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def parse_limit(spec: str) -> tuple[int, int]:
    """'10/minute' → (10, 60); '1/5minute' → (1, 300)."""
    m = _SPEC.match(spec or "")
    if not m:
        raise ValueError(f"bad rate limit spec {spec!r}")
    count, multiple, unit = int(m.group(1)), int(m.group(2) or 1), m.group(3).lower()
    return count, multiple * _UNITS[unit]


@lru_cache(maxsize=8)
def _overrides(raw: str) -> dict[str, str]:
    raw = (raw or "").strip()
    if not raw or raw == "{}":
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        log.error("RATE_LIMIT_OVERRIDES_JSON is not valid JSON; using code defaults")
        return {}
    out: dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                parse_limit(str(v))
                out[str(k)] = str(v)
            except ValueError:
                log.error("RATE_LIMIT_OVERRIDES_JSON: ignoring bad spec for %r", k)
    return out


def scope_limit(scope: str) -> tuple[int, int]:
    spec = _overrides(settings.rate_limit_overrides_json).get(scope) or DEFAULT_SCOPES.get(scope)
    if spec is None:
        raise ValueError(f"unknown rate-limit scope {scope!r}")
    return parse_limit(spec)


@dataclass
class RateResult:
    allowed: bool
    scope: str
    count: int
    limit: int
    window_seconds: int
    retry_after: int  # 0 when allowed

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.count)


def _ts(now: datetime) -> float:
    return (now - _EPOCH).total_seconds()


def check(
    db: Session,
    scope: str,
    identity: str,
    *,
    limit: int | None = None,
    window_seconds: int | None = None,
    now: datetime | None = None,
) -> RateResult:
    """Count one hit for `identity` in `scope`'s current window and say
    whether it is within the limit. Always counts — a refused request
    still consumed its slot, so a client hammering after a 429 does not
    get a free retry the moment the window turns."""
    if limit is None or window_seconds is None:
        cfg_limit, cfg_window = scope_limit(scope)
        limit = cfg_limit if limit is None else limit
        window_seconds = cfg_window if window_seconds is None else window_seconds
    now = now or datetime.utcnow()
    now_ts = _ts(now)
    window_start = int(now_ts // window_seconds) * window_seconds
    key = f"{scope}:{identity}:{window_start}"
    expires = _EPOCH + timedelta(seconds=window_start + window_seconds)

    count = _upsert_count(db, key, expires)
    db.commit()
    allowed = count <= limit
    retry_after = 0 if allowed else max(1, int(window_start + window_seconds - now_ts + 0.999))
    return RateResult(allowed, scope, count, limit, window_seconds, retry_after)


def _upsert_count(db: Session, key: str, expires: datetime) -> int:
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        stmt = pg_insert(RateLimitWindow).values(key=key, count=1, expires_at=expires)
        stmt = stmt.on_conflict_do_update(
            index_elements=["key"], set_={"count": RateLimitWindow.count + 1},
        ).returning(RateLimitWindow.count)
        return int(db.execute(stmt).scalar_one())
    if dialect == "sqlite" and SQLITE_RETURNING:
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        stmt = sqlite_insert(RateLimitWindow).values(key=key, count=1, expires_at=expires)
        stmt = stmt.on_conflict_do_update(
            index_elements=["key"], set_={"count": RateLimitWindow.count + 1},
        ).returning(RateLimitWindow.count)
        return int(db.execute(stmt).scalar_one())
    # Portable fallback (SQLite < 3.35, anything else): insert-or-ignore,
    # then increment, then read — the UPDATE holds the write lock for the
    # rest of the transaction so the SELECT cannot see another writer.
    if dialect == "sqlite":
        db.execute(text(
            "INSERT OR IGNORE INTO rate_limit_windows (key, count, expires_at) VALUES (:k, 0, :e)"
        ), {"k": key, "e": expires})
    else:  # pragma: no cover — no such dialect in this repo
        exists = db.execute(select(RateLimitWindow.key).where(RateLimitWindow.key == key)).first()
        if not exists:
            db.add(RateLimitWindow(key=key, count=0, expires_at=expires))
            db.flush()
    db.execute(update(RateLimitWindow).where(RateLimitWindow.key == key)
               .values(count=RateLimitWindow.count + 1))
    return int(db.execute(select(RateLimitWindow.count).where(RateLimitWindow.key == key)).scalar_one())


class RateLimited(HTTPException):
    """429 with the structured body and a `Retry-After` header."""

    def __init__(self, result: RateResult, *, code: str = "rate_limited", message: str = "") -> None:
        detail = {
            "code": code,
            "scope": result.scope,
            "retry_after": int(result.retry_after),
            "window_seconds": int(result.window_seconds),
            "limit": int(result.limit),
            "message": message or (
                f"Too many requests for {result.scope}: limit is {result.limit} per "
                f"{result.window_seconds}s. Try again in {result.retry_after}s."
            ),
        }
        super().__init__(status_code=429, detail=detail, headers={"Retry-After": str(int(result.retry_after))})
        self.result = result


def enforce(db: Session, request: Request, scope: str, *, user_id: int | None, now: datetime | None = None) -> RateResult:
    """User-first, IP-second. Two users behind one IP are independent
    (their user keys differ); one user on two IPs shares one bucket
    (the user key). The IP bucket is `IP_MULTIPLIER` × wider. Raises
    `RateLimited` on refusal, after recording the hit for telemetry."""
    limit, window = scope_limit(scope)
    if user_id is not None:
        user_res = check(db, f"user:{scope}", str(user_id), limit=limit, window_seconds=window, now=now)
        if not user_res.allowed:
            _record_hit(db, scope, "user", request, user_id)
            raise RateLimited(user_res)
    ip_res = check(
        db, f"ip:{scope}", client_ip(request),
        limit=limit * (IP_MULTIPLIER if user_id is not None else 1), window_seconds=window, now=now,
    )
    if not ip_res.allowed:
        _record_hit(db, scope, "ip", request, user_id)
        raise RateLimited(ip_res)
    return ip_res if user_id is None else user_res


def _record_hit(db: Session, scope: str, kind: str, request: Request, user_id: int | None) -> None:
    try:
        from . import analytics
        from .principal import current_principal
        principal = current_principal(request)
        analytics.track(
            "rate_limit_hit", db=db, principal=principal,
            props={"scope": scope, "kind": kind, "route": request.url.path, "method": request.method},
        )
    except Exception as exc:  # pragma: no cover — telemetry must never break a 429
        log.debug("rate_limit_hit telemetry failed: %s", type(exc).__name__)


# ---------------------------------------------------------------------------
# Concurrency leases
# ---------------------------------------------------------------------------

def lease(
    db: Session,
    *,
    user_id: int,
    feature: str,
    max_concurrent: int,
    ttl_seconds: int = 120,
    resource_ref: str | None = None,
    now: datetime | None = None,
) -> str | None:
    """Take a lease if fewer than `max_concurrent` are live; else None.

    Count-then-insert is not atomic across two processes, so under a
    true race one extra lease can slip through. The bound is small (one
    over), the TTL heals it, and the alternative — a per-user advisory
    lock — is not portable to SQLite. Accepted.
    """
    now = now or datetime.utcnow()
    live = db.execute(select(func.count(ActiveAction.id)).where(
        ActiveAction.user_id == user_id, ActiveAction.feature == feature, ActiveAction.expires_at > now,
    )).scalar_one()
    if int(live) >= max_concurrent:
        return None
    token = secrets.token_hex(16)
    db.add(ActiveAction(
        user_id=user_id, feature=feature, resource_ref=resource_ref, lease_token=token,
        started_at=now, expires_at=now + timedelta(seconds=ttl_seconds),
    ))
    db.commit()
    return token


def release_lease(db: Session, token: str | None) -> bool:
    if not token:
        return False
    result = db.execute(delete(ActiveAction).where(ActiveAction.lease_token == token))
    db.commit()
    return result.rowcount == 1


def gc_expired(db: Session, *, now: datetime | None = None) -> int:
    """Drop windows and leases whose time has passed. Billing-loop duty."""
    now = now or datetime.utcnow()
    n = db.execute(delete(RateLimitWindow).where(RateLimitWindow.expires_at <= now)).rowcount
    n += db.execute(delete(ActiveAction).where(ActiveAction.expires_at <= now)).rowcount
    db.commit()
    return int(n or 0)


# ---------------------------------------------------------------------------
# slowapi bridge
# ---------------------------------------------------------------------------

def structured_slowapi_handler(request: Request, exc: Any) -> JSONResponse:
    """Replacement for slowapi's `_rate_limit_exceeded_handler`: the same
    body shape as `RateLimited`, `Retry-After`, and slowapi's own
    `X-RateLimit-*` headers when it can compute them."""
    item = getattr(getattr(exc, "limit", None), "limit", None)
    window = 60
    amount: int | None = None
    if item is not None:
        try:
            window = int(item.get_expiry())
            amount = int(item.amount)
        except Exception:  # pragma: no cover — limits API drift
            pass
    retry_after = window
    try:
        view = getattr(request.state, "view_rate_limit", None)
        limiter = request.app.state.limiter
        if view is not None:
            reset_at, _remaining = limiter.limiter.get_window_stats(view[0], *view[1])
            import time
            retry_after = max(1, int(reset_at - time.time()) + 1)
    except Exception:
        pass
    detail = {
        "code": "rate_limited",
        "scope": "ip",
        "retry_after": int(retry_after),
        "window_seconds": int(window),
        "limit": amount,
        "message": f"Too many requests from this address ({getattr(exc, 'detail', 'rate limit exceeded')}). "
                   f"Try again in {int(retry_after)}s.",
    }
    response = JSONResponse({"detail": detail}, status_code=429, headers={"Retry-After": str(int(retry_after))})
    try:
        response = request.app.state.limiter._inject_headers(response, request.state.view_rate_limit)
    except Exception:
        pass
    try:
        from ..database import SessionLocal
        from . import analytics
        from .principal import current_principal
        with SessionLocal() as db:
            analytics.track(
                "rate_limit_hit", db=db, principal=current_principal(request),
                props={"scope": "ip", "kind": "slowapi", "route": request.url.path, "method": request.method},
            )
    except Exception:  # pragma: no cover — telemetry must never break a 429
        pass
    return response
