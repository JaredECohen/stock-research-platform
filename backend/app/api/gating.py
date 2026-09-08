"""Route-side helpers for the FEAT-002 gate (slice S2).

`auth/entitlements.require_feature` is the per-feature dependency
(plan, meter, follows-memo, concurrency lease). What it deliberately
does not do is the per-user *rate* limit, because a rate scope is a
property of the route, not of the feature — `GET /api/stocks/{t}` and
`GET /api/stocks/{t}/prices` share no feature but both sit in a scope.
`rate_scope(...)` is that dependency. The two compose on a route as

    Depends(rate_scope("llm_light")), Depends(require_feature("pm_chat", ...))

with the rate check first: it is one primary-key upsert, whereas the
feature check may take a lease and reserve a meter that then have to be
given back when the cheaper check would have refused anyway.

Behaviour-preserving by construction: with `AUTH_ENABLED=false` neither
helper touches the database, so the default deployment runs exactly the
code it ran before this feature. The per-user limiter also honours
`RATE_LIMIT_ENABLED`, the same switch slowapi uses, so the test suite
(which shares one client address across every test) is not tripped by
the `ip:` ceiling and a local dev server can turn every limiter off in
one place.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends
from sqlalchemy.orm import Session
from starlette.requests import Request

from ..auth import ratelimit
from ..auth.entitlements import EntitlementError
from ..auth.principal import current_principal
from ..config import settings
from ..database import get_db


def rate_scope(scope: str) -> Callable[..., None]:
    """FastAPI dependency: count this request in `scope` for the signed-in
    user (and, at `IP_MULTIPLIER` × the limit, for the caller's address)
    and refuse with a structured 429 when the window is full. No-op with
    the login wall or the limiter switched off."""
    ratelimit.scope_limit(scope)  # a typo fails at import, not at first request

    def dependency(request: Request, db: Session = Depends(get_db)) -> None:
        if not settings.auth_enabled or not settings.rate_limit_enabled:
            return
        principal = current_principal(request)
        ratelimit.enforce(db, request, scope, user_id=principal.user_id)

    dependency.__name__ = f"rate_{scope}"
    return dependency


def feature_disabled(message: str, *, feature: str | None = None, status_code: int = 404,
                     **extra: Any) -> EntitlementError:
    """A capability that exists in the code but is not offered to customer
    accounts (backtest memos, synchronous generation). 404 by default —
    the route is not there for this caller — but the analyze route uses
    403 for `sync=true` so a client that knowingly asks for the inline
    path gets "refused", not "unknown"."""
    return EntitlementError(status_code, "feature_disabled", message, feature=feature,
                            extra=extra or {})


def customer_wall_on() -> bool:
    """True when customer requests must not start provider or LLM work
    inside the web process. Kept as a function (not a constant) because
    tests flip the setting per case."""
    return settings.auth_enabled
