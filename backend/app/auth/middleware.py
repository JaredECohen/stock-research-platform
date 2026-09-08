"""The login wall.

Runs on every request and always attaches `request.state.principal`.
With AUTH_ENABLED=false that is the whole job — an anonymous principal
flagged `auth_disabled` — and nothing else in this module executes, so
the default deployment behaves exactly as before FEAT-002.

With AUTH_ENABLED=true:

  1. `auth/policy.py` classifies (method, path). `admin` paths belong to
     `admin_auth` and pass straight through (the admin token, never a
     customer JWT, opens them). `OPTIONS` is always public (CORS).
  2. If Clerk is not configured, every non-public route is 503
     `auth_unavailable`. Fail closed: a login wall with no keys is an
     open door, and the marketing site does not need keys.
  3. A bearer token is read from the `Authorization` header ONLY. Query
     strings are persisted to `ui_logs.payload` by the logging middleware,
     so a `?token=` would end up in the database; it is ignored here.
  4. A present-but-bad token is 401 `auth_invalid` on non-public routes
     and treated as anonymous on public ones (a stale token in the
     browser must not break the landing page). No token on a non-public
     route is 401 `auth_required`.
  5. A verified token loads (or lazily creates) the `users` row, resolves
     the plan once for the request, and the middleware itself enforces
     the two coarse levels — signed-in and Pro. Per-feature meters are
     `authorize()`'s job inside the route.

Registration order in `main.py` — this middleware is registered LAST, and
Starlette runs the last-registered `app.middleware("http")` OUTERMOST. So
this runs before `admin_auth_middleware` and before the request logger;
a refusal here is still logged by uvicorn's access log but does not get
a `ui_logs` row. That is deliberate: the logger writes a DB row per
request, and a login wall that wrote a row for every unauthenticated
probe would be a cheap way to fill the database.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta

from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from ..agents.log_safety import log_safely
from ..config import settings
from ..database import SessionLocal
from ..models.accounts import User
from . import policy
from .entitlements import resolve_for_user
from .principal import Principal, email_hash_for
from .sanitize import safe_logger

log = safe_logger(__name__)

LAST_SEEN_WRITE_INTERVAL = timedelta(minutes=5)


def _error(status: int, code: str, message: str, *, challenge: bool = False) -> JSONResponse:
    headers = {"WWW-Authenticate": "Bearer"} if challenge else None
    return JSONResponse({"detail": {"code": code, "message": message}}, status_code=status, headers=headers)


def bearer_token(request: Request) -> str | None:
    """The token from `Authorization: Bearer …`, or None. Header only."""
    header = request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    value = value.strip()
    return value or None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def ensure_user(db, claims: dict, *, now: datetime | None = None) -> User:
    """Load-or-create the `users` row for a verified token's `sub`.

    Creating here (rather than only in `/api/me/bootstrap`) means a
    signed-in user always has a row to meter against, whatever order the
    client calls things in. The trial is NOT started here — that is
    bootstrap's job, gated on a verified email.
    """
    now = now or datetime.utcnow()
    sub = str(claims["sub"])
    email = claims.get("email") if isinstance(claims.get("email"), str) else None
    verified = _truthy(claims.get("email_verified"))
    ehash = email_hash_for(email)

    user = db.query(User).filter(User.external_id == sub).one_or_none()
    changed = False
    if user is None:
        user = User(
            external_id=sub, auth_provider="clerk", email_hash=ehash,
            email_verified_at=now if verified else None, created_at=now, last_seen_at=now,
        )
        db.add(user)
        try:
            db.commit()
        except Exception:
            # Two first requests raced; the other one won. Re-read.
            db.rollback()
            user = db.query(User).filter(User.external_id == sub).one()
    else:
        if ehash and user.email_hash != ehash:
            user.email_hash = ehash
            changed = True
        if verified and user.email_verified_at is None:
            user.email_verified_at = now
            changed = True
        if user.last_seen_at is None or now - user.last_seen_at > LAST_SEEN_WRITE_INTERVAL:
            user.last_seen_at = now
            changed = True
        if changed:
            db.commit()
    return user


def resolve_principal(token: str, request_id: str) -> Principal:
    """Verify + load. Sync (PyJWT, DB); run in a thread from the middleware.
    Raises `jwks.AuthError` subclasses; never logs the token."""
    from . import jwks  # lazy: PyJWT/cryptography stay off the import path

    claims = jwks.verify_token(token)
    now = datetime.utcnow()
    with SessionLocal() as db:
        user = ensure_user(db, claims, now=now)
        state = resolve_for_user(db, user, now)
        email = claims.get("email") if isinstance(claims.get("email"), str) else None
        return Principal(
            kind="user", user_id=user.id, external_id=user.external_id,
            email_verified=user.email_verified_at is not None or _truthy(claims.get("email_verified")),
            email_hash=user.email_hash, email=email, account_state=user.account_state or "active",
            plan_state=state, request_id=request_id,
        )


async def customer_auth_middleware(request: Request, call_next):
    request_id = (request.headers.get("x-request-id") or "")[:64] or secrets.token_hex(8)

    if not settings.auth_enabled:
        request.state.principal = Principal.anonymous(request_id, auth_disabled=True)
        return await call_next(request)

    from . import jwks  # lazy, see resolve_principal

    method, path = request.method, request.url.path
    pol = policy.classify(method, path)
    anon = Principal.anonymous(request_id)
    request.state.principal = anon

    if pol.level == policy.ADMIN:
        return await call_next(request)

    if not settings.auth_configured:
        if pol.is_public:
            return await call_next(request)
        log.error("AUTH_ENABLED but CLERK_ISSUER/CLERK_JWKS_URL unset — refusing %s %s", method, path)
        return _error(503, "auth_unavailable", "sign-in is not configured on this deployment")

    token = bearer_token(request)
    principal = anon
    if token:
        try:
            principal = await run_in_threadpool(resolve_principal, token, request_id)
        except jwks.TokenInvalid as exc:
            if not pol.is_public:
                log.info("rejected token on %s %s: %s", method, path, exc.message)
                return _error(401, "auth_invalid", exc.message, challenge=True)
        except jwks.AuthUnavailable as exc:
            if not pol.is_public:
                log.error("sign-in unavailable on %s %s: %s", method, path, exc.message)
                return _error(503, "auth_unavailable", "sign-in is temporarily unavailable")
        except Exception as exc:  # DB down, etc. — never let it become a 500 with a token in it
            log_safely(log, f"principal resolution failed on {method} {path}", exc)
            if not pol.is_public:
                return _error(503, "auth_unavailable", "sign-in is temporarily unavailable")
    request.state.principal = principal

    if pol.is_public:
        return await call_next(request)
    if principal.is_anon:
        return _error(401, "auth_required", "sign in to continue", challenge=True)
    state = principal.plan_state
    if principal.account_state != "active" or (state is not None and state.suspended):
        return _error(403, "account_suspended", "this account is suspended")
    if pol.level == policy.PRO and (state is None or not state.is_pro):
        return JSONResponse(
            {"detail": {
                "code": "plan_required", "feature": pol.feature, "plan": state.plan if state else "free",
                "upgrade_url": "/pricing", "message": "This is a Pro feature.",
            }},
            status_code=402,
        )
    return await call_next(request)
