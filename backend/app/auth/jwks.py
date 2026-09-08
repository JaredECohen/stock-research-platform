"""Clerk session-token verification against the tenant's JWKS.

RS256 only, via PyJWT + cryptography (imported lazily so `import
app.main` stays as cheap as before — see test_dependency_footprint.py).
No Clerk SDK: the JWT template named "marketmosaic" carries `email` and
`email_verified`, so a request costs one signature check and zero
network calls.

Key cache (process-local — JWKS is public and identical everywhere, so
this is not the kind of state the two-process rule is about):
  - refreshed when older than `LIFESPAN`;
  - an unknown `kid` forces one refresh, throttled to once per
    `REFRESH_THROTTLE` seconds so a flood of garbage tokens cannot turn
    into a flood of JWKS fetches;
  - a fetch failure keeps serving the cached keys for up to `STALE_MAX`
    (stale-while-error); with no keys at all it raises `AuthUnavailable`
    and the middleware answers 503 — fail closed, never open.

`fetch` is the only function that touches the network; tests replace it.
Nothing here ever logs a token.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from ..agents.log_safety import log_safely
from ..config import settings
from .sanitize import safe_logger

log = safe_logger(__name__)

LIFESPAN = 3600.0          # normal refresh interval
STALE_MAX = 24 * 3600.0    # how long stale keys keep verifying during an outage
REFRESH_THROTTLE = 60.0    # min seconds between forced refreshes
LEEWAY = 60                # clock-skew tolerance on exp/nbf/iat
ALGORITHMS = ("RS256",)
FETCH_TIMEOUT = 5.0


class AuthError(Exception):
    status = 401
    code = "auth_invalid"

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message = message or self.code


class TokenInvalid(AuthError):
    """The token cannot be trusted: bad signature, expired, wrong issuer,
    wrong audience/party, unsupported algorithm, unknown key."""
    status = 401
    code = "auth_invalid"


class AuthUnavailable(AuthError):
    """We cannot decide: no keys and the JWKS endpoint is unreachable."""
    status = 503
    code = "auth_unavailable"


_lock = threading.Lock()
# `last_forced` throttles only the unknown-kid refreshes: a scheduled
# refresh that just succeeded must not block the one forced refresh a
# genuinely rotated key needs.
_STATE: dict[str, Any] = {"keys": {}, "fetched_at": 0.0, "last_attempt": 0.0, "last_forced": 0.0}


def reset_cache() -> None:
    with _lock:
        _STATE["keys"] = {}
        _STATE["fetched_at"] = 0.0
        _STATE["last_attempt"] = 0.0
        _STATE["last_forced"] = 0.0


def fetch(url: str) -> dict[str, Any]:
    """GET the JWKS document. The single network call in this module."""
    import httpx  # lazy: keep import cost off the startup path

    resp = httpx.get(url, timeout=FETCH_TIMEOUT, headers={"Accept": "application/json"})
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or not isinstance(data.get("keys"), list):
        raise ValueError("JWKS document has no `keys` list")
    return data


def _refresh(*, force: bool = False, now: float | None = None) -> bool:
    """Reload keys. Returns True when a fetch succeeded. Never raises."""
    now = time.time() if now is None else now
    url = settings.clerk_jwks_url
    with _lock:
        if force:
            if now - _STATE["last_forced"] < REFRESH_THROTTLE:
                return False
            _STATE["last_forced"] = now
        _STATE["last_attempt"] = now
        try:
            doc = fetch(url)
            keys = {k.get("kid"): k for k in doc.get("keys", []) if isinstance(k, dict) and k.get("kid")}
            if not keys:
                raise ValueError("JWKS document contains no usable keys")
            _STATE["keys"] = keys
            _STATE["fetched_at"] = now
            return True
        except Exception as exc:
            # Type name only at WARNING; the redacted detail at DEBUG.
            log_safely(log, "JWKS refresh failed", exc)
            return False


def _keys(now: float) -> dict[str, dict[str, Any]]:
    """Cached keys, refreshing when due; stale-while-error inside STALE_MAX."""
    age = now - _STATE["fetched_at"]
    if not _STATE["keys"] or age > LIFESPAN:
        if not _refresh(now=now) and not _STATE["keys"]:
            raise AuthUnavailable("sign-in keys unavailable")
        if not _STATE["keys"] or now - _STATE["fetched_at"] > STALE_MAX:
            raise AuthUnavailable("sign-in keys are stale")
    return _STATE["keys"]


def _signing_key(kid: str, now: float):
    keys = _keys(now)
    jwk = keys.get(kid)
    if jwk is None:
        # A rotated key we have not seen yet — one forced refresh, throttled.
        _refresh(force=True, now=now)
        jwk = _STATE["keys"].get(kid)
    if jwk is None:
        raise TokenInvalid("unknown signing key")
    import jwt  # lazy

    try:
        return jwt.PyJWK(jwk).key
    except Exception as exc:  # malformed JWK from the provider
        log_safely(log, "JWKS entry could not be loaded", exc)
        raise AuthUnavailable("sign-in keys unusable") from None


def verify_token(token: str, *, now: float | None = None) -> dict[str, Any]:
    """Verify a Clerk session token and return its claims.

    Raises `TokenInvalid` (401) or `AuthUnavailable` (503). The token is
    never logged, and neither is any exception message derived from it.
    """
    import jwt  # lazy

    now = time.time() if now is None else now
    if not token or not isinstance(token, str) or token.count(".") != 2:
        raise TokenInvalid("malformed token")
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError:
        raise TokenInvalid("malformed token") from None
    alg = header.get("alg")
    if alg not in ALGORITHMS:
        # `none`, HS256 (a symmetric alg that would let a client sign with
        # the public key), or anything exotic — refused before any key work.
        raise TokenInvalid("unsupported token algorithm")
    kid = header.get("kid")
    if not kid or not isinstance(kid, str):
        raise TokenInvalid("token has no key id")

    key = _signing_key(kid, now)
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=list(ALGORITHMS),
            issuer=settings.clerk_issuer,
            leeway=LEEWAY,
            options={
                "require": ["exp", "iat", "sub"],
                # Clerk session tokens carry `azp`, not `aud`; we check
                # `azp` ourselves below.
                "verify_aud": False,
            },
        )
    except jwt.ExpiredSignatureError:
        raise TokenInvalid("token expired") from None
    except jwt.InvalidIssuerError:
        raise TokenInvalid("token issuer not trusted") from None
    except jwt.PyJWTError:
        raise TokenInvalid("token could not be verified") from None

    parties = settings.clerk_authorized_parties_list
    if parties:
        azp = claims.get("azp")
        if not isinstance(azp, str) or azp not in parties:
            raise TokenInvalid("token authorized party not trusted")
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        raise TokenInvalid("token has no subject")
    return claims
