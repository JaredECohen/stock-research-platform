"""Clerk session-token verification against the tenant's JWKS.

RS256 only, via PyJWT + cryptography (imported lazily so `import
app.main` stays as cheap as before — see test_dependency_footprint.py).
No Clerk SDK: the JWT template named "marketmosaic" carries `email` and
`email_verified`, so a request costs one signature check and zero
network calls.

Key cache (process-local — JWKS is public and identical everywhere, so
this is not the kind of state the two-process rule is about):
  - refreshed when older than `LIFESPAN`, attempting at most once per
    `REFRESH_THROTTLE` seconds; between attempts the cached keys are
    served without touching the network;
  - an unknown `kid` forces one refresh, throttled separately to once per
    `REFRESH_THROTTLE` seconds so a flood of garbage tokens cannot turn
    into a flood of JWKS fetches;
  - a fetch failure keeps serving the cached keys for up to `STALE_MAX`
    (stale-while-error); with no keys at all it raises `AuthUnavailable`
    and the middleware answers 503 — fail closed, never open;
  - the network call is single-flight per process and never made while
    holding the state lock, so a dead JWKS endpoint costs one thread a
    timeout per throttle window, not every request a serial 5s wait.

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
REFRESH_THROTTLE = 60.0    # min seconds between refresh attempts (scheduled and forced, tracked separately)
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


# Two locks. `_state_lock` guards `_STATE` and is only ever held for a few
# field reads/writes. `_fetch_lock` serialises the network call so a burst
# of requests becomes one JWKS fetch per process (single-flight). The
# fetch is never made under `_state_lock`: the first version did exactly
# that, and once the cached keys passed LIFESPAN with the endpoint down,
# every verification queued behind a 5s timeout — ~1 request per 5s per
# process for up to STALE_MAX. Re-entrant because the cold-start path in
# `_keys` holds it across its re-check and the `_refresh` it may call.
_state_lock = threading.Lock()
_fetch_lock = threading.RLock()
# `last_attempt` throttles the scheduled refreshes and `last_forced` the
# unknown-kid ones — separately, so a scheduled refresh that just
# succeeded does not block the one forced refresh a genuinely rotated
# key needs.
_STATE: dict[str, Any] = {"keys": {}, "fetched_at": 0.0, "last_attempt": 0.0, "last_forced": 0.0}


def reset_cache() -> None:
    with _state_lock:
        _STATE["keys"] = {}
        _STATE["fetched_at"] = 0.0
        _STATE["last_attempt"] = 0.0
        _STATE["last_forced"] = 0.0


def _snapshot() -> tuple[dict[str, dict[str, Any]], float, float]:
    """(keys, fetched_at, last_attempt) read atomically."""
    with _state_lock:
        return _STATE["keys"], _STATE["fetched_at"], _STATE["last_attempt"]


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
    """Reload keys. Returns True when a fetch succeeded. Never raises.

    Bookkeeping under `_state_lock`; the network call under `_fetch_lock`
    only; the result written back under `_state_lock`. `last_attempt` is
    stamped BEFORE the fetch so a concurrent caller sees the attempt as
    taken and serves stale rather than fetching too."""
    now = time.time() if now is None else now
    url = settings.clerk_jwks_url
    with _state_lock:
        if force:
            if now - _STATE["last_forced"] < REFRESH_THROTTLE:
                return False
            _STATE["last_forced"] = now
        _STATE["last_attempt"] = now
    with _fetch_lock:
        try:
            doc = fetch(url)
            keys = {k.get("kid"): k for k in doc.get("keys", []) if isinstance(k, dict) and k.get("kid")}
            if not keys:
                raise ValueError("JWKS document contains no usable keys")
        except Exception as exc:
            # Type name only at WARNING; the redacted detail at DEBUG.
            log_safely(log, "JWKS refresh failed", exc)
            return False
    with _state_lock:
        _STATE["keys"] = keys
        _STATE["fetched_at"] = now
    return True


def _keys(now: float) -> dict[str, dict[str, Any]]:
    """Cached keys, refreshing when due; stale-while-error inside STALE_MAX.

    A scheduled refresh (keys older than LIFESPAN) is attempted at most
    once per REFRESH_THROTTLE per process; between attempts the cached
    keys are served without a network call. Without that gate an outage
    at the JWKS endpoint made every verification wait out a fetch
    timeout — "stale-while-error" in name only."""
    keys, fetched_at, last_attempt = _snapshot()
    if keys and now - fetched_at <= LIFESPAN:
        return keys
    if keys:
        if now - last_attempt >= REFRESH_THROTTLE:
            _refresh(now=now)
            keys, fetched_at, _ = _snapshot()
        if now - fetched_at > STALE_MAX:
            raise AuthUnavailable("sign-in keys are stale")
        return keys
    # Cold start (or a cache reset): nothing to serve stale. Single-flight
    # on the fetch lock — a request that queued behind an in-flight fetch
    # finds the keys on its re-check and never fetches itself; one that
    # queued behind a FAILED fetch is refused at once rather than paying
    # the timeout again inside the throttle window.
    with _fetch_lock:
        keys, _, last_attempt = _snapshot()
        if keys:
            return keys
        if last_attempt > 0 and now - last_attempt < REFRESH_THROTTLE:
            raise AuthUnavailable("sign-in keys unavailable")
        if not _refresh(now=now):
            raise AuthUnavailable("sign-in keys unavailable")
        keys, _, _ = _snapshot()
        if not keys:  # pragma: no cover — _refresh returned True with keys
            raise AuthUnavailable("sign-in keys unavailable")
        return keys


def _signing_key(kid: str, now: float):
    keys = _keys(now)
    jwk = keys.get(kid)
    if jwk is None:
        # A rotated key we have not seen yet — one forced refresh, throttled.
        _refresh(force=True, now=now)
        jwk = _snapshot()[0].get(kid)
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
