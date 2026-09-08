"""Customer authentication, plans and entitlements (FEAT-002).

Deliberately separate from `api/admin_auth.py`: the admin bearer token
and a customer session JWT are different credentials for different
surfaces, and neither ever satisfies the other.

  jwks         Clerk RS256 verification (PyJWT, lazily imported)
  principal    who is asking — attached to every request
  policy       which routes need what (default-deny under /api/)
  middleware   the login wall itself
  features     the entitlement matrix (one table, enforced + rendered)
  plans        subscription/trial/override → effective plan (pure)
  entitlements `authorize()` / `require_feature()` — the one call site
  usage        atomic, idempotent monthly meters
  ratelimit    DB-backed per-user limits, leases, structured 429s
  analytics    funnel + abuse telemetry writer
  sanitize     credential redaction for logs
"""
from __future__ import annotations

from .entitlements import EntitlementError, Grant, authorize, require_feature
from .principal import Principal, current_principal

__all__ = [
    "authorize",
    "require_feature",
    "current_principal",
    "Principal",
    "Grant",
    "EntitlementError",
]
