"""Thin Stripe HTTP client — no SDK (FEAT-002, S4).

Five calls (create customer, create Checkout session, create Portal
session, retrieve / list subscriptions) and a webhook signature check
are all the billing feature needs, and `httpx` is already a dependency.
The official SDK would add a package, its own httpx pins and ~10 MB of
RSS to the web process for the same five requests.

Rules this module enforces so no caller has to remember them:

  - **Unconfigured means unavailable.** Every call raises
    `BillingUnavailable` when `STRIPE_SECRET_KEY` is empty, so a route
    can map that one exception to 503 `billing_unavailable`. Nothing in
    here ever pretends a request happened.
  - **Every write carries an `Idempotency-Key`.** Stripe replays the
    original response for a key it has seen (24h), so a client retry
    after a timeout creates one customer / one Checkout session, not two.
    Callers pick keys with a natural meaning (`customer:<user_id>`).
  - **Form encoding, Stripe style.** Nested params flatten to
    `subscription_data[trial_end]`, `line_items[0][price]`,
    `metadata[user_id]`; booleans become `true`/`false`.
  - **Secrets never reach a log.** Errors are re-raised with the HTTP
    status and Stripe's error `code`/`type` only; response bodies are not
    logged (they echo request params, which include customer email).
  - **Signature verification is local**: Stripe's v1 scheme is
    HMAC-SHA256 over `"<t>.<raw body>"` with the endpoint's signing
    secret, compared in constant time, with a 300s replay window.

`request` is the single network entry point; tests replace it.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

from ..agents.log_safety import log_safely
from ..auth.sanitize import safe_logger
from ..config import settings

log = safe_logger(__name__)

API_BASE = "https://api.stripe.com/v1"
TIMEOUT_SECONDS = 10.0
SIGNATURE_TOLERANCE_SECONDS = 300
# The Stripe API version the client is written against. Pinning it per
# request means an account-level version upgrade in the dashboard cannot
# silently change the shapes parsed in `billing_service`.
STRIPE_VERSION = "2024-06-20"


class BillingUnavailable(Exception):
    """Stripe is not configured, or could not be reached. Routes answer
    503 `billing_unavailable`; entitlements are unaffected (DB-derived)."""

    status_code = 503
    code = "billing_unavailable"

    def __init__(self, message: str = "billing is not available") -> None:
        super().__init__(message)
        self.message = message


class StripeError(Exception):
    """Stripe answered with an error. Carries the HTTP status and the
    error `code`/`type` — never the body, which echoes request params."""

    def __init__(self, status_code: int, code: str = "", error_type: str = "", message: str = "") -> None:
        super().__init__(message or code or error_type or f"stripe error {status_code}")
        self.status_code = int(status_code)
        self.code = code or ""
        self.error_type = error_type or ""
        self.message = message or ""


class SignatureError(Exception):
    """The `Stripe-Signature` header did not verify (missing, malformed,
    wrong secret, or outside the replay tolerance)."""


def configured() -> bool:
    return bool(settings.stripe_secret_key)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def flatten(data: dict[str, Any] | None, prefix: str = "") -> dict[str, str]:
    """Stripe's bracketed form encoding: `{"a": {"b": 1}, "c": [x, y]}` →
    `{"a[b]": "1", "c[0]": "x", "c[1]": "y"}`. `None` values are dropped
    (Stripe rejects empty strings for most fields), booleans lower-cased."""
    out: dict[str, str] = {}
    for key, value in (data or {}).items():
        name = f"{prefix}[{key}]" if prefix else str(key)
        if value is None:
            continue
        if isinstance(value, dict):
            out.update(flatten(value, name))
        elif isinstance(value, list | tuple):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    out.update(flatten(item, f"{name}[{i}]"))
                elif item is not None:
                    out[f"{name}[{i}]"] = _scalar(item)
        else:
            out[name] = _scalar(value)
    return out


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def request(
    method: str,
    path: str,
    data: dict[str, Any] | None = None,
    *,
    idempotency_key: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One Stripe API call. `path` is relative to `/v1` (`/customers`).

    Raises `BillingUnavailable` when unconfigured or unreachable and
    `StripeError` on a 4xx/5xx. Returns the decoded JSON object.
    """
    if not configured():
        raise BillingUnavailable("STRIPE_SECRET_KEY is not set")
    import httpx  # lazy: keep the web process's import path unchanged when billing is off

    headers = {
        "Authorization": f"Bearer {settings.stripe_secret_key}",
        "Stripe-Version": STRIPE_VERSION,
    }
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key[:255]
    body = flatten(data) if data else None
    try:
        with httpx.Client(base_url=API_BASE, timeout=TIMEOUT_SECONDS) as client:
            resp = client.request(method.upper(), path, data=body, params=flatten(params) if params else None,
                                  headers=headers)
    except httpx.HTTPError as exc:
        # httpx errors quote the URL, never the headers — but the type is
        # all an operator needs, and `log_safely` masks the detail anyway.
        log_safely(log, f"stripe {method.upper()} {path} failed", exc)
        raise BillingUnavailable("could not reach Stripe") from None

    if resp.status_code >= 400:
        code, error_type, message = "", "", ""
        try:
            err = resp.json().get("error", {}) or {}
            code = str(err.get("code") or "")
            error_type = str(err.get("type") or "")
            message = str(err.get("message") or "")[:200]
        except Exception:
            pass
        # Type + code only. Stripe's `message` can repeat request
        # parameters (customer email, ids) so it stays out of the log.
        log.warning("stripe %s %s -> %s %s/%s", method.upper(), path, resp.status_code,
                    error_type or "?", code or "?")
        if resp.status_code in (401, 403):
            # A revoked/rotated key is "unavailable", not the customer's fault.
            raise BillingUnavailable("Stripe rejected the API key")
        raise StripeError(resp.status_code, code=code, error_type=error_type, message=message)
    try:
        payload = resp.json()
    except ValueError:
        raise StripeError(resp.status_code, code="invalid_json", message="Stripe returned a non-JSON body") from None
    if not isinstance(payload, dict):
        raise StripeError(resp.status_code, code="invalid_shape", message="Stripe returned a non-object body")
    return payload


# ---------------------------------------------------------------------------
# The five calls
# ---------------------------------------------------------------------------

def create_customer(*, email: str | None, user_id: int, external_id: str, idempotency_key: str) -> dict[str, Any]:
    """`POST /customers`. `email` is the plaintext claim from the token and
    goes to Stripe only (the DB never stores it); the metadata is what
    links the Stripe object back to `users` without it."""
    return request("POST", "/customers", {
        "email": email,
        "metadata": {"user_id": str(user_id), "external_id": external_id, "app": "marketmosaic"},
    }, idempotency_key=idempotency_key)


def create_checkout_session(
    *,
    customer_id: str,
    price_id: str,
    success_url: str,
    cancel_url: str,
    client_reference_id: str,
    user_id: int,
    external_id: str,
    trial_end: int | None,
    idempotency_key: str,
) -> dict[str, Any]:
    """`POST /checkout/sessions` in subscription mode.

    `client_reference_id` (= `users.id`) and `subscription_data[metadata]`
    are what the webhook's ownership check reads back; `trial_end` is the
    unix timestamp the local trial ends at, passed only when at least 48h
    remain (Stripe's minimum) so the customer keeps the rest of the trial
    they were promised instead of paying from today.
    """
    data: dict[str, Any] = {
        "mode": "subscription",
        "customer": customer_id,
        "client_reference_id": client_reference_id,
        "line_items": [{"price": price_id, "quantity": 1}],
        "success_url": success_url,
        "cancel_url": cancel_url,
        "allow_promotion_codes": True,
        "metadata": {"user_id": str(user_id), "external_id": external_id},
        "subscription_data": {
            "metadata": {"user_id": str(user_id), "external_id": external_id},
        },
    }
    if trial_end is not None:
        data["subscription_data"]["trial_end"] = int(trial_end)
    return request("POST", "/checkout/sessions", data, idempotency_key=idempotency_key)


def create_portal_session(*, customer_id: str, return_url: str, configuration_id: str | None = None,
                          idempotency_key: str | None = None) -> dict[str, Any]:
    """`POST /billing_portal/sessions`. `billing_service.create_portal`
    always passes a key (every Stripe write carries one); it is optional
    here only so the transport stays a plain function of its inputs."""
    data: dict[str, Any] = {"customer": customer_id, "return_url": return_url}
    if configuration_id:
        data["configuration"] = configuration_id
    return request("POST", "/billing_portal/sessions", data, idempotency_key=idempotency_key)


def retrieve_subscription(subscription_id: str) -> dict[str, Any]:
    return request("GET", f"/subscriptions/{subscription_id}")


def list_subscriptions(customer_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
    """Every subscription (any status) for a customer, newest first —
    reconcile needs the canceled ones too, to end a row Stripe ended."""
    payload = request("GET", "/subscriptions", params={"customer": customer_id, "status": "all", "limit": limit})
    data = payload.get("data")
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Webhook signatures (Stripe v1 scheme)
# ---------------------------------------------------------------------------

def _parse_signature_header(header: str) -> tuple[int | None, list[str]]:
    timestamp: int | None = None
    signatures: list[str] = []
    for part in (header or "").split(","):
        key, _, value = part.strip().partition("=")
        if key == "t":
            try:
                timestamp = int(value)
            except ValueError:
                timestamp = None
        elif key == "v1" and value:
            signatures.append(value)
    return timestamp, signatures


def compute_signature(payload: bytes, secret: str, timestamp: int) -> str:
    signed = f"{timestamp}.".encode() + payload
    return hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()


def sign_payload(payload: bytes, secret: str, *, timestamp: int | None = None) -> str:
    """A `Stripe-Signature` header value for `payload` — what Stripe would
    send. Used by the tests (fixtures are signed at test time) and by an
    operator replaying a stored event locally."""
    ts = int(time.time()) if timestamp is None else int(timestamp)
    return f"t={ts},v1={compute_signature(payload, secret, ts)}"


def verify_signature(
    payload: bytes,
    header: str,
    secret: str,
    *,
    tolerance: int = SIGNATURE_TOLERANCE_SECONDS,
    now: float | None = None,
) -> None:
    """Raise `SignatureError` unless `header` proves `payload` came from
    Stripe within `tolerance` seconds. Constant-time compare; every `v1`
    entry is tried so a secret rotation (two live secrets) verifies."""
    if not secret:
        raise SignatureError("no webhook secret configured")
    if not header:
        raise SignatureError("missing Stripe-Signature header")
    timestamp, signatures = _parse_signature_header(header)
    if timestamp is None or not signatures:
        raise SignatureError("malformed Stripe-Signature header")
    current = time.time() if now is None else float(now)
    if abs(current - timestamp) > tolerance:
        raise SignatureError("timestamp outside tolerance")
    expected = compute_signature(payload, secret, timestamp)
    if not any(hmac.compare_digest(expected, sig) for sig in signatures):
        raise SignatureError("signature mismatch")
