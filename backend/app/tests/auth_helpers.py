"""Shared harness for the FEAT-002 tests: a Clerk stand-in.

Generates an RSA keypair per stub, publishes it as a JWKS document, and
mints tokens shaped like Clerk's `marketmosaic` template output. Tests
monkeypatch `app.auth.jwks.fetch` to return the document, so nothing
touches the network — CI has no keys and no Clerk tenant.

Test modules define two thin fixtures on top of this (importing fixtures
by name trips ruff's F811, and a shared conftest is not this slice's):

    @pytest.fixture()
    def clerk():
        return ClerkStub()

    @pytest.fixture()
    def auth_on(monkeypatch, clerk):
        yield from enable_auth(monkeypatch, clerk)
"""
from __future__ import annotations

import time
import uuid
from typing import Any

ISSUER = "https://test-tenant.clerk.accounts.dev"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
PARTY = "https://app.marketmosaic.test"


class ClerkStub:
    def __init__(self, kid: str = "kid-1") -> None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        self.kid = kid
        self._private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.private_pem = self._private.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
        )
        self.public_pem = self._private.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.fetch_calls = 0
        self.fail_fetch = False

    @property
    def jwks(self) -> dict[str, Any]:
        from jwt.algorithms import RSAAlgorithm

        jwk = RSAAlgorithm.to_jwk(self._private.public_key(), as_dict=True)
        jwk.update({"kid": self.kid, "use": "sig", "alg": "RS256"})
        return {"keys": [jwk]}

    def fetch(self, url: str) -> dict[str, Any]:
        self.fetch_calls += 1
        if self.fail_fetch:
            raise ConnectionError("jwks endpoint unreachable")
        assert url == JWKS_URL, url
        return self.jwks

    def token(
        self,
        sub: str | None = None,
        *,
        email: str | None = "jane@example.com",
        verified: bool | str = True,
        exp_in: int = 3600,
        iat: int | None = None,
        iss: str = ISSUER,
        azp: str | None = PARTY,
        kid: str | None = None,
        alg: str = "RS256",
        key: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        import jwt

        now = int(time.time())
        claims: dict[str, Any] = {
            "sub": sub or new_sub(),
            "iss": iss,
            "iat": now if iat is None else iat,
            "nbf": now - 5,
            "exp": now + exp_in,
            "sid": "sess_" + uuid.uuid4().hex[:10],
        }
        if azp is not None:
            claims["azp"] = azp
        if email is not None:
            claims["email"] = email
        claims["email_verified"] = verified
        if extra:
            claims.update(extra)
        headers = {"kid": kid or self.kid}
        if alg == "none":
            return jwt.encode(claims, None, algorithm="none", headers=headers)
        if alg == "HS256":
            # The algorithm-confusion attack signs with a symmetric secret
            # under a kid the verifier knows. PyJWT itself refuses to HMAC
            # with PEM material, so a plain secret stands in; the verifier
            # must refuse on `alg` before any key is consulted.
            return jwt.encode(claims, key or "attacker-chosen-secret", algorithm="HS256", headers=headers)
        return jwt.encode(claims, key or self.private_pem, algorithm="RS256", headers=headers)


def new_sub() -> str:
    return "user_" + uuid.uuid4().hex[:16]


def new_email() -> str:
    return f"{uuid.uuid4().hex[:10]}@example.com"


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def enable_auth(monkeypatch, clerk: ClerkStub):
    """Generator behind the `auth_on` fixture: login wall on against the
    stub tenant, usage limits on, billing off. Restored by monkeypatch;
    the JWKS cache is reset on both sides so no test inherits another's
    keys."""
    from app.auth import jwks
    from app.config import settings

    jwks.reset_cache()
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "usage_limits_enabled", True)
    monkeypatch.setattr(settings, "billing_enabled", False)
    monkeypatch.setattr(settings, "clerk_issuer", ISSUER)
    monkeypatch.setattr(settings, "clerk_jwks_url", JWKS_URL)
    monkeypatch.setattr(settings, "clerk_authorized_parties", PARTY)
    monkeypatch.setattr(jwks, "fetch", clerk.fetch)
    yield clerk
    jwks.reset_cache()


FORBIDDEN_LOG_PATTERNS = ("Bearer ", "sk_", "whsec_", "eyJ")


def assert_no_secrets_in_logs(records) -> None:
    """No log record — template or rendered — may carry a credential."""
    for rec in records:
        rendered = rec.getMessage()
        for needle in FORBIDDEN_LOG_PATTERNS:
            assert needle not in rendered, f"log record leaks {needle!r}: {rendered[:120]!r}"
            assert needle not in str(rec.msg), f"log template leaks {needle!r}"
