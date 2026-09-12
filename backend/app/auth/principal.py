"""Who is making this request.

A `Principal` is attached to `request.state.principal` by
`customer_auth_middleware` on EVERY request — including when the login
wall is off (an anonymous principal with `auth_disabled=True`) — so route
code has exactly one way to ask "who is this" and never has to special-
case the flag.

It never holds the raw token. `email` is the plaintext claim from the
JWT and is kept only for the request's lifetime (Stripe Checkout needs
it as `customer_email`); it is excluded from `repr` so a stray
`log.info("%s", principal)` cannot print it. Nothing here is persisted.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover
    from starlette.requests import Request

    from .plans import PlanState


@dataclass
class Principal:
    kind: Literal["anon", "user"] = "anon"
    user_id: int | None = None
    external_id: str | None = None
    email_verified: bool = False
    email_hash: str | None = None
    email: str | None = field(default=None, repr=False)
    account_state: str = "active"
    plan_state: PlanState | None = None
    request_id: str = ""
    # True when AUTH_ENABLED=false: "anonymous because there is no login
    # wall", as opposed to "anonymous because no token was presented".
    auth_disabled: bool = False

    @property
    def is_user(self) -> bool:
        return self.kind == "user" and self.user_id is not None

    @property
    def is_anon(self) -> bool:
        return not self.is_user

    @property
    def plan(self) -> str:
        return self.plan_state.plan if self.plan_state is not None else "free"

    @classmethod
    def anonymous(cls, request_id: str = "", *, auth_disabled: bool = False) -> Principal:
        return cls(kind="anon", request_id=request_id, auth_disabled=auth_disabled)


def email_hash_for(email: str | None) -> str | None:
    """sha256 of the lower-cased, trimmed address. Unsalted on purpose: the
    hash has to be the same across deployments for the one-trial-per-person
    check and for support lookups by address; it is not a password."""
    if not email:
        return None
    normalised = email.strip().lower()
    if not normalised:
        return None
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def current_principal(request: Request) -> Principal:
    """The principal the middleware attached, or an anonymous one if the
    request somehow bypassed it (direct handler calls in tests)."""
    p = getattr(request.state, "principal", None)
    if isinstance(p, Principal):
        return p
    return Principal.anonymous()
