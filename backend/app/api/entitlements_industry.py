"""FEAT-003 — the entitlement seam for the Industry Analysis surfaces.

One question, asked in one place: *what does this surface cost, and is
that being enforced right now?*

The answer comes from `services/industry_report_store.access_policy()`,
never from `settings.industry_analysis_access` directly — `latest` takes
the setting's tier, `history` and `changes` are Pro, and `enforced` is
`AUTH_ENABLED`. The chat tool already reports its tier that way, and
`auth/policy.py` classifies the routes from the same function, so the
login wall and this dependency cannot disagree about a surface.

Layering (the repo's existing shape, not a new one):

* `auth/policy.py` + the customer middleware do the COARSE check — is
  this route public, or does it need a signed-in Pro account. That runs
  before the handler.
* This dependency does the PER-FEATURE check inside the handler, through
  `auth/entitlements.authorize`, so the `industry_analysis` feature is
  charged and refused by the same machinery as every other feature. It
  also hands the handler the `access` block every response carries.

Refusals keep their structured FEAT-002 shape (`auth_required`,
`plan_required`, …) and gain `required_tier` and `surface`, so a client
can say "this is a Pro surface" rather than "an error occurred".

With `AUTH_ENABLED=false` — today's default deployment — nothing is
enforced and nothing is charged; the `access` block says so
(`enforced: false`) rather than implying the caller was entitled.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from starlette.requests import Request

from ..auth.entitlements import EntitlementError, authorize
from ..auth.features import get as get_feature
from ..config import settings
from ..database import get_db
from ..services import industry_report_store

FEATURE = "industry_analysis"

# The surfaces this API exposes, mapped to the store's policy keys. The
# store owns the tiers; this names which of them each route is.
SURFACE_LATEST = "latest"
SURFACE_HISTORY = "history"
SURFACE_CHANGES = "changes"
# The cross-industry snapshot is the compact view the PM block quotes, so
# it rides the `pm_chat` tier rather than inventing a surface the store
# has never heard of (unknown surfaces fail closed at Pro anyway).
SURFACE_SNAPSHOT = "pm_chat"

SURFACES: tuple[str, ...] = (SURFACE_LATEST, SURFACE_HISTORY, SURFACE_CHANGES, SURFACE_SNAPSHOT)


def access_block(
    surface: str, *, allowed: bool = True, plan: str | None = None, route_gated: bool = True,
) -> dict[str, Any]:
    """The `access` object every Industry Analysis response carries.

    `tier` is the policy for this surface; `required_tier` is `None` when
    the surface is public. `enforced` says whether `AUTH_ENABLED` makes
    any of it bite — a UI that renders "Pro" while `enforced` is false
    would be describing a gate that is not there.

    `route_gated` is the third, separate question: did THIS route apply
    the tier? `/taxonomy` answers to everyone by design — it is how the
    UI learns that the reports behind it are Pro — so it reports the
    `latest` tier with `route_gated: false` rather than implying the
    caller cleared a gate it never met.
    """
    policy = industry_report_store.access_policy()
    tier = policy["surfaces"].get(surface, industry_report_store.PRO)
    return {
        "surface": surface,
        "tier": tier,
        "required_tier": None if tier == industry_report_store.PUBLIC else tier,
        "allowed": allowed,
        "enforced": bool(policy["enforced"]),
        "route_gated": bool(route_gated),
        "setting": policy["setting"],
        "surfaces": dict(policy["surfaces"]),
        "plan": plan,
        "note": policy["note"],
    }


def tier_for(surface: str) -> str:
    """The tier one surface needs right now. Unknown surfaces fail closed."""
    return industry_report_store.surface_tier(surface)


def _refusal(exc: EntitlementError, surface: str, tier: str) -> HTTPException:
    """Re-raise an entitlement refusal with the surface and the tier it
    needs. `StructuredError` has no `required_tier` field, so the detail
    dict is extended rather than rebuilt — the FEAT-002 keys (`code`,
    `message`, `plan`, `upgrade_url`, …) and the `WWW-Authenticate`
    challenge survive untouched."""
    detail: dict[str, Any]
    if isinstance(exc.detail, dict):
        detail = dict(exc.detail)
    else:  # pragma: no cover — EntitlementError always carries a dict
        detail = {"code": "auth_required", "message": str(exc.detail)}
    detail.setdefault("feature", FEATURE)
    detail["surface"] = surface
    detail["required_tier"] = tier
    return HTTPException(status_code=exc.status_code, detail=detail, headers=exc.headers)


def enforce(request: Request, surface: str, *, db: Session | None = None) -> dict[str, Any]:
    """Authorize one read of `surface` and return its `access` block.

    A public surface is allowed without touching the database. A Pro
    surface goes through `authorize()`, which is where anonymity (401),
    suspension (403) and plan (402) are judged; the feature is not
    metered, so the grant is finalised immediately and nothing is
    charged against a monthly allowance.
    """
    tier = tier_for(surface)
    if tier == industry_report_store.PUBLIC or not settings.auth_enabled:
        # Not "allowed because we checked" — allowed because there is no
        # gate on this surface right now. `enforced` in the block says which.
        return access_block(surface)
    try:
        grant = authorize(request, FEATURE, db=db)
    except EntitlementError as exc:
        raise _refusal(exc, surface, tier) from None
    grant.commit(db)
    return access_block(surface, plan=grant.plan)


def require_surface(surface: str) -> Callable[..., dict[str, Any]]:
    """FastAPI dependency form: `access = Depends(require_surface("latest"))`."""
    get_feature(FEATURE)  # a typo fails at import, not at first request
    if surface not in SURFACES:  # pragma: no cover - guarded by the constants above
        raise ValueError(f"unknown industry surface {surface!r}")

    def dependency(request: Request, db: Session = Depends(get_db)) -> dict[str, Any]:
        return enforce(request, surface, db=db)

    dependency.__name__ = f"industry_access_{surface}"
    return dependency


def describe_surface(surface: str) -> Callable[..., dict[str, Any]]:
    """Dependency for a route that must ALWAYS answer, while still saying
    what the gated surface behind it costs.

    `/taxonomy` is the only one: it is what a signed-out visitor reads to
    find out that history and changes are Pro, and `auth/policy.py`
    classifies it PUBLIC unconditionally for the same reason. Gating it
    when `INDUSTRY_ANALYSIS_ACCESS=pro` would leave the UI with a 401 and
    no way to explain it. The block it returns is the `latest` tier with
    `route_gated: false` — the price list, not a receipt.
    """
    get_feature(FEATURE)
    if surface not in SURFACES:  # pragma: no cover - guarded by the constants above
        raise ValueError(f"unknown industry surface {surface!r}")

    def dependency() -> dict[str, Any]:
        return access_block(surface, route_gated=False)

    dependency.__name__ = f"industry_access_describe_{surface}"
    return dependency
