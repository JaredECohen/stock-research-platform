"""FEAT-001 — `POST /api/fundamentals/commentary` (slice S3).

The one LLM-touching route of the Fundamentals Explorer: one cheap-route
call over exactly the displayed series plus bounded stored-memo
excerpts, cached by fingerprint in `chart_commentaries` (shared by every
web replica; swept after 90 days by `monitoring/llm_log_gc`). The work
is `services/chart_commentary.generate`; this module is the wire: the
plan shape, the fingerprint refusal, the meter-outage refusal, and the
limits.

Limits, all of which apply before any LLM call:
  - slowapi `fundamentals_commentary` — 10/minute per IP, always;
  - `rate_scope("llm_light")` — the per-user window under the wall;
  - the `chart_commentary` feature (`auth/features.py`): Free 5 / Pro
    100 a month, at most two in flight per user via the DB-backed
    `active_actions` lease. Both are taken by `authorize()` inside the
    service on a cache miss with an LLM available — a cache hit, and
    every degraded answer, is free.

Plan shape: `require_feature("fundamentals_explorer")` yields the plan
(`unrestricted` with the wall off, i.e. the Pro shape), and the same
`features.shape` / year-cap rules as `POST /api/fundamentals/series`
apply, so the fingerprint the client sends was computed over the same
range the server recomputes here.

Anonymous, wall off: the default `FUNDAMENTALS_ANON_COMMENTARY=false`
returns the deterministic degraded shape (`degraded_reason`
"commentary requires an account"), no LLM call, nothing charged — the
owner decision for the logged-out surface. Wall on: anonymous is 401 at
the middleware like every other customer route.

Errors (all `{"detail": StructuredError}`):
  402 `plan_required`      shape exceeded (companies/metrics beyond the plan)
  402 `quota_exceeded`     monthly meter full (used/limit/resets_at)
  409 `series_changed`     the recomputed fingerprint differs from the
                           one sent; `extra.fingerprint` is the current
                           one, so the client refetches and retries
  422                      pydantic (empty lists, >5 tickers, >4 metrics,
                           a fingerprint of the wrong length)
  429 `concurrency_limited` two commentaries already in flight
  429 `rate_limited`       per-IP / per-user window
  503 `usage_unavailable`  the meter could not be read or written;
                           nothing charged, nothing generated, retry
  503 is never returned for an LLM outage: that is a 200 with
  `degraded=true` and a reason, and it is free.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from ..auth import features
from ..auth.entitlements import EntitlementError, Grant, require_feature
from ..auth.principal import current_principal
from ..database import get_db
from ..rate_limit import LIMITS, limiter
from ..schemas.fundamentals import CommentaryOut, CommentaryRequest
from ..services import chart_commentary
from .gating import rate_scope
from .routes_fundamentals import FEATURE as EXPLORER_FEATURE
from .routes_fundamentals import _cap_years, _shape_exceeded

router = APIRouter()

FEATURE = chart_commentary.FEATURE


@router.post("/api/fundamentals/commentary", response_model=CommentaryOut)
@limiter.limit(LIMITS["fundamentals_commentary"])
def post_commentary(
    request: Request,
    response: Response,
    req: CommentaryRequest,
    db: Session = Depends(get_db),
    _rate: None = Depends(rate_scope("llm_light")),
    grant: Grant = Depends(require_feature(EXPLORER_FEATURE, resource_param=None)),
) -> CommentaryOut:
    """Commentary on exactly the displayed chart. See the module
    docstring for the limits and the error table."""
    shape = features.shape(EXPLORER_FEATURE, grant.plan)
    n_companies, n_metrics = len(req.tickers), len(req.metrics)
    if n_companies > shape.max_companies or n_metrics > shape.max_metrics:
        raise _shape_exceeded(
            grant, shape, {"companies": n_companies, "metrics": n_metrics, "years": req.years},
        )
    years, _capped = _cap_years(req.years, shape.max_years)

    try:
        return chart_commentary.generate(
            req, current_principal(request), request=request, db=db, years=years,
        )
    except chart_commentary.SeriesChanged as exc:
        raise EntitlementError(
            409, "series_changed",
            "The displayed data has changed since the chart was drawn; refetch the series and try again.",
            feature=FEATURE,
            extra={"fingerprint": exc.actual, "requested": exc.requested},
        ) from None
    except chart_commentary.UsageUnavailable:
        raise EntitlementError(
            503, "usage_unavailable",
            "The usage meter is temporarily unavailable. Nothing was charged and no commentary "
            "was generated; try again shortly.",
            feature=FEATURE, retry_after=10, headers={"Retry-After": "10"},
        ) from None
