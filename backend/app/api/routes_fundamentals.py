"""FEAT-001 — the Fundamentals Explorer's catalog and series routes.

    GET  /api/fundamentals/catalog   every metric the explorer can draw
    POST /api/fundamentals/series    the exact data a chart displays

Both are DB reads (`fundamentals_series_service.build_series`; the one
provider-touching path is the cached daily price series for market-
derived metrics, the same read `/api/stocks/{t}/prices` makes). Nothing
here backfills, sweeps or calls an LLM: a ticker with no stored history
is reported as `not_backfilled` with the remedy — run research on it —
rather than fetched inside a page request.

Plan shape (the FEAT-002 seam). `require_feature("fundamentals_explorer")`
answers 401/403 exactly as every other customer route does, and the
`Grant.plan` it returns picks the request shape from
`features.shape(...)`: Free 2 companies × 2 metrics × 5 years, Pro 5 ×
4 × the full history. With the login wall off the plan is
`unrestricted`, which is the Pro shape — the owner decision for
anonymous visitors. Too many companies or metrics is a structured 402
`plan_required` (the request shape is wrong, so nothing is drawn); too
many *years* is capped silently and reported as `limits.capped_by_plan`,
because a long range is a shareable URL from a Pro user and should still
render for a Free viewer. Client-side hiding is UX; this is the gate.

Errors:
  422 `invalid_request`   unknown metric, bad normalize, years < 1 (the
                          service's `ValueError`); pydantic's own 422 for
                          >5 tickers / >4 metrics / empty lists
  404 `unknown_tickers`   every ticker is unknown to `companies` AND has no
                          stored rows — there is nothing to draw and no
                          per-ticker state to report
  402 `plan_required`     shape exceeded (see above)
  429                     slowapi per-IP (`fundamentals_series`) always;
                          per-user `series` / `data` scopes under the wall
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import features
from ..auth.entitlements import UPGRADE_URL, EntitlementError, Grant, require_feature
from ..database import get_db
from ..models import Company, FinancialPeriod
from ..rate_limit import LIMITS, limiter
from ..schemas.accounts import StructuredError
from ..schemas.fundamentals import CatalogOut, MetricSpecOut, SeriesRequest, SeriesResponse
from ..services import fundamentals_catalog as catalog
from ..services.fundamentals_series_service import NOT_BACKFILLED_REMEDY, build_series
from .gating import rate_scope

router = APIRouter()

FEATURE = "fundamentals_explorer"


def _utcnow() -> datetime:
    """Clock seam for tests (the service has its own for the data)."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Plan shape
# ---------------------------------------------------------------------------

def _plan_label(plan: str) -> str:
    return "Free Explorer" if plan == "free" else "Pro"


def _years_label(max_years: int | None) -> str:
    return "full history" if max_years is None else f"{max_years} years"


def _shape_exceeded(grant: Grant, shape: features.Shape, requested: dict[str, int | None]) -> EntitlementError:
    """402 with what was asked, what the plan allows and what Pro unlocks —
    enough for the UpgradePrompt to say something specific."""
    pro = features.shape(FEATURE, "pro")
    return EntitlementError(
        402, "plan_required",
        f"{_plan_label(grant.plan)} draws up to {shape.max_companies} companies × {shape.max_metrics} "
        f"metrics × {_years_label(shape.max_years)}; this request asks for "
        f"{requested['companies']} companies × {requested['metrics']} metrics. "
        f"Pro draws {pro.max_companies} × {pro.max_metrics} × {_years_label(pro.max_years)}.",
        feature=FEATURE, plan=grant.plan, upgrade_url=UPGRADE_URL,
        extra={
            "limits": shape.as_dict(),
            "requested": requested,
            "upgrade": {"plan": "pro", "limits": pro.as_dict(), "url": UPGRADE_URL},
        },
    )


def _cap_years(requested: int | None, max_years: int | None) -> tuple[int | None, bool]:
    """(years to draw, capped_by_plan). `None` requested = the plan's
    default range: everything on record for an uncapped plan, the plan
    ceiling otherwise — and that is NOT a cap, since nothing was asked for
    beyond the plan (a Free page load with the default range must not nag
    about upgrading). Only an explicit request above the ceiling is capped."""
    if max_years is None:
        return requested, False
    if requested is None:
        return max_years, False
    if requested > max_years:
        return max_years, True
    return requested, False


# ---------------------------------------------------------------------------
# Ticker resolution
# ---------------------------------------------------------------------------

def _known_tickers(db: Session, tickers: list[str]) -> set[str]:
    """Tickers the platform knows at all: in `companies`, or with stored
    statement rows (a ticker can have history without a profile when a
    backfill ran ahead of the universe seed). Two small IN queries."""
    known = {row[0] for row in db.execute(select(Company.ticker).where(Company.ticker.in_(tickers)))}
    rest = [t for t in tickers if t not in known]
    if rest:
        known |= {
            row[0] for row in db.execute(
                select(FinancialPeriod.ticker).where(FinancialPeriod.ticker.in_(rest)).distinct()
            )
        }
    return known


def _structured(status: int, code: str, message: str, **fields) -> HTTPException:
    """A non-entitlement refusal in the same `{"detail": StructuredError}`
    envelope, so the frontend has one error shape to switch on."""
    err = StructuredError(code=code, message=message, **fields)
    return HTTPException(status_code=status, detail=err.model_dump(mode="json", exclude_none=True))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/api/fundamentals/catalog", response_model=CatalogOut)
@limiter.limit(LIMITS["fundamentals_series"])
def get_catalog(
    request: Request,
    response: Response,
    _rate: None = Depends(rate_scope("data")),
    grant: Grant = Depends(require_feature(FEATURE, resource_param=None)),
) -> CatalogOut:
    """The metric table the explorer draws from: formulas in statement
    line-item terms, units, provenance rules, and the `source_kind`
    vocabulary reserved for the expectations ledger (empty in v1 — see
    `fundamentals_catalog`)."""
    return CatalogOut(
        catalog_version=catalog.CATALOG_VERSION,
        as_of=_utcnow(),
        families=list(catalog.FAMILIES),
        metrics=[MetricSpecOut.model_validate(entry) for entry in catalog.catalog_entries()],
        reserved_source_kinds=list(catalog.RESERVED_SOURCE_KINDS),
    )


@router.post("/api/fundamentals/series", response_model=SeriesResponse)
@limiter.limit(LIMITS["fundamentals_series"])
def post_series(
    request: Request,
    response: Response,
    req: SeriesRequest,
    db: Session = Depends(get_db),
    _rate: None = Depends(rate_scope("series")),
    grant: Grant = Depends(require_feature(FEATURE, resource_param=None)),
) -> SeriesResponse:
    """One series per (ticker, metric) on a shared fiscal-year axis; every
    missing value carries a reason. See the module docstring for the plan
    shape and the error table."""
    shape = features.shape(FEATURE, grant.plan)
    n_companies, n_metrics = len(req.tickers), len(req.metrics)
    if n_companies > shape.max_companies or n_metrics > shape.max_metrics:
        raise _shape_exceeded(
            grant, shape, {"companies": n_companies, "metrics": n_metrics, "years": req.years},
        )
    years, capped = _cap_years(req.years, shape.max_years)

    # Metric ids first (no DB): a typo in the selection is a 422 whatever
    # the tickers are, so the client learns about the wrong thing.
    # Echo the offending ids bounded (they are user input): at most a few,
    # each clipped, so a junk request cannot inflate the response or logs.
    unknown = [m[:64] for m in req.metrics if m not in catalog.CATALOG][:10]
    if unknown:
        raise _structured(
            422, "invalid_request", "unknown metric id(s): " + ", ".join(unknown),
            feature=FEATURE, extra={"unknown_metrics": unknown, "known_metrics": list(catalog.metric_ids())},
        )

    if not _known_tickers(db, req.tickers):
        raise _structured(
            404, "unknown_tickers",
            "None of the requested tickers is known to the platform: " + ", ".join(req.tickers) + ". "
            + NOT_BACKFILLED_REMEDY,
            feature=FEATURE, extra={"tickers": req.tickers},
        )

    try:
        return build_series(
            req.tickers, req.metrics, years=years, normalize=req.normalize,
            capped_by_plan=capped, db=db,
        )
    except ValueError as exc:
        # The service's own validation (bad normalize, years < 1 — both
        # already refused by `SeriesRequest`, kept as the honest fallback
        # for a future service rule). The message names the caller's
        # input, nothing else.
        raise _structured(422, "invalid_request", str(exc)[:200], feature=FEATURE) from None
