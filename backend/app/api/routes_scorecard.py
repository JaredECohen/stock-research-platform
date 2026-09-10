"""Fundamental Factor Scorecard — public reads and the v1 export (Phase 6).

Every route here is a database read; nothing computes a score in a
request (the worker's `scorecard_loop` does, through the durable
`scorecard_runs` queue). Under `AUTH_ENABLED` the reads are Pro
(`auth/policy.py`), and the export additionally accepts — and, when
`SCORECARD_EXPORT_TOKEN` is set, REQUIRES — a bearer token of its own,
separate from the admin token, so the data can be handed to a downstream
system without granting the ops surface or a customer login.

Shapes keep the observed layer (raw values, periods, availability dates)
apart from the model read (z, score, percentiles). Missing evidence is
`null` with a reason, never zero. Research and education only.
"""
from __future__ import annotations

import logging
import secrets
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse

from ..config import settings
from ..rate_limit import LIMITS, limiter
from ..schemas.scorecard import ScorecardDetailOut, ScorecardEvaluationOut, ScorecardUniverseOut
from ..services import scorecard_service
from .gating import feature_disabled, rate_scope

log = logging.getLogger(__name__)
router = APIRouter()

EXPORT_FORMATS = ("csv", "json")
EXPORT_CONTRACTS = (scorecard_service.EXPORT_CONTRACT,)

# Whether THIS process has already tried to register the in-code spec.
# Only a memo of an attempt — the registry row in `scorecard_versions`
# is the state, shared with the worker through the database — so a GET
# does not repeat an idempotent write on every request.
_registry_attempted = False


def _register_version_lazily() -> None:
    """Register the in-code methodology from the web process the first
    time a scorecard route is hit. The worker does the same at boot and
    on every daily tick; doing it here too means `/api/scorecard/spec`
    reports `source: registry` even on a deployment whose worker has not
    ticked yet. Best-effort: any failure (a race with the worker's own
    upsert, a read-only replica) is logged by type and the readers fall
    back to the in-code spec exactly as before."""
    global _registry_attempted
    if _registry_attempted:
        return
    _registry_attempted = True
    try:
        scorecard_service.ensure_version_registered()
    except Exception as exc:
        log.warning("scorecard lazy version registration failed (serving the in-code spec): %s",
                    type(exc).__name__)


def _require_enabled() -> None:
    """`ENABLE_SCORECARD=false` is the kill switch for the read surface:
    404 `feature_disabled`, the route is simply not there."""
    if not settings.enable_scorecard:
        raise feature_disabled("the fundamental scorecard is not enabled on this deployment", feature="scorecard")
    _register_version_lazily()


def _version_or_404(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except scorecard_service.UnknownVersion as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


@router.get("/api/scorecard", response_model=ScorecardUniverseOut)
def get_scorecard_universe(
    as_of: date | None = Query(None, description="Latest run on or before this date; default the latest run."),
    version: str | None = Query(None, description="Methodology version key; default the active one."),
    sector: str | None = Query(None, description="Case-insensitive substring match on the stored sector."),
    sort_by: str = Query("overall_score"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(200, ge=1, le=600),
    min_coverage: float = Query(0.0, ge=0.0, le=1.0),
    _enabled: None = Depends(_require_enabled),
    _rate: None = Depends(rate_scope("data")),
) -> ScorecardUniverseOut:
    """The cross-section from the latest succeeded run.

    `sort_by` is whitelisted (`overall_score`, `overall_z`,
    `universe_percentile`, `sector_percentile`, `coverage`, `ticker`, or a
    family name to sort on that family's z) — anything else is 422 rather
    than a silent fallback. Rows whose overall is "insufficient data" are
    included with `overall_score: null` and sort last.
    """
    if sort_by not in scorecard_service.UNIVERSE_SORT_COLUMNS:
        raise HTTPException(
            status_code=422,
            detail=f"sort_by must be one of {sorted(scorecard_service.UNIVERSE_SORT_COLUMNS)}",
        )
    out = _version_or_404(
        scorecard_service.universe_table, version_key=version, as_of=as_of, sector=sector,
        sort_by=sort_by, order=order, limit=limit, min_coverage=min_coverage,
    )
    if out is None:
        raise HTTPException(status_code=404, detail=f"no scorecard run on file (version {version or 'active'})")
    return out


@router.get("/api/scorecard/spec")
def get_scorecard_spec(
    version: str | None = Query(None),
    _enabled: None = Depends(_require_enabled),
    _rate: None = Depends(rate_scope("data")),
) -> dict:
    """The methodology: families, features (formula, sign, weight,
    inputs, sector applicability), normalisation parameters and the
    null-handling / percentile rules, with the hash that fingerprints
    them. Served from the registry when the worker has registered the
    version, else from code."""
    return _version_or_404(scorecard_service.spec_view, version)


@router.get("/api/scorecard/evaluation", response_model=ScorecardEvaluationOut)
def get_scorecard_evaluation(
    version: str | None = Query(None),
    kind: str | None = Query(None, description="quintile_ls | ff6_regression | double_lasso"),
    _enabled: None = Depends(_require_enabled),
    _rate: None = Depends(rate_scope("data")),
) -> ScorecardEvaluationOut:
    """Latest persisted evaluation per kind, always with the caveats the
    evaluation is only honest with. Model output, not a recommendation."""
    return _version_or_404(scorecard_service.evaluation_view, version, kind=kind)


def _check_export_token(request: Request) -> None:
    """When `SCORECARD_EXPORT_TOKEN` is configured the export requires it
    as a bearer (and nothing else — the customer wall treats the route as
    public then, see `auth/policy.py`). Constant-time compare, no
    WWW-Authenticate challenge (machine-to-machine), token never logged."""
    token = settings.scorecard_export_token
    if not token:
        return
    supplied = request.headers.get("authorization") or ""
    expected = f"Bearer {token}"
    if len(supplied) != len(expected) or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="scorecard export token required")


@router.get("/api/scorecard/export")
@limiter.limit(LIMITS["scorecard_export"])
def export_scorecard(
    request: Request,
    response: Response,
    format: str = Query("csv", pattern="^(csv|json)$"),
    contract: str = Query("v1"),
    version: str | None = Query(None),
    as_of: date | None = Query(None),
    include_features: bool = Query(False, description="JSON only: append feature_raw / feature_z objects."),
    _enabled: None = Depends(_require_enabled),
    _rate: None = Depends(rate_scope("data")),
):
    """Stream the latest succeeded run under a FROZEN column contract.

    `contract=v1` promises `scorecard_service.EXPORT_COLUMNS_V1` in that
    exact order; new columns only ever appear under a new contract. Floats
    are 6 dp, nulls empty, rows sorted by ticker, streamed with
    `yield_per(200)` so the web process never materialises the run. The
    response carries `X-Scorecard-Contract`, `-Version`, `-As-Of`, `-Run-Id`
    and an `ETag` equal to the run id.
    """
    if contract not in EXPORT_CONTRACTS:
        raise HTTPException(status_code=422, detail=f"unknown export contract {contract!r}; supported: v1")
    _check_export_token(request)
    run = _version_or_404(scorecard_service.export_run, version_key=version, as_of=as_of)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no scorecard run on file (version {version or 'active'})")
    as_of_str = run["as_of"].isoformat()
    headers = {
        "X-Scorecard-Contract": scorecard_service.EXPORT_CONTRACT,
        "X-Scorecard-Version": run["version_key"],
        "X-Scorecard-As-Of": as_of_str,
        "X-Scorecard-Run-Id": run["run_id"],
        "ETag": f'"{run["run_id"]}"',
        "Cache-Control": "private, max-age=300",
    }
    if format == "json":
        headers["Content-Disposition"] = f'attachment; filename="scorecard_{run["version_key"]}_{as_of_str}.json"'
        return StreamingResponse(
            scorecard_service.iter_export_json(run, include_features=include_features),
            media_type="application/json", headers=headers,
        )
    headers["Content-Disposition"] = f'attachment; filename="scorecard_{run["version_key"]}_{as_of_str}.csv"'
    return StreamingResponse(
        scorecard_service.iter_export_csv(run), media_type="text/csv; charset=utf-8", headers=headers,
    )


@router.get("/api/scorecard/{ticker}", response_model=ScorecardDetailOut)
def get_scorecard_ticker(
    ticker: str,
    as_of: date | None = Query(None, description="Latest row on or before this date."),
    version: str | None = Query(None),
    months: int = Query(36, ge=1, le=120, description="Month-end history points to include."),
    _enabled: None = Depends(_require_enabled),
    _rate: None = Depends(rate_scope("data")),
) -> ScorecardDetailOut:
    """Latest score for one name with every feature's observed value and
    model read side by side, plus the month-end history. 404 when no
    succeeded run has scored the ticker."""
    out = _version_or_404(scorecard_service.ticker_detail, ticker, version_key=version, as_of=as_of, months=months)
    if out is None:
        raise HTTPException(
            status_code=404, detail=f"no scorecard for {ticker.upper()} (version {version or 'active'})",
        )
    return out
