"""DCF endpoints."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from ..schemas import DCFAssumptions, DCFResult
from ..services import dcf_store
from ..services.valuation_service import build_dcf, default_dcf_assumptions

router = APIRouter()


@router.get("/api/dcf/{ticker}/default-assumptions", response_model=DCFAssumptions)
def get_default_assumptions(ticker: str) -> DCFAssumptions:
    """Engine-derived defaults. Wave 8I: when analyst consensus
    estimates are available the 5-year revenue-growth path starts from
    consensus rather than historical-trend extrapolation. Wave 8Q
    default-preserves current profitability (no implicit margin
    expansion) so DCFs aren't systematically bearish."""
    a = default_dcf_assumptions(ticker.upper())
    if a is None:
        raise HTTPException(status_code=404, detail=f"No financials for {ticker}")
    return a


@router.get("/api/dcf/{ticker}/consensus")
def get_consensus_baseline(ticker: str) -> Dict[str, Any]:
    """Wave 8Q — return the analyst-consensus 5-year growth path that
    the engine uses as a starting point, plus the trailing 3-year op
    margin (which is held flat by default). The DCF Lab renders this
    next to the live assumptions so any divergence is explicit and the
    rationale (LLM-driven updater audit trail) is one click away.
    """
    from ..finance.dcf import _consensus_growth_path
    from ..services.data_service import get_data_service
    from ..services.fundamentals_service import get_full_financials
    fin = get_full_financials(ticker.upper())
    income = sorted(fin.get("income") or [], key=lambda r: r.get("period", ""))
    consensus_growth = None
    try:
        estimates = get_data_service().get_estimates(ticker.upper())
        consensus_growth = _consensus_growth_path(estimates)
    except Exception:  # pragma: no cover — estimates optional
        consensus_growth = None
    # Trailing 3-yr op margin as the starting flat baseline.
    margin_baseline = None
    if income:
        ratios = []
        for row in income[-3:]:
            rev = row.get("revenue") or 0
            op = row.get("operating_income") or 0
            if rev:
                ratios.append(op / rev)
        if ratios:
            margin_baseline = sum(ratios) / len(ratios)
    return {
        "ticker": ticker.upper(),
        "consensus_revenue_growth": consensus_growth,  # 5-element list or None
        "trailing_op_margin": margin_baseline,         # float or None
        "has_consensus": consensus_growth is not None,
    }


@router.get("/api/dcf/{ticker}/saved")
def get_saved_assumptions(ticker: str) -> Dict[str, Any]:
    """Wave 8J — return the latest persisted DCF version's assumptions.

    Used by the DCF Lab to pre-populate the editor with what the
    platform last saved (post-LLM-updater) rather than re-deriving
    from defaults every time. Falls back gracefully when no version
    has been persisted yet.

    Returns `{has_saved, version, trigger, generated_at, assumptions}`
    where `has_saved=False` means no DCFModel row exists; the frontend
    can then offer the default-assumptions path.
    """
    snap = dcf_store.latest_version(ticker.upper())
    if snap is None:
        return {"has_saved": False, "ticker": ticker.upper()}
    assumptions = dcf_store.assumptions_to_pydantic(snap)
    return {
        "has_saved": True,
        "ticker": ticker.upper(),
        "version": snap.version,
        "trigger": snap.trigger,
        "parent_version": snap.parent_version,
        "generated_at": snap.generated_at.isoformat() if snap.generated_at else None,
        "assumption_changes": snap.assumption_changes or [],
        "assumptions": assumptions.model_dump(),
    }


@router.post("/api/dcf/{ticker}", response_model=DCFResult)
def run_dcf(ticker: str, assumptions: Optional[DCFAssumptions] = None) -> DCFResult:
    """Compute a DCF result from `assumptions`. Wave 8J: edits made in
    the DCF Lab are explicitly **non-persistent** when the supplied
    assumptions diverge from the engine defaults — `valuation_service.build_dcf`
    only writes a `DCFModel` row when the inputs match the default-derived
    set. So the lab is safe for ad-hoc what-if exploration."""
    res = build_dcf(ticker.upper(), assumptions)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Cannot build DCF for {ticker}")
    return res
