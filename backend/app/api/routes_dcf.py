"""DCF endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import DCFAssumptions, DCFResult
from ..services.valuation_service import build_dcf, default_dcf_assumptions

router = APIRouter()


@router.get("/api/dcf/{ticker}/default-assumptions", response_model=DCFAssumptions)
def get_default_assumptions(ticker: str) -> DCFAssumptions:
    a = default_dcf_assumptions(ticker.upper())
    if a is None:
        raise HTTPException(status_code=404, detail=f"No financials for {ticker}")
    return a


@router.post("/api/dcf/{ticker}", response_model=DCFResult)
def run_dcf(ticker: str, assumptions: DCFAssumptions | None = None) -> DCFResult:
    res = build_dcf(ticker.upper(), assumptions)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Cannot build DCF for {ticker}")
    return res
