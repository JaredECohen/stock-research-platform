"""Comps endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import CompsResult
from ..services.valuation_service import build_comps

router = APIRouter()


@router.get("/api/comps/{ticker}", response_model=CompsResult)
def get_comps(ticker: str) -> CompsResult:
    res = build_comps(ticker.upper())
    if res is None:
        raise HTTPException(status_code=404, detail=f"No peers configured for {ticker}")
    return res
