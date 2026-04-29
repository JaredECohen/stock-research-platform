"""Stock endpoints — list, detail, memo generation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from ..agents.graph import run_stock_memo
from ..schemas import CompanyOut, StockMemoOut
from ..services.data_service import get_data_service
from ..services.fundamentals_service import get_full_financials
from ..services.market_data_service import get_basic_stats, get_price_series

router = APIRouter()


@router.get("/api/stocks", response_model=List[CompanyOut])
def list_stocks() -> List[CompanyOut]:
    ds = get_data_service()
    out: List[CompanyOut] = []
    for ticker in ds.list_tickers():
        profile = ds.get_company_profile(ticker)
        if profile:
            out.append(CompanyOut(**{k: profile.get(k) for k in CompanyOut.model_fields.keys() if k in profile}))
    return out


@router.get("/api/stocks/{ticker}")
def get_stock(ticker: str) -> Dict[str, Any]:
    fin = get_full_financials(ticker.upper())
    if not fin.get("profile"):
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")
    stats = get_basic_stats(ticker.upper())
    return dict(
        profile=fin["profile"],
        ratios=fin["ratios"],
        income=fin["income"],
        balance=fin["balance"],
        cash=fin["cash"],
        earnings=fin["earnings"],
        market_stats=stats,
    )


@router.get("/api/stocks/{ticker}/prices")
def get_stock_prices(ticker: str, days: int = 252) -> List[Dict[str, Any]]:
    rows = get_price_series(ticker.upper(), days)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No prices for {ticker}")
    return rows


@router.get("/api/stocks/{ticker}/memo", response_model=StockMemoOut)
def get_stock_memo(ticker: str, scenario: str = "soft_landing") -> StockMemoOut:
    try:
        return run_stock_memo(ticker.upper(), scenario=scenario)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/api/stocks/{ticker}/analyze", response_model=StockMemoOut)
def analyze_stock(ticker: str, scenario: Optional[str] = None) -> StockMemoOut:
    try:
        return run_stock_memo(ticker.upper(), scenario=scenario or "soft_landing")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
