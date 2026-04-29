"""Portfolio endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from ..schemas import ModelPortfolio, PortfolioRequest
from ..services.portfolio_service import build_model_portfolio

router = APIRouter()


@router.post("/api/portfolio/build", response_model=ModelPortfolio)
def build(req: PortfolioRequest) -> ModelPortfolio:
    return build_model_portfolio(req)
