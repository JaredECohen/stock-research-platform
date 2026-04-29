"""Portfolio construction agent — wraps the deterministic engine with narrative."""
from __future__ import annotations

from typing import Dict, List

from ..schemas import ModelPortfolio, PortfolioRequest
from ..services.portfolio_service import build_model_portfolio


def run_portfolio_agent(request: PortfolioRequest) -> ModelPortfolio:
    return build_model_portfolio(request)
