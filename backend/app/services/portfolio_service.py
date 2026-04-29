"""Portfolio construction service."""
from __future__ import annotations

from typing import List

from ..finance.portfolio_construction import build_portfolio
from ..schemas import ModelPortfolio, PortfolioRequest
from .screener_service import get_universe_dicts


def build_model_portfolio(request: PortfolioRequest) -> ModelPortfolio:
    candidates: List[dict] = get_universe_dicts(theme=None)
    name = "Scenario Portfolio"
    return build_portfolio(request, candidates, name=name)
