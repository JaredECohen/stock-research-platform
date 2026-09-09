"""Portfolio endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..agents.llm import llm_call_context
from ..auth.entitlements import Grant, require_feature
from ..auth.principal import current_principal
from ..schemas import ModelPortfolio, PortfolioRequest
from ..services.portfolio_service import build_model_portfolio
from .gating import rate_scope

router = APIRouter()


@router.post("/api/portfolio/build", response_model=ModelPortfolio)
def build(
    request: Request,
    req: PortfolioRequest,
    _rate: None = Depends(rate_scope("llm_light")),
    _grant: Grant = Depends(require_feature("portfolio", resource_param=None)),
) -> ModelPortfolio:
    """FEAT-002: Pro only (the middleware refuses Free before this runs);
    the feature adds a two-in-flight lease per user and the `llm_light`
    rate scope because the brief extraction is an LLM call."""
    principal = current_principal(request)
    with llm_call_context(user_id=principal.user_id, feature="portfolio"):
        return build_model_portfolio(req)
