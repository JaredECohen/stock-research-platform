"""Comps endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ..agents.llm import llm_call_context
from ..auth.entitlements import Grant, require_feature
from ..auth.principal import current_principal
from ..schemas import CompsResult
from ..services.valuation_service import build_comps
from .gating import rate_scope

router = APIRouter()


@router.get("/api/comps/{ticker}", response_model=CompsResult)
def get_comps(
    request: Request,
    ticker: str,
    _rate: None = Depends(rate_scope("data")),
    _grant: Grant = Depends(require_feature("comps")),
) -> CompsResult:
    """FEAT-002: `comps` follows the memo on Free — available for a ticker
    whose memo the user opened this month — and is unrestricted on Pro.
    The cheap exposure-peers LLM call inside `build_comps` is attributed
    to the customer via the call context."""
    principal = current_principal(request)
    with llm_call_context(user_id=principal.user_id, feature="comps"):
        res = build_comps(ticker.upper())
    if res is None:
        raise HTTPException(status_code=404, detail=f"No peers configured for {ticker}")
    return res
