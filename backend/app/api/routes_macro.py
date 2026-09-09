"""Macro endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from ..agents.llm import llm_call_context
from ..agents.macro_agent import run_macro_scenario
from ..auth.entitlements import Grant, require_feature
from ..auth.principal import current_principal
from ..schemas import MacroScenarioRequest, MacroScenarioResult
from ..services.macro_service import get_series, list_series
from .gating import rate_scope

router = APIRouter()


@router.get("/api/macro/series")
def macro_series(
    series_id: str | None = None,
    _rate: None = Depends(rate_scope("series")),
    _grant: Grant = Depends(require_feature("macro", resource_param=None)),
) -> Any:
    if series_id:
        s = get_series(series_id)
        return s or {}
    return list_series()


@router.post("/api/macro/analyze", response_model=MacroScenarioResult)
def macro_analyze(
    request: Request,
    req: MacroScenarioRequest,
    _rate: None = Depends(rate_scope("llm_light")),
    _grant: Grant = Depends(require_feature("pm_chat", resource_param=None)),
) -> MacroScenarioResult:
    """FEAT-002: Pro only, and one LLM call — so it is metered as an
    Ask-the-PM turn (`pm_chat`) rather than left uncounted."""
    principal = current_principal(request)
    with llm_call_context(user_id=principal.user_id, feature="pm_chat"):
        return run_macro_scenario(req.scenario)
