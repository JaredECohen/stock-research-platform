"""Macro endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from ..agents.macro_agent import run_macro_scenario
from ..schemas import MacroScenarioRequest, MacroScenarioResult, MacroSeries
from ..services.macro_service import get_series, list_series

router = APIRouter()


@router.get("/api/macro/series")
def macro_series(series_id: Optional[str] = None) -> Any:
    if series_id:
        s = get_series(series_id)
        return s or {}
    return list_series()


@router.post("/api/macro/analyze", response_model=MacroScenarioResult)
def macro_analyze(req: MacroScenarioRequest) -> MacroScenarioResult:
    return run_macro_scenario(req.scenario)
