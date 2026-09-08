"""Macro schemas — series points and scenario request/result."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MacroSeriesPoint(BaseModel):
    date: str
    value: Optional[float] = None


class MacroSeries(BaseModel):
    series_id: str
    name: str
    points: List[MacroSeriesPoint] = Field(default_factory=list)
    units: str = ""


class MacroScenarioRequest(BaseModel):
    scenario: str
    detail: Optional[str] = None


class MacroScenarioResult(BaseModel):
    scenario: str
    narrative: str
    sector_impacts: Dict[str, str]
    favored_sectors: List[str]
    pressured_sectors: List[str]
    suggested_research_views: List[str]
    risks: List[str]
    # Wave 10 — continuous regime probabilities. Real macro states are
    # mixtures (e.g., 0.55 soft / 0.30 sticky / 0.15 recession). The
    # `scenario` field carries the modal regime label for backward
    # compat; downstream consumers (sector tilts, memo invalidation
    # triggers) blend across regimes weighted by these probabilities.
    # Empty dict on memos that pre-date the field.
    regime_probabilities: Dict[str, float] = Field(default_factory=dict)
