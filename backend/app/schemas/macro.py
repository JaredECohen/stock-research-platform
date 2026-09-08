"""Macro schemas — series points and scenario request/result."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MacroSeriesPoint(BaseModel):
    date: str
    value: float | None = None


class MacroSeries(BaseModel):
    series_id: str
    name: str
    points: list[MacroSeriesPoint] = Field(default_factory=list)
    units: str = ""


class MacroScenarioRequest(BaseModel):
    scenario: str
    detail: str | None = None


class MacroScenarioResult(BaseModel):
    scenario: str
    narrative: str
    sector_impacts: dict[str, str]
    favored_sectors: list[str]
    pressured_sectors: list[str]
    suggested_research_views: list[str]
    risks: list[str]
    # Wave 10 — continuous regime probabilities. Real macro states are
    # mixtures (e.g., 0.55 soft / 0.30 sticky / 0.15 recession). The
    # `scenario` field carries the modal regime label for backward
    # compat; downstream consumers (sector tilts, memo invalidation
    # triggers) blend across regimes weighted by these probabilities.
    # Empty dict on memos that pre-date the field.
    regime_probabilities: dict[str, float] = Field(default_factory=dict)
