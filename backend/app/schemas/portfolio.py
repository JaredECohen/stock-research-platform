"""Portfolio schemas — request, brief, holdings, model portfolio."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PortfolioRequest(BaseModel):
    market_view: str
    risk_level: Literal["conservative", "balanced", "aggressive"] = "balanced"
    num_holdings: int = 10
    max_position_size: float = 0.15
    excluded_sectors: List[str] = Field(default_factory=list)
    excluded_tickers: List[str] = Field(default_factory=list)
    desired_sectors: List[str] = Field(default_factory=list)
    horizon: Literal["short", "medium", "long"] = "medium"


class PortfolioBrief(BaseModel):
    """Wave 10 — structured brief extracted from a free-form market_view.

    The user's prompt drives portfolio composition. Today's
    `build_portfolio` only flexes via a 5-key scenario tag, which is
    why two different prompts that hit the same scenario produce
    nearly identical portfolios. This brief carries the *real* signal
    in the prompt so the scoring weights, factor tilts, sector
    targets, and constraints all flow from it.
    """
    horizon_years: int = 5  # 1, 3, 5, 10
    risk: Literal["conservative", "balanced", "aggressive"] = "balanced"
    themes: List[str] = Field(default_factory=list)
    factor_tilts: Dict[str, float] = Field(default_factory=dict)  # 0-1 weights
    sector_targets: Dict[str, float] = Field(default_factory=dict)  # sector → bias multiplier
    exclusions: Dict[str, List[str]] = Field(default_factory=dict)  # {tickers: [...], sectors: [...]}
    beta_target: Optional[float] = None
    yield_target: Optional[float] = None
    constraints: List[str] = Field(default_factory=list)  # e.g. "tax-efficient", "ESG-aware"
    rationale: str = ""  # LLM's explanation of how it read the prompt


class PortfolioHolding(BaseModel):
    ticker: str
    company_name: str
    sector: str
    weight: float
    rationale: str
    pm_conviction: float = 0.0


class ModelPortfolio(BaseModel):
    name: str
    market_view: str
    risk_level: str
    holdings: List[PortfolioHolding]
    sector_allocation: Dict[str, float]
    concentration: Dict[str, float]
    expected_volatility: float = 0.0
    risk_notes: List[str]
    top_thesis_drivers: List[str]
    what_could_invalidate: List[str]
    watch_items: List[str]
    disclaimer: str = (
        "Educational scenario-based portfolio. Not personalized financial advice."
    )
