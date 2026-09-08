"""Screener schemas — AI-rank rows and the rule-based custom screener."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ScreenerRow(BaseModel):
    rank: int
    ticker: str
    company_name: str
    sector: str
    pm_score: float
    quality: float
    growth: float
    valuation: float
    earnings_momentum: float
    risk: float
    macro_fit: float
    one_line_thesis: str
    main_catalyst: str
    main_risk: str
    theme: Optional[str] = None
    # Wave 10 — exposure score for the requested theme (0-100). None
    # when no theme filter was supplied or the ticker has no exposure
    # row computed yet. Powers the AI-rank screener's "this name is
    # actually exposed to your theme" affordance.
    theme_exposure_score: Optional[float] = None
    # Consecutive EPS beats (most-recent quarters, surprise > 2%).
    # Surfaces the "beat & raise" intuition: companies that string
    # together beats are a documented momentum signal. The screener
    # only has surprise history (no LLM guidance extraction), so this
    # is a "beat-only" streak; the memo path adds the guidance side.
    beat_streak: int = 0


class ScreenerRequest(BaseModel):
    theme: Optional[str] = None
    sectors: Optional[List[str]] = None
    sort_by: Optional[str] = "pm_score"
    limit: int = 50


class ScreenerResult(BaseModel):
    theme: Optional[str] = None
    rows: List[ScreenerRow]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Wave 9b — Custom rule-based screener (Phase 4)
# ---------------------------------------------------------------------------

ScreenerMetricName = Literal[
    "pe_ttm", "forward_pe", "peg", "ev_ebitda", "ev_revenue",
    "gross_margin", "op_margin", "fcf_margin", "roic", "roe",
    "debt_to_ebitda", "revenue_growth_yoy", "dividend_yield",
    "market_cap", "beta",
]

ScreenerOp = Literal[">", "<", ">=", "<=", "=", "between"]


class ScreenerRule(BaseModel):
    metric: ScreenerMetricName
    op: ScreenerOp
    value: float = 0.0
    # `value2` is required only when `op == "between"`; ignored otherwise.
    value2: Optional[float] = None


class CustomScreenRequest(BaseModel):
    rules: List[ScreenerRule] = Field(default_factory=list)
    sectors: Optional[List[str]] = None
    sort_by: ScreenerMetricName = "market_cap"
    order: Literal["asc", "desc"] = "desc"
    limit: int = Field(50, ge=1, le=500)


class CustomScreenRow(BaseModel):
    ticker: str
    company_name: str
    sector: str
    pm_score: Optional[float] = None
    rating_label: Optional[str] = None
    metrics: Dict[str, Optional[float]] = Field(default_factory=dict)


class CustomScreenResult(BaseModel):
    rows: List[CustomScreenRow]
    rule_count: int
    matched: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)
