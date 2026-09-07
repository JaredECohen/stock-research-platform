"""Comparable-company schemas — peer rows, self-history stats, result."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CompsRow(BaseModel):
    ticker: str
    company_name: str
    market_cap: Optional[float] = None
    revenue_growth: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    ebitda_margin: Optional[float] = None
    roic: Optional[float] = None
    pe: Optional[float] = None
    ev_revenue: Optional[float] = None
    ev_ebitda: Optional[float] = None
    p_fcf: Optional[float] = None
    fcf_yield: Optional[float] = None


class CompsHistoryStats(BaseModel):
    """Wave 3E — self-historical valuation context.

    Distribution stats for the target's *own* multi-year history of each
    metric, plus where the live `CompsRow` value sits within that history.
    Surfaced alongside peer-relative stats so a reader can see whether a
    name that looks cheap vs. peers is actually expensive vs. its own
    history (or vice versa).

    All dicts are keyed by the same metric vocabulary as `CompsRow`
    (revenue_growth, gross_margin, ev_ebitda, …). Values are None for
    metrics with insufficient history.
    """
    lookback_periods: int
    lookback_label: str  # e.g. "20 quarters" / "5y"
    own_median: Dict[str, Optional[float]] = Field(default_factory=dict)
    own_p25: Dict[str, Optional[float]] = Field(default_factory=dict)
    own_p75: Dict[str, Optional[float]] = Field(default_factory=dict)
    current_percentile: Dict[str, float] = Field(default_factory=dict)
    current_vs_own_median: Dict[str, float] = Field(default_factory=dict)
    interpretation: str = ""


class CompsResult(BaseModel):
    target: CompsRow
    peers: List[CompsRow]
    median: CompsRow
    target_percentiles: Dict[str, float] = Field(default_factory=dict)
    premium_discount: Dict[str, float] = Field(default_factory=dict)
    interpretation: str = ""
    # Wave 3E: optional self-historical context. None when the target lacks
    # enough usable history for any metric (typical for a recent IPO or a
    # sparse demo dataset).
    history: Optional[CompsHistoryStats] = None
    # Wave 10 — Track B exposure peers. Cross-sector names that share
    # the target's key exposures (AI capex, China consumer, long-rate
    # sensitivity, etc.) — picked at runtime by an LLM with theme-
    # exposure fallback. Empty when no such peers were identified.
    exposure_peers: List[CompsRow] = Field(default_factory=list)
    exposure_rationale: str = ""
