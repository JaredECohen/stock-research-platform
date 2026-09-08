"""Comparable-company schemas — peer rows, self-history stats, result."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CompsRow(BaseModel):
    ticker: str
    company_name: str
    market_cap: float | None = None
    revenue_growth: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    ebitda_margin: float | None = None
    roic: float | None = None
    pe: float | None = None
    ev_revenue: float | None = None
    ev_ebitda: float | None = None
    p_fcf: float | None = None
    fcf_yield: float | None = None


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
    own_median: dict[str, float | None] = Field(default_factory=dict)
    own_p25: dict[str, float | None] = Field(default_factory=dict)
    own_p75: dict[str, float | None] = Field(default_factory=dict)
    current_percentile: dict[str, float] = Field(default_factory=dict)
    current_vs_own_median: dict[str, float] = Field(default_factory=dict)
    interpretation: str = ""


class CompsResult(BaseModel):
    target: CompsRow
    peers: list[CompsRow]
    median: CompsRow
    target_percentiles: dict[str, float] = Field(default_factory=dict)
    premium_discount: dict[str, float] = Field(default_factory=dict)
    interpretation: str = ""
    # Wave 3E: optional self-historical context. None when the target lacks
    # enough usable history for any metric (typical for a recent IPO or a
    # sparse demo dataset).
    history: CompsHistoryStats | None = None
    # Wave 10 — Track B exposure peers. Cross-sector names that share
    # the target's key exposures (AI capex, China consumer, long-rate
    # sensitivity, etc.) — picked at runtime by an LLM with theme-
    # exposure fallback. Empty when no such peers were identified.
    exposure_peers: list[CompsRow] = Field(default_factory=list)
    exposure_rationale: str = ""
