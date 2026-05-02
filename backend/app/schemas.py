"""Pydantic schemas used at the API boundary and as agent structured outputs.

These are deliberately verbose: agents emit Pydantic-validated JSON so the
frontend can render rich, deterministic memos even when the LLM is offline.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Foundational types
# ---------------------------------------------------------------------------

RatingLabel = Literal[
    "Bullish", "Mixed Positive", "Neutral", "Mixed Negative", "Bearish"
]

IntentType = Literal[
    "single_stock_analysis",
    "stock_comparison",
    "thematic_screen",
    "macro_question",
    "portfolio_construction",
    "dcf_analysis",
    "comps_analysis",
    "general_research_chat",
]


class CompanyOut(BaseModel):
    ticker: str
    company_name: str
    exchange: str
    sector: str
    industry: str
    sub_industry: Optional[str] = None
    country: str = "US"
    currency: str = "USD"
    market_cap: Optional[float] = None
    business_description: str = ""
    last_price: Optional[float] = None
    is_etf: bool = False
    beta: Optional[float] = None
    shares_outstanding: Optional[float] = None
    # Universe tier (Phase F + Wave 1B). Frontend uses this to render
    # the appropriate analyze affordance: `auto_analysis` shows a memo
    # immediately, `analyzed_on_demand` shows a cached memo, `data_only`
    # shows an explicit "Analyze this stock" gate.
    universe_tier: str = "data_only"


class CatalystItem(BaseModel):
    title: str
    detail: str = ""
    horizon: Literal["near_term", "medium_term", "long_term"] = "medium_term"
    impact: Literal["low", "medium", "high"] = "medium"


class RiskItem(BaseModel):
    title: str
    detail: str = ""
    severity: Literal["low", "medium", "high"] = "medium"
    type: Literal["company", "valuation", "macro", "regulatory", "thesis_breaker"] = "company"


class BullBearCase(BaseModel):
    headline: str
    key_points: List[str] = Field(default_factory=list)


class FalsifiableTest(BaseModel):
    """A concrete observation that would invalidate one side of the thesis.

    Required by the sector-integrated bull/bear (Wave 3A): a side without a
    falsifiable test is hand-waving. Forcing the analyst to articulate
    "this is wrong if X is observed" disciplines the case construction.
    """
    statement: str
    invalidates_side: Literal["bull", "bear"]


class BullBearAnalysis(BaseModel):
    """Wave 3A — sector-integrated bull/bear with bias mitigations.

    Stored under `sector_agent_view.data["bull_bear_analysis"]`. PM
    synthesis treats `sector_synthesis` as a prior, not a directive: the
    PM rating may diverge from `sector_lean` when other findings outvote
    the sector view, and the PM should explain that divergence.
    """
    bull_case: BullBearCase
    bear_case: BullBearCase
    key_disagreement: str
    falsifiable_tests: List[FalsifiableTest] = Field(default_factory=list)
    sector_synthesis: str
    sector_lean: Literal["bull", "bear", "balanced"] = "balanced"


class AgentFinding(BaseModel):
    """One agent's contribution to a memo.

    `data` is an optional structured payload — agents that do real research
    (e.g. sector cohort analysis) attach distributional stats, peer placements,
    and trend tables here so the frontend can render rich evidence beyond
    prose.
    """
    agent: str
    headline: str
    summary: str
    key_points: List[str] = Field(default_factory=list)
    confidence: float = 0.7
    sources: List[str] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)


class CriticReview(BaseModel):
    overall_assessment: str
    challenges: List[str] = Field(default_factory=list)
    underweighted_risks: List[str] = Field(default_factory=list)
    suggested_revisions: List[str] = Field(default_factory=list)
    advice_compliance_check: str = "Output framed as research/education only."


# ---------------------------------------------------------------------------
# Multi-agent message contracts (Phase 4+)
# Lightweight Pydantic shapes used as the interchange between PM, sectors,
# tool agents, and the monitoring (news/social/macro) loops.
# ---------------------------------------------------------------------------

NewsSeverity = Literal["advisory", "material", "breaking"]


class NewsAlert(BaseModel):
    """Single news/social/macro item pushed into the hot cache."""
    ticker: Optional[str] = None
    sector: Optional[str] = None
    title: str
    summary: str = ""
    url: str = ""
    severity: NewsSeverity = "advisory"
    published_at: Optional[str] = None
    source: str = "news_service"


class MacroBroadcast(BaseModel):
    """Macro snapshot + regime label broadcast to PM and sector agents."""
    snapshot: Dict[str, float] = Field(default_factory=dict)
    regime: str = "mixed"
    favored_sectors: List[str] = Field(default_factory=list)
    pressured_sectors: List[str] = Field(default_factory=list)
    note: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class SectorQuery(BaseModel):
    """PM → sector: ask a sector for a structured view on `target_ticker`."""
    sector: str
    target_ticker: str
    question: str = ""
    include_peer_relevance: bool = True


class SectorReport(BaseModel):
    """Sector → PM: structured response. Phase 6 populates cross-sector
    relevance so the PM can pull through tickers in *other* sectors."""
    sector: str
    target_ticker: str
    finding: AgentFinding
    cross_sector_relevance: List[str] = Field(default_factory=list)
    macro_alignment: Optional[str] = None


class ToolFinding(BaseModel):
    """Tool agent → sector: an `AgentFinding` plus the tool name used."""
    tool: str
    finding: AgentFinding


# ---------------------------------------------------------------------------
# DCF
# ---------------------------------------------------------------------------

class DCFAssumptions(BaseModel):
    revenue_growth: List[float] = Field(default_factory=lambda: [0.10, 0.09, 0.08, 0.07, 0.06])
    operating_margin: List[float] = Field(default_factory=lambda: [0.25, 0.26, 0.27, 0.27, 0.27])
    tax_rate: float = 0.21
    da_pct_revenue: float = 0.04
    capex_pct_revenue: float = 0.05
    nwc_pct_revenue: float = 0.02
    terminal_growth: float = 0.025
    exit_ebitda_multiple: float = 15.0
    wacc: float = 0.085

    base_revenue: float = 0.0
    net_debt: float = 0.0
    diluted_shares: float = 0.0
    current_price: float = 0.0


class DCFYearProjection(BaseModel):
    year: int
    revenue: float
    ebit: float
    nopat: float
    da: float
    capex: float
    change_nwc: float
    fcff: float
    discount_factor: float
    pv_fcff: float


class DCFScenario(BaseModel):
    name: Literal["base", "bull", "bear"]
    label: str
    assumptions: DCFAssumptions
    projections: List[DCFYearProjection]
    pv_explicit: float
    terminal_value_gordon: float
    terminal_value_exit_multiple: float
    pv_terminal_gordon: float
    pv_terminal_exit: float
    enterprise_value_gordon: float
    enterprise_value_exit: float
    enterprise_value_blended: float
    equity_value: float
    implied_share_price: float
    upside_pct: float


class SensitivityCell(BaseModel):
    row_label: str
    col_label: str
    value: float


class DCFSensitivity(BaseModel):
    name: str
    row_axis: str
    col_axis: str
    rows: List[float]
    cols: List[float]
    cells: List[SensitivityCell]


class DCFResult(BaseModel):
    ticker: str
    current_price: float
    base: DCFScenario
    bull: DCFScenario
    bear: DCFScenario
    sensitivities: List[DCFSensitivity] = Field(default_factory=list)
    summary: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Comps
# ---------------------------------------------------------------------------

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


class CompsResult(BaseModel):
    target: CompsRow
    peers: List[CompsRow]
    median: CompsRow
    target_percentiles: Dict[str, float] = Field(default_factory=dict)
    premium_discount: Dict[str, float] = Field(default_factory=dict)
    interpretation: str = ""


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class PortfolioRequest(BaseModel):
    market_view: str
    risk_level: Literal["conservative", "balanced", "aggressive"] = "balanced"
    num_holdings: int = 10
    max_position_size: float = 0.15
    excluded_sectors: List[str] = Field(default_factory=list)
    excluded_tickers: List[str] = Field(default_factory=list)
    desired_sectors: List[str] = Field(default_factory=list)
    horizon: Literal["short", "medium", "long"] = "medium"


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


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------

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
# Memos
# ---------------------------------------------------------------------------

class TechnicalSignals(BaseModel):
    """Wave 3B — pure-math indicators surfaced by the Technical Analyst.

    All numeric fields are optional because the indicator may need more
    bars than the price history exposes (e.g., SMA200 needs 200 bars).
    `trend` and `momentum` are best-effort buckets derived from whichever
    indicators came back populated.
    """
    last_price: Optional[float] = None
    last_date: Optional[str] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    sma_50_above_200: Optional[bool] = None
    ema_10: Optional[float] = None
    ema_20: Optional[float] = None
    rsi_14: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_position: Optional[float] = None
    vwma_20: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    position_52w: Optional[float] = None
    trend: Literal["up", "down", "sideways"] = "sideways"
    momentum: Literal["positive", "negative", "neutral"] = "neutral"
    notes: List[str] = Field(default_factory=list)


class StockMemoOut(BaseModel):
    ticker: str
    company_name: str
    sector: str
    final_pm_view: str
    rating_label: RatingLabel
    confidence_score: float
    one_sentence_thesis: str
    business_summary: str
    sector_agent_view: AgentFinding
    earnings_agent_view: AgentFinding
    filing_agent_view: AgentFinding
    valuation_agent_view: AgentFinding
    comps_agent_view: AgentFinding
    macro_sensitivity: AgentFinding
    # Wave 3B — Technical Analyst. Optional so older memos that pre-date
    # this addition still validate; the graph populates it on every run.
    technical_agent_view: Optional[AgentFinding] = None
    bull_case: BullBearCase
    bear_case: BullBearCase
    catalysts: List[CatalystItem]
    key_risks: List[RiskItem]
    thesis_breakers: List[RiskItem]
    dcf_summary: Dict[str, Any] = Field(default_factory=dict)
    portfolio_fit: str = ""
    risk_committee_challenge: CriticReview
    final_verdict: str
    scores: Dict[str, float] = Field(default_factory=dict)
    sources_used: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_mode: Literal["demo", "live"] = "demo"
    # List of agents that failed during this memo's generation. Empty when
    # everything ran normally; populated by the safe-runner so the UI can
    # show "X analyst was unavailable" rather than dropping the memo.
    degraded_agents: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "MarketMosaic is for investment research and education only. "
        "It does not provide personalized financial, investment, legal, or tax advice."
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)


class AgentTrace(BaseModel):
    agent: str
    status: Literal["queued", "running", "done"] = "done"
    detail: str = ""


class ChatResponse(BaseModel):
    intent: IntentType
    answer: str
    agent_trace: List[AgentTrace] = Field(default_factory=list)
    memo: Optional[StockMemoOut] = None
    portfolio: Optional[ModelPortfolio] = None
    macro: Optional[MacroScenarioResult] = None
    dcf: Optional[DCFResult] = None
    comps: Optional[CompsResult] = None
    screener: Optional[ScreenerResult] = None
    sources: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "MarketMosaic is for research and education only and does not provide personalized financial advice."
    )
