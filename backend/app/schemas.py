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
    "Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish"
]


def rating_from_stock_score(score: float) -> str:
    """Wave 8P — deterministic mapping from quant Stock Score → rating label.

    Locked decision: rating is now a function of the Stock Score (the
    quantitative factor blend) rather than the LLM's PM synthesis. The
    LLM's `confidence_score` separately reflects the agents' conviction
    in the directional call.

      80–100 → Very Bullish
      60–80  → Bullish
      40–60  → Neutral
      20–40  → Bearish
       0–20  → Very Bearish
    """
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "Neutral"
    if s >= 80:
        return "Very Bullish"
    if s >= 60:
        return "Bullish"
    if s >= 40:
        return "Neutral"
    if s >= 20:
        return "Bearish"
    return "Very Bearish"


_RATING_LABEL_TO_SCORE: Dict[str, float] = {
    "Very Bullish": 90.0,
    "Bullish": 70.0,
    "Neutral": 50.0,
    "Bearish": 30.0,
    "Very Bearish": 10.0,
}


def score_from_rating_label(label: Optional[str]) -> float:
    """Inverse of `rating_from_stock_score` — bucket centers on 0-100.

    Used by the PM rating-blend (Option A) so the LLM's directional call
    can be mixed with the quant factor score before label assignment.
    Unknown / missing labels collapse to Neutral (50).
    """
    if not label:
        return 50.0
    return _RATING_LABEL_TO_SCORE.get(str(label).strip(), 50.0)


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


class CritiqueQuestion(BaseModel):
    """Wave 9 — one follow-up question the PM emits during deep research.

    `target_agent` names which specialist's runner re-fires with this
    question as additional prompt context. `why_it_matters` is captured
    in the audit trail so reviewers can see *why* the PM dug in.
    """
    target_agent: Literal[
        "sector", "earnings", "valuation", "comps",
        "risk", "filing", "macro", "technical",
    ]
    question: str
    why_it_matters: str = ""


class CritiqueOutput(BaseModel):
    """PM critique step's structured output. `no_further_questions`
    is the explicit early-exit signal the loop respects so the PM can
    end the dialog before the round budget runs out."""
    questions: List[CritiqueQuestion] = Field(default_factory=list)
    no_further_questions: bool = False
    rationale: str = ""


class RoundFindings(BaseModel):
    """Wave 9 — one round of the deep-research dialog.

    `round=0` is the initial parallel fan-out (no PM questions). Rounds
    1+ each carry the PM's questions (`pm_questions`) and the
    re-fired agents' new findings (`findings`, keyed by agent_name).
    `early_exit` is set when the PM declared no further questions on
    THIS round, so reviewers know whether the loop terminated by
    consensus or by hitting the round cap.
    """
    round: int
    pm_questions: List[CritiqueQuestion] = Field(default_factory=list)
    findings: Dict[str, AgentFinding] = Field(default_factory=dict)
    early_exit: bool = False
    pm_rationale: str = ""


class RiskRecommendation(BaseModel):
    """Wave 8H — actionable rec from the risk analyst.

    The graph applies a deterministic enforcement step that REALLY
    moves these into the memo (confidence cap, rating downshift, bear
    augmentation, thesis-breaker propagation) so risk findings aren't
    just notes — they shape the final memo. The PM synthesis prompt
    also receives them so the LLM-written narrative acknowledges what
    the risk lens demanded.

    `target` — what part of the memo this rec touches.
    `direction` — which way to push.
    `magnitude` — how hard. Confidence deltas: small=5, medium=10, large=15.
    `detail` — short title shown in UI / prose.
    `rationale` — why; required so reviewers can audit why a rec moved.
    """
    target: Literal[
        "confidence", "rating", "thesis_breakers", "sizing", "bear_case",
    ]
    direction: Literal["raise", "lower", "flag", "neutral"]
    magnitude: Literal["small", "medium", "large"] = "medium"
    detail: str
    rationale: str


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


class Citation(BaseModel):
    """Wave 10 — typed citation for an evidence chain entry.

    `kind` is the source type — filing, transcript, ratio, peer,
    macro, dcf — so the UI can render an appropriate link / drawer.
    `ref` is the stable identifier (accession, period, ticker, etc.).
    `excerpt` is an optional short quote (≤300 chars) that the agent
    pulled into its prompt; we cap to keep the memo payload compact.
    """
    kind: Literal[
        "filing", "transcript", "ratio", "peer", "macro", "dcf",
        "news", "research_note", "other",
    ] = "other"
    ref: str = ""
    excerpt: str = ""
    section: Optional[str] = None


class AgentFinding(BaseModel):
    """One agent's contribution to a memo.

    `data` is an optional structured payload — agents that do real research
    (e.g. sector cohort analysis) attach distributional stats, peer placements,
    and trend tables here so the frontend can render rich evidence beyond
    prose.

    `long_form_report` (Wave 3C) is an optional 4-8 paragraph markdown
    drill-down. Always at least a deterministic build from the structured
    fields above; when `ENABLE_LONG_FORM_REPORTS=true`, enriched with an
    LLM expansion per agent. Frontend renders it in a collapsible drawer
    on each agent tile.

    Wave 10 — `evidence` is the typed citation list (one entry per
    citable claim or chunk). Empty when the agent didn't emit
    citations; backwards-compatible with memos that pre-date the
    field. Frontend renders this as "View sources" expander on the
    agent tile.
    """
    agent: str
    headline: str
    summary: str
    key_points: List[str] = Field(default_factory=list)
    confidence: float = 0.7
    sources: List[str] = Field(default_factory=list)
    evidence: List[Citation] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)
    long_form_report: Optional[str] = None


# Wave 10 — earnings structured extraction.

class GuidanceChange(BaseModel):
    metric: str  # e.g. "FY revenue", "Q4 op margin", "FY FCF"
    prior: Optional[str] = None
    current: Optional[str] = None
    direction: Literal["raised", "lowered", "reaffirmed", "introduced", "withdrawn", "unclear"] = "unclear"
    rationale: str = ""


class ToneSignal(BaseModel):
    speaker: str = ""  # e.g. "CEO", "CFO", "VP Finance"
    segment: str = ""  # e.g. "AWS", "Search", "Auto"
    classification: Literal["constructive", "measured", "cautious", "defensive", "evasive"] = "measured"
    evidence: str = ""  # short transcript quote


class QAThemeAnalysis(BaseModel):
    theme: str
    analyst: str = ""
    response_quality: Literal["clear", "partial", "deflected", "evasive"] = "clear"


class EarningsStructured(BaseModel):
    """Wave 10 — typed extraction over the transcript.

    Replaces the freeform `key_points` with a structure the UI can
    render as cards (guidance timeline, tone trends, Q&A heatmap).
    All fields default to empty so a partial LLM response still
    validates.
    """
    period: str = ""
    overall_tone: Literal["constructive", "measured", "cautious"] = "measured"
    guidance_changes: List[GuidanceChange] = Field(default_factory=list)
    tone_signals: List[ToneSignal] = Field(default_factory=list)
    qa_themes: List[QAThemeAnalysis] = Field(default_factory=list)
    most_defended_segment: Dict[str, str] = Field(default_factory=dict)  # {name, why}
    most_pressed_segment: Dict[str, str] = Field(default_factory=dict)
    forward_catalysts: List[Dict[str, str]] = Field(default_factory=list)  # [{event, expected_quarter, materiality}]


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


class DCFGuardrail(BaseModel):
    """Wave 10 — sanity-check flag emitted by `check_dcf_realism`.

    `severity`: warn | error. Warn = 'consider revising'; error =
    'the model is internally inconsistent'.
    """
    severity: Literal["warn", "error"] = "warn"
    message: str = ""
    metric: str = ""  # e.g. "implied_y5_ev_ebitda", "terminal_disagreement"
    value: Optional[float] = None
    cohort_p90: Optional[float] = None


class DCFResult(BaseModel):
    ticker: str
    current_price: float
    base: DCFScenario
    bull: DCFScenario
    bear: DCFScenario
    sensitivities: List[DCFSensitivity] = Field(default_factory=list)
    summary: str = ""
    # Wave 10 — reality-check flags. Empty list when nothing tripped.
    # The PM sees these and decides whether to defend or revise the
    # model; the UI surfaces them as a "model warnings" block.
    guardrails: List[DCFGuardrail] = Field(default_factory=list)
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
    # Wave 10 — continuous regime probabilities. Real macro states are
    # mixtures (e.g., 0.55 soft / 0.30 sticky / 0.15 recession). The
    # `scenario` field carries the modal regime label for backward
    # compat; downstream consumers (sector tilts, memo invalidation
    # triggers) blend across regimes weighted by these probabilities.
    # Empty dict on memos that pre-date the field.
    regime_probabilities: Dict[str, float] = Field(default_factory=dict)


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


class MispricingThesis(BaseModel):
    """The 'why is the market wrong' field — required output of the PM.

    Wave 10. A serious retail user pays for the *mispricing* call, not
    a metric recap. PM is required to fill these on every memo. Empty
    strings are allowed (the PM can say 'no mispricing — fairly priced
    on our work') but the structure must be present.
    """
    consensus_view: str = ""
    our_view: str = ""
    gap: str = ""
    falsifiers: List[str] = Field(default_factory=list)


class StockMemoOut(BaseModel):
    ticker: str
    company_name: str
    sector: str
    final_pm_view: str
    rating_label: RatingLabel
    confidence_score: float
    one_sentence_thesis: str
    # Wave 10 — mispricing-first synthesis. Empty Mispricing() means the
    # memo predates the schema or the PM declined to commit a view.
    mispricing_thesis: MispricingThesis = Field(default_factory=MispricingThesis)
    # Wave 10 — memo-time price snapshot. Frozen at memo creation so a
    # later live-overlay can show drift ("memo wrote DCF vs $145; current
    # $158, +9% since memo"). Null on memos that pre-date the field or
    # ran without quote chain access.
    price_at_memo: Optional[float] = None
    price_at_memo_at: Optional[datetime] = None
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
    # Wave 10 — initial (consensus-anchored) DCF kept alongside the
    # PM-adjusted view in `dcf_summary`. Empty when the PM made no
    # adjustments or no LLM was available. Lets the UI show "what the
    # team's research changed about the model".
    dcf_initial_summary: Dict[str, Any] = Field(default_factory=dict)
    dcf_pm_adjustments: List[Dict[str, Any]] = Field(default_factory=list)
    dcf_pm_adjustment_headline: str = ""
    portfolio_fit: str = ""
    risk_committee_challenge: CriticReview
    final_verdict: str
    scores: Dict[str, float] = Field(default_factory=dict)
    sources_used: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_mode: Literal["demo", "live"] = "demo"
    # Wave 9 — full diligence-dialog audit trail. `round=0` is the
    # initial parallel fan-out; rounds 1+ are PM↔specialist
    # critique-and-revise turns. Empty list when deep_research is off.
    round_findings: List[RoundFindings] = Field(default_factory=list)
    # Wave 10 — forward catalysts (next 90d) populated from
    # `catalyst_events`. Each item: {ticker, event_type, event_date,
    # title, description, materiality, source}.
    forward_catalysts: List[Dict[str, Any]] = Field(default_factory=list)
    # Wave 10 — earnings quarter-over-quarter delta (separate finding;
    # rendered as its own UI tile). Optional — None when prior-quarter
    # data isn't available yet.
    earnings_qoq_delta: Optional[AgentFinding] = None
    # Wave 10 — PM intake decision (which specialists were skipped and
    # why). `{skipped: List[str], rationale: str}`. Empty dict when
    # all 8 ran (the default). Audit trail for cost-aware memo runs.
    intake_decision: Dict[str, Any] = Field(default_factory=dict)
    # Wave 10 — per-agent influence on the rating. Computed post-PM
    # synthesis from each agent's confidence + tone; values are signed
    # contributions (positive = bullish pull, negative = bearish pull),
    # roughly normalized so the largest |value| is the most-influential
    # agent on this memo. Empty on memos that pre-date the field.
    agent_influence: Dict[str, float] = Field(default_factory=dict)
    # Wave 10 — macro context frozen at memo creation. Lets the
    # postmortem regime-conditional dashboards bucket memos by the
    # regime that was active when they were written, even if the macro
    # broadcast cache has rolled over by the time outcomes evaluate.
    macro_snapshot_at_memo: Dict[str, float] = Field(default_factory=dict)
    macro_regime_at_memo: str = ""
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
