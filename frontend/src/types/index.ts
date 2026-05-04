// Shared types mirrored from the FastAPI Pydantic schemas.
// Keep these in sync with backend/app/schemas.py.

export type RatingLabel =
  | "Very Bullish"
  | "Bullish"
  | "Neutral"
  | "Bearish"
  | "Very Bearish";

export type IntentType =
  | "single_stock_analysis"
  | "stock_comparison"
  | "thematic_screen"
  | "macro_question"
  | "portfolio_construction"
  | "dcf_analysis"
  | "comps_analysis"
  | "general_research_chat";

export type UniverseTier = "data_only" | "auto_analysis" | "analyzed_on_demand";

export interface CompanyOut {
  ticker: string;
  company_name: string;
  exchange: string;
  sector: string;
  industry: string;
  sub_industry?: string | null;
  country?: string;
  currency?: string;
  market_cap?: number | null;
  business_description?: string;
  last_price?: number | null;
  is_etf?: boolean;
  beta?: number | null;
  shares_outstanding?: number | null;
  // Universe tier (Phase F + Wave 1B). Drives the analyze affordance.
  universe_tier?: UniverseTier;
}

// Multi-agent message contracts surfaced from sector_agent_view.data when
// the backend's sector agent has subscribed to MacroBroadcast / NewsAlerts
// (Phase 6). All optional — older memos won't carry these fields.
export type NewsSeverity = "advisory" | "material" | "breaking";

export interface NewsAlert {
  ticker?: string | null;
  sector?: string | null;
  title: string;
  summary?: string;
  url?: string;
  severity: NewsSeverity;
  published_at?: string | null;
  source?: string;
}

export interface MacroBroadcast {
  snapshot: Record<string, number>;
  regime: string;
  favored_sectors: string[];
  pressured_sectors: string[];
  note?: string;
  generated_at?: string;
}

export interface FalsifiableTest {
  statement: string;
  invalidates_side: "bull" | "bear";
}

export interface BullBearAnalysis {
  bull_case: { headline: string; key_points: string[] };
  bear_case: { headline: string; key_points: string[] };
  key_disagreement: string;
  falsifiable_tests: FalsifiableTest[];
  sector_synthesis: string;
  sector_lean: "bull" | "bear" | "balanced";
}

export interface SectorFindingData {
  cross_sector_relevance?: string[];
  macro_alignment?: string;
  macro_broadcast?: MacroBroadcast;
  pending_news_alerts?: NewsAlert[];
  bull_bear_analysis?: BullBearAnalysis;
  // The full sector research payload also rides here; consumers tolerate
  // arbitrary extra keys via the index signature.
  [key: string]: unknown;
}

export interface AgentFinding {
  agent: string;
  headline: string;
  summary: string;
  key_points: string[];
  confidence: number;
  sources: string[];
  data?: SectorFindingData;
  // Wave 3C — drill-down report (markdown). Optional; older memos won't carry it.
  long_form_report?: string | null;
}

export interface BullBearCase {
  headline: string;
  key_points: string[];
}

export interface CatalystItem {
  title: string;
  detail: string;
  horizon: "near_term" | "medium_term" | "long_term";
  impact: "low" | "medium" | "high";
}

export interface RiskItem {
  title: string;
  detail: string;
  severity: "low" | "medium" | "high";
  type: "company" | "valuation" | "macro" | "regulatory" | "thesis_breaker";
}

export interface CriticReview {
  overall_assessment: string;
  challenges: string[];
  underweighted_risks: string[];
  suggested_revisions: string[];
  advice_compliance_check: string;
}

export interface DCFAssumptions {
  revenue_growth: number[];
  operating_margin: number[];
  tax_rate: number;
  da_pct_revenue: number;
  capex_pct_revenue: number;
  nwc_pct_revenue: number;
  terminal_growth: number;
  exit_ebitda_multiple: number;
  wacc: number;
  base_revenue: number;
  net_debt: number;
  diluted_shares: number;
  current_price: number;
}

export interface DCFYearProjection {
  year: number;
  revenue: number;
  ebit: number;
  nopat: number;
  da: number;
  capex: number;
  change_nwc: number;
  fcff: number;
  discount_factor: number;
  pv_fcff: number;
}

export interface DCFScenario {
  name: "base" | "bull" | "bear";
  label: string;
  assumptions: DCFAssumptions;
  projections: DCFYearProjection[];
  pv_explicit: number;
  terminal_value_gordon: number;
  terminal_value_exit_multiple: number;
  pv_terminal_gordon: number;
  pv_terminal_exit: number;
  enterprise_value_gordon: number;
  enterprise_value_exit: number;
  enterprise_value_blended: number;
  equity_value: number;
  implied_share_price: number;
  upside_pct: number;
}

export interface SensitivityCell {
  row_label: string;
  col_label: string;
  value: number;
}

export interface DCFSensitivity {
  name: string;
  row_axis: string;
  col_axis: string;
  rows: number[];
  cols: number[];
  cells: SensitivityCell[];
}

export interface DCFResult {
  ticker: string;
  current_price: number;
  base: DCFScenario;
  bull: DCFScenario;
  bear: DCFScenario;
  sensitivities: DCFSensitivity[];
  summary: string;
  generated_at: string;
}

export interface CompsRow {
  ticker: string;
  company_name: string;
  market_cap?: number | null;
  revenue_growth?: number | null;
  gross_margin?: number | null;
  operating_margin?: number | null;
  ebitda_margin?: number | null;
  roic?: number | null;
  pe?: number | null;
  ev_revenue?: number | null;
  ev_ebitda?: number | null;
  p_fcf?: number | null;
  fcf_yield?: number | null;
}

export interface CompsHistoryStats {
  lookback_periods: number;
  lookback_label: string;
  own_median: Record<string, number | null>;
  own_p25: Record<string, number | null>;
  own_p75: Record<string, number | null>;
  current_percentile: Record<string, number>;
  current_vs_own_median: Record<string, number>;
  interpretation: string;
}

export interface CompsResult {
  target: CompsRow;
  peers: CompsRow[];
  median: CompsRow;
  target_percentiles: Record<string, number>;
  premium_discount: Record<string, number>;
  interpretation: string;
  // Wave 3E: optional self-historical context.
  history?: CompsHistoryStats | null;
}

export interface MacroScenarioResult {
  scenario: string;
  narrative: string;
  sector_impacts: Record<string, string>;
  favored_sectors: string[];
  pressured_sectors: string[];
  suggested_research_views: string[];
  risks: string[];
}

export interface PortfolioRequest {
  market_view: string;
  risk_level: "conservative" | "balanced" | "aggressive";
  num_holdings: number;
  max_position_size: number;
  excluded_sectors?: string[];
  excluded_tickers?: string[];
  desired_sectors?: string[];
  horizon?: "short" | "medium" | "long";
}

export interface PortfolioHolding {
  ticker: string;
  company_name: string;
  sector: string;
  weight: number;
  rationale: string;
  pm_conviction: number;
}

export interface ModelPortfolio {
  name: string;
  market_view: string;
  risk_level: string;
  holdings: PortfolioHolding[];
  sector_allocation: Record<string, number>;
  concentration: Record<string, number>;
  expected_volatility: number;
  risk_notes: string[];
  top_thesis_drivers: string[];
  what_could_invalidate: string[];
  watch_items: string[];
  disclaimer: string;
}

export interface ScreenerRow {
  rank: number;
  ticker: string;
  company_name: string;
  sector: string;
  pm_score: number;
  quality: number;
  growth: number;
  valuation: number;
  earnings_momentum: number;
  risk: number;
  macro_fit: number;
  one_line_thesis: string;
  main_catalyst: string;
  main_risk: string;
  theme?: string | null;
}

export interface ScreenerResult {
  theme?: string | null;
  rows: ScreenerRow[];
  generated_at: string;
}

// Wave 9b — Custom rule-based screen
export type ScreenerMetricName =
  | "pe_ttm" | "forward_pe" | "peg" | "ev_ebitda" | "ev_revenue"
  | "gross_margin" | "op_margin" | "fcf_margin" | "roic" | "roe"
  | "debt_to_ebitda" | "revenue_growth_yoy" | "dividend_yield"
  | "market_cap" | "beta";

export type ScreenerOp = ">" | "<" | ">=" | "<=" | "=" | "between";

export interface ScreenerRule {
  metric: ScreenerMetricName;
  op: ScreenerOp;
  value: number;
  value2?: number | null;
}

export interface CustomScreenRequest {
  rules: ScreenerRule[];
  sectors?: string[];
  sort_by?: ScreenerMetricName;
  order?: "asc" | "desc";
  limit?: number;
}

export interface CustomScreenRow {
  ticker: string;
  company_name: string;
  sector: string;
  pm_score?: number | null;
  rating_label?: string | null;
  metrics: Partial<Record<ScreenerMetricName, number | null>>;
}

export interface CustomScreenResult {
  rows: CustomScreenRow[];
  rule_count: number;
  matched: number;
  generated_at: string;
}

export interface StockMemoOut {
  ticker: string;
  company_name: string;
  sector: string;
  final_pm_view: string;
  rating_label: RatingLabel;
  confidence_score: number;
  one_sentence_thesis: string;
  business_summary: string;
  sector_agent_view: AgentFinding;
  earnings_agent_view: AgentFinding;
  filing_agent_view: AgentFinding;
  valuation_agent_view: AgentFinding;
  comps_agent_view: AgentFinding;
  macro_sensitivity: AgentFinding;
  // Wave 3B — Technical Analyst. Optional: older memos may not have it.
  technical_agent_view?: AgentFinding | null;
  bull_case: BullBearCase;
  bear_case: BullBearCase;
  catalysts: CatalystItem[];
  key_risks: RiskItem[];
  thesis_breakers: RiskItem[];
  dcf_summary: Record<string, unknown>;
  // Wave 10 — consensus-anchored DCF kept alongside the PM-adjusted view
  // (in `dcf_summary`). Empty object when the PM made no adjustments or
  // no LLM was available.
  dcf_initial_summary?: Record<string, unknown>;
  dcf_pm_adjustments?: Array<{
    field: string;
    from: number | string | null;
    to: number | string | null;
    rationale: string;
  }>;
  dcf_pm_adjustment_headline?: string;
  portfolio_fit: string;
  risk_committee_challenge: CriticReview;
  final_verdict: string;
  scores: Record<string, number>;
  sources_used: string[];
  generated_at: string;
  generation_mode: "demo" | "live";
  // Names of specialist agents that failed during this memo's run. Empty when
  // everything ran normally; populated by the backend safe-runner so the UI
  // can surface "X analyst unavailable" instead of dropping the memo.
  degraded_agents?: string[];
  // Wave 9 — PM↔specialist deep-research dialog. Empty when the loop is
  // disabled (default) or the memo is from a backtest run.
  round_findings?: RoundFindings[];
  disclaimer: string;
}

export type DeepResearchTarget =
  | "sector"
  | "earnings"
  | "valuation"
  | "comps"
  | "risk"
  | "filing"
  | "macro"
  | "technical";

export interface CritiqueQuestion {
  target_agent: DeepResearchTarget;
  question: string;
  why_it_matters?: string;
}

export interface RoundFindings {
  round: number;
  pm_questions: CritiqueQuestion[];
  findings: Record<string, AgentFinding>;
  early_exit?: boolean;
  pm_rationale?: string;
}

export interface AgentTrace {
  agent: string;
  status: "queued" | "running" | "done";
  detail: string;
}

export interface ChatResponse {
  intent: IntentType;
  answer: string;
  agent_trace: AgentTrace[];
  memo?: StockMemoOut;
  portfolio?: ModelPortfolio;
  macro?: MacroScenarioResult;
  dcf?: DCFResult;
  comps?: CompsResult;
  screener?: ScreenerResult;
  sources: string[];
  disclaimer: string;
}

export interface ProviderStatus {
  name: string;
  configured: boolean;
  healthy: boolean;
  notes: string;
  capabilities: string[];
}

export interface LLMStatus {
  configured: boolean;
  provider_choice: string;
  active_provider: "openai" | "anthropic" | "none";
  openai_configured: boolean;
  anthropic_configured: boolean;
  openai_strong_model: string;
  openai_cheap_model: string;
  anthropic_strong_model: string;
  anthropic_cheap_model: string;
}

export interface ProvidersStatusResponse {
  mode: "demo" | "live";
  providers: Record<string, ProviderStatus>;
  missing_api_keys: string[];
  llm_configured: boolean;
  llm?: LLMStatus;
  feature_flags: Record<string, boolean>;
}
