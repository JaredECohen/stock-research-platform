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

export interface GuidanceChange {
  metric: string;
  prior: string;
  current: string;
  direction:
    | "raised"
    | "lowered"
    | "reaffirmed"
    | "introduced"
    | "withdrawn"
    | "unclear";
  rationale: string;
}

export interface ToneSignal {
  speaker: string;
  segment: string;
  classification:
    | "constructive"
    | "measured"
    | "cautious"
    | "defensive"
    | "evasive";
  evidence: string;
}

export interface QAThemeAnalysis {
  theme: string;
  analyst?: string;
  response_quality: "clear" | "partial" | "deflected" | "evasive";
}

export interface EarningsStructured {
  period: string;
  overall_tone: "constructive" | "measured" | "cautious";
  guidance_changes: GuidanceChange[];
  tone_signals: ToneSignal[];
  qa_themes: QAThemeAnalysis[];
  most_defended_segment: { name?: string; why?: string };
  most_pressed_segment: { name?: string; why?: string };
  forward_catalysts: Array<{
    event?: string;
    expected_quarter?: string;
    materiality?: string;
  }>;
}

export interface AgentFinding {
  agent: string;
  headline: string;
  summary: string;
  key_points: string[];
  confidence: number;
  sources: string[];
  data?: SectorFindingData & { structured?: EarningsStructured };
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
  // null when the engine could not compute the number — no diluted share
  // count (implied price) or no positive quote (upside). Render "n/a";
  // a 0 here used to print as "+0.0%" and read as a real valuation.
  implied_share_price: number | null;
  upside_pct: number | null;
  // True when WACC − terminal growth hit the engine's 0.5% floor, so the
  // Gordon terminal value was capped and the price is not trustworthy.
  // Absent on results that pre-date the field.
  tv_clamped?: boolean;
}

export interface SensitivityCell {
  row_label: string;
  col_label: string;
  // null mirrors DCFScenario.implied_share_price (no share count).
  value: number | null;
}

// Wave 10 — sanity-check flag from the engine's `check_dcf_realism`.
export interface DCFGuardrail {
  severity: "warn" | "error";
  message: string;
  metric: string;
  value?: number | null;
  cohort_p90?: number | null;
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
  // null when no quote reached the model (older payloads may carry 0).
  current_price: number | null;
  base: DCFScenario;
  bull: DCFScenario;
  bear: DCFScenario;
  sensitivities: DCFSensitivity[];
  summary: string;
  guardrails?: DCFGuardrail[];
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
  /** Relative change, (current - median) / |median|; 0.067 = 6.7% above, not 6.7 points. */
  current_vs_own_median: Record<string, number>;
  /** Gap in percentage points for rate-type metrics; absent on older payloads. */
  current_minus_own_median_pp?: Record<string, number>;
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

// "Why is the market wrong" structure — consensus vs our view, the gap,
// and what would prove us wrong. Backfilled deterministically when the PM
// declines, so post-fix memos always carry content here.
export interface MispricingThesis {
  consensus_view: string;
  our_view: string;
  gap: string;
  falsifiers: string[];
}

// Single reconciled valuation call. The one place that answers "cheap or
// expensive?" — the thesis, valuation card, and rating all agree with it.
export interface ValuationVerdict {
  verdict: "undervalued" | "fairly_priced" | "overvalued";
  dcf_base_upside?: number | null;
  comps_ev_ebitda_premium?: number | null;
  factor_valuation?: number | null;
  summary: string;
}

export interface StockMemoOut {
  ticker: string;
  company_name: string;
  sector: string;
  final_pm_view: string;
  rating_label: RatingLabel;
  confidence_score: number;
  one_sentence_thesis: string;
  // Optional: memos predating these fields won't carry them.
  mispricing_thesis?: MispricingThesis;
  valuation_verdict?: ValuationVerdict;
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
  // RP-001 — why each `degraded_agents` entry is there, same order. Absent
  // on memos that pre-date the field; not rendered yet.
  degradation_events?: Array<{ agent: string; error_type: string; message: string }>;
  // RP-003 — findings from roster agents that have no dedicated field
  // above, keyed by roster key. Empty for the current roster; not rendered.
  extra_agent_views?: Record<string, AgentFinding>;
  // Wave 9 — PM↔specialist deep-research dialog. Empty when the loop is
  // disabled (default) or the memo is from a backtest run.
  round_findings?: RoundFindings[];
  // Phase 6 — the Fundamental Factor Scorecard summary the memo carries
  // (`StockMemoOut.scorecard`, additive). Absent on memos that pre-date
  // the field and null when the run had no row for the ticker; the memo
  // view hides its Scorecard section in both cases. Informs the memo
  // only — it does not move the rating.
  scorecard?: ScorecardSummary | null;
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
  | "technical"
  | "industry_group";

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
  /** Tickers the orchestrator could not answer about because no memo is
   *  stored and inline generation is off under the login wall; the UI
   *  offers a research run for each. Absent on older backends. */
  needs_analysis?: string[];
}

export interface ProviderStatus {
  name: string;
  configured: boolean;
  healthy: boolean;
  notes: string;
  capabilities: string[];
}

// Per-provider circuit-breaker snapshot from app/agents/llm.py. Scope
// caveat: web and worker each keep their own breakers, so a status served
// by web only describes web's view of the providers.
export interface LLMBreakerStatus {
  failure_count: number;
  is_open: boolean;
  seconds_since_last_failure: number | null;
  cooldown_seconds: number;
}

export interface LLMFailoverStatus {
  enabled: boolean;
  count: number;
  last_from: string | null;
  last_to: string | null;
  // ISO-8601 (naive strings are UTC) or epoch seconds; parseTimestamp() normalises.
  last_at: string | number | null;
  last_reason: string | null;
}

// Everything past the first five fields is optional: the deployed backend
// may lag the frontend, and the health banner must render nothing (never
// crash) when an older payload comes back without them.
export interface LLMStatus {
  configured: boolean;
  provider_choice: string;
  active_provider: "openai" | "anthropic" | "gemini" | "none" | (string & {});
  openai_configured: boolean;
  anthropic_configured: boolean;
  gemini_configured?: boolean;
  openai_strong_model?: string;
  openai_cheap_model?: string;
  anthropic_strong_model?: string;
  anthropic_cheap_model?: string;
  role_models?: Record<string, string>;
  breakers?: Record<string, LLMBreakerStatus>;
  failover?: LLMFailoverStatus;
  degraded?: boolean;
  degradation_reasons?: string[];
}

export interface ProvidersStatusResponse {
  mode: "demo" | "live" | (string & {});
  providers: Record<string, ProviderStatus>;
  missing_api_keys: string[];
  llm_configured: boolean;
  llm?: LLMStatus;
  feature_flags: Record<string, boolean>;
  /** What a reported flag actually does, where its name misleads (e.g. a flag no code reads). Absent on older backends. */
  feature_flag_notes?: Record<string, string>;
}

// ---------------------------------------------------------------------------
// FEAT-002 — accounts, entitlements, billing, structured errors.
// Mirrors backend/app/schemas/accounts.py. The backend is the authority on
// every one of these; the frontend only renders them.
// ---------------------------------------------------------------------------

export type PlanName = "free" | "pro" | "none";
export type PlanSource = "trial" | "subscription" | "override" | "grace" | "default" | "suspended";

/** Names in backend/app/auth/features.py. Kept as a union so the UI copy
 *  table (`FEATURE_LABELS`) cannot silently miss one. */
export type FeatureName =
  | "memo_view"
  | "research_run"
  | "pm_chat"
  | "chart_commentary"
  | "fundamentals_explorer"
  | "dcf"
  | "comps"
  | "portfolio"
  | "macro"
  | "track_record"
  | "memo_history"
  | "data_catalog"
  | "scorecard";

export interface Entitlement {
  feature: string;
  allowed: boolean;
  /** null = unlimited (when allowed). */
  limit: number | null;
  used: number;
  remaining: number | null;
  resets_at: string | null;
  /** Free may use DCF/comps only for a ticker already counted as a memo view this month. */
  follows_memo?: boolean;
  metered?: boolean;
}

export interface PlanState {
  plan: PlanName;
  source: PlanSource;
  trial_ends_at: string | null;
  period_end: string | null;
  cancel_at_period_end: boolean;
  grace_until: string | null;
  ends_at?: string | null;
  warning: string | null;
}

export interface AccountUser {
  id: number;
  external_id: string;
  email_verified: boolean;
  created_at: string;
  account_state: string;
  trial_started_at?: string | null;
  trial_ends_at?: string | null;
}

export interface BillingInfo {
  has_subscription: boolean;
  stripe_status: string | null;
  interval: string | null;
  /** True only when Stripe is configured AND this user has a customer id. */
  portal_available: boolean;
  billing_enabled?: boolean;
}

/** GET /api/me */
export interface Account {
  user: AccountUser;
  plan: PlanState;
  entitlements: Record<string, Entitlement>;
  billing: BillingInfo;
  period_key: string;
  usage_limits_enabled: boolean;
}

/** POST /api/me/bootstrap */
export interface BootstrapResponse extends Account {
  trial_started_now: boolean;
}

export interface UsageHistoryItem {
  feature: string;
  resource_ref: string | null;
  created_at: string;
  status: string;
  quantity: number;
}

/** GET /api/me/usage */
export interface UsageResponse {
  period_key: string;
  features: Record<string, Entitlement>;
  history: UsageHistoryItem[];
}

/** GET /api/public/config — safe defaults live in auth/ConfigProvider. */
export interface PublicConfig {
  auth_enabled: boolean;
  billing_enabled: boolean;
  usage_limits_enabled: boolean;
  clerk_publishable_key: string | null;
  clerk_frontend_api: string | null;
  sample_tickers: string[];
  prices: { monthly_cents: number; annual_cents: number; currency: string };
  legal_reviewed: boolean;
  app_env: string;
  /** Trial length the backend grants; null when the config fetch fell back
   *  to defaults (the UI then says "your Pro trial" without a number). */
  trial_days: number | null;
  /** The entitlement matrix the backend enforces (`features.registry_for_config`),
   *  including ENTITLEMENT_OVERRIDES_JSON. Every allowance number in UI copy
   *  comes from here or from `/api/me` — never from a literal. */
  features: FeatureMatrix;
}

/** Mirrors backend `auth/features.py` allowances: an int is metered per
 *  UTC month, null is unlimited, booleans are allowed / not allowed, and
 *  "follows_memo" means usable for a ticker whose memo was opened this month. */
export type FeatureAllowance = number | boolean | null | "follows_memo";

export interface FeatureMatrixEntry {
  description: string;
  free: FeatureAllowance;
  pro: FeatureAllowance;
  metered: boolean;
  period: string;
  distinct_resources: boolean;
}

export type FeatureMatrix = Record<string, FeatureMatrixEntry>;

export type BillingInterval = "month" | "year";

/** Every entitlement / quota / rate-limit refusal puts this inside FastAPI's
 *  `{"detail": ...}` envelope. `code` is what the UI switches on. */
export type ApiErrorCode =
  | "auth_required"
  | "auth_invalid"
  | "auth_unavailable"
  | "email_unverified"
  | "account_suspended"
  | "plan_required"
  | "quota_exceeded"
  | "rate_limited"
  | "concurrency_limited"
  | "feature_disabled"
  | "no_memo"
  | "already_subscribed"
  | "billing_unavailable"
  | (string & {});

export interface StructuredErrorDetail {
  code: ApiErrorCode;
  message?: string;
  feature?: string | null;
  plan?: string | null;
  used?: number | null;
  limit?: number | null;
  remaining?: number | null;
  resets_at?: string | null;
  upgrade_url?: string | null;
  scope?: string | null;
  retry_after?: number | null;
  window_seconds?: number | null;
  extra?: Record<string, unknown>;
}

export interface EntitlementRefusal {
  code: "plan_required" | "quota_exceeded";
  feature: string | null;
  plan: string | null;
  used: number | null;
  limit: number | null;
  resets_at: string | null;
  upgrade_url: string;
  message: string;
}

export interface RateLimitRefusal {
  code: "rate_limited" | "concurrency_limited";
  scope: string;
  retry_after: number;
  window_seconds: number | null;
  message: string;
}

/** 202 from POST /api/stocks/{t}/analyze (and from GET /memo?ondemand=true
 *  when the login wall routes generation through the worker). */
export interface AnalyzeJob {
  ticker: string;
  status: "started" | "in_progress";
  started_at: string;
  job_id: number;
  current_version: number | null;
  current_generated_at: string | null;
  note: string;
}

// ---------------------------------------------------------------------------
// FEAT-001 — Fundamentals Explorer. The chart engine's contract lives in
// ./fundamentals and is re-exported so pages import one module. The `*Wire`
// shapes add what the backend (`schemas/fundamentals.py`) serialises beyond
// that mirror — the series fingerprint the commentary route verifies, the
// shared period axis, server warnings, the per-ticker remedy — and the
// request the commentary route actually accepts (`years` is nullable: null
// is the full stored history the server applied).
// ---------------------------------------------------------------------------

export * from "./fundamentals";
import type { NormalizeMode, SeriesResponse, UnavailableTicker } from "./fundamentals";

export interface UnavailableTickerWire extends UnavailableTicker {
  /** What fixes it (for `not_backfilled`: run research on the company). */
  remedy?: string;
}

export interface SeriesResponseWire extends SeriesResponse {
  /** sha256 over the displayed values; echoed to the commentary route. */
  fingerprint: string;
  /** The shared fiscal-year axis, oldest first. */
  periods: string[];
  warnings: string[];
  normalize?: NormalizeMode;
  frequency?: "annual";
  unavailable: UnavailableTickerWire[];
}

export interface CommentaryRequestWire {
  tickers: string[];
  metrics: string[];
  /** The years the server applied to the displayed series (null = full history). */
  years: number | null;
  fingerprint: string;
}

// ---------------------------------------------------------------------------
// Phase 6 — Fundamental Factor Scorecard. The client-side mirror of
// `schemas/scorecard.py` lives in ./scorecard (types, the fs-v1 client
// rules and the score-scale captions) and is re-exported so pages import
// one module.
// ---------------------------------------------------------------------------

export * from "./scorecard";
import type { ScorecardSummary } from "./scorecard";

// ---------------------------------------------------------------------------
// FEAT-003 — Industry Analysis. The mirror of `schemas/industry.py` lives
// in ./industries (report, taxonomy, companies, history, changes, and the
// access block the UI reads to explain a gate) and is re-exported so
// pages import one module.
// ---------------------------------------------------------------------------

export * from "./industries";
