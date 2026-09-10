// Phase 6 Fundamental Factor Scorecard — TypeScript mirror of the backend
// contracts served by /api/scorecard/* (plan §3 as amended by the
// orchestrator decisions: quintile evaluation legs, export contract v1).
// Mirrored by hand; keep the two in sync when either side changes.
//
// Research-process rules these shapes carry:
//   * a number the model could not compute is `null` — never 0, never
//     "neutral" — and the UI renders it as "n/a" with a reason;
//   * observed inputs (`raw`, `latest_period`, `data_available_at`) stay
//     separable from the model read (`z`, `score`, `contribution`) all the
//     way to the pixel, so a reader always knows which is which;
//   * evaluation results are model outputs about a scoring rule, not
//     recommendations; their `caveats` are rendered verbatim.

/** The eight feature families of spec fs-v1, in canonical order. This order
 *  is also the export column order (`z_valuation … z_earnings_quality`), so
 *  it is part of contract v1 and must not be reordered. */
export const SCORECARD_FAMILIES = [
  "valuation",
  "quality",
  "growth",
  "profitability",
  "efficiency",
  "leverage",
  "capital_allocation",
  "earnings_quality",
] as const;

export type ScorecardFamily = (typeof SCORECARD_FAMILIES)[number];

export const SCORECARD_FAMILY_LABELS: Record<ScorecardFamily, string> = {
  valuation: "Valuation",
  quality: "Quality",
  growth: "Growth",
  profitability: "Profitability",
  efficiency: "Efficiency",
  leverage: "Leverage",
  capital_allocation: "Capital allocation",
  earnings_quality: "Earnings quality",
};

/** The frozen export contract the page links to (`?contract=v1`). A new
 *  column order is a new contract, never a change to this one. */
export const SCORECARD_EXPORT_CONTRACT = "v1";

/** How the 0–100 score maps from z. `factor_scores._z_to_100` is linear:
 *  `50 + z / 2.5 * 50`, clipped, so 50 sits at z = 0 — the *mean* of the
 *  winsorized composite within the sector (or the universe when the sector
 *  is too small) — not at the universe median. Only the rank-based
 *  percentiles are median-anchored. One string, shared by the panel
 *  headline and the universe Score header so the two cannot disagree. */
export const SCORECARD_SCORE_SCALE = "0–100 · 50 = z of 0 (sector mean)";
export const SCORECARD_SCORE_SCALE_LONG = "0–100, linear in z: 50 = z of 0 (the sector mean, or the universe mean when the sector is too small); the percentile columns, not the score, are median-anchored.";

/** fs-v1 client-side interpretation rules. These are not part of the wire
 *  contract: the spec (`/api/scorecard/spec`) carries the normalisation
 *  parameters but not these thresholds, and an evaluation row's `params`
 *  may or may not name its minimums. They are collected here — one place,
 *  documented — so the memo section and the /app/scorecard page read the
 *  same numbers, and each component prefers the API's `params` value when
 *  it is present (see `evalParam`). Changing one is a client-rule change
 *  and must be reflected in the backend's pm_context block. */
export const SCORECARD_CLIENT_RULES = {
  /** A sub-composite z at or above this reads as that profile. */
  profileThresholdZ: 0.5,
  /** Months of month-end history below which an evaluation is "insufficient". */
  evalMinMonths: 24,
  /** Names a quintile leg needs or the month is skipped (orchestrator rule). */
  evalMinLeg: 15,
  /** Panel observations below which the LASSO draws no verdict. */
  lassoMinObs: 2000,
} as const;

/** Read a numeric minimum from an evaluation row's `params`, falling back
 *  to the documented fs-v1 client rule. Non-numeric or non-finite values
 *  fall back too, so a malformed row cannot print "NaN of NaN months". */
export function evalParam(params: Record<string, unknown> | null | undefined, key: "min_leg" | "min_months" | "min_obs", fallback: number): number {
  const v = params?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

export type ScorecardFeatureUnit = "ratio" | "percent" | "multiple" | "currency" | "count";

/** One family composite for one ticker. `z`/`score`/`percentile` are null
 *  when fewer than half of the applicable features were available. */
export interface ScorecardCategory {
  z: number | null;
  score: number | null;
  percentile: number | null;
  weight: number;
  n_features: number;
  n_available: number;
}

/** One feature row: the observed input (`raw`) beside the model read
 *  (`z`, `contribution`). `applicable: false` means the sector mask
 *  excluded it (e.g. EV features for Financials); it then counts neither
 *  for nor against coverage. `reason` is the server's short machine text
 *  for why `raw` is null (optional on the wire; the UI falls back to a
 *  generic reason). */
export interface ScorecardFeature {
  name: string;
  family: ScorecardFamily | string;
  /** +1: higher raw is better; -1: lower raw is better. */
  sign: 1 | -1;
  raw: number | null;
  z: number | null;
  weight: number;
  contribution: number | null;
  applicable: boolean;
  formula: string;
  unit: ScorecardFeatureUnit | string;
  reason?: string | null;
}

/** A top-N contributor as the API lists it (`top_positive`/`top_negative`). */
export interface ScorecardContribution {
  feature: string;
  family: ScorecardFamily | string;
  z: number;
  contribution: number;
}

export type DisagreementSeverity = "material" | "watch";
export type DisagreementDirection = "narrative_above_quant" | "narrative_below_quant";

/** Memo-vs-scorecard disagreement (plan §5.5). A finding, not an outage:
 *  it never appears in `degraded_agents`. */
export interface ScorecardDisagreement {
  severity: DisagreementSeverity;
  /** memo rating score (bucket centre) minus universe percentile. */
  gap: number;
  direction: DisagreementDirection;
  /** "overall" or the family that contradicts the memo (e.g. "valuation"). */
  dimension: string;
  note: string;
}

/** Research-process sub-composites (computed, not weighted into overall):
 *  compounder = mean of quality/profitability/capital-allocation/earnings-
 *  quality z; inflection = mean of growth z and the two acceleration
 *  features. Null when the inputs are. */
export interface ScorecardProfiles {
  compounder: number | null;
  inflection: number | null;
}

/** The memo-embedded summary (`StockMemoOut.scorecard`) and the base of
 *  the ticker detail. */
export interface ScorecardSummary {
  version_key: string;
  as_of: string;
  overall_z: number | null;
  overall_score: number | null;
  universe_percentile: number | null;
  sector_percentile: number | null;
  /** Share of applicable features that were available, 0–1. */
  coverage: number;
  /** Keyed by family; a family the run could not score may be absent or
   *  carry nulls — both render as n/a. */
  categories: Partial<Record<ScorecardFamily, ScorecardCategory>> & Record<string, ScorecardCategory>;
  top_positive: ScorecardContribution[];
  top_negative: ScorecardContribution[];
  profiles?: ScorecardProfiles | null;
  disagreement?: ScorecardDisagreement | null;
  /** The PM's reconciliation paragraph when a disagreement was surfaced. */
  reconciliation?: string | null;
  /** Fiscal period the fundamentals came from, e.g. "FY2025". */
  latest_period: string | null;
  /** Point-in-time date the fundamentals became available. */
  data_available_at: string | null;
  /** True when `as_of` is older than 45 days. */
  stale: boolean;
  is_month_end?: boolean;
}

/** `GET /api/scorecard/{ticker}` */
export interface ScorecardDetail extends ScorecardSummary {
  ticker: string;
  sector: string | null;
  run_id: string;
  price_date: string | null;
  features: ScorecardFeature[];
}

export interface ScorecardHistoryPoint {
  as_of: string;
  overall_score: number | null;
  universe_percentile: number | null;
  sector_percentile: number | null;
  coverage: number;
  category_z: Partial<Record<ScorecardFamily, number | null>> & Record<string, number | null>;
}

/** `GET /api/scorecard/{ticker}/history` — month-end rows only, oldest first. */
export interface ScorecardHistory {
  ticker: string;
  version_key: string;
  points: ScorecardHistoryPoint[];
}

export interface ScorecardUniverseRow {
  rank: number | null;
  ticker: string;
  company_name: string | null;
  sector: string | null;
  overall_score: number | null;
  universe_percentile: number | null;
  sector_percentile: number | null;
  coverage: number;
  category_score: Partial<Record<ScorecardFamily, number | null>> & Record<string, number | null>;
  top_positive: ScorecardContribution[];
  top_negative: ScorecardContribution[];
}

/** `GET /api/scorecard` */
export interface ScorecardUniverse {
  version_key: string;
  as_of: string;
  run_id: string;
  universe_size: number;
  rows: ScorecardUniverseRow[];
}

export interface ScorecardSpecFeature {
  name: string;
  sign: 1 | -1;
  weight: number;
  formula: string;
  description: string;
  inputs: string[];
  applicability: { exclude_sectors: string[] };
}

export interface ScorecardSpecFamily {
  name: ScorecardFamily | string;
  weight: number;
  features: ScorecardSpecFeature[];
}

/** `GET /api/scorecard/spec` */
export interface ScorecardSpec {
  version_key: string;
  spec_hash: string;
  families: ScorecardSpecFamily[];
  normalization: {
    winsor_pct: number;
    sector_neutral: boolean;
    min_sector_n: number;
    clip_z: number;
  };
}

// ---------------------------------------------------------------------------
// Evaluation (`GET /api/scorecard/evaluation`)
// ---------------------------------------------------------------------------

export type ScorecardEvaluationKind = "quintile_ls" | "ff6_regression" | "double_lasso";

export interface QuintileMonth {
  as_of: string;
  long_ret: number;
  short_ret: number;
  spread: number;
  n_long: number;
  n_short: number;
  universe_ew: number;
}

/** A month the evaluation could not score (e.g. a leg had fewer than the
 *  minimum 15 names). Reported, never silently dropped. */
export interface SkippedMonth {
  as_of: string;
  reason: string;
}

export interface QuintileBucket {
  q: number;
  mean_ret: number | null;
  n: number;
}

/** Top-vs-bottom quintile long/short monthly spread series with stats. */
export interface QuintileLSResult {
  months: QuintileMonth[];
  skipped_months: SkippedMonth[];
  mean_spread: number | null;
  stdev: number | null;
  sharpe_annualized: number | null;
  t_stat: number | null;
  hit_rate: number | null;
  max_drawdown: number | null;
  n_months: number;
  quintile_table: QuintileBucket[];
  monotonic: boolean | null;
  caveats: string[];
}

export type FF6Factor = "MKT_RF" | "SMB" | "HML" | "RMW" | "CMA" | "MOM";

/** OLS of the spread (or long-leg excess return) on FF5 + momentum. */
export interface FF6RegressionResult {
  series: "spread" | "long_excess";
  alpha_monthly: number | null;
  alpha_annualized: number | null;
  alpha_t: number | null;
  betas: Partial<Record<FF6Factor, number>>;
  beta_t: Partial<Record<FF6Factor, number>>;
  r_squared: number | null;
  n_months: number;
  start: string | null;
  end: string | null;
  /** True when n_months < 24 — the row is still written so the UI says so. */
  insufficient: boolean;
  caveats: string[];
}

export type LassoVerdict = "independent" | "subsumed" | "insufficient_data";

/** Double-selection LASSO (Belloni–Chernozhukov–Hansen) of next-month
 *  excess return on the overall z, controlling for the standard
 *  characteristics. */
export interface DoubleLassoResult {
  coef_d: number | null;
  se_hc0: number | null;
  se_cluster_month: number | null;
  t_stat: number | null;
  p_value: number | null;
  selected_y: string[];
  selected_d: string[];
  lambda_y: number | null;
  lambda_d: number | null;
  n_obs: number;
  n_months: number;
  p_controls: number;
  naive_coef: number | null;
  full_ols_coef: number | null;
  verdict: LassoVerdict;
  interpretation: string;
  caveats: string[];
}

interface EvaluationBase {
  created_at: string;
  sample_start: string | null;
  sample_end: string | null;
  n_obs: number;
  params: Record<string, unknown>;
}

export type ScorecardEvaluation =
  | (EvaluationBase & { kind: "quintile_ls"; result: QuintileLSResult })
  | (EvaluationBase & { kind: "ff6_regression"; result: FF6RegressionResult })
  | (EvaluationBase & { kind: "double_lasso"; result: DoubleLassoResult });

/** `GET /api/scorecard/evaluation` — latest row per kind. */
export interface ScorecardEvaluationResponse {
  version_key: string;
  evaluations: ScorecardEvaluation[];
}
