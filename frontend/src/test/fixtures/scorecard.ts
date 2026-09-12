// Fixtures for the Fundamental Factor Scorecard (Phase 6), shaped exactly
// as /api/scorecard/* serves them (see types/scorecard.ts). Values are
// hand-picked so tests can assert on exact strings; the larger universe
// is generated from a tiny seeded PRNG so it is stable across runs.
import {
  SCORECARD_EXPORT_CONTRACT,
  SCORECARD_FAMILIES,
  type DoubleLassoResult,
  type FF6RegressionResult,
  type QuintileLSResult,
  type ScorecardCategory,
  type ScorecardContribution,
  type ScorecardDetail,
  type ScorecardEvaluation,
  type ScorecardEvaluationResponse,
  type ScorecardFamily,
  type ScorecardFeature,
  type ScorecardHistory,
  type ScorecardHistoryPoint,
  type ScorecardSpec,
  type ScorecardSummary,
  type ScorecardUniverse,
  type ScorecardUniverseRow,
} from "@/types/scorecard";

export const VERSION = "fs-v1";
export const AS_OF = "2026-08-31";
export const RUN_ID = "run_2026-08-31_fs-v1";

/** The evaluation caveat strings the backend renders verbatim. Tests
 *  assert these exact sentences appear on the page unchanged. */
export const CAVEATS = {
  unadjusted: "Prices are FMP close values, unadjusted for splits; a split month can produce a spurious return.",
  constituents: "Universe is the current constituent list (survivorship: names that left the universe are absent).",
  sectors: "Sector labels are today's Company.sector strings, applied to every historical month.",
  restated: "Fundamental values are as restated today, dated at their original availability.",
  noCosts: "No transaction costs or borrow fees are modelled.",
} as const;

export const ALL_CAVEATS: string[] = [CAVEATS.unadjusted, CAVEATS.constituents, CAVEATS.sectors, CAVEATS.restated, CAVEATS.noCosts];

export function makeCategory(over: Partial<ScorecardCategory> = {}): ScorecardCategory {
  return { z: 0.4, score: 58.0, percentile: 62.0, weight: 0.125, n_features: 4, n_available: 4, ...over };
}

/** All eight families scored; valuation is the weak leg. */
export function fullCategories(): Record<ScorecardFamily, ScorecardCategory> {
  return {
    valuation: makeCategory({ z: -1.1, score: 28.0, percentile: 14.2, n_features: 4, n_available: 4 }),
    quality: makeCategory({ z: 1.2, score: 78.5, percentile: 88.0, n_features: 4, n_available: 4 }),
    growth: makeCategory({ z: 0.3, score: 55.9, percentile: 60.1, n_features: 5, n_available: 5 }),
    profitability: makeCategory({ z: 0.9, score: 71.2, percentile: 80.3, n_features: 4, n_available: 4 }),
    efficiency: makeCategory({ z: 0.1, score: 51.8, percentile: 52.0, n_features: 3, n_available: 3 }),
    leverage: makeCategory({ z: 0.5, score: 60.4, percentile: 66.7, n_features: 4, n_available: 3 }),
    capital_allocation: makeCategory({ z: 0.7, score: 65.0, percentile: 72.4, n_features: 4, n_available: 4 }),
    earnings_quality: makeCategory({ z: 1.4, score: 82.1, percentile: 91.5, n_features: 3, n_available: 3 }),
  };
}

/** Only five families scored: leverage/efficiency were masked out for a
 *  Financials name and growth had too few inputs (nulls with a count). */
export function partialCategories(): Partial<Record<ScorecardFamily, ScorecardCategory>> {
  return {
    valuation: makeCategory({ z: 0.2, score: 53.9, percentile: 55.0, n_features: 2, n_available: 2 }),
    quality: makeCategory({ z: 0.8, score: 69.0, percentile: 76.0 }),
    growth: makeCategory({ z: null, score: null, percentile: null, n_features: 5, n_available: 1 }),
    profitability: makeCategory({ z: -0.4, score: 42.0, percentile: 35.5, n_features: 3, n_available: 3 }),
    capital_allocation: makeCategory({ z: 0.1, score: 51.9, percentile: 52.2 }),
    earnings_quality: makeCategory({ z: 0.6, score: 62.7, percentile: 70.0, n_features: 3, n_available: 2 }),
  };
}

export const TOP_POSITIVE: ScorecardContribution[] = [
  { feature: "accruals_ratio", family: "earnings_quality", z: 1.4, contribution: 0.058 },
  { feature: "roic", family: "quality", z: 1.3, contribution: 0.041 },
  { feature: "gross_margin", family: "profitability", z: 1.1, contribution: 0.034 },
];

export const TOP_NEGATIVE: ScorecardContribution[] = [
  { feature: "ebitda_ev_yield", family: "valuation", z: -1.3, contribution: -0.041 },
  { feature: "fcf_yield", family: "valuation", z: -0.9, contribution: -0.028 },
  { feature: "sbc_to_revenue", family: "capital_allocation", z: -0.6, contribution: -0.019 },
];

export function makeSummary(over: Partial<ScorecardSummary> = {}): ScorecardSummary {
  return {
    version_key: VERSION,
    as_of: AS_OF,
    overall_z: 0.62,
    overall_score: 62.4,
    universe_percentile: 71.3,
    sector_percentile: 58.0,
    coverage: 0.93,
    categories: fullCategories(),
    top_positive: TOP_POSITIVE,
    top_negative: TOP_NEGATIVE,
    profiles: { compounder: 1.05, inflection: 0.2 },
    disagreement: null,
    reconciliation: null,
    latest_period: "FY2025",
    data_available_at: "2026-02-12",
    stale: false,
    is_month_end: true,
    ...over,
  };
}

/** A summary the run wrote with `overall` null: fewer than five families
 *  were scoreable, so there is no rank — the UI must say so. */
export function makeInsufficientSummary(over: Partial<ScorecardSummary> = {}): ScorecardSummary {
  return makeSummary({
    overall_z: null,
    overall_score: null,
    universe_percentile: null,
    sector_percentile: null,
    coverage: 0.42,
    categories: {
      valuation: makeCategory({ z: 0.2, score: 53.9, percentile: 55.0 }),
      quality: makeCategory({ z: null, score: null, percentile: null, n_features: 4, n_available: 1 }),
      profitability: makeCategory({ z: -0.4, score: 42.0, percentile: 35.5 }),
    },
    top_positive: [],
    top_negative: [],
    profiles: { compounder: null, inflection: null },
    latest_period: "FY2024",
    data_available_at: "2025-03-01",
    stale: true,
    ...over,
  });
}

export function makeDisagreementSummary(over: Partial<ScorecardSummary> = {}): ScorecardSummary {
  return makeSummary({
    universe_percentile: 18.0,
    overall_z: -0.9,
    overall_score: 32.1,
    disagreement: {
      severity: "material",
      gap: 52,
      direction: "narrative_above_quant",
      dimension: "overall",
      note: "Memo rating Bullish (70) vs universe percentile 18: gap 52 points.",
    },
    reconciliation: "The PM attributes the gap to a one-off restructuring charge depressing TTM earnings; the falsifier is FY2026 operating margin below 18%.",
    ...over,
  });
}

function feature(name: string, family: string, over: Partial<ScorecardFeature> = {}): ScorecardFeature {
  return {
    name,
    family,
    sign: 1,
    raw: 0.05,
    z: 0.5,
    weight: 0.03125,
    contribution: 0.0156,
    applicable: true,
    formula: `${name} formula`,
    unit: "ratio",
    ...over,
  };
}

/** A representative feature list: one observed value per unit, one masked
 *  feature, one with a server reason and one with no reason at all. */
export function makeFeatures(): ScorecardFeature[] {
  return [
    feature("fcf_yield", "valuation", { raw: 0.021, z: -0.9, contribution: -0.028, formula: "free_cash_flow_ttm / market_cap", unit: "percent" }),
    feature("ebitda_ev_yield", "valuation", { raw: 0.048, z: -1.3, contribution: -0.041, formula: "ebitda_ttm / EV", unit: "percent" }),
    feature("roic", "quality", { raw: 0.187, z: 1.3, contribution: 0.041, formula: "nopat_ttm / invested_capital", unit: "percent" }),
    feature("gross_margin_stability", "quality", { raw: -0.012, z: 0.4, contribution: 0.0125, formula: "-stdev(gross_margin, 5y)", unit: "ratio" }),
    feature("revenue_cagr_3y", "growth", { raw: null, z: null, contribution: null, reason: "fewer_than_3_annual_points", formula: "(revenue_ttm / revenue_3y_ago)^(1/3) - 1", unit: "percent" }),
    feature("net_debt_to_ebitda", "leverage", { raw: null, z: null, contribution: null, applicable: false, sign: -1, formula: "(total_debt - cash) / ebitda_ttm", unit: "multiple" }),
    feature("interest_coverage", "leverage", { raw: 24.6, z: 0.8, contribution: 0.025, formula: "min(ebit_ttm / |interest_expense|, 50)", unit: "multiple" }),
    feature("shareholder_yield", "capital_allocation", { raw: null, z: null, contribution: null, formula: "(|dividends| + |buybacks|) / market_cap", unit: "percent" }),
    feature("accruals_ratio", "earnings_quality", { raw: -0.031, z: 1.4, contribution: 0.058, sign: -1, formula: "(net_income - cfo) / avg_assets", unit: "ratio" }),
  ];
}

export function makeDetail(over: Partial<ScorecardDetail> = {}): ScorecardDetail {
  return {
    ...makeSummary(),
    ticker: "COST",
    sector: "Consumer Staples",
    run_id: RUN_ID,
    price_date: AS_OF,
    features: makeFeatures(),
    ...over,
  };
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

function monthEnd(year: number, month: number): string {
  // Day 0 of the next month is the last day of this one (month is 1-based).
  const d = new Date(Date.UTC(year, month, 0));
  return d.toISOString().slice(0, 10);
}

/** 36 month-ends ending at AS_OF, oldest first. Percentile drifts upward
 *  deterministically; two months are null (the run had insufficient
 *  coverage) so gap handling can be asserted. */
export function makeHistory(over: Partial<ScorecardHistory> = {}): ScorecardHistory {
  const points: ScorecardHistoryPoint[] = [];
  for (let i = 0; i < 36; i += 1) {
    const idx = 9 + i; // Sept 2023 … Aug 2026 (1-based month index from Jan 2023)
    const year = 2023 + Math.floor((idx - 1) / 12);
    const month = ((idx - 1) % 12) + 1;
    const missing = i === 5 || i === 20;
    const pct = missing ? null : Math.round((35 + i * 1.05 + (i % 3) * 2) * 10) / 10;
    points.push({
      as_of: monthEnd(year, month),
      overall_score: pct === null ? null : Math.round((40 + i * 0.6) * 10) / 10,
      universe_percentile: pct,
      sector_percentile: pct === null ? null : Math.min(100, pct - 8),
      coverage: missing ? 0.4 : 0.9,
      category_z: {
        valuation: pct === null ? null : -1.0 + i * 0.01,
        quality: pct === null ? null : 0.8,
      },
    });
  }
  return { ticker: "COST", version_key: VERSION, points, ...over };
}

// ---------------------------------------------------------------------------
// Universe
// ---------------------------------------------------------------------------

/** mulberry32 — a tiny seeded PRNG so the generated universe is identical
 *  in every run without pulling in a dependency. */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SECTORS = ["Technology", "Financial Services", "Healthcare", "Consumer Staples"] as const;

export function makeUniverseRow(over: Partial<ScorecardUniverseRow> = {}): ScorecardUniverseRow {
  return {
    rank: 1,
    ticker: "COST",
    company_name: "Costco Wholesale",
    sector: "Consumer Staples",
    overall_score: 62.4,
    universe_percentile: 71.3,
    sector_percentile: 58.0,
    coverage: 0.93,
    category_score: Object.fromEntries(SCORECARD_FAMILIES.map((f) => [f, 55])) as Record<ScorecardFamily, number>,
    top_positive: TOP_POSITIVE.slice(0, 1),
    top_negative: TOP_NEGATIVE.slice(0, 1),
    ...over,
  };
}

/** Three hand-picked rows tests assert on by name, then `extra` generated
 *  rows across four sectors. The hand-picked rows: COST (top), a Financials
 *  name whose EV/leverage families are masked (nulls), and a row with
 *  `overall_score` null (insufficient coverage, unranked). */
export function makeUniverse(extra = 21, over: Partial<ScorecardUniverse> = {}): ScorecardUniverse {
  const rows: ScorecardUniverseRow[] = [
    makeUniverseRow(),
    makeUniverseRow({
      rank: 2,
      ticker: "JPM",
      company_name: "JPMorgan Chase",
      sector: "Financial Services",
      overall_score: 58.7,
      universe_percentile: 64.0,
      sector_percentile: 80.0,
      coverage: 0.8,
      category_score: { ...makeUniverseRow().category_score, leverage: null, efficiency: null },
    }),
    makeUniverseRow({
      rank: 8, // positional: universe_table ranks every row, unscored names last
      ticker: "NEWCO",
      company_name: "Newly Listed Co",
      sector: "Technology",
      overall_score: null,
      universe_percentile: null,
      sector_percentile: null,
      coverage: 0.35,
      category_score: { valuation: 44, quality: null, growth: 61, profitability: null, efficiency: null, leverage: null, capital_allocation: null, earnings_quality: null },
      top_positive: [],
      top_negative: [],
    }),
  ];
  const rnd = seeded(20260831);
  for (let i = 0; i < extra; i += 1) {
    const score = Math.round((20 + rnd() * 70) * 10) / 10;
    const sector = SECTORS[i % SECTORS.length];
    rows.push(
      makeUniverseRow({
        rank: i + 3,
        ticker: `T${String(i + 1).padStart(2, "0")}`,
        company_name: `Company ${i + 1}`,
        sector,
        overall_score: score,
        universe_percentile: Math.round(score * 10) / 10,
        sector_percentile: Math.round(Math.min(100, score + (rnd() - 0.5) * 20) * 10) / 10,
        coverage: Math.round((0.6 + rnd() * 0.4) * 100) / 100,
        category_score: Object.fromEntries(SCORECARD_FAMILIES.map((f) => [f, Math.round(rnd() * 100)])) as Record<ScorecardFamily, number>,
      }),
    );
  }
  return { version_key: VERSION, as_of: AS_OF, run_id: RUN_ID, universe_size: rows.length, rows, ...over };
}

/** The href the page hands the table; matches `scorecardExportUrl` in the
 *  client once that slice lands. */
export function exportHref(format: "csv" | "json" = "csv", asOf: string = AS_OF): string {
  return `/api/scorecard/export?format=${format}&contract=${SCORECARD_EXPORT_CONTRACT}&version=${VERSION}&as_of=${asOf}`;
}

// ---------------------------------------------------------------------------
// Spec
// ---------------------------------------------------------------------------

export function makeSpec(over: Partial<ScorecardSpec> = {}): ScorecardSpec {
  return {
    version_key: VERSION,
    spec_hash: "9f1c2e7a",
    families: [
      {
        name: "valuation",
        weight: 0.125,
        features: [
          { name: "fcf_yield", sign: 1, weight: 0.25, formula: "free_cash_flow_ttm / market_cap", description: "Free cash flow yield", inputs: ["free_cash_flow", "market_cap"], applicability: { exclude_sectors: [] } },
          { name: "ebitda_ev_yield", sign: 1, weight: 0.25, formula: "ebitda_ttm / EV", description: "EBITDA / enterprise value", inputs: ["ebitda", "EV"], applicability: { exclude_sectors: ["Financial Services", "Real Estate"] } },
        ],
      },
    ],
    normalization: { winsor_pct: 0.025, sector_neutral: true, min_sector_n: 5, clip_z: 3 },
    ...over,
  };
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

export function makeQuintileResult(over: Partial<QuintileLSResult> = {}): QuintileLSResult {
  const months = [];
  for (let i = 0; i < 30; i += 1) {
    const idx = 2 + i;
    const year = 2024 + Math.floor((idx - 1) / 12);
    const month = ((idx - 1) % 12) + 1;
    const spread = Math.round((0.004 + Math.sin(i) * 0.02) * 10000) / 10000;
    months.push({ as_of: monthEnd(year, month), long_ret: 0.012 + spread / 2, short_ret: 0.012 - spread / 2, spread, n_long: 31, n_short: 31, universe_ew: 0.011 });
  }
  return {
    months,
    skipped_months: [{ as_of: "2024-01-31", reason: "long leg had 12 names (minimum 15)" }],
    mean_spread: 0.0041,
    stdev: 0.0142,
    sharpe_annualized: 1.0,
    t_stat: 1.58,
    hit_rate: 0.6,
    max_drawdown: -0.061,
    n_months: 30,
    quintile_table: [
      { q: 1, mean_ret: 0.0071, n: 31 },
      { q: 2, mean_ret: 0.0092, n: 31 },
      { q: 3, mean_ret: 0.0105, n: 31 },
      { q: 4, mean_ret: 0.0118, n: 31 },
      { q: 5, mean_ret: 0.0134, n: 31 },
    ],
    monotonic: true,
    caveats: ALL_CAVEATS,
    ...over,
  };
}

export function makeFF6Result(over: Partial<FF6RegressionResult> = {}): FF6RegressionResult {
  return {
    series: "spread",
    alpha_monthly: 0.0028,
    alpha_annualized: 0.0341,
    alpha_t: 2.14,
    betas: { MKT_RF: -0.05, SMB: 0.12, HML: 0.31, RMW: 0.44, CMA: 0.08, MOM: -0.02 },
    beta_t: { MKT_RF: -0.6, SMB: 0.9, HML: 2.3, RMW: 3.1, CMA: 0.5, MOM: -0.2 },
    r_squared: 0.37,
    n_months: 30,
    start: "2024-02-29",
    end: "2026-07-31",
    insufficient: false,
    caveats: [CAVEATS.unadjusted, CAVEATS.constituents],
    ...over,
  };
}

export function makeLassoResult(over: Partial<DoubleLassoResult> = {}): DoubleLassoResult {
  return {
    coef_d: 0.0031,
    se_hc0: 0.0011,
    se_cluster_month: 0.0014,
    t_stat: 2.21,
    p_value: 0.027,
    selected_y: ["log_mktcap", "book_to_market", "momentum_12_1"],
    selected_d: ["roa", "book_to_market"],
    lambda_y: 0.081,
    lambda_d: 0.077,
    n_obs: 4680,
    n_months: 30,
    p_controls: 14,
    naive_coef: 0.0047,
    full_ols_coef: 0.0029,
    verdict: "independent",
    interpretation: "The overall z carries next-month return information not explained by size, value, momentum, reversal, beta, ROA, asset growth, leverage or sector.",
    caveats: ALL_CAVEATS,
    ...over,
  };
}

export function makeInsufficientLasso(over: Partial<DoubleLassoResult> = {}): DoubleLassoResult {
  return makeLassoResult({
    coef_d: null,
    se_hc0: null,
    se_cluster_month: null,
    t_stat: null,
    p_value: null,
    selected_y: [],
    selected_d: [],
    lambda_y: null,
    lambda_d: null,
    n_obs: 1360,
    n_months: 8,
    naive_coef: null,
    full_ols_coef: null,
    verdict: "insufficient_data",
    interpretation: "Fewer than 24 month-ends (8) and fewer than 2,000 observations (1,360): no verdict.",
    ...over,
  });
}

export function makeEvaluation(over: Partial<ScorecardEvaluationResponse> = {}): ScorecardEvaluationResponse {
  const base = { created_at: "2026-09-01T04:05:00Z", sample_start: "2024-02-29", sample_end: "2026-07-31", params: { min_leg: 15, min_coverage: 0.6 } };
  const evaluations: ScorecardEvaluation[] = [
    { ...base, kind: "quintile_ls", n_obs: 30, result: makeQuintileResult() },
    { ...base, kind: "ff6_regression", n_obs: 30, result: makeFF6Result() },
    { ...base, kind: "double_lasso", n_obs: 4680, result: makeLassoResult() },
  ];
  return { version_key: VERSION, evaluations, ...over };
}

/** Too little history: every kind reports its insufficient state. */
export function makeInsufficientEvaluation(): ScorecardEvaluationResponse {
  const base = { created_at: "2026-09-01T04:05:00Z", sample_start: "2025-12-31", sample_end: "2026-07-31", params: { min_leg: 15, min_coverage: 0.6 } };
  return {
    version_key: VERSION,
    evaluations: [
      {
        ...base,
        kind: "quintile_ls",
        n_obs: 8,
        result: makeQuintileResult({
          months: makeQuintileResult().months.slice(0, 8),
          n_months: 8,
          sharpe_annualized: null,
          t_stat: null,
          max_drawdown: null,
          monotonic: null,
          quintile_table: [],
          skipped_months: [{ as_of: "2025-11-30", reason: "short leg had 9 names (minimum 15)" }],
        }),
      },
      { ...base, kind: "ff6_regression", n_obs: 8, result: makeFF6Result({ alpha_monthly: null, alpha_annualized: null, alpha_t: null, betas: {}, beta_t: {}, r_squared: null, n_months: 8, insufficient: true }) },
      { ...base, kind: "double_lasso", n_obs: 1360, result: makeInsufficientLasso() },
    ],
  };
}
