// FEAT-001 Fundamentals Explorer — TypeScript mirror of the backend
// contracts (schemas under backend/app/schemas/ for the series and
// commentary routes). Mirrored by hand; keep the two in sync when either
// side changes. v1 is ANNUAL data only.
//
// Research-process rule carried by these shapes: a missing number is
// `value: null` with a machine `reason`, never 0 — the UI renders it as
// "n/a (reason)". `estimated` marks a documented fallback, so observed data
// stays separable from inference all the way to the pixel.

export type UnitType = "currency" | "percent" | "ratio" | "multiple" | "count";

export type MetricKind = "reported" | "derived" | "market";

/** Closed set of server-side reasons a point carries `value: null`. */
export type PointReason =
  | "base_nonpositive"
  | "denominator_nonpositive"
  | "no_price"
  | "no_shares"
  | "missing_line"
  | "not_backfilled";

/** Reasons the client adds while transforming (never sent by the server).
 *  `before_index_base`: the point was observed but precedes the period an
 *  indexed view rebases to, so it has no indexed value. */
export type ClientReason = "before_index_base";

export type MissingReason = PointReason | ClientReason;

export interface MetricSpec {
  id: string;
  label: string;
  family: string;
  unit_type: UnitType;
  kind: MetricKind;
  formula_text: string;
  inputs: string[];
  sign_note: string | null;
  frequency: "annual";
}

export interface CatalogResponse {
  catalog_version: string;
  metrics: MetricSpec[];
  as_of: string;
}

export interface MetricPoint {
  /** Fiscal period label, e.g. "FY2024". */
  period: string;
  /** ISO date of the fiscal period end. */
  period_end: string;
  value: number | null;
  reason?: MissingReason | null;
  estimated: boolean;
}

export interface SeriesCoverage {
  first: string | null;
  last: string | null;
  /** Periods with a value. */
  n: number;
  /** Periods the request asked for. */
  expected: number;
}

export interface SeriesProvenance {
  source: string;
  fetched_at: string | null;
  stale: boolean;
  stale_reason: string | null;
}

export interface MetricSeries {
  ticker: string;
  metric: string;
  unit_type: UnitType;
  /** Reporting currency of the company (ISO 4217). Only meaningful for
   *  currency-unit series; absent when the server does not know it. The
   *  layout engine compares currencies across companies only when both
   *  are known. */
  currency?: string | null;
  points: MetricPoint[];
  coverage: SeriesCoverage;
  provenance: SeriesProvenance;
}

export interface AppliedLimits {
  max_companies: number;
  max_metrics: number;
  /** null = the plan allows the full stored history. */
  max_years: number | null;
}

export interface SeriesLimits {
  applied: AppliedLimits;
  capped_by_plan: boolean;
}

export type UnavailableReason = "not_backfilled" | "unknown_ticker" | (string & {});

export interface UnavailableTicker {
  ticker: string;
  reason: UnavailableReason;
}

export type NormalizeMode = "none" | "indexed";

export interface SeriesRequest {
  tickers: string[];
  metrics: string[];
  years?: number;
  normalize?: NormalizeMode;
}

export interface SeriesResponse {
  catalog_version: string;
  as_of: string;
  series: MetricSeries[];
  limits: SeriesLimits;
  unavailable: UnavailableTicker[];
}

/** URL `v=` values. `auto` lets the layout engine pick; `table` shows the
 *  data table instead of a chart. */
export type ViewMode = "auto" | "dual-axis" | "small-multiples" | "indexed" | "table";

export const VIEW_MODES: readonly ViewMode[] = ["auto", "dual-axis", "small-multiples", "indexed", "table"];

export function isViewMode(v: unknown): v is ViewMode {
  return typeof v === "string" && (VIEW_MODES as readonly string[]).includes(v);
}

// ---------------------------------------------------------------------------
// Commentary
// ---------------------------------------------------------------------------

export interface CommentaryRequest {
  tickers: string[];
  metrics: string[];
  years: number;
  /** Fingerprint of the displayed series; the server recomputes and
   *  refuses (409) when they no longer match. */
  fingerprint: string;
}

export interface CommentaryRef {
  ticker: string;
  metric: string;
  period: string;
}

/** A statement grounded in the displayed series ("Observed in the data"). */
export interface ObservedItem {
  text: string;
  refs: CommentaryRef[];
}

/** A statement taken from a stored memo ("From stored memos"). */
export interface MemoViewItem {
  text: string;
  ticker: string;
  memo_version: number | null;
  memo_generated_at: string | null;
  memo_stale: boolean;
  memo_stale_reason: string | null;
}

export interface CommentaryResponse {
  commentary_id: number | null;
  fingerprint: string;
  cache_hit: boolean;
  /** True when no model output was produced (no LLM configured, breaker
   *  open, anonymous commentary disabled). Degraded output charges nothing. */
  degraded: boolean;
  degraded_reason: string | null;
  observed: ObservedItem[];
  memo_view: MemoViewItem[];
  caveats: string[];
  generated_at: string;
  model: string | null;
}
