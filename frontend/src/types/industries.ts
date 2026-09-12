// FEAT-003 Industry Analysis — the TypeScript mirror of
// `backend/app/schemas/industry.py` as `/api/industries/*` serves it.
// Mirrored by hand, field names exactly as pydantic serialises them;
// `backend/app/tests/test_industry_ui_fixture_contract.py` validates the
// checked-in wire fixture against those models and compares field sets,
// so a rename on either side fails a test rather than drifting silently.
//
// Three research rules travel with these shapes all the way to the pixel:
//
//   * **Observed data and analyst interpretation never merge.** A report
//     section is `{facts, interpretation}` and the UI prints them under
//     two headings that say which is which. `facts` are server-computed;
//     `interpretation` is a model (or template) reading of them.
//   * **A missing value carries its reason.** Absent numbers arrive as
//     `null` beside a `reason` / `unpriced_reason` / `stale_reason`, and
//     the UI renders "n/a (reason)" — never 0, never a bare dash.
//   * **Membership is not coverage.** `count` is who is classified into
//     the group; `n_priced` is how many of them the statistics row could
//     price. They are different numbers and are rendered as such.
//
// Forward-looking lines in a report are scenarios, not recommendations.

/** Surfaces the API tiers separately (`entitlements_industry.SURFACES`). */
export const INDUSTRY_SURFACES = ["latest", "history", "changes", "pm_chat"] as const;
export type IndustrySurface = (typeof INDUSTRY_SURFACES)[number];

/**
 * What a surface costs and whether that is being enforced right now.
 *
 * Three separate questions, and the UI must not collapse them: `tier` is
 * the policy, `enforced` is whether `AUTH_ENABLED` makes it bite, and
 * `route_gated` is whether the route that returned this block applied it
 * (`/taxonomy` always answers, so it reports the `latest` tier with
 * `route_gated: false` — a price list, not a receipt).
 */
export interface IndustryAccess {
  surface: string;
  tier: "public" | "pro";
  required_tier: string | null;
  allowed: boolean;
  enforced: boolean;
  route_gated: boolean;
  setting: string;
  /** Tier per surface — the only place a signed-out visitor learns that
   *  history and changes are Pro. */
  surfaces: Record<string, string>;
  plan: string | null;
  note: string;
}

// ---------------------------------------------------------------------------
// Taxonomy
// ---------------------------------------------------------------------------

export interface TaxonomyVersion {
  key: string;
  checksum: string;
  source: string;
  is_active: boolean;
  effective_from: string | null;
  effective_to: string | null;
  node_counts: Record<string, number>;
  provenance: Record<string, unknown>;
  attribution: string;
  display_mode: string;
  mapping_caveat: string;
  imported_at: string | null;
  activated_at: string | null;
}

/** The picker's pointer at a group's published edition. `stale_by_age` is
 *  arithmetic on `as_of` alone — a failed refresh is the other half of
 *  staleness and only the report endpoint answers for it, which is what
 *  `stale_basis` says. */
export interface LatestReportPointer {
  version: number;
  period_key: string;
  as_of: string | null;
  status: string;
  generated_at: string | null;
  degraded: boolean;
  degraded_reasons: string[];
  stale_by_age: boolean;
  age_days: number | null;
  stale_basis: string;
}

/**
 * Whether this universe can EVER put the sample floor's worth of priced
 * names in the group.
 *
 * Structural only — the taxonomy response knows the classified
 * membership, not which of it had a price series in any given week. A
 * group with `coverable: false` is short of CLASSIFIED CONSTITUENTS: its
 * statistics will read `insufficient_sample` every week until more
 * companies are classified into it, and the page must not word that like
 * a warm-up in progress. Nor like a universe short of companies —
 * `uncounted_in_universe` counts the companies already here that no group
 * counts, and classifying those closes the same gap. The per-week half of
 * the story is `IndustrySampleFloor`.
 */
export interface GroupUniverseCoverage {
  min_sample: number;
  constituent_count: number;
  coverable: boolean;
  constituents_short_by: number;
  /** Uncounted rows that already name this group (a `stale` row does). */
  uncounted_for_group: number;
  /** Uncounted rows anywhere in this universe — the pool a
   *  re-classification could draw the shortfall from. */
  uncounted_in_universe: number;
  /** The server's own sentence. The page prints it; it never writes one. */
  explanation: string;
}

export interface IndustryGroupNode {
  code: string;
  name: string;
  sector_code: string;
  is_active: boolean;
  effective_from: string | null;
  effective_to: string | null;
  industry_count: number;
  sub_industry_count: number;
  constituent_count: number;
  universe_coverage: GroupUniverseCoverage;
  latest_report: LatestReportPointer | null;
}

export interface SectorNode {
  code: string;
  name: string;
  industry_groups: IndustryGroupNode[];
}

/** The taxonomy-level structural count: `not_coverable` of `groups` hold
 *  fewer classified constituents than `min_sample`, so no number of
 *  warm-up weeks can produce statistics for them. */
export interface UniverseCoverage {
  min_sample: number;
  setting: string;
  groups: number;
  coverable: number;
  not_coverable: number;
  not_coverable_codes: string[];
  /** Classified CONSTITUENTS, not companies — see `uncounted`. */
  constituents_needed: number;
  /** `industry_classification.uncounted_rows()`: the companies in this
   *  universe no group counts, by state and (where the row names one) by
   *  group. How much of `constituents_needed` re-classification could
   *  supply, rather than new companies. */
  uncounted: Record<string, unknown>;
  /** The sentence the index page prints verbatim. The page must not
   *  compose a remedy the server did not state. */
  explanation: string;
  basis: string;
}

export interface IndustryTaxonomy {
  taxonomy_version: TaxonomyVersion;
  sectors: SectorNode[];
  /** Counts by level, straight from the registry. The UI never hardcodes
   *  a number of sectors, groups, industries or sub-industries. */
  node_counts: Record<string, number>;
  constituents: Record<string, unknown>;
  /** How much of this taxonomy the current universe can never report on,
   *  in one place. Counted from the classification table on every call —
   *  the page never derives it. */
  universe_coverage: UniverseCoverage;
  reports: Record<string, number>;
  access: IndustryAccess;
  attribution: string;
  mapping_caveat: string;
  disclaimer: string;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/** The thirteen sections, in the writer's frozen order
 *  (`agents/industry_report_validator.SECTION_ORDER`). The page renders
 *  `payload.section_order` when the edition carries one and falls back to
 *  this, so a future section appears without a deploy. */
export const INDUSTRY_SECTIONS = [
  "overview",
  "drivers",
  "kpis",
  "performance",
  "companies",
  "statistics",
  "themes",
  "cross_industry",
  "outlook",
  "risks",
  "what_changed",
  "sources",
  "metadata",
] as const;
export type IndustrySection = (typeof INDUSTRY_SECTIONS)[number];

/** Sections that carry an analyst layer at all; the rest are facts-only
 *  and their `interpretation` is `null` by construction, not by failure. */
export const INDUSTRY_FACTS_ONLY_SECTIONS: readonly string[] = ["statistics", "sources", "metadata"];

export const INDUSTRY_SECTION_LABELS: Record<IndustrySection, string> = {
  overview: "Overview",
  drivers: "Drivers",
  kpis: "KPIs",
  performance: "Performance",
  companies: "Companies",
  statistics: "Statistics",
  themes: "Themes",
  cross_industry: "Cross-industry",
  outlook: "Outlook",
  risks: "Risks",
  what_changed: "What changed",
  sources: "Sources",
  metadata: "Metadata",
};

/** One claim the analyst layer makes, with the evidence it rests on and
 *  the observation that would break it. A `causal_inference` without a
 *  usable `falsifier` never reaches the client — the validator rejects
 *  the edition — so the UI prints what it is given. */
export interface IndustryClaim {
  type: "observed_fact" | "causal_inference" | "forecast_assumption" | string;
  text: string;
  basis: string[];
  falsifier: string;
}

export interface IndustryInterpretation {
  text: string;
  claims?: IndustryClaim[];
  /** The eight-stage causal order, when the section is ordered by it. */
  stages?: Array<{ id: string; text: string }>;
  /** Scenarios — labelled scenarios, never recommendations. */
  scenarios?: Record<string, { text: string; falsifiers?: string[] }>;
}

export interface IndustryReportSection {
  facts: Record<string, unknown>;
  interpretation: IndustryInterpretation | null;
}

export interface IndustryReportPayload {
  report_schema_version?: number;
  section_order?: string[];
  sections: Record<string, IndustryReportSection>;
  /** `llm` | `deterministic` | `llm_unavailable` for the edition overall. */
  analyst_narrative?: string;
  /** Per section, who wrote the interpretation — the edition-level mode
   *  cannot say "llm" for a run whose call budget reached only some. */
  narrative_by_section?: Record<string, string>;
  disclaimer?: string;
  attribution?: string;
  mapping_caveat?: string;
}

/**
 * Where one edition stands against the sample floor — on
 * `report.coverage.sample_floor` and `stats.sample.sample_floor`.
 *
 * Below the floor is two situations, not one, and the page must not print
 * the same words for both:
 *
 *   * `prices_not_warmed` — enough members, too few priced THIS week. The
 *     weekly warm-up fetches more series each run, so it clears on its
 *     own;
 *   * `universe_too_small` — fewer members than the floor. Price every
 *     one and the group is still short; only a wider universe fixes it.
 *
 * `explanation` is the server's sentence and is what the page prints. The
 * state is a key for choosing a badge, never text for a reader.
 */
export const INDUSTRY_SAMPLE_FLOOR_STATES = ["met", "prices_not_warmed", "universe_too_small"] as const;
export type IndustrySampleFloorState = (typeof INDUSTRY_SAMPLE_FLOOR_STATES)[number];

export interface IndustrySampleFloor {
  state: IndustrySampleFloorState;
  structural: boolean;
  clears_with_warm_up: boolean;
  /** The API always sends these; `null` is what the client's own reader
   *  substitutes for a count an older edition did not carry, so nothing
   *  downstream can mistake a missing count for a zero. */
  min_sample: number | null;
  n_constituents: number | null;
  n_with_prices: number | null;
  priced_short_by: number | null;
  constituents_short_by: number | null;
  explanation: string;
}

/** The refresh attempt behind a `stale` verdict. */
export interface IndustryLastAttempt {
  job_id: number;
  status: string;
  at: string | null;
  period_key: string;
  attempts: number;
  max_attempts: number;
  error_type: string;
  error_message: string;
  report_id: number | null;
  source: string;
}

/** The observed layer, with the `method` that produced it. The two travel
 *  together: a benchmark-relative return without its cohort basis, or
 *  breadth without its session window, is a number a reader cannot check. */
export interface IndustryStats {
  id: number;
  period_key: string;
  as_of: string | null;
  method: Record<string, unknown>;
  sample: Record<string, unknown>;
  payload: Record<string, unknown>;
  inputs_hash: string;
  computed_at: string | null;
  per_ticker_note: string;
}

export interface IndustryReport {
  code: string;
  name: string;
  sector_code: string;
  taxonomy_version: string;
  version: number;
  parent_report_id: number | null;
  period_key: string;
  as_of: string | null;
  status: string;
  is_latest_good: boolean;
  report_schema_version: number;
  stale: boolean | null;
  stale_reason: string | null;
  last_attempt: IndustryLastAttempt | null;
  payload: IndustryReportPayload;
  stats: IndustryStats | null;
  stats_unavailable_reason: string | null;
  coverage: Record<string, unknown>;
  freshness: Record<string, unknown>;
  generation: Record<string, unknown>;
  /** Labelled degradations, e.g. `analyst_narrative:llm_unavailable`. */
  degraded: string[];
  errors: unknown[];
  sources: Array<Record<string, unknown>>;
  llm_cost_usd: number | null;
  generated_at: string | null;
  access: IndustryAccess;
  disclaimer: string;
  attribution: string;
  mapping_caveat: string;
}

export interface IndustryHistoryItem {
  version: number;
  period_key: string;
  as_of: string | null;
  status: string;
  is_latest_good: boolean;
  degraded: string[];
  errors: unknown[];
  stats_id: number | null;
  llm_cost_usd: number | null;
  generation: Record<string, unknown>;
  generated_at: string | null;
}

export interface IndustryHistory {
  code: string;
  name: string;
  taxonomy_version: string;
  count: number;
  limit: number;
  /** How many editions the cap dropped. A truncated list says so. */
  truncated: number;
  items: IndustryHistoryItem[];
  last_attempt: IndustryLastAttempt | null;
  access: IndustryAccess;
  disclaimer: string;
}

/** One fact, both sides. `delta` is `null` with a `reason` whenever either
 *  side is missing — a missing fact is never differenced to zero. */
export interface IndustryFactDelta {
  from: unknown;
  to: unknown;
  delta: number | null;
  reason: string | null;
}

export interface IndustryChanges {
  code: string;
  name: string;
  taxonomy_version: string;
  from: Record<string, unknown>;
  to: Record<string, unknown>;
  adjacent: boolean;
  facts_delta: Record<string, IndustryFactDelta>;
  constituents: Record<string, unknown>;
  leaders_laggards: Record<string, unknown>;
  analyst_view: Record<string, unknown>;
  access: IndustryAccess;
  disclaimer: string;
}

// ---------------------------------------------------------------------------
// Companies
// ---------------------------------------------------------------------------

/** Where a row's group assignment came from. `source_label` is the string
 *  the UI shows verbatim — it carries the research map's own caveat. */
export interface IndustryClassificationOut {
  state: string;
  source: string;
  source_label: string;
  method: string;
  author: string;
  as_of: string;
  confidence: number | null;
  classified_at: string | null;
  mapping_caveat: string;
}

export interface IndustryCompanyRow {
  ticker: string;
  company_name: string | null;
  is_active: boolean;
  industry_code: string | null;
  industry_name: string | null;
  sub_industry_code: string | null;
  sub_industry_name: string | null;
  sub_industry_codes: string[];
  classification: IndustryClassificationOut;
  market_cap: number | null;
  weight_mcw: number | null;
  last_close: number | null;
  last_date: string | null;
  price_source: string | null;
  returns: Record<string, number | null>;
  return_reasons: Record<string, string>;
  above_50d_mean: boolean | null;
  priced: boolean;
  /** Never blank on an unpriced row: a name the statistics row never saw
   *  is as unpriced as one it excluded, and both say why. */
  unpriced_reason: string | null;
}

export interface IndustryCompanies {
  code: string;
  name: string;
  sector_code: string;
  taxonomy_version: string;
  as_of: string | null;
  /** The WHOLE membership, not the page. */
  count: number;
  /** How many of the whole membership the statistics row priced. */
  n_priced: number;
  counts_basis: string;
  membership_source: string;
  membership_states: Record<string, number>;
  stats: Record<string, unknown> | null;
  stats_unavailable_reason: string | null;
  limit: number;
  truncated: number;
  items: IndustryCompanyRow[];
  excluded: Array<Record<string, unknown>>;
  access: IndustryAccess;
  attribution: string;
  mapping_caveat: string;
  /** The research map author's own label for the symbol crosswalk; shown
   *  verbatim wherever a research-map row is. */
  security_reference_caveat: string;
  disclaimer: string;
}

// ---------------------------------------------------------------------------
// Structured refusals the page renders as states rather than errors
// ---------------------------------------------------------------------------

/** `503 taxonomy_not_imported` still carries the access policy and the
 *  operator's remedy, so the page explains a deployment state instead of
 *  showing "an error occurred". */
export interface TaxonomyNotImportedDetail {
  code: "taxonomy_not_imported";
  message: string;
  remedy?: string;
  access?: IndustryAccess & { allowed: boolean };
}

/** `404 no_report` names the group and the attempt that last tried. */
export interface NoReportDetail {
  code: "no_report";
  message: string;
  industry_group_code?: string;
  name?: string;
  taxonomy_version?: string;
  last_attempt?: IndustryLastAttempt | null;
}
