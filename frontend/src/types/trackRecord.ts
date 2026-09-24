// W6 / FIX-007 Track Record — TypeScript mirror of the payload
// `GET /api/admin/track-record` serves (backend
// `app/services/outcome_service.py::track_record`). Mirrored by hand; the
// captured `test/fixtures/trackRecord.wire.json` is the producer's own
// output and `backend/app/tests/test_track_record_fixture_contract.py`
// fails when it drifts, so re-capture rather than editing either side.
//
// Every figure is computed over ELIGIBLE outcomes only. Excluded rows are
// kept in the database and counted by reason in `eligibility`, never
// deleted. `thesis_hit_rate` is an absolute-return measure; the page shows
// it beside SPY-relative `alpha` and the always-Bullish `base_rate`.

/** Who produced the rating label (derived from the stored memo). */
export type RatingSource = "llm_pm" | "keyword_pm" | "fallback_pm" | "patch" | "unknown";

export interface TrackRecordAlpha {
  /** Rows with an SPY-relative alpha at this horizon. */
  n: number;
  /** Rows whose benchmark lacked the exact sessions. */
  unavailable: number;
  mean: number | null;
  median: number | null;
  /** Directional calls with alpha; alpha is signed by the call's direction. */
  directional_n: number;
  directional_median: number | null;
  /** Share of directional calls whose direction-adjusted alpha was > 0. */
  beat_benchmark_rate: number | null;
  /** Median over companies of each company's mean adjusted alpha. */
  company_weighted_median: number | null;
}

export interface TrackRecordBaseRate {
  /** What labelling every directional call Bullish would have scored. */
  always_bullish_hit_rate: number | null;
  always_bullish_n: number;
  positive_alpha_rate: number | null;
  positive_alpha_n: number;
}

export interface TrackRecordCoverageRow {
  horizon_days: number;
  memos: number;
  companies: number;
  directional: number;
  /** companies / universe_companies; null when the universe is empty. */
  universe_pct: number | null;
  late_evaluation_candidates: number;
}

export interface TrackRecordCoverage {
  universe_companies: number;
  memos_any_horizon: number;
  companies_any_horizon: number;
  /** FIX-007 triage flag at the selected horizon (not proof of a bad row). */
  late_evaluation_candidates: number;
  horizons: TrackRecordCoverageRow[];
}

export interface TrackRecordEligibility {
  rule_version: number;
  eligible: number;
  excluded: number;
  unclassified: number;
  excluded_by_reason: Record<string, number>;
  eligible_by_reason: Record<string, number>;
}

export type ProvisionalReason = "companies_below_threshold" | "directional_below_threshold";

export interface TrackRecordProvisional {
  is_provisional: boolean;
  reasons: ProvisionalReason[];
  min_companies: number;
  min_directional: number;
  companies: number;
  directional: number;
}

export interface TrackRecordOut {
  horizon_days: number;
  total: number;
  directional_evaluations: number;
  thesis_hit_rate: number | null;
  avg_forward_return: number;
  avg_alpha: number | null;
  ticker_filter: string | null;
  sector_filter: string | null;
  benchmark: string;
  alpha: TrackRecordAlpha;
  base_rate: TrackRecordBaseRate;
  rating_mix: Record<string, number>;
  rating_mix_by_source: Partial<Record<RatingSource | string, Record<string, number>>>;
  coverage: TrackRecordCoverage;
  eligibility: TrackRecordEligibility;
  provisional: TrackRecordProvisional;
}

/** Plain-language names for exclusion reasons; an unknown code falls back
 *  to itself so a new server reason is visible rather than hidden. */
export const EXCLUSION_REASON_LABELS: Record<string, string> = {
  demo_dev_copy_2026_05_04: "demo-mode memos copied from a development machine on 2026-05-04",
  test_fixture_migrated_2026_05_04: "test-fixture memos migrated from a development machine on 2026-05-04",
  no_llm_or_demo_generation: "memos generated without LLM analysis or on demo data",
  generation_mode_unrecorded: "memos with no recorded generation mode",
  generation_mode_unrecognized: "memos with an unrecognized generation mode",
  backtest: "backtest reproductions",
  patch_parent_missing: "news patches whose original memo is missing",
};

export const RATING_SOURCE_LABELS: Record<string, string> = {
  llm_pm: "LLM PM",
  keyword_pm: "deterministic keyword PM",
  fallback_pm: "PM fallback",
  patch: "news patch",
  unknown: "unknown",
};
