// FEAT-003 Industry Analysis fixtures.
//
// `industry.wire.json` is NOT hand-written: it is what
// `/api/industries/*` actually served after the real weekly path
// (classify → enqueue_period → drain) ran against the demo universe, and
// `backend/app/tests/test_industry_ui_fixture_contract.py` re-validates
// every response in it against the pydantic models and compares field
// sets, so a rename on either side fails a test instead of drifting.
//
// What was edited after capture, and nothing else: per-ticker
// `weekly_closes` arrays were emptied to keep the file readable, and each
// row records how many points it lost in `weekly_closes_dropped`
// (`meta.trimmed` carries the total). The UI never reads them.
//
// The captured group (4530 Semiconductors & Semiconductor Equipment) is
// deliberately mixed: five classified members, four of them priced, one
// assigned from the research map with an 8-digit sub-industry and four
// derived from the provider's industry label — so the table's provenance
// column, its unpriced row and its "membership is not coverage" caption
// all have real data behind them. Edition v1 is the honest
// `insufficient_sample` week (no price history yet), v2 the priced one.
//
// The capture also carries all three sample-floor states, because the
// page has to word them differently and a hand-built stand-in would
// drift from the producer: `report` is above the floor, `warmingUpReport`
// is the same group's first week (enough members, no prices yet — the
// warm-up clears it) and `universeShortReport` is a group whose
// membership is itself below the floor, which no warm-up can fix.
//
// Owner decision 1 (template editions are never displayed) adds three
// captured states, all produced by the real worker path: `report` carries
// a section the (stubbed) analyst model did not write, which the server
// hides; `notUpdatedReport` is the thin group after a week in which the
// model was down — its last analyst edition with `display.not_updated`;
// and `noAgenticDetail` is the 404 for a group whose only edition is an
// audit-only template. The "analyst" is the deterministic stand-in the
// capture declares in `meta.analyst_stub` — its prose is the writer's own
// template text relabelled, never invented.
import wire from "./industry.wire.json";
import type {
  IndustryAccess,
  IndustryChanges,
  IndustryCompanies,
  IndustryHistory,
  IndustryReport,
  IndustrySampleFloor,
  IndustryTaxonomy,
  NoReportDetail,
  TaxonomyNotImportedDetail,
} from "@/types/industries";

interface Wire {
  meta: {
    generated_by: string;
    taxonomy_version: string;
    code: string;
    short_code: string;
    empty_code: string;
    no_agentic_code: string;
    outage_period: string;
    read_at: Record<string, string>;
    analyst_stub: { enabled: boolean; label: string; model: string; reason: string };
    min_sample: number;
    trimmed: Record<string, number>;
    trimmed_note: string;
    omitted_responses: string[];
    omitted_reason: string;
  };
  taxonomy: IndustryTaxonomy;
  report: IndustryReport;
  report_warming_up: IndustryReport;
  report_universe_short: IndustryReport;
  report_not_updated: IndustryReport;
  report_no_agentic: { detail: NoReportDetail };
  companies: IndustryCompanies;
  companies_warming_up: IndustryCompanies;
  history: IndustryHistory;
  changes: IndustryChanges;
  report_missing: { detail: NoReportDetail };
}

const w = wire as unknown as Wire;

export const WIRE_META = w.meta;
/** The captured group's code — never spelled out in a test, so a
 *  re-capture against a different universe does not need a sweep. */
export const CODE = w.meta.code;
/** A real group in the same taxonomy that has no edition on file. */
export const EMPTY_CODE = w.meta.empty_code;
/** A real group this universe holds too few constituents to ever cover —
 *  below the sample floor on MEMBERSHIP, so no warm-up will fix it. */
export const SHORT_CODE = w.meta.short_code;
/** The deployment's sample floor at capture time. */
export const MIN_SAMPLE = w.meta.min_sample;
export const TAXONOMY_VERSION = w.meta.taxonomy_version;

export const taxonomy: IndustryTaxonomy = w.taxonomy;
export const report: IndustryReport = w.report;
/** The captured group's FIRST week: enough members, no prices yet. The
 *  transient way to sit below the sample floor — the weekly warm-up
 *  clears it, and the page has to say so in those words. */
export const warmingUpReport: IndustryReport = w.report_warming_up;
/** A thin group's priced week: every member priced and STILL below the
 *  floor. The structural way to sit below it, which no warm-up fixes. */
export const universeShortReport: IndustryReport = w.report_universe_short;
/** The thin group after a week whose only product was an audit-only
 *  template (the model was down): the page shows its LAST analyst edition,
 *  with `display.not_updated` naming the week that produced none. */
export const notUpdatedReport: IndustryReport = w.report_not_updated;
/** `404 no_report` for a group whose only edition is audit-only:
 *  `reason: no_validated_analyst_edition`, `withheld_editions` counted. */
export const noAgenticDetail: NoReportDetail = w.report_no_agentic.detail;
/** The week the capture ran with the model down. */
export const OUTAGE_PERIOD = w.meta.outage_period;
export const companies: IndustryCompanies = w.companies;
/** `/companies` as it answered BEFORE any price existed — the SAME
 *  endpoint, read in the earlier week. `/companies` prices its rows from
 *  the latest statistics row, so this body counts 0 priced while the
 *  published edition counts its own. Served together they are the race
 *  the page has to survive; each body is internally consistent, because
 *  both are what the API said. */
export const companiesWarmingUp: IndustryCompanies = w.companies_warming_up;
export const history: IndustryHistory = w.history;
export const changes: IndustryChanges = w.changes;
/** The `404 no_report` body, as FastAPI serialises it. */
export const noReportDetail: NoReportDetail = w.report_missing.detail;

/** The three captured sample-floor blocks — met, warm-up short and
 *  structurally short — as the API served them. */
export const INDUSTRY_FLOOR_SAMPLES: IndustrySampleFloor[] = [report, warmingUpReport, universeShortReport].map(
  (r) => (r.coverage as { sample_floor: IndustrySampleFloor }).sample_floor,
);

/** A deep clone, so a test that reshapes a fixture cannot leak into the
 *  next one through the shared import. */
export function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/** The first group in the picker, and the first with a published edition
 *  — derived rather than named, for the same reason as `CODE`. */
export const firstGroup = taxonomy.sectors[0].industry_groups[0];
export const groupWithEdition =
  taxonomy.sectors.flatMap((s) => s.industry_groups).find((g) => g.latest_report !== null) ?? firstGroup;
export const allGroups = taxonomy.sectors.flatMap((s) => s.industry_groups);

// ---------------------------------------------------------------------------
// Derived states. Each one changes ONLY the fields the backend would have
// changed, so a test asserting on a badge is asserting on a shape the API
// can actually produce.
// ---------------------------------------------------------------------------

/** The same edition after a refresh failed: content unchanged, `stale`
 *  set with the store's reason, `last_attempt` naming the failure. */
export function staleReport(over: Partial<IndustryReport> = {}): IndustryReport {
  const r = clone(report);
  r.stale = true;
  r.stale_reason = "a later refresh attempt failed (2026-W37)";
  r.last_attempt = {
    job_id: 41,
    status: "failed",
    outcome: "failed",
    at: "2026-09-13T06:41:00",
    period_key: "2026-W37",
    attempts: 3,
    max_attempts: 3,
    error_type: "ReportRejected",
    // The public form: the server replaces a rejection's raw message (it
    // quotes the rejected model prose) with a count of problems.
    error_message: "the analyst draft did not pass validation (1 problem)",
    report_id: null,
    source: "weekly_cron",
  };
  return { ...r, ...over };
}

/** An edition whose analyst layer never ran. The backend labels this on
 *  the edition itself; nothing about it is invented here. */
export function degradedReport(): IndustryReport {
  const r = clone(report);
  r.degraded = [
    "performance:insufficient_sample",
    "analyst_narrative:llm_unavailable",
    "cross_industry:snapshot:none_yet",
  ];
  return r;
}

/** The v1 edition, exactly as `?version=1` served it. Captured rather
 *  than folded together from the history row and the v2 payload, because
 *  the two editions differ in more than their metadata — v1 is the week
 *  with no prices, and its coverage block says so. */
export function priorEdition(): IndustryReport {
  return clone(warmingUpReport);
}

/**
 * The two reads whose priced counts disagree, for the page-level race
 * test: the published edition and `/companies` as it answered in the
 * earlier week.
 *
 * The disagreement lives BETWEEN the reads, which is where the real one
 * lives — `/companies` prices from the latest statistics row, and an
 * edition on screen may have counted a different week's. It is never
 * grafted INTO one of them: an edition whose companies facts contradict
 * its own coverage block and its own per-ticker rows is a body no
 * producer can emit, and a page proved against one is proved against
 * nothing.
 */
export function racedReads(): { report: IndustryReport; companies: IndustryCompanies } {
  return { report: clone(report), companies: clone(companiesWarmingUp) };
}

/** The access block a deployment with the login wall ON serves: the tiers
 *  are unchanged (they are the policy), `enforced` flips to true. */
export function enforcedAccess(over: Partial<IndustryAccess> = {}): IndustryAccess {
  return { ...clone(report.access), enforced: true, ...over };
}

/** A taxonomy response from a deployment that gates the reports behind
 *  Pro — the one configuration in which the picker renders but the
 *  report must not be fetched. */
export function gatedTaxonomy(tier: "pro" = "pro"): IndustryTaxonomy {
  const t = clone(taxonomy);
  t.access = {
    ...t.access,
    enforced: true,
    tier,
    required_tier: tier,
    surfaces: { ...t.access.surfaces, latest: tier },
  };
  return t;
}

/** `503 taxonomy_not_imported` — a deployment state, with the operator's
 *  remedy and the access policy still attached. */
export const notImportedDetail: TaxonomyNotImportedDetail = {
  code: "taxonomy_not_imported",
  message: "the GICS taxonomy has not been imported on this deployment yet",
  remedy: "POST /api/admin/industries/taxonomy/import (or python -m app.scripts.import_gics_taxonomy --activate)",
  access: { ...clone(taxonomy.access), allowed: false },
};
