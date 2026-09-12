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
import wire from "./industry.wire.json";
import type {
  IndustryAccess,
  IndustryChanges,
  IndustryCompanies,
  IndustryHistory,
  IndustryReport,
  IndustryTaxonomy,
  NoReportDetail,
  TaxonomyNotImportedDetail,
} from "@/types/industries";

interface Wire {
  meta: {
    generated_by: string;
    taxonomy_version: string;
    code: string;
    empty_code: string;
    trimmed: Record<string, number>;
    trimmed_note: string;
    omitted_responses: string[];
    omitted_reason: string;
  };
  taxonomy: IndustryTaxonomy;
  report: IndustryReport;
  companies: IndustryCompanies;
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
export const TAXONOMY_VERSION = w.meta.taxonomy_version;

export const taxonomy: IndustryTaxonomy = w.taxonomy;
export const report: IndustryReport = w.report;
export const companies: IndustryCompanies = w.companies;
export const history: IndustryHistory = w.history;
export const changes: IndustryChanges = w.changes;
/** The `404 no_report` body, as FastAPI serialises it. */
export const noReportDetail: NoReportDetail = w.report_missing.detail;

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
    at: "2026-09-13T06:41:00",
    period_key: "2026-W37",
    attempts: 3,
    max_attempts: 3,
    error_type: "ReportRejected",
    error_message: "drivers section opened with a KPI forecast",
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

/** The v1 edition as `?version=1` would serve it — the history row's own
 *  metadata folded onto the captured payload. */
export function priorEdition(): IndustryReport {
  const older = history.items.find((i) => !i.is_latest_good) ?? history.items[history.items.length - 1];
  const r = clone(report);
  r.version = older.version;
  r.period_key = older.period_key;
  r.as_of = older.as_of;
  r.status = older.status;
  r.is_latest_good = older.is_latest_good;
  r.degraded = [...older.degraded];
  r.generated_at = older.generated_at;
  return r;
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
