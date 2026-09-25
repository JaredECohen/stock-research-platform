// W2b (owner decision 7, 2026-09-24): the memo's research-quality record,
// `memo.quality`, written by the backend's number check (7a), rating
// reconciliation (7b) and earned-confidence caps (7c). This module READS
// it for the renderers; it never re-derives a verdict.
//
// Everything a reader sees is a human label. The record carries machine
// vocabulary — cap codes (`critic_not_live`), memo field paths
// (`valuation_agent_view.key_points[2]`), ledger source refs
// (`industry_group:<slug>`, `filing:<accession>`) and review modes — and
// owner decision 8 keeps codes (industry codes above all) off the page. So
// every one of those goes through a closed map here, and anything the map
// does not know is shown as a generic phrase, never verbatim. The same goes
// for the backend's free-text notes: the rating check is worded here from
// the record's fields, and its `note` is never printed.
import type {
  ConfidenceAssessment,
  ConfidenceCap,
  MemoQuality,
  NumberCheck,
  NumberClaim,
  NumberClaimStatus,
  RatingReconciliation,
  StockMemoOut,
} from "@/types";
import { hiddenKeys, isHidden } from "@/lib/memoSections";

/** The quality record, or null for a memo that pre-dates it. */
export function qualityOf(memo: Pick<StockMemoOut, "quality"> | null | undefined): MemoQuality | null {
  return memo?.quality ?? null;
}

// ---------------------------------------------------------------------------
// Inline figure marks
// ---------------------------------------------------------------------------

// Statuses a renderer marks in the prose. `untraceable` and `mis_anchored`
// are the flagged ones (`number_check.FLAGGED_STATUSES`); `assumption` is a
// forward figure the PM declared, labelled so it does not read as reported.
// `weak` (a value-only match on a round number) is counted in the panel but
// not marked: a round number matching by value proves little either way,
// and marking half the memo's percentages would bury the real flags.
export const MARKED_STATUSES: ReadonlySet<NumberClaimStatus> = new Set<NumberClaimStatus>([
  "untraceable",
  "mis_anchored",
  "assumption",
]);

export const FLAGGED_STATUSES: ReadonlySet<NumberClaimStatus> = new Set<NumberClaimStatus>([
  "untraceable",
  "mis_anchored",
]);

export const CLAIM_TITLE: Partial<Record<NumberClaimStatus, string>> = {
  untraceable: "Not found in the data this memo's analysts were given",
  mis_anchored: "This value appears in the source data, but not for the metric named here",
  assumption: "PM assumption: a forward figure the PM declared, not a reported number",
};

/** Stored claims on one memo field that a renderer marks. */
export function claimsFor(
  memo: Pick<StockMemoOut, "quality"> | null | undefined,
  field: string,
): NumberClaim[] {
  const claims = qualityOf(memo)?.number_check?.claims ?? [];
  return claims.filter((c) => c.field === field && MARKED_STATUSES.has(c.status));
}

/** The declared PM assumption a stored `assumption` claim matched, or
 * null. Mirrors `number_check._is_declared`: the units must agree when the
 * declaration names one, and the values match within the claim's printed
 * precision — "18.5%" is 18.45..18.55, so a declared 18.47 is its
 * assumption. Exact equality would drop the horizon and basis of any
 * declared value the prose rounds. The closest match wins. */
export function declaredAssumptionFor(
  nc: Pick<NumberCheck, "assumptions">,
  claim: Pick<NumberClaim, "raw" | "value" | "unit">,
): Record<string, unknown> | null {
  const value = Math.abs(Number(claim.value));
  if (!Number.isFinite(value)) return null;
  const tol = printedTolerance(claim.raw, value);
  let best: Record<string, unknown> | null = null;
  let bestDist = Infinity;
  for (const a of nc.assumptions ?? []) {
    const v = a?.value;
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    const unit = typeof a.unit === "string" ? a.unit : "";
    if (unit && unit !== claim.unit) continue;
    const dist = Math.abs(Math.abs(v) - value);
    if (dist <= tol && dist < bestDist) {
      best = a;
      bestDist = dist;
    }
  }
  return best;
}

// Half a unit in the last printed place of `raw`, in the claim's (scaled)
// value units — `number_check.extract_claims`' `tol`, including its
// loosening for round integers with a scale word ("1,200 million").
function printedTolerance(raw: string, value: number): number {
  const m = /(\d[\d,]*)(?:\.(\d+))?/.exec(raw ?? "");
  if (!m) return 1e-9;
  const intDigits = m[1].replace(/,/g, "");
  const decimals = m[2]?.length ?? 0;
  const printed = Number(`${intDigits}${m[2] ? `.${m[2]}` : ""}`);
  const scale = printed > 0 ? value / printed : 1;
  const zeros = intDigits.length - intDigits.replace(/0+$/, "").length;
  const zEff = decimals === 0 && scale >= 1e3 ? Math.min(zeros, Math.max(0, intDigits.length - 2)) : 0;
  return 0.5 * 10 ** (zEff - decimals) * scale + 1e-9;
}

// ---------------------------------------------------------------------------
// Human labels
// ---------------------------------------------------------------------------

// Memo field (the path's first segment) -> where the reader finds it.
const FIELD_LABEL: Record<string, string> = {
  final_pm_view: "PM view",
  one_sentence_thesis: "Thesis",
  mispricing_thesis: "Where we differ from consensus",
  bull_case: "Bull case",
  bear_case: "Bear case",
  sector_agent_view: "Sector analyst",
  earnings_agent_view: "Earnings analyst",
  filing_agent_view: "Filing analyst",
  valuation_agent_view: "Valuation analyst",
  comps_agent_view: "Comps analyst",
  macro_sensitivity: "Macro analyst",
  technical_agent_view: "Technical analyst",
  earnings_qoq_delta: "Quarter-over-quarter earnings",
  catalysts: "Catalysts",
  key_risks: "Key risks",
  thesis_breakers: "Thesis breakers",
  dcf_pm_adjustment_headline: "PM DCF adjustment",
  dcf_pm_adjustments: "PM DCF adjustment",
  scorecard: "Scorecard reconciliation",
  quality: "PM's valuation reason",
};

// `extra_agent_views.<roster key>` — routed analysts with no field of their own.
const EXTRA_VIEW_LABEL: Record<string, string> = {
  industry_group: "Industry group analyst",
};

/** "Valuation analyst" for `valuation_agent_view.key_points[2]`. */
export function fieldLabel(path: string): string {
  const head = path.split(/[.[]/, 1)[0];
  if (head === "extra_agent_views") {
    const key = path.split(".")[1]?.split("[")[0] ?? "";
    return EXTRA_VIEW_LABEL[key] ?? "Additional analyst";
  }
  return FIELD_LABEL[head] ?? "Memo text";
}

// Ledger source-ref prefix (`backend/app/agents/source_ledger.py` call
// sites: `register_source(kind, "<prefix>:<id>", ...)`) -> a label. The id
// after the colon is never shown: it is a ticker, an accession number, a
// chunk id or an industry slug, and none of those tells a reader more than
// the label does.
const SOURCE_LABEL: Record<string, string> = {
  financials: "Financial statements",
  risk_inputs: "Financial statements",
  filing: "Company filings",
  chunk: "Company filings",
  transcript: "Earnings-call transcript",
  "dcf:initial": "DCF model (consensus-anchored)",
  "dcf:pm_adjusted": "DCF model (PM-adjusted)",
  "dcf:adjustments": "PM DCF adjustments",
  dcf_summary: "DCF model",
  comps: "Peer comparables",
  sector: "Sector research",
  sector_context: "Sector research",
  macro: "Macro data",
  news_alerts: "News alerts",
  technical: "Price technicals",
  price: "Market price",
  estimates: "Consensus estimates",
  scorecard: "Fundamental scorecard",
  scorecard_block: "Fundamental scorecard",
  factor_scores: "Quant factor scores",
  catalysts: "Catalyst calendar",
  industry_group: "Industry group analysis",
  industry_snapshot: "Weekly industry snapshot",
  notes: "Research notes",
  prior_extraction: "Prior-quarter earnings extraction",
};

export function sourceLabel(ref: string): string {
  const exact = SOURCE_LABEL[ref];
  if (exact) return exact;
  const prefix = ref.split(":", 1)[0];
  return SOURCE_LABEL[prefix] ?? "Other data given to the analysts";
}

/** Distinct human labels for `sources_cited`, in first-cited order. */
export function sourceLabels(refs: readonly string[] | null | undefined): string[] {
  const out: string[] = [];
  for (const ref of refs ?? []) {
    const label = sourceLabel(ref);
    if (!out.includes(label)) out.push(label);
  }
  return out;
}

// `ledger.PRIMARY_KINDS` -> label, for the "one primary source kind" cap.
const PRIMARY_KIND_LABEL: Record<string, string> = {
  financials: "financial statements",
  filing: "filings",
  transcript: "earnings-call transcripts",
};

// Section keys the `template_sections` cap names in its detail
// (`memo_quality.CORE_SECTIONS`).
const CORE_SECTION_LABEL: Record<string, string> = {
  sector_agent_view: "sector",
  earnings_agent_view: "earnings",
  filing_agent_view: "filing",
  valuation_agent_view: "valuation",
  "extra_agent_views.industry_group": "industry group",
};

function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

/** One sentence per cap code (`memo_quality.earned_confidence` and the
 * graph's fallback). The backend `detail` names section keys, review modes
 * and source kinds, so it is only mined for those facts, never printed. */
export function capText(cap: ConfidenceCap): string {
  switch (cap.code) {
    case "pm_template":
      return "The PM synthesis was template-filled.";
    case "template_sections": {
      const names = Object.keys(CORE_SECTION_LABEL)
        .filter((k) => new RegExp(`(^|[\\s:,])${k.replace(/\./g, "\\.")}(?=[,.]|$)`).test(cap.detail))
        .map((k) => CORE_SECTION_LABEL[k]);
      const n = Number(/(\d+) core/.exec(cap.detail)?.[1] ?? names.length) || names.length;
      const list = names.length ? ` (${names.join(", ")})` : "";
      return `${n > 0 ? plural(n, "core analyst section") : "Core analyst sections"} ${n === 1 ? "was" : "were"} template-filled${list}.`;
    }
    case "critic_not_live":
      return "No live risk-committee review was completed.";
    case "no_transcript":
      return "No earnings-call transcript was available.";
    case "no_filing_review":
      return "No filing was reviewed by the filing analyst.";
    case "divergence_unreviewed":
      return "The rating goes against the valuation evidence on a reason no live reviewer checked.";
    case "figures_unchecked":
      return "Figures were not source-checked for this version.";
    case "no_primary_trace":
      return "No figure traced to the financial statements, a filing or a transcript.";
    case "single_primary_kind": {
      const kind = /\(([a-z_]+)\)/.exec(cap.detail)?.[1] ?? "";
      const label = PRIMARY_KIND_LABEL[kind];
      return `Figures trace to one kind of primary source only${label ? ` (${label})` : ""}.`;
    }
    case "untraceable_figures": {
      const n = /^(\d+) distinct/.exec(cap.detail)?.[1];
      return n
        ? `${plural(Number(n), "figure")} not found in the data the analysts were given.`
        : "Figures not found in the data the analysts were given.";
    }
    case "last_full_run":
      return "A news update may lower confidence but never raise it above the last full run.";
    case "quality_check_failed":
      return "The confidence checks could not run for this memo.";
    default:
      return "Another research check limited confidence.";
  }
}

/** Caps with the binding one first, the rest by strictness. */
export function orderedCaps(conf: ConfidenceAssessment): ConfidenceCap[] {
  const caps = [...(conf.caps ?? [])];
  caps.sort((a, b) => {
    if (conf.binding && (a.code === conf.binding) !== (b.code === conf.binding)) {
      return a.code === conf.binding ? -1 : 1;
    }
    return a.cap - b.cap;
  });
  return caps;
}

/** The cap that actually lowered confidence, or null when none did. */
export function bindingCap(conf: ConfidenceAssessment | null | undefined): ConfidenceCap | null {
  if (!conf || !(conf.final < conf.raw)) return null;
  return conf.caps.find((c) => c.code === conf.binding) ?? orderedCaps(conf)[0] ?? null;
}

/** "Capped at 45 — 3 core analyst sections were template-filled (…)." for a
 * memo whose confidence the checks lowered, or "" otherwise (including a
 * memo whose confidence the presenter hid: the cap would print the number). */
export function cappedConfidenceLine(memo: StockMemoOut): string {
  if (isHidden(memo, "confidence_score")) return "";
  const conf = qualityOf(memo)?.confidence;
  const cap = bindingCap(conf);
  if (!conf || !cap) return "";
  return `Capped at ${Math.round(conf.final)} — ${capText(cap)}`;
}

/** True when the PM's own confidence (`quality.confidence.raw`) is a
 * template's number: the PM synthesis fell back (`pm_template` cap), or the
 * presenter hid the PM view as template. The quality stage stamps the
 * FINAL confidence as earned, so `confidence_score` stays available in
 * that state (`memo_sections._classify`) and only this catches it. */
export function pmConfidenceIsTemplate(memo: StockMemoOut): boolean {
  const conf = qualityOf(memo)?.confidence;
  return Boolean(conf?.caps?.some((c) => c.code === "pm_template")) || isHidden(memo, "final_pm_view");
}

/** True when the presenter hid any section or item of this memo. The
 * number check ran over the whole memo before presentation, and its tallies
 * and cited sources are stored memo-wide (traced figures are not stored
 * per section), so they then include text the page does not show. */
export function checksCoverHiddenText(memo: Pick<StockMemoOut, "section_availability">): boolean {
  if (hiddenKeys(memo).length > 0) return true;
  return Object.values(memo.section_availability ?? {}).some((av) => (av?.hidden_items ?? 0) > 0);
}

// The Confidence card's tooltip. The old wording ("dampened by
// source-evidence quality") still describes every memo stored before the
// checks existed, so it stays for those; a checked memo gets the wording
// that matches how its number was produced.
export const LEGACY_CONFIDENCE_TOOLTIP = (rating: string) =>
  `How sure the PM is that "${rating}" is the right call. From signal counts across all 8 specialist findings, dampened by source-evidence quality.`;
export const CHECKED_CONFIDENCE_TOOLTIP =
  "PM conviction, lowered by the research checks when sections are templated, sources are thin, the critic did not run, or figures are untraceable.";

export function confidenceTooltip(memo: StockMemoOut): string {
  return qualityOf(memo)?.confidence ? CHECKED_CONFIDENCE_TOOLTIP : LEGACY_CONFIDENCE_TOOLTIP(memo.rating_label);
}

// ---------------------------------------------------------------------------
// Rating reconciliation
// ---------------------------------------------------------------------------

const VERDICT_TEXT: Record<string, string> = {
  overvalued: "overvalued",
  undervalued: "undervalued",
  fairly_priced: "fairly priced",
  mixed: "mixed",
};

export function verdictText(v: string | null | undefined): string {
  return VERDICT_TEXT[v ?? ""] ?? "unclear";
}

// A record is about the rating on the page only while its final rating IS
// that rating. A news patch can move the rating without re-running the
// full-run check (`memo_quality.enforce_after_patch` re-checks only a move
// that diverges from the evidence), which leaves the full run's record in
// place under a different rating.
function recordIsCurrent(memo: Pick<StockMemoOut, "rating_label">, rec: RatingReconciliation): boolean {
  return Boolean(rec.final_rating) && rec.final_rating === memo.rating_label;
}

// `rating_reconciliation_mode=record` (the kill switch) writes `downgraded`
// but leaves the rating as blended: the check recorded, it did not act.
function downgradeEnforced(rec: RatingReconciliation): boolean {
  return rec.final_rating !== rec.blended_rating;
}

// A patch-guard record (`enforce_after_patch`) states no reason and runs no
// reason checks: a news patch cannot state one. A full-run record always
// carries the checks it ran (`reconcile_rating`).
function isPatchRecord(rec: RatingReconciliation): boolean {
  return Object.keys(rec.reason_checks ?? {}).length === 0;
}

/** One line under the rating badge when the check changed or overrode the
 * call; "" when the rating simply agreed with the evidence, and "" when the
 * record no longer describes the rating shown (a later news patch moved
 * it) — the badge must never be captioned with another rating's story. */
export function reconciliationBadgeNote(memo: StockMemoOut): string {
  const rec = qualityOf(memo)?.rating_reconciliation;
  if (!rec || !recordIsCurrent(memo, rec)) return "";
  const evidence = verdictText(rec.valuation_verdict);
  const from = rec.blended_rating || rec.pm_rating;
  const call = from ? `a ${from} call` : "the call";
  if (rec.outcome === "downgraded") {
    if (!downgradeEnforced(rec)) {
      return `Kept at ${rec.final_rating}: the call conflicts with the valuation evidence (${evidence}), but the rating check is recording only, not changing ratings.`;
    }
    if (isPatchRecord(rec)) {
      return `Set to ${rec.final_rating}: a news update moved the call to ${from || "another rating"} against the valuation evidence (${evidence}).`;
    }
    return `Set to ${rec.final_rating}: ${call} conflicted with the valuation evidence (${evidence}) without a supported reason.`;
  }
  if (rec.outcome === "accepted") {
    const review =
      rec.critic_assessment === "supported" ? "reviewed and supported" : "not independently reviewed";
    return `${rec.final_rating} despite ${evidence} valuation evidence, on the PM's stated reason (${review}).`;
  }
  return "";
}

// `memo_quality.REASON_CHECKS` -> what a failed check says about the reason.
const REASON_CHECK_FAILED: Record<string, string> = {
  substantive: "was too short or generic to count as a reason",
  names_signal: "did not name the valuation signal it goes against",
  quotes_value: "did not quote that signal's value",
};

function joinClauses(parts: string[]): string {
  if (parts.length <= 1) return parts.join("");
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}

function downgradeWhy(rec: RatingReconciliation): string {
  const checks = rec.reason_checks ?? {};
  if (!rec.reason.trim()) return "no reason was given";
  if (checks.check_failed) return "the stated reason could not be verified";
  const failed = Object.keys(REASON_CHECK_FAILED).filter((k) => checks[k] === false);
  if (failed.length === 0) return "the risk committee judged the stated reason unsupported";
  return `the stated reason ${joinClauses(failed.map((k) => REASON_CHECK_FAILED[k]))}`;
}

/** The rating check in sentences, composed from the record's structured
 * fields. The backend's `note` is never printed: it carries the reason
 * checks' names ("failed: names_signal, quotes_value") and the config
 * switch's name, and any future wording would reach the page unreviewed. */
export function ratingCheckLines(memo: StockMemoOut): string[] {
  const rec = qualityOf(memo)?.rating_reconciliation;
  if (!rec) return [];
  const evidence = verdictText(rec.valuation_verdict);
  const blended = rec.blended_rating || rec.pm_rating || "the blended rating";
  const lines: string[] = [];
  switch (rec.outcome) {
    case "not_applicable":
      lines.push("No valuation evidence was available, so the rating was not checked against it.");
      break;
    case "consistent":
      lines.push(`The rating agrees with the valuation evidence (${evidence}).`);
      break;
    case "accepted":
      lines.push(`Rated ${blended} although the valuation evidence reads ${evidence}, on the PM's stated reason.`);
      break;
    case "downgraded": {
      const enforced = downgradeEnforced(rec);
      const what = isPatchRecord(rec)
        ? `A news update moved the rating to ${blended} against the valuation evidence (${evidence}) with no valuation reason (a news update cannot state one).`
        : `The blended rating was ${blended}, but the valuation evidence reads ${evidence}, and ${downgradeWhy(rec)}.`;
      lines.push(
        enforced
          ? `${what} The rating was set to ${rec.final_rating}.`
          : `${what} Recorded only: the rating check is not changing ratings, so it stays ${rec.final_rating || blended}.`,
      );
      break;
    }
    default:
      break;
  }
  if (rec.final_rating && !recordIsCurrent(memo, rec)) {
    lines.push(
      `A news update has since moved the rating to ${memo.rating_label}; this check describes the last full run.`,
    );
  }
  return lines;
}

export const CRITIC_ASSESSMENT_TEXT: Record<string, string> = {
  supported: "The risk committee reviewed the reason and supported it.",
  unsupported: "The risk committee reviewed the reason and did not support it.",
  not_assessed: "No live risk-committee review of the reason.",
};

// ---------------------------------------------------------------------------
// Degraded banners
// ---------------------------------------------------------------------------

// Degradation `error_type`s that are findings about the memo's content, not
// outages. W2b records untraceable figures in `quality` (the panel shows
// them) and never on the banner; this filter keeps it that way for any
// body that carries such an event, so "N agents degraded" counts failures
// only. A crash of the check itself ("Number Check" with a real error type)
// is an outage and stays on the banner.
export const QUALITY_EVENT_TYPES: ReadonlySet<string> = new Set(["UntraceableNumbers"]);

/** `degraded_agents` minus agents whose every event is a quality event. */
export function bannerDegradedAgents(
  memo: Pick<StockMemoOut, "degraded_agents" | "degradation_events">,
): string[] {
  const events = memo.degradation_events ?? [];
  return (memo.degraded_agents ?? []).filter((agent) => {
    const mine = events.filter((e) => e?.agent === agent);
    return !(mine.length > 0 && mine.every((e) => QUALITY_EVENT_TYPES.has(e.error_type)));
  });
}
