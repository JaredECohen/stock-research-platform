// W2a (owner decision 2, 2026-09-24): a memo is published, but a section a
// template filled — the deterministic fallback — is shown as "Unavailable in
// this version." with a one-line reason instead of reading as analysis.
//
// The backend presenter (`backend/app/services/memo_sections.py`) does the
// classifying on every read and ships the verdict as
// `memo.section_availability`, keyed by section. This module only READS that
// map; it never re-derives a verdict from the text, so the frontend cannot
// disagree with the server about what is template.
//
// The presenter already replaces hidden prose with `UNAVAILABLE_TEXT`, so a
// renderer that ignored the map would still print the placeholder. What the
// map adds is the reason line, the hidden-item counts and the banner count,
// and it lets a renderer drop the text it would otherwise wrap around the
// placeholder (a confidence number, a "Read full report" button).
import type {
  AgentFinding,
  SectionAvailability,
  SectionReason,
  SectionStatus,
  StockMemoOut,
} from "@/types";

// Must equal `memo_sections.UNAVAILABLE_TEXT`; the fixture test in
// `lib/memoSections.test.ts` compares it with the captured presenter output.
export const UNAVAILABLE_TEXT = "Unavailable in this version.";

// One plain sentence per reason, shown under the placeholder (hidden
// sections) or as a short note (degraded ones). Keyed by the closed
// vocabulary in `schemas/memo.py::SectionReason`; a `Record` over the TS
// union makes a new backend reason a type error here until it is worded.
export const REASON_TEXT: Record<SectionReason, string> = {
  template_fallback: "The analyst did not complete this section for this version.",
  template_always: "This section is standard wording, not analysis of this company.",
  agent_failed: "The analyst failed during this run.",
  no_source_data: "The data this section needs was not on file.",
  skipped_by_intake: "The PM skipped this analyst for this run.",
  critic_not_run: "No live risk-committee review was completed.",
  derived_from_hidden: "Built from sections that are unavailable in this version.",
  not_produced: "Not produced for this version.",
  rule_based: "Produced by a rule-based check, not an analyst.",
  llm_patched: "Updated by a news patch after the original run.",
  reduced_inputs: "The analyst worked from reduced inputs in this run.",
  unclassified: "This section could not be checked for template text.",
  pm_view_unavailable: "PM view unavailable — rating reflects the factor blend.",
  partial_template: "Template text was removed from this section.",
  follow_up_unanswered: "The follow-up went unanswered; the first-round read is shown.",
  templated_scenarios: "The bull and bear DCF scenarios use sector-default assumptions.",
};

// Reasons that do not count toward the "N sections unavailable" banner.
// `template_always` (Portfolio Fit) is never written by an analyst, so it
// would add one to every memo and the banner would stop meaning anything;
// `skipped_by_intake` is a PM decision, not a failure; `not_produced` means
// there was nothing to hide (an optional section the run had no input for).
const NOT_COUNTED: ReadonlySet<SectionReason> = new Set<SectionReason>([
  "template_always",
  "skipped_by_intake",
  "not_produced",
]);

/** The presenter's verdict on one section, or undefined when the memo has
 * no map (a pre-W2a body) or the map has no entry for the key. */
export function availability(
  memo: Pick<StockMemoOut, "section_availability"> | null | undefined,
  key: string,
): SectionAvailability | undefined {
  return memo?.section_availability?.[key];
}

/** Absent map or key reads as "available", which keeps every body served
 * before the presenter existed (and every older test fixture) rendering as
 * it always did. */
export function sectionStatus(
  memo: Pick<StockMemoOut, "section_availability"> | null | undefined,
  key: string,
): SectionStatus {
  return availability(memo, key)?.status ?? "available";
}

/** True when the presenter marked the section unavailable for any reason,
 * including `not_produced`. */
export function isUnavailable(
  memo: Pick<StockMemoOut, "section_availability"> | null | undefined,
  key: string,
): boolean {
  return sectionStatus(memo, key) === "unavailable";
}

/** True when the section's content was withheld: unavailable for a reason
 * other than `not_produced`. A `not_produced` section had nothing to hide,
 * so renderers keep their existing empty-state behaviour for it (the
 * mispricing card is the one exception, per the design). */
export function isHidden(
  memo: Pick<StockMemoOut, "section_availability"> | null | undefined,
  key: string,
): boolean {
  const av = availability(memo, key);
  return av?.status === "unavailable" && av.reason !== "not_produced";
}

export function isDegraded(
  memo: Pick<StockMemoOut, "section_availability"> | null | undefined,
  key: string,
): boolean {
  return sectionStatus(memo, key) === "degraded";
}

// Reason sentences that depend on the section as well as the reason. A
// degraded thesis (`partial_template`, basis `claim:llm_headline`) is a
// builder rewrite that kept the analyst's claim and appended the builder's
// canned lever sentence: the presenter removes nothing from it, so the
// generic "Template text was removed" would be false. The integration plan
// kept that thesis visible only because it is "shown with a note"
// (critique delta 11).
export const SECTION_REASON_TEXT: Readonly<Record<string, Partial<Record<SectionReason, string>>>> = {
  one_sentence_thesis: {
    partial_template: "The claim is the analyst's; the sentence after it is standard builder wording, not analysis.",
  },
};

/** The reason sentence for a verdict, or "" when it carries none. With
 * `section`, a section-specific wording wins over the generic one. */
export function reasonText(av: SectionAvailability | undefined, section?: string): string {
  if (!av?.reason) return "";
  const specific = section ? SECTION_REASON_TEXT[section]?.[av.reason] : undefined;
  return specific ?? REASON_TEXT[av.reason] ?? "";
}

/** "2 template items not shown" for a list section the presenter filtered,
 * or "" when nothing was removed. */
export function hiddenItemsNote(av: SectionAvailability | undefined): string {
  const n = av?.hidden_items ?? 0;
  if (n <= 0) return "";
  return `${n} template item${n === 1 ? "" : "s"} not shown`;
}

/** Keys whose content the presenter withheld, in map order. */
export function hiddenKeys(memo: Pick<StockMemoOut, "section_availability">): string[] {
  return Object.keys(memo.section_availability ?? {}).filter((k) => isHidden(memo, k));
}

/** True when an unavailable section counts toward the banner: withheld for
 * a reason in neither `NOT_COUNTED` nor a drill-down (`<finding>.long_form_report`
 * belongs to its card, which carries its own note). */
export function isCounted(
  memo: Pick<StockMemoOut, "section_availability"> | null | undefined,
  key: string,
): boolean {
  const av = availability(memo, key);
  return (
    av?.status === "unavailable" &&
    !(av.reason && NOT_COUNTED.has(av.reason)) &&
    !key.endsWith(".long_form_report")
  );
}

// The sections each renderer can show a placeholder for. The banner counts
// only these, so "N sections unavailable" never names a section the reader
// cannot find on the page in front of them: no memo renderer shows
// `extra_agent_views.*` (FEAT-003 routing adds one per unmapped ticker),
// `earnings_qoq_delta` or `forward_catalysts`; the card has no scorecard or
// thesis breakers; the full memo has no sector synthesis. A renderer that
// starts showing a section adds its key here with the placeholder, and each
// renderer's test asserts that the count equals the placeholders it shows.
const FINDING_SECTIONS = [
  "sector_agent_view",
  "earnings_agent_view",
  "filing_agent_view",
  "valuation_agent_view",
  "comps_agent_view",
  "macro_sensitivity",
  "technical_agent_view",
] as const;

export const MEMO_CARD_SECTIONS: ReadonlySet<string> = new Set([
  "one_sentence_thesis",
  "valuation_verdict",
  "confidence_score",
  "final_pm_view",
  "mispricing_thesis",
  ...FINDING_SECTIONS,
  "bull_case",
  "bear_case",
  "sector_synthesis",
  "catalysts",
  "key_risks",
  "dcf_summary",
  "risk_committee_challenge",
  "final_verdict",
]);

export const FULL_MEMO_SECTIONS: ReadonlySet<string> = new Set([
  "confidence_score",
  "one_sentence_thesis",
  "final_pm_view",
  "mispricing_thesis",
  "bull_case",
  "bear_case",
  ...FINDING_SECTIONS,
  "valuation_verdict",
  "dcf_summary",
  "scorecard",
  "catalysts",
  "key_risks",
  "thesis_breakers",
  "risk_committee_challenge",
  "portfolio_fit",
  "final_verdict",
]);

export const SAMPLE_SUMMARY_SECTIONS: ReadonlySet<string> = new Set([
  "confidence_score",
  "one_sentence_thesis",
  "final_pm_view",
  "bull_case",
  "bear_case",
  "valuation_verdict",
  "catalysts",
  "key_risks",
]);

/** How many sections the "N sections unavailable in this version" banner
 * reports: the counted sections (`isCounted`) among those the calling
 * renderer shows (`shown`, one of the sets above). */
export function bannerCount(
  memo: Pick<StockMemoOut, "section_availability">,
  shown: ReadonlySet<string>,
): number {
  return Object.keys(memo.section_availability ?? {}).filter(
    (key) => shown.has(key) && isCounted(memo, key),
  ).length;
}

export function bannerText(count: number): string {
  return `${count} section${count === 1 ? "" : "s"} unavailable in this version`;
}

/** The PM's stated reason for skipping an analyst. The finding's own
 * `data.intake_rationale` survives the presenter's allowlist, but the
 * public-sample size reducer can empty `data`, so fall back to the memo's
 * `intake_decision`. "" when neither carries one. */
export function intakeRationale(
  memo: StockMemoOut,
  finding: AgentFinding | null | undefined,
): string {
  const own = finding?.data?.intake_rationale;
  if (typeof own === "string" && own.trim()) return own.trim();
  const decision = (memo as { intake_decision?: { rationale?: unknown } }).intake_decision;
  const r = decision?.rationale;
  return typeof r === "string" ? r.trim() : "";
}

/** A diligence-round finding the presenter blanked. Round findings have no
 * per-finding map entry (the `round_findings` key only counts them), so the
 * presenter's own output shape is the signal: both prose fields replaced
 * with the placeholder. */
export function isBlankedFinding(f: AgentFinding | null | undefined): boolean {
  return !!f && f.headline === UNAVAILABLE_TEXT && f.summary === UNAVAILABLE_TEXT;
}
