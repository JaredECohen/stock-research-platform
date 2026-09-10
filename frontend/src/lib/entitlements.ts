// FEAT-002 — copy and helpers shared by the meters, the upgrade prompt and
// the account page. The numbers themselves always come from the backend
// (`/api/me` entitlements, a 402 body, or the `features` matrix in
// `/api/public/config`); this file only knows how to name things. No
// allowance, cap or trial length is ever written as a literal here —
// ENTITLEMENT_OVERRIDES_JSON / TRIAL_DAYS change them without a deploy and
// the UI must follow.

import { parseTimestamp } from "@/components/ProviderHealthBanner";
import type { Entitlement, FeatureAllowance, FeatureMatrix, FeatureName } from "@/types";

export interface FeatureCopy {
  /** Short noun, e.g. "memo views". */
  label: string;
  /** Singular form for "1 research run". */
  singular: string;
  /** One line on why Pro is worth it for this feature — the pitch plus,
   *  when the matrix is known, the allowance the backend enforces. */
  value: string;
}

interface StaticCopy {
  label: string;
  singular: string;
  /** Number-free description; the allowance sentence is appended from the matrix. */
  pitch: string;
}

const STATIC_COPY: Record<FeatureName, StaticCopy> = {
  memo_view: {
    label: "stored memo views",
    singular: "stored memo view",
    pitch: "Pro opens every stored memo in the universe.",
  },
  research_run: {
    label: "research runs",
    singular: "research run",
    pitch: "A research run puts the full agent committee — sector, earnings, filings, valuation, comps, macro, risk — on a ticker and writes a fresh memo.",
  },
  pm_chat: {
    label: "Ask-the-PM turns",
    singular: "Ask-the-PM turn",
    pitch: "Ask-the-PM answers from the committee's stored work; macro scenarios and the natural-language screener count as turns.",
  },
  chart_commentary: {
    label: "chart commentaries",
    singular: "chart commentary",
    pitch: "AI commentary on any chart.",
  },
  fundamentals_explorer: {
    label: "fundamentals explorer",
    singular: "fundamentals explorer",
    pitch: "Explore reported fundamentals across the universe.",
  },
  dcf: {
    label: "DCF models",
    singular: "DCF model",
    pitch: "Pro runs a DCF on any ticker.",
  },
  comps: {
    label: "comps tables",
    singular: "comps table",
    pitch: "Pro builds comparable-company tables for any ticker.",
  },
  portfolio: {
    label: "portfolio builds",
    singular: "portfolio build",
    pitch: "The Portfolio Builder turns a market view into a diversified scenario portfolio.",
  },
  macro: {
    label: "macro analysis",
    singular: "macro analysis",
    pitch: "Macro series and scenario analysis for the whole universe.",
  },
  track_record: {
    label: "track record",
    singular: "track record",
    pitch: "The realised track record — forward returns, alpha vs SPY, thesis-held rate.",
  },
  memo_history: {
    label: "memo history",
    singular: "memo history",
    pitch: "Memo version history, agent memory and DCF versions.",
  },
  data_catalog: {
    label: "data catalog",
    singular: "data catalog",
    pitch: "The data catalog and sector overlays.",
  },
  scorecard: {
    label: "scorecard",
    singular: "scorecard",
    pitch: "The fundamental factor scorecard: a versioned, sector-neutral read of reported fundamentals across the universe, with its evaluation and the frozen-contract export.",
  },
};

function capitalize(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

/** "Pro includes 20 research runs a month." — from the matrix, or null
 *  when the matrix is unknown (config fetch fell back) so nothing is
 *  invented. */
export function proAllowanceSentence(feature: string, matrix: FeatureMatrix | undefined, label: string): string | null {
  const entry = matrix?.[feature];
  if (!entry) return null;
  return allowanceSentence("Pro", entry.pro, label, entry.metered, entry.distinct_resources);
}

/** The Free side of the same row, for features where Free has a rule
 *  worth stating (follows-memo or a monthly number). */
export function freeAllowanceSentence(feature: string, matrix: FeatureMatrix | undefined, label: string): string | null {
  const entry = matrix?.[feature];
  if (!entry) return null;
  if (entry.free === false) return null;
  return allowanceSentence("Free", entry.free, label, entry.metered, entry.distinct_resources);
}

function allowanceSentence(plan: "Pro" | "Free", value: FeatureAllowance, label: string, metered: boolean, distinct: boolean): string | null {
  if (value === "follows_memo") {
    return `On ${plan}, a ticker is available once its memo has been opened this month.`;
  }
  if (value === null) {
    return metered
      ? `${plan} has no monthly cap on ${label}${distinct ? " (distinct tickers)" : ""}.`
      : `${capitalize(label)} ${plural(label) ? "are" : "is"} part of ${plan}.`;
  }
  if (typeof value === "boolean") {
    return value
      ? `${capitalize(label)} ${plural(label) ? "are" : "is"} part of ${plan}.`
      : `${capitalize(label)} ${plural(label) ? "are" : "is"} not included in ${plan}.`;
  }
  if (value <= 0) return `${capitalize(label)} ${plural(label) ? "are" : "is"} not included in ${plan}.`;
  return `${plan} includes ${value} ${label} a month${distinct ? " (distinct tickers)" : ""}.`;
}

function plural(label: string): boolean {
  return /s$/.test(label) && !/(analysis|status)$/.test(label);
}

/**
 * Copy for a feature. Pass the config's `features` matrix to get the
 * allowance sentence; without it the value is the pitch alone (or, for an
 * unknown feature, a generic "is part of Pro").
 */
export function featureCopy(feature: string | null | undefined, matrix?: FeatureMatrix): FeatureCopy {
  const known = feature && feature in STATIC_COPY ? STATIC_COPY[feature as FeatureName] : null;
  const name = (feature || "this feature").replace(/_/g, " ");
  const label = known ? known.label : name;
  const singular = known ? known.singular : name;
  const parts: string[] = [];
  if (known) parts.push(known.pitch);
  const pro = feature ? proAllowanceSentence(feature, matrix, label) : null;
  if (pro) parts.push(pro);
  // Free's follows-memo rule is worth stating next to the Pro pitch.
  const entry = feature ? matrix?.[feature] : undefined;
  if (entry?.free === "follows_memo") {
    const free = freeAllowanceSentence(feature as string, matrix, label);
    if (free) parts.push(free);
  }
  if (parts.length === 0) parts.push(`${capitalize(name)} is part of Pro.`);
  return { label, singular, value: parts.join(" ") };
}

/**
 * "up to 3 stored memo views, 1 research run and 10 Ask-the-PM turns a
 * month" from the user's OWN entitlements (`/api/me`), which already
 * include any override. Null when none of the headline meters carries a
 * number, so the caller can drop the clause instead of guessing.
 */
export function freeHeadlineAllowances(
  entitlements: Record<string, Entitlement> | undefined,
  order: string[] = ["memo_view", "research_run", "pm_chat"],
): string | null {
  if (!entitlements) return null;
  const clauses: string[] = [];
  for (const feature of order) {
    const e = entitlements[feature];
    if (!e || !e.allowed || typeof e.limit !== "number" || e.limit <= 0) continue;
    const copy = featureCopy(feature);
    clauses.push(`${e.limit} ${e.limit === 1 ? copy.singular : copy.label}`);
  }
  if (clauses.length === 0) return null;
  const list = clauses.length === 1
    ? clauses[0]
    : `${clauses.slice(0, -1).join(", ")} and ${clauses[clauses.length - 1]}`;
  return `up to ${list} a month`;
}

const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

/** "September 15, 2026 at 14:03 UTC" — hand-built so the wording does not
 *  depend on the viewer's ICU data. The backend emits naive UTC stamps;
 *  `parseTimestamp` treats them as UTC. */
export function formatExactUtc(value: string | number | null | undefined): string | null {
  const t = parseTimestamp(value);
  if (t === null) return null;
  const d = new Date(t);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${d.getUTCFullYear()} at ${hh}:${mm} UTC`;
}

/** "Sep 15, 2026" for meter footers. */
export function formatShortUtc(value: string | number | null | undefined): string | null {
  const t = parseTimestamp(value);
  if (t === null) return null;
  const d = new Date(t);
  return `${MONTHS[d.getUTCMonth()].slice(0, 3)} ${d.getUTCDate()}, ${d.getUTCFullYear()}`;
}

/** Whole days from now until `value`, floored at 0; null when unparseable. */
export function daysUntil(value: string | number | null | undefined, now: number = Date.now()): number | null {
  const t = parseTimestamp(value);
  if (t === null) return null;
  return Math.max(0, Math.ceil((t - now) / 86_400_000));
}

export function planLabel(plan: string, source: string): string {
  if (plan === "pro") return source === "trial" ? "Pro trial" : "Pro";
  if (plan === "none") return "Suspended";
  return "Free";
}
