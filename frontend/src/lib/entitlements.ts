// FEAT-002 — copy and helpers shared by the meters, the upgrade prompt and
// the account page. The numbers themselves always come from the backend
// (`/api/me` entitlements or a 402 body); this file only knows how to
// name things.

import { parseTimestamp } from "@/components/ProviderHealthBanner";
import type { FeatureName } from "@/types";

export interface FeatureCopy {
  /** Short noun, e.g. "memo views". */
  label: string;
  /** Singular form for "1 research run". */
  singular: string;
  /** One line on why Pro is worth it for this feature. */
  value: string;
}

export const FEATURE_COPY: Record<FeatureName, FeatureCopy> = {
  memo_view: {
    label: "stored memo views",
    singular: "stored memo view",
    value: "Pro opens every stored memo in the universe, with no monthly cap on distinct tickers.",
  },
  research_run: {
    label: "research runs",
    singular: "research run",
    value: "A research run puts the full agent committee — sector, earnings, filings, valuation, comps, macro, risk — on a ticker and writes a fresh memo. Pro includes 20 a month.",
  },
  pm_chat: {
    label: "Ask-the-PM turns",
    singular: "Ask-the-PM turn",
    value: "Pro includes 300 Ask-the-PM turns a month (macro scenarios and the natural-language screener count as turns).",
  },
  chart_commentary: {
    label: "chart commentaries",
    singular: "chart commentary",
    value: "Pro includes 100 AI chart commentaries a month.",
  },
  fundamentals_explorer: {
    label: "fundamentals explorer",
    singular: "fundamentals explorer",
    value: "Explore reported fundamentals across the universe.",
  },
  dcf: {
    label: "DCF models",
    singular: "DCF model",
    value: "Pro runs a DCF on any ticker; Free can model a ticker once its memo has been opened this month.",
  },
  comps: {
    label: "comps tables",
    singular: "comps table",
    value: "Pro builds comparable-company tables for any ticker; Free follows the memos opened this month.",
  },
  portfolio: {
    label: "portfolio builds",
    singular: "portfolio build",
    value: "The Portfolio Builder turns a market view into a diversified scenario portfolio. Pro only.",
  },
  macro: {
    label: "macro analysis",
    singular: "macro analysis",
    value: "Macro series and scenario analysis are part of Pro.",
  },
  track_record: {
    label: "track record",
    singular: "track record",
    value: "The realised track record — forward returns, alpha vs SPY, thesis-held rate — is part of Pro.",
  },
  memo_history: {
    label: "memo history",
    singular: "memo history",
    value: "Memo version history, agent memory and DCF versions are part of Pro.",
  },
  data_catalog: {
    label: "data catalog",
    singular: "data catalog",
    value: "The data catalog and sector overlays are part of Pro.",
  },
};

export function featureCopy(feature: string | null | undefined): FeatureCopy {
  if (feature && feature in FEATURE_COPY) return FEATURE_COPY[feature as FeatureName];
  const name = (feature || "this feature").replace(/_/g, " ");
  return { label: name, singular: name, value: `${name} is part of Pro.` };
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
