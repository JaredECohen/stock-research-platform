// FEAT-002 (S6) — turning the backend's entitlement matrix and prices
// into pricing-page copy. No number in this file: every figure comes from
// `/api/public/config` (`features`, `prices`, `trial_days`), which is the
// same table the backend enforces, so the page cannot promise something
// ENTITLEMENT_OVERRIDES_JSON took away.
import type { FeatureAllowance, FeatureMatrix, FeatureMatrixEntry } from "@/types";

/** "$29.99" / "$299" — whole-dollar amounts drop the cents. */
export function formatCents(cents: number, currency = "usd"): string {
  const whole = cents % 100 === 0;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: currency.toUpperCase(),
    minimumFractionDigits: whole ? 0 : 2,
    maximumFractionDigits: 2,
  }).format(cents / 100);
}

/** The monthly equivalent of an annual price, to the cent. */
export function monthlyEquivalent(annualCents: number, currency = "usd"): string {
  return formatCents(Math.round(annualCents / 12), currency);
}

/** Rows the comparison table shows, in order, with the customer-facing
 *  label. Features the matrix marks "reserved" (not shipped) are left out
 *  on purpose — the table describes what exists. */
export const COMPARISON_ROWS: Array<{ feature: string; label: string }> = [
  { feature: "memo_view", label: "Stored research memos" },
  { feature: "research_run", label: "Research runs (the full committee on a ticker)" },
  { feature: "pm_chat", label: "Ask-the-PM turns (also macro analysis and the natural-language screener)" },
  { feature: "dcf", label: "DCF models" },
  { feature: "comps", label: "Comparable-company tables" },
  { feature: "portfolio", label: "Portfolio builder" },
  { feature: "macro", label: "Macro series and scenario analysis" },
  { feature: "track_record", label: "Track record and outcome evaluation" },
  { feature: "memo_history", label: "Memo history, agent memory and DCF versions" },
  { feature: "data_catalog", label: "Data catalog and sector overlays" },
];

/**
 * One cell of the comparison table. `distinct_resources` is what makes
 * the Free memo allowance "3 distinct companies' memos per month" rather
 * than three page loads — the wording the pricing page must carry.
 */
export function allowanceText(value: FeatureAllowance, entry: FeatureMatrixEntry, feature: string): string {
  if (value === "follows_memo") return "For companies whose memo you opened this month";
  if (value === null) return "Unlimited";
  if (value === true) return "Included";
  if (value === false) return "Not included";
  if (value <= 0) return "Not included";
  const per = entry.period === "month" ? "per month" : `per ${entry.period}`;
  if (entry.distinct_resources) {
    return feature === "memo_view" ? `${value} distinct companies' memos ${per}` : `${value} distinct companies ${per}`;
  }
  return `${value} ${per}`;
}

/** Cells for one row, or null when the matrix does not carry the feature. */
export function rowCells(matrix: FeatureMatrix, feature: string): { free: string; pro: string } | null {
  const entry = matrix[feature];
  if (!entry) return null;
  return { free: allowanceText(entry.free, entry, feature), pro: allowanceText(entry.pro, entry, feature) };
}

export function matrixKnown(matrix: FeatureMatrix | undefined): boolean {
  return !!matrix && Object.keys(matrix).length > 0;
}
