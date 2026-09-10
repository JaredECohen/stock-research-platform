// Formatting shared by the scorecard components. Extends lib/format.ts with
// the scorecard's own quantities (z-scores, 0–100 scores, percentile ranks)
// and the research rule that a missing value renders as "n/a" *with its
// reason* — never as 0, "—", or a blank the reader could mistake for a
// neutral score.
import { fmtCurrency, fmtMultiple, fmtNumber, fmtPct, NA } from "@/lib/format";
import { SCORECARD_FAMILY_LABELS, type ScorecardFamily } from "@/types/scorecard";

export { NA };

export function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

/** "n/a (reason)". The reason is required so no caller can forget it. */
export function na(reason: string): string {
  return `${NA} (${reason})`;
}

/** Human label for a family id; an unknown id on the wire is humanised
 *  rather than dropped, so a new backend family still shows up. */
export function familyLabel(family: string): string {
  return (SCORECARD_FAMILY_LABELS as Record<string, string>)[family as ScorecardFamily] ?? humanize(family);
}

export function humanize(id: string): string {
  return id.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

/** Signed z with two decimals: "+0.62", "-1.10", "0.00". */
export function fmtZ(v: number | null | undefined, reason = "not computed"): string {
  if (!isNum(v)) return na(reason);
  const s = v.toFixed(2);
  return v > 0 ? `+${s}` : s;
}

/** 0–100 composite score with one decimal. */
export function fmtScore(v: number | null | undefined, reason = "not computed"): string {
  if (!isNum(v)) return na(reason);
  return v.toFixed(1);
}

/** Percentile rank as an ordinal: 71.3 → "71st", 2 → "2nd", 100 → "100th". */
export function fmtPercentile(v: number | null | undefined, reason = "not ranked"): string {
  if (!isNum(v)) return na(reason);
  const n = Math.round(v);
  const mod100 = n % 100;
  const suffix = mod100 >= 11 && mod100 <= 13 ? "th" : n % 10 === 1 ? "st" : n % 10 === 2 ? "nd" : n % 10 === 3 ? "rd" : "th";
  return `${n}${suffix}`;
}

/** Coverage share 0–1 as a whole percent. */
export function fmtCoverage(v: number | null | undefined): string {
  if (!isNum(v)) return na("not computed");
  return `${Math.round(v * 100)}%`;
}

/** Observed feature input in its declared unit. */
export function fmtRaw(v: number | null | undefined, unit: string, reason = "not computed"): string {
  if (!isNum(v)) return na(reason);
  switch (unit) {
    case "percent":
      return fmtPct(v, 1);
    case "multiple":
      return fmtMultiple(v);
    case "currency":
      return fmtCurrency(v, { compact: true });
    case "count":
      return fmtNumber(v, 0);
    case "ratio":
    default:
      return fmtNumber(v, 3);
  }
}

/** Monthly return as a signed percent with two decimals: "+1.23%". */
export function fmtReturn(v: number | null | undefined, reason = "not computed"): string {
  if (!isNum(v)) return na(reason);
  const s = `${(v * 100).toFixed(2)}%`;
  return v > 0 ? `+${s}` : s;
}

/** Plain statistic (t-stat, Sharpe, R²) with two decimals. */
export function fmtStat(v: number | null | undefined, reason = "not computed"): string {
  if (!isNum(v)) return na(reason);
  return v.toFixed(2);
}

/** Why a feature has no observed value. The server's `reason` wins; the
 *  sector mask is the one reason the client can name on its own. */
export function featureMissingReason(f: { applicable: boolean; reason?: string | null }): string {
  if (!f.applicable) return "not applicable to sector";
  if (f.reason) return humanize(f.reason).toLowerCase();
  return "input not available";
}

/** Tone class for a 0–100 score; text colour only — every value also
 *  prints as text so colour is never the sole carrier. */
export function scoreTone(v: number | null | undefined): string {
  if (!isNum(v)) return "text-slate-500";
  if (v >= 70) return "text-accent-500";
  if (v >= 50) return "text-slate-200";
  if (v >= 30) return "text-warn-500";
  return "text-danger-500";
}
