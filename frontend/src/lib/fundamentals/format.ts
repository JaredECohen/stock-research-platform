// Per-unit formatting for the Fundamentals Explorer. Extends lib/format.ts
// (which is USD-only and DCF-shaped) with the catalog's unit types and the
// research rule that a missing value renders as "n/a" *with its reason* —
// never as 0, "—", or a blank the reader could mistake for zero.
import type { MissingReason, UnitType } from "@/types/fundamentals";

export const NA = "n/a";

/** Human text for every reason a point can be null. Kept as a complete
 *  record so a new reason on the wire fails the type check here instead of
 *  rendering an empty parenthesis. */
export const REASON_TEXT: Record<MissingReason, string> = {
  base_nonpositive: "prior-period base is zero or negative",
  denominator_nonpositive: "denominator is zero or negative",
  no_price: "no price at period end",
  no_shares: "no diluted share count",
  missing_line: "line item not reported",
  not_backfilled: "history not loaded",
  before_index_base: "before the index base period",
};

export function reasonText(reason?: MissingReason | null): string {
  if (reason && reason in REASON_TEXT) return REASON_TEXT[reason];
  // An unknown reason still says *something* is missing; the wire value
  // is shown so a reader can report it, rather than being swallowed.
  return reason ? String(reason).replace(/_/g, " ") : "not obtained";
}

/** "n/a (no price at period end)". */
export function formatMissing(reason?: MissingReason | null): string {
  return `${NA} (${reasonText(reason)})`;
}

const CURRENCY_SYMBOLS: Record<string, string> = {
  USD: "$",
  EUR: "€",
  GBP: "£",
  JPY: "¥",
  CNY: "CN¥",
  CAD: "CA$",
  AUD: "A$",
  CHF: "CHF ",
  HKD: "HK$",
  INR: "₹",
  KRW: "₩",
  // SEK/DKK/NOK all print "kr" and would be indistinguishable side by side,
  // so they fall through to the ISO-code prefix ("SEK 1.2B").
  BRL: "R$",
  TWD: "NT$",
};

/** Symbol or ISO-code prefix for a known currency; an empty string when the
 *  reporting currency is unknown. USD is never assumed: a series without a
 *  `currency` may be a EUR or JPY reporter, and printing "$" would put a
 *  wrong unit in front of the reader (and the screen reader). The axis and
 *  table header say "currency" in that case, so the number stays honest. */
export function currencySymbol(currency?: string | null): string {
  if (!currency) return "";
  const code = currency.toUpperCase();
  return CURRENCY_SYMBOLS[code] ?? `${code} `;
}

/** Compact magnitude with the existing K/M/B/T thresholds from lib/format
 *  (1dp; trillions get 2dp so $1.23T does not read as $1.2T next to $1.3T).
 *  Hand-rolled rather than Intl compact notation so the output is byte-stable
 *  across Node ICU builds — these strings are compared in tests and read
 *  aloud by screen readers. */
export function compactMagnitude(v: number, opts: { digits?: number } = {}): string {
  const abs = Math.abs(v);
  const d = opts.digits ?? 1;
  if (abs >= 1e12) return `${(abs / 1e12).toFixed(Math.max(d, 2))}T`;
  if (abs >= 1e9) return `${(abs / 1e9).toFixed(d)}B`;
  if (abs >= 1e6) return `${(abs / 1e6).toFixed(d)}M`;
  if (abs >= 1e3) return `${(abs / 1e3).toFixed(d)}K`;
  return abs.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function formatCurrency(v: number | null | undefined, currency?: string | null): string {
  if (!isNum(v)) return NA;
  const sign = v < 0 ? "-" : "";
  return `${sign}${currencySymbol(currency)}${compactMagnitude(v)}`;
}

/** Fractions on the wire (0.462 → "46.2%"), matching `fmtPct`. */
export function formatPercent(v: number | null | undefined, digits = 1): string {
  if (!isNum(v)) return NA;
  return `${(v * 100).toFixed(digits)}%`;
}

export function formatRatio(v: number | null | undefined, digits = 2): string {
  if (!isNum(v)) return NA;
  return v.toFixed(digits);
}

export function formatMultiple(v: number | null | undefined, digits = 1): string {
  if (!isNum(v)) return NA;
  return `${v.toFixed(digits)}x`;
}

/** Share counts and similar: compact, unit-less ("15.2B"). */
export function formatCount(v: number | null | undefined): string {
  if (!isNum(v)) return NA;
  return `${v < 0 ? "-" : ""}${compactMagnitude(v)}`;
}

/** Indexed values are unit-less numbers rebased to 100. */
export function formatIndexed(v: number | null | undefined): string {
  if (!isNum(v)) return NA;
  return v.toFixed(1);
}

export type FormatUnit = UnitType | "index";

export interface FormatOptions {
  currency?: string | null;
  /** Reason to print when the value is missing. */
  reason?: MissingReason | null;
}

/** One entry point for cells, tooltips and labels. A null value always
 *  comes back as "n/a (reason)" so no caller can forget the reason. */
export function formatValue(v: number | null | undefined, unit: FormatUnit, opts: FormatOptions = {}): string {
  if (!isNum(v)) return formatMissing(opts.reason);
  switch (unit) {
    case "currency":
      return formatCurrency(v, opts.currency);
    case "percent":
      return formatPercent(v);
    case "ratio":
      return formatRatio(v);
    case "multiple":
      return formatMultiple(v);
    case "count":
      return formatCount(v);
    case "index":
      return formatIndexed(v);
    default:
      return formatRatio(v);
  }
}

/** Axis ticks: same units, fewer digits, so a crowded axis stays legible. */
export function axisTickFormatter(unit: FormatUnit, currency?: string | null): (v: number) => string {
  switch (unit) {
    case "currency":
      return (v) => (isNum(v) ? `${v < 0 ? "-" : ""}${currencySymbol(currency)}${compactMagnitude(v, { digits: 0 })}` : "");
    case "percent":
      // Plan §6.4 rule 7: a percent axis is labelled with 1 decimal, since
      // margin series often move by tenths of a point between ticks.
      return (v) => (isNum(v) ? `${(v * 100).toFixed(1)}%` : "");
    case "multiple":
      return (v) => (isNum(v) ? `${v.toFixed(0)}x` : "");
    case "count":
      return (v) => (isNum(v) ? compactMagnitude(v, { digits: 0 }) : "");
    case "index":
      return (v) => (isNum(v) ? v.toFixed(0) : "");
    default:
      return (v) => (isNum(v) ? v.toFixed(1) : "");
  }
}

/** Short axis / legend name for a unit. */
export function unitLabel(unit: FormatUnit, currency?: string | null): string {
  switch (unit) {
    case "currency":
      return currency ? currency.toUpperCase() : "currency";
    case "percent":
      return "%";
    case "multiple":
      return "x";
    case "ratio":
      return "ratio";
    case "count":
      return "count";
    case "index":
      return "index (base = 100)";
    default:
      return String(unit);
  }
}

/** Fallback label for a metric id when the catalog has not loaded. */
export function humanizeMetric(id: string): string {
  return id.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}
