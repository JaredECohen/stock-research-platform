// Number formatting helpers used across the frontend.

export function fmtCurrency(v?: number | null, opts: { compact?: boolean } = {}): string {
  if (v == null || Number.isNaN(v)) return "—";
  if (opts.compact) {
    if (Math.abs(v) >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
    if (Math.abs(v) >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
    if (Math.abs(v) >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
    if (Math.abs(v) >= 1e3) return `$${(v / 1e3).toFixed(1)}K`;
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(v);
}

export function fmtPct(v?: number | null, fractionDigits = 1): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(fractionDigits)}%`;
}

export function fmtNumber(v?: number | null, fractionDigits = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toLocaleString("en-US", { maximumFractionDigits: fractionDigits });
}

export function fmtMultiple(v?: number | null): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${v.toFixed(1)}x`;
}

// DCF-specific formatters. The engine emits `null` for an implied price it
// could not compute (no diluted share count) and for an upside it could not
// compute (no implied price, or no positive quote). These render the same
// "n/a" the backend prints into memo prose, so a reader sees one spelling
// everywhere. Distinct from the "—" the generic helpers use for "field
// absent": n/a means "we ran the model and there is no number".
export const NA = "n/a";

// Coerce a loosely typed `dcf_summary` value into a number or null. The
// summary is a JSON bag on the memo, so a value may be a number, `null`
// (unavailable), or a string on very old memos. `Number(null)` is 0 — the
// exact lie this replaces — so never fall through to `Number()`.
export function numOrNull(v: unknown): number | null {
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

export function fmtPrice(v?: number | null): string {
  if (v == null || Number.isNaN(v)) return NA;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(v);
}

export function fmtUpside(v?: number | null, fractionDigits = 1): string {
  if (v == null || Number.isNaN(v)) return NA;
  const pct = (v * 100).toFixed(fractionDigits);
  return v > 0 ? `+${pct}%` : `${pct}%`;
}

export function ratingBadgeClass(rating: string): string {
  // Wave 8P — five-label rating ladder driven by the quant Stock Score.
  switch (rating) {
    case "Very Bullish":
      return "badge-bull";
    case "Bullish":
      return "badge-bull";
    case "Bearish":
      return "badge-bear";
    case "Very Bearish":
      return "badge-bear";
    // Legacy labels kept for back-compat with cached memos.
    case "Mixed Positive":
      return "badge-bull";
    case "Mixed Negative":
      return "badge-mixed";
    case "Neutral":
    default:
      return "badge-neutral";
  }
}
