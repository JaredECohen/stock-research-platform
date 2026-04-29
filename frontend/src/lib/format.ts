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

export function ratingBadgeClass(rating: string): string {
  switch (rating) {
    case "Bullish":
      return "badge-bull";
    case "Mixed Positive":
      return "badge-bull";
    case "Mixed Negative":
      return "badge-mixed";
    case "Bearish":
      return "badge-bear";
    default:
      return "badge-neutral";
  }
}
