// Screen-reader text for the chart engine: the `aria-label` sentence on
// each chart panel, the table caption, and series names. Recharts tooltips
// are mouse-only, so these sentences plus the data table are the accessible
// equivalent of the chart. Everything here is derived from observed points;
// a missing point is counted and named, never described as flat or zero.
import type { MetricSeries } from "@/types/fundamentals";
import { formatValue, humanizeMetric, type FormatUnit } from "./format";
import { observedChange, seriesId, statusCounts } from "./transform";
import type { PanelSpec } from "./layout";

export function joinNames(names: string[]): string {
  if (names.length === 0) return "";
  if (names.length === 1) return names[0];
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export function metricLabel(metric: string, labels?: Record<string, string>): string {
  return labels?.[metric] ?? humanizeMetric(metric);
}

/** "AAPL Revenue" — how a series is named in legends, headers and labels. */
export function seriesName(s: { ticker: string; metric: string }, labels?: Record<string, string>): string {
  return `${s.ticker} ${metricLabel(s.metric, labels)}`;
}

/** Units whose change is described in the unit itself rather than as a
 *  relative percentage: "gross margin rose 21%" for 38.2% → 46.2% is heard
 *  as a 21-point rise, so percent series report percentage points, and
 *  ratios/multiples report the plain difference. `flat` is half of the
 *  displayed precision, so a change that would print as "0.0 points" is
 *  called flat rather than a rise. */
const ABSOLUTE_CHANGE: Partial<Record<FormatUnit, { flat: number; delta: (d: number) => string }>> = {
  percent: { flat: 0.0005, delta: (d) => `${(Math.abs(d) * 100).toFixed(1)} points` },
  ratio: { flat: 0.005, delta: (d) => Math.abs(d).toFixed(2) },
  multiple: { flat: 0.05, delta: (d) => `${Math.abs(d).toFixed(1)}x` },
};

/** "rose 42% from FY2020 to FY2024" (currency, count, index), "rose 8.0
 *  points from 38.2% (FY2020) to 46.2% (FY2024)" (percent), "fell 1.2x from
 *  14.0x (FY2020) to 12.8x (FY2024)" (multiple), "moved from -$1.2B to $3.4B"
 *  when a relative change is undefined (sign change or non-positive start),
 *  or "has no observed points". Uses the drawn unit so an indexed panel
 *  reads in index points. */
export function describeChange(s: MetricSeries, unit: FormatUnit = s.unit_type): string {
  const change = observedChange(s.points);
  if (!change) return "has no observed points";
  const { first, last, pct } = change;
  const fmt = (v: number) => formatValue(v, unit, { currency: s.currency });
  if (first.period === last.period) return `has one observed point (${fmt(first.value)} in ${first.period})`;
  const absolute = ABSOLUTE_CHANGE[unit];
  if (absolute) {
    const delta = last.value - first.value;
    if (Math.abs(delta) < absolute.flat) return `was flat from ${first.period} to ${last.period}`;
    return `${delta > 0 ? "rose" : "fell"} ${absolute.delta(delta)} from ${fmt(first.value)} (${first.period}) to ${fmt(last.value)} (${last.period})`;
  }
  if (pct === null) {
    return `moved from ${fmt(first.value)} (${first.period}) to ${fmt(last.value)} (${last.period})`;
  }
  const abs = Math.abs(pct * 100);
  const word = pct > 0.0005 ? "rose" : pct < -0.0005 ? "fell" : "was flat";
  if (word === "was flat") return `was flat from ${first.period} to ${last.period}`;
  const digits = abs >= 10 ? 0 : 1;
  return `${word} ${abs.toFixed(digits)}% from ${first.period} to ${last.period}`;
}

/** Full `aria-label` for one panel: "Revenue, FY2020 to FY2024, AAPL and
 *  MSFT; AAPL rose 42%, MSFT rose 61%; 1 missing point". */
export function panelLabel(panel: PanelSpec, series: MetricSeries[], labels?: Record<string, string>): string {
  const byId = new Map(series.map((s) => [seriesId(s), s]));
  const inPanel = panel.seriesIds.map((id) => byId.get(id)).filter((s): s is MetricSeries => !!s);
  const metrics = joinNames(Array.from(new Set(inPanel.map((s) => metricLabel(s.metric, labels)))));
  const tickers = joinNames(Array.from(new Set(inPanel.map((s) => s.ticker))));
  const periods = inPanel.flatMap((s) => s.points.map((p) => p.period)).sort();
  const span = periods.length ? `${periods[0]} to ${periods[periods.length - 1]}` : "no periods";
  const oneMetric = new Set(inPanel.map((s) => s.metric)).size === 1;
  const unit = panel.indexed ? "index" : undefined;
  const changes = inPanel.map((s) => `${oneMetric ? s.ticker : seriesName(s, labels)} ${describeChange(s, unit ?? s.unit_type)}`);
  const parts = [`${metrics}${panel.indexed ? " (indexed to 100)" : ""}, ${span}, ${tickers}`, changes.join(", ")];
  let missing = 0;
  let estimated = 0;
  let stale = 0;
  for (const s of inPanel) {
    const c = statusCounts(s);
    missing += c.missing;
    estimated += c.estimated;
    if (c.stale) stale += 1;
  }
  const flags: string[] = [];
  if (missing) flags.push(`${missing} missing point${missing === 1 ? "" : "s"}`);
  if (estimated) flags.push(`${estimated} estimated point${estimated === 1 ? "" : "s"}`);
  if (stale) flags.push(`${stale} stale series`);
  if (flags.length) parts.push(flags.join(", "));
  return parts.join("; ");
}

/** `<caption>` for the data table. */
export function tableCaption(series: MetricSeries[], labels?: Record<string, string>, indexBasePeriod: string | null = null): string {
  const metrics = joinNames(Array.from(new Set(series.map((s) => metricLabel(s.metric, labels)))));
  const tickers = joinNames(Array.from(new Set(series.map((s) => s.ticker))));
  const periods = series.flatMap((s) => s.points.map((p) => p.period)).sort();
  const span = periods.length ? `${periods[0]} to ${periods[periods.length - 1]}` : "no periods";
  const idx = indexBasePeriod ? `, indexed to 100 at ${indexBasePeriod}` : "";
  return `${metrics || "No metrics"} for ${tickers || "no companies"}, ${span}, annual${idx}. n/a means the value was not obtained; the reason follows in parentheses. ≈ marks an estimated value.`;
}
