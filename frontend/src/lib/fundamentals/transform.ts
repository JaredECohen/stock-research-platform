// Pure data transforms for the Fundamentals Explorer chart engine: a shared
// x-axis across series, status counts, indexing (rebase to 100) and the
// scale-ratio check the layout engine uses. No React, no recharts.
//
// Gaps are preserved, never interpolated: a period a series does not have
// stays `null` with a reason, and the chart breaks the line there.
import type { MetricPoint, MetricSeries, MissingReason } from "@/types/fundamentals";

export function seriesId(s: { ticker: string; metric: string }): string {
  return `${s.ticker}:${s.metric}`;
}

export function splitSeriesId(id: string): { ticker: string; metric: string } {
  const i = id.indexOf(":");
  return i < 0 ? { ticker: id, metric: "" } : { ticker: id.slice(0, i), metric: id.slice(i + 1) };
}

export function isObserved(p: MetricPoint | undefined | null): p is MetricPoint & { value: number } {
  return !!p && typeof p.value === "number" && Number.isFinite(p.value);
}

export function observedPoints(points: MetricPoint[]): Array<MetricPoint & { value: number }> {
  return points.filter(isObserved);
}

export function firstObserved(points: MetricPoint[]): (MetricPoint & { value: number }) | null {
  return points.find(isObserved) ?? null;
}

export function lastObserved(points: MetricPoint[]): (MetricPoint & { value: number }) | null {
  for (let i = points.length - 1; i >= 0; i--) {
    const p = points[i];
    if (isObserved(p)) return p;
  }
  return null;
}

/** A series worth drawing has at least one observed point. */
export function isDrawable(s: MetricSeries): boolean {
  return s.points.some(isObserved);
}

export interface StatusCounts {
  observed: number;
  missing: number;
  estimated: number;
  stale: boolean;
}

export function statusCounts(s: MetricSeries): StatusCounts {
  let observed = 0;
  let missing = 0;
  let estimated = 0;
  for (const p of s.points) {
    if (isObserved(p)) {
      observed += 1;
      if (p.estimated) estimated += 1;
    } else {
      missing += 1;
    }
  }
  return { observed, missing, estimated, stale: !!s.provenance?.stale };
}

/** One row per fiscal period on the shared x-axis. Values are keyed by
 *  series id in nested records (not flat keys) because tickers can contain
 *  "." (BRK.B) and recharts resolves string dataKeys as lodash paths —
 *  callers pass a function dataKey instead. */
export interface ChartRow {
  period: string;
  period_end: string;
  values: Record<string, number | null>;
  estimated: Record<string, boolean>;
  reasons: Record<string, MissingReason | null>;
}

function periodSortKey(p: { period: string; period_end: string }): string {
  // period_end is ISO, so lexical order is chronological; the label breaks
  // ties for malformed dates rather than throwing.
  return `${p.period_end || ""}|${p.period}`;
}

/** Union of every period across the series, chronological. A series that
 *  lacks a period gets `null` with reason null ("not obtained") so the row
 *  count is identical for every series in the panel. */
export function buildRows(series: MetricSeries[]): ChartRow[] {
  const byPeriod = new Map<string, ChartRow>();
  for (const s of series) {
    const id = seriesId(s);
    for (const p of s.points) {
      let row = byPeriod.get(p.period);
      if (!row) {
        row = { period: p.period, period_end: p.period_end, values: {}, estimated: {}, reasons: {} };
        byPeriod.set(p.period, row);
      }
      row.values[id] = isObserved(p) ? p.value : null;
      row.estimated[id] = isObserved(p) && !!p.estimated;
      row.reasons[id] = isObserved(p) ? null : (p.reason ?? null);
    }
  }
  const rows = Array.from(byPeriod.values()).sort((a, b) => (periodSortKey(a) < periodSortKey(b) ? -1 : periodSortKey(a) > periodSortKey(b) ? 1 : 0));
  for (const s of series) {
    const id = seriesId(s);
    for (const row of rows) {
      if (!(id in row.values)) {
        row.values[id] = null;
        row.estimated[id] = false;
        row.reasons[id] = null;
      }
    }
  }
  return rows;
}

export function periodsOf(series: MetricSeries[]): string[] {
  return buildRows(series).map((r) => r.period);
}

export interface IndexResult {
  /** Period every indexed series is rebased to, or null when none exists. */
  basePeriod: string | null;
  series: MetricSeries[];
  skipped: Array<{ id: string; reason: string }>;
}

/**
 * Rebase each series to `base` at the first period where *every* series is
 * observed and positive, so the lines share one starting point and the
 * reader compares trajectories rather than magnitudes. Points before the
 * base period are blanked with `before_index_base`; missing points keep
 * their original reason. Non-positive bases are refused rather than
 * producing a sign-flipped index.
 */
export function indexSeries(series: MetricSeries[], base = 100): IndexResult {
  const skipped: IndexResult["skipped"] = [];
  const candidates: MetricSeries[] = [];
  for (const s of series) {
    if (isDrawable(s)) candidates.push(s);
    else skipped.push({ id: seriesId(s), reason: "no observed points to index" });
  }
  if (candidates.length === 0) return { basePeriod: null, series: [], skipped };

  const rows = buildRows(candidates);
  const ids = candidates.map(seriesId);
  const baseRow = rows.find((r) => ids.every((id) => typeof r.values[id] === "number" && (r.values[id] as number) > 0));
  if (!baseRow) {
    for (const id of ids) skipped.push({ id, reason: "no period where every series is observed and positive" });
    return { basePeriod: null, series: [], skipped };
  }

  const baseIdx = rows.findIndex((r) => r.period === baseRow.period);
  const before = new Set(rows.slice(0, baseIdx).map((r) => r.period));
  const out = candidates.map((s) => {
    const id = seriesId(s);
    const baseValue = baseRow.values[id] as number;
    const points: MetricPoint[] = s.points.map((p) => {
      if (before.has(p.period)) {
        return { ...p, value: null, reason: isObserved(p) ? "before_index_base" : (p.reason ?? null) };
      }
      if (!isObserved(p)) return { ...p };
      return { ...p, value: (p.value / baseValue) * base };
    });
    return { ...s, points };
  });
  return { basePeriod: baseRow.period, series: out, skipped };
}

/** max/min of the latest observed *positive* value across the series, or
 *  null when fewer than two series have one. Drives the "these scales differ
 *  20×" suggestion; signs are excluded because a ratio across a sign change
 *  is meaningless. */
export function latestValueRatio(series: MetricSeries[]): number | null {
  const latest: number[] = [];
  for (const s of series) {
    const p = lastObserved(s.points);
    if (p && p.value > 0) latest.push(p.value);
  }
  if (latest.length < 2) return null;
  return Math.max(...latest) / Math.min(...latest);
}

/** Percentage change from the first to the last observed point, or null
 *  unless both are positive: a ratio across ≤ 0 is undefined, and a series
 *  that crosses zero ("fell 300%") reads as nonsense, so callers describe
 *  those as "moved from X to Y" instead. */
export function observedChange(points: MetricPoint[]): { first: MetricPoint & { value: number }; last: MetricPoint & { value: number }; pct: number | null } | null {
  const first = firstObserved(points);
  const last = lastObserved(points);
  if (!first || !last) return null;
  const pct = first.value > 0 && last.value > 0 ? last.value / first.value - 1 : null;
  return { first, last, pct };
}
