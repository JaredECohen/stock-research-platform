import React, { useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useReducedMotion } from "@/components/public/hooks";
import type { ScorecardHistory, ScorecardHistoryPoint } from "@/types/scorecard";
import { fmtCoverage, fmtPercentile, fmtScore, isNum, na } from "./format";

/**
 * Month-end history of one ticker's scorecard as a line: universe
 * percentile by default, sector percentile or the 0–100 score on request.
 * A month the run could not rank is a gap in the line (`connectNulls`
 * off) and a counted "missing point" in the label, never a drop to zero.
 * The container is `role="img"` with a generated sentence as its label;
 * "View as table" swaps in the accessible equivalent.
 */
export type HistoryMetric = "universe_percentile" | "sector_percentile" | "overall_score";

export interface ScorecardHistoryChartProps {
  history: ScorecardHistory | null | undefined;
  metric?: HistoryMetric;
  /** Fixed size (tests, print); otherwise fills the container. */
  width?: number;
  height?: number;
  className?: string;
}

const METRIC_LABEL: Record<HistoryMetric, string> = {
  universe_percentile: "universe percentile",
  sector_percentile: "sector percentile",
  overall_score: "overall score",
};

const FOCUS = "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500";
const STROKE = "#52E0C4";
const GRID = "#243056";
const AXIS = "#94a3b8";

function fmtMetric(metric: HistoryMetric, v: number | null): string {
  return metric === "overall_score" ? fmtScore(v, "not scored") : fmtPercentile(v, "not ranked");
}

/** "COST universe percentile (fs-v1), 2023-09-30 to 2026-08-31, 36
 *  month-ends, 2 missing points; latest 72nd (2026-08-31)." */
export function historyLabel(h: ScorecardHistory, metric: HistoryMetric): string {
  const pts = h.points;
  if (pts.length === 0) return `${h.ticker} ${METRIC_LABEL[metric]} (${h.version_key}): no month-end history yet.`;
  const first = pts[0].as_of;
  const last = pts[pts.length - 1].as_of;
  const missing = pts.filter((p) => !isNum(p[metric])).length;
  const latest = [...pts].reverse().find((p) => isNum(p[metric]));
  const parts = [`${h.ticker} ${METRIC_LABEL[metric]} (${h.version_key})`, `${first} to ${last}`, `${pts.length} month-end${pts.length === 1 ? "" : "s"}`];
  if (missing > 0) parts.push(`${missing} missing point${missing === 1 ? "" : "s"}`);
  const tail = latest ? `latest ${fmtMetric(metric, latest[metric])} (${latest.as_of})` : "no ranked month";
  return `${parts.join(", ")}; ${tail}.`;
}

export default function ScorecardHistoryChart({ history, metric = "universe_percentile", width, height = 240, className = "" }: ScorecardHistoryChartProps) {
  const reduced = useReducedMotion();
  const [tableShown, setTableShown] = useState(false);
  const points = history?.points ?? [];
  const label = useMemo(() => (history ? historyLabel(history, metric) : ""), [history, metric]);
  const rows = useMemo(() => points.map((p) => ({ as_of: p.as_of, value: isNum(p[metric]) ? (p[metric] as number) : null, point: p })), [points, metric]);

  if (!history || points.length === 0) {
    return (
      <div className={`card-tight text-xs text-slate-400 ${className}`} role="status" data-testid="history-empty">
        {history ? `${history.ticker}: ` : ""}
        {na("no month-end history yet")}. History accrues one point per month-end run.
      </div>
    );
  }

  const isPct = metric !== "overall_score";
  const chart = (
    <LineChart data={rows} width={width} height={width ? height : undefined} margin={{ top: 12, right: 16, left: 4, bottom: 4 }}>
      <CartesianGrid stroke={GRID} vertical={false} />
      <XAxis dataKey="as_of" stroke={AXIS} tick={{ fontSize: 11 }} tickLine={false} minTickGap={24} />
      <YAxis domain={[0, 100]} stroke={AXIS} tick={{ fontSize: 11 }} tickLine={false} width={40} label={{ value: isPct ? "pct" : "score", angle: -90, position: "insideLeft", fill: AXIS, fontSize: 10 }} />
      <Tooltip
        filterNull={false}
        contentStyle={{ background: "#0E1525", border: `1px solid ${GRID}`, borderRadius: 8, fontSize: 12 }}
        labelStyle={{ color: "#e2e8f0" }}
        itemStyle={{ color: "#cbd5e1" }}
        formatter={(value: unknown, _name: unknown, item: { payload?: { point?: ScorecardHistoryPoint } }) => {
          const p = item?.payload?.point;
          const text = fmtMetric(metric, typeof value === "number" ? value : null);
          return [p ? `${text} · coverage ${fmtCoverage(p.coverage)}` : text, METRIC_LABEL[metric]];
        }}
      />
      <Line type="monotone" dataKey="value" name={METRIC_LABEL[metric]} stroke={STROKE} strokeWidth={2} connectNulls={false} isAnimationActive={!reduced} dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
    </LineChart>
  );

  return (
    <figure className={`card-tight min-w-0 ${className}`} data-testid="scorecard-history">
      <div className="flex items-center justify-between gap-2 mb-1">
        <figcaption className="text-sm font-medium text-slate-200">
          {history.ticker} · {METRIC_LABEL[metric]} · month-ends
        </figcaption>
        <button type="button" aria-pressed={tableShown} onClick={() => setTableShown((v) => !v)} className={`btn-ghost !py-1 !px-2 text-xs ${FOCUS}`} data-testid="history-table-toggle">
          {tableShown ? "View as chart" : "View as table"}
        </button>
      </div>
      {tableShown ? (
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs" data-testid="history-table">
            <caption className="sr-only">{label}</caption>
            <thead className="uppercase tracking-wider text-slate-400">
              <tr>
                <th scope="col" className="text-left px-2 py-1">
                  Month-end
                </th>
                <th scope="col" className="text-right px-2 py-1">
                  {METRIC_LABEL[metric]}
                </th>
                <th scope="col" className="text-right px-2 py-1">
                  Coverage
                </th>
              </tr>
            </thead>
            <tbody>
              {points.map((p) => (
                <tr key={p.as_of} className="border-t border-ink-700/60">
                  <th scope="row" className="text-left px-2 py-1 font-normal font-mono text-slate-200">
                    {p.as_of}
                  </th>
                  <td className="px-2 py-1 text-right font-mono text-slate-300" data-missing={isNum(p[metric]) ? undefined : "true"}>
                    {fmtMetric(metric, p[metric])}
                  </td>
                  <td className="px-2 py-1 text-right font-mono text-slate-400">{fmtCoverage(p.coverage)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div role="img" aria-label={label} tabIndex={0} className={`rounded-md ${FOCUS}`} style={{ height: width ? undefined : height }} data-animate={reduced ? "off" : "on"}>
          {width ? chart : <ResponsiveContainer width="100%" height="100%">{chart}</ResponsiveContainer>}
        </div>
      )}
    </figure>
  );
}
