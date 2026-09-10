import React, { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmtCurrency } from "@/lib/format";
import type { FundamentalsSeries } from "@/types/public";
import { FOCUS_RING } from "./ctas";
import { useReducedMotion } from "./hooks";

/**
 * Headline lines from `financial_periods` as stored by the worker. One
 * metric at a time (buttons, `aria-pressed`), with the numbers under the
 * chart in a real table because a bar chart is not something a screen
 * reader can read and jsdom cannot lay out. A missing value is "n/a
 * (not obtained)" — the research rule that a blank is not a zero.
 */
const METRIC_LABELS: Record<string, string> = {
  revenue: "Revenue",
  gross_profit: "Gross profit",
  operating_income: "Operating income",
  net_income: "Net income",
  free_cash_flow: "Free cash flow",
};

export function metricLabel(metric: string): string {
  return METRIC_LABELS[metric] || metric.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

function compact(v: number): string {
  return fmtCurrency(v, { compact: true });
}

export default function SampleFundamentalsChart({
  fundamentals,
  headingLevel = 2,
}: {
  fundamentals: { series: FundamentalsSeries[] };
  headingLevel?: 2 | 3;
}) {
  const H = `h${headingLevel}` as "h2" | "h3";
  const reduced = useReducedMotion();
  const series = fundamentals.series.filter((s) => Array.isArray(s.points) && s.points.length > 0);
  const [metric, setMetric] = useState<string>(series[0]?.metric || "");
  const active = series.find((s) => s.metric === metric) || series[0];

  const data = useMemo(
    () => (active ? active.points.map((p) => ({ period: p.period, value: typeof p.value === "number" ? p.value : null })) : []),
    [active],
  );

  if (!active) {
    return (
      <section aria-labelledby="sample-fundamentals-heading">
        <H id="sample-fundamentals-heading" className="text-lg font-semibold mb-2">Fundamentals</H>
        <p className="text-sm text-slate-400">No fundamentals series stored for this sample.</p>
      </section>
    );
  }

  return (
    <section aria-labelledby="sample-fundamentals-heading">
      <div className="flex flex-wrap items-end justify-between gap-2 mb-3">
        <div>
          <H id="sample-fundamentals-heading" className="text-lg font-semibold">Fundamentals</H>
          <p className="text-xs text-slate-400 mt-0.5">
            {active.cadence === "quarterly" ? "Quarterly" : "Annual"} figures as reported, from stored financial periods.
          </p>
        </div>
        <div role="group" aria-label="Metric" className="flex flex-wrap gap-1.5">
          {series.map((s) => {
            const on = s.metric === active.metric;
            return (
              <button
                key={s.metric}
                type="button"
                aria-pressed={on}
                onClick={() => setMetric(s.metric)}
                className={`px-2.5 py-1 rounded-md border text-xs motion-safe:transition-colors ${FOCUS_RING} ${
                  on ? "border-accent-600/50 bg-accent-600/15 text-accent-500" : "border-ink-700 text-slate-300 hover:bg-ink-800"
                }`}
              >
                {metricLabel(s.metric)}
              </button>
            );
          })}
        </div>
      </div>

      <div className="h-56 card-tight" aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}>
            <CartesianGrid stroke="#243056" vertical={false} />
            <XAxis dataKey="period" stroke="#94a3b8" tick={{ fontSize: 11 }} />
            <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} tickFormatter={compact} width={64} />
            <Tooltip
              formatter={(v) => (typeof v === "number" ? compact(v) : "n/a")}
              contentStyle={{ background: "#0E1525", border: "1px solid #243056", borderRadius: 8, fontSize: 12 }}
            />
            <Bar dataKey="value" fill="#52E0C4" radius={[4, 4, 0, 0]} isAnimationActive={!reduced} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <details className="mt-2 text-sm">
        <summary className={`cursor-pointer text-slate-300 rounded-sm ${FOCUS_RING}`}>
          Show the numbers — {metricLabel(active.metric)}
        </summary>
        <div className="overflow-x-auto mt-2">
          <table className="min-w-[20rem] text-sm">
            <caption className="sr-only">{metricLabel(active.metric)} by period</caption>
            <thead className="text-xs uppercase tracking-wider text-slate-400">
              <tr>
                <th scope="col" className="text-left px-2 py-1">Period</th>
                <th scope="col" className="text-left px-2 py-1">Period end</th>
                <th scope="col" className="text-right px-2 py-1">{metricLabel(active.metric)}</th>
              </tr>
            </thead>
            <tbody>
              {active.points.map((p) => (
                <tr key={`${p.period}-${p.period_end ?? ""}`} className="text-slate-300">
                  <td className="px-2 py-1 font-mono">{p.period}</td>
                  <td className="px-2 py-1 font-mono">{p.period_end ?? "—"}</td>
                  <td className="px-2 py-1 text-right font-mono">
                    {typeof p.value === "number" ? compact(p.value) : "n/a (not obtained)"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}
