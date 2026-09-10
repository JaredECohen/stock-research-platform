import React, { useMemo, useState } from "react";
import type { MetricSeries } from "@/types/fundamentals";
import { seriesName, tableCaption } from "@/lib/fundamentals/a11y";
import { formatValue, unitLabel } from "@/lib/fundamentals/format";
import { buildRows, seriesId, type ChartRow } from "@/lib/fundamentals/transform";

/**
 * The accessible equivalent of the chart: one row per fiscal period, one
 * column per series. Headers sort (click or Enter/Space) and announce it
 * through `aria-sort`; nulls sort last in either direction. A missing cell
 * reads "n/a (reason)"; an estimated cell is prefixed "≈" with an sr-only
 * "estimated" so the two never look like plain observed numbers.
 */
export interface SeriesTableProps {
  series: MetricSeries[];
  metricLabels?: Record<string, string>;
  /** Present when the series were rebased for an indexed view. */
  indexBasePeriod?: string | null;
  caption?: string;
  className?: string;
}

type SortKey = "period" | string;
type SortDir = "ascending" | "descending";

const FOCUS = "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 rounded-sm";

function compareRows(a: ChartRow, b: ChartRow, key: SortKey, dir: SortDir): number {
  const sign = dir === "ascending" ? 1 : -1;
  if (key === "period") {
    const ka = `${a.period_end}|${a.period}`;
    const kb = `${b.period_end}|${b.period}`;
    return ka < kb ? -sign : ka > kb ? sign : 0;
  }
  const va = a.values[key];
  const vb = b.values[key];
  // Missing values sort last regardless of direction — a gap is not a
  // small number.
  if (va === null && vb === null) return 0;
  if (va === null) return 1;
  if (vb === null) return -1;
  return (va - vb) * sign;
}

export default function SeriesTable({ series, metricLabels, indexBasePeriod = null, caption, className = "" }: SeriesTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("period");
  const [sortDir, setSortDir] = useState<SortDir>("ascending");
  const rows = useMemo(() => buildRows(series), [series]);
  const sorted = useMemo(() => [...rows].sort((a, b) => compareRows(a, b, sortKey, sortDir)), [rows, sortKey, sortDir]);
  const unit = indexBasePeriod ? "index" : null;

  function toggle(key: SortKey) {
    if (key === sortKey) setSortDir((d) => (d === "ascending" ? "descending" : "ascending"));
    else {
      setSortKey(key);
      setSortDir("ascending");
    }
  }

  const headerButton = (key: SortKey, label: React.ReactNode, align: "left" | "right") => (
    <button
      type="button"
      onClick={() => toggle(key)}
      className={`inline-flex items-center gap-1 ${align === "right" ? "justify-end w-full" : ""} ${FOCUS}`}
    >
      <span>{label}</span>
      <span aria-hidden="true" className="text-slate-500">
        {sortKey === key ? (sortDir === "ascending" ? "▲" : "▼") : "↕"}
      </span>
    </button>
  );

  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="min-w-full text-sm" data-testid="series-table">
        <caption className="text-left text-xs text-slate-400 mb-2">{caption ?? tableCaption(series, metricLabels, indexBasePeriod)}</caption>
        <thead className="text-xs uppercase tracking-wider text-slate-400">
          <tr>
            <th scope="col" aria-sort={sortKey === "period" ? sortDir : "none"} className="text-left px-2 py-1 whitespace-nowrap">
              {headerButton("period", "Period", "left")}
            </th>
            {series.map((s) => {
              const id = seriesId(s);
              return (
                <th key={id} scope="col" aria-sort={sortKey === id ? sortDir : "none"} className="text-right px-2 py-1 whitespace-nowrap">
                  {headerButton(
                    id,
                    <>
                      {seriesName(s, metricLabels)}
                      <span className="block normal-case tracking-normal text-[10px] text-slate-500">{unitLabel(unit ?? s.unit_type, s.currency)}</span>
                    </>,
                    "right",
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={row.period} className="text-slate-300 border-t border-ink-700/60">
              <th scope="row" className="text-left px-2 py-1 font-mono font-normal whitespace-nowrap">
                {row.period}
                <span className="sr-only"> ending {row.period_end}</span>
              </th>
              {series.map((s) => {
                const id = seriesId(s);
                const v = row.values[id];
                const text = formatValue(v, unit ?? s.unit_type, { currency: s.currency, reason: row.reasons[id] });
                const est = row.estimated[id];
                return (
                  <td key={id} className={`px-2 py-1 text-right font-mono whitespace-nowrap ${v === null ? "text-slate-500" : ""}`} data-missing={v === null ? "true" : undefined}>
                    {est ? (
                      <span title="estimated">
                        <span aria-hidden="true">≈ </span>
                        <span className="sr-only">estimated </span>
                        {text}
                      </span>
                    ) : (
                      text
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
          {sorted.length === 0 && (
            <tr>
              <td colSpan={series.length + 1} className="px-2 py-3 text-slate-500">
                No periods to show.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
