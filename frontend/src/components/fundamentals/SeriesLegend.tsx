import React from "react";
import type { MetricSeries } from "@/types/fundamentals";
import { seriesName } from "@/lib/fundamentals/a11y";
import { seriesId, statusCounts } from "@/lib/fundamentals/transform";
import { colorForTicker, dashForMetric, STALE_OPACITY } from "./chart/palette";

/**
 * One chip per series: a swatch carrying the company colour and metric dash,
 * the series name, and status chips — `n missing`, `n estimated`, `stale`
 * (with the reason), `not drawn` for a series without observed points.
 * Status is always text, never colour alone, so it survives a screen
 * reader, print, and colour-vision deficiency.
 */
export interface SeriesLegendProps {
  series: MetricSeries[];
  metricLabels?: Record<string, string>;
  /** Series ids the layout excluded, keyed to the reason. */
  excluded?: Array<{ id: string; reason: string }>;
  className?: string;
}

function Swatch({ color, dash, faded }: { color: string; dash: string; faded: boolean }) {
  return (
    <svg width="28" height="10" aria-hidden="true" className="shrink-0">
      <line x1="1" y1="5" x2="27" y2="5" stroke={color} strokeWidth="2" strokeDasharray={dash || undefined} strokeOpacity={faded ? STALE_OPACITY : 1} strokeLinecap="round" />
    </svg>
  );
}

function Chip({ children, title, tone = "neutral" }: { children: React.ReactNode; title?: string; tone?: "neutral" | "warn" }) {
  const cls = tone === "warn" ? "border-warn-500/40 bg-warn-500/10 text-warn-500" : "border-ink-700 bg-ink-900/60 text-slate-300";
  return (
    <span className={`inline-flex items-center rounded px-1.5 py-0.5 text-[11px] border ${cls}`} title={title}>
      {children}
    </span>
  );
}

export default function SeriesLegend({ series, metricLabels, excluded = [], className = "" }: SeriesLegendProps) {
  const tickers = Array.from(new Set(series.map((s) => s.ticker)));
  const metrics = Array.from(new Set(series.map((s) => s.metric)));
  const excludedById = new Map(excluded.map((e) => [e.id, e.reason]));
  return (
    <ul aria-label="Series" className={`flex flex-wrap gap-x-4 gap-y-2 text-xs ${className}`}>
      {series.map((s) => {
        const id = seriesId(s);
        const counts = statusCounts(s);
        const notDrawn = excludedById.get(id);
        return (
          <li key={id} data-series-id={id} className="flex items-center gap-2">
            <Swatch color={colorForTicker(s.ticker, tickers)} dash={dashForMetric(s.metric, metrics)} faded={counts.stale || !!notDrawn} />
            <span className="text-slate-200">{seriesName(s, metricLabels)}</span>
            {counts.missing > 0 && (
              <Chip title="Points with no value; the table shows the reason for each">
                {counts.missing} missing
              </Chip>
            )}
            {counts.estimated > 0 && (
              <Chip title="Values computed with a documented fallback, drawn as hollow markers">
                {counts.estimated} estimated
              </Chip>
            )}
            {counts.stale && (
              <Chip tone="warn" title={s.provenance.stale_reason ?? "Stored data may be out of date"}>
                stale
                {s.provenance.stale_reason ? <span className="sr-only">: {s.provenance.stale_reason}</span> : null}
              </Chip>
            )}
            {notDrawn && (
              <Chip tone="warn" title={notDrawn}>
                not drawn: {notDrawn}
              </Chip>
            )}
          </li>
        );
      })}
    </ul>
  );
}
