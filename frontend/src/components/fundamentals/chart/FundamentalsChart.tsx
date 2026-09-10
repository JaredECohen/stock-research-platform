import React, { useEffect, useMemo, useState } from "react";
import type { MetricSeries, ViewMode } from "@/types/fundamentals";
import { computeLayout, type LayoutResult } from "@/lib/fundamentals/layout";
import { useReducedMotion } from "@/components/public/hooks";
import SeriesLegend from "../SeriesLegend";
import SeriesTable from "../SeriesTable";
import ChartPanel from "./ChartPanel";

/**
 * The chart engine's presentational root. Runs the layout rules on the
 * displayed series, renders one `ChartPanel` per resolved panel (shared
 * axis, dual axis, small multiples, or indexed), the legend, and the
 * "View as table" toggle whose table is the accessible equivalent of the
 * chart. No fetching, no routing: the page owns state and passes props.
 */
export interface FundamentalsChartProps {
  series: MetricSeries[];
  view: ViewMode;
  metricLabels?: Record<string, string>;
  /** Selection order of metrics (drives dash assignment and axis sides). */
  metricOrder?: string[];
  /** Selection order of tickers (drives colour assignment). */
  tickerOrder?: string[];
  /** Viewport below `md`; dual axis collapses to small multiples. */
  narrow?: boolean;
  /** Fixed panel size (tests, print). */
  width?: number;
  height?: number;
  /** Called with the layout each time it is recomputed, so the page can
   *  disable view-mode options with the engine's reason. */
  onLayout?: (layout: LayoutResult) => void;
  /** Suggestion banner action (e.g. switch the URL view). */
  onSuggestion?: (mode: "indexed" | "small-multiples") => void;
  className?: string;
}

const FOCUS = "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500";

export default function FundamentalsChart({
  series,
  view,
  metricLabels,
  metricOrder,
  tickerOrder,
  narrow = false,
  width,
  height,
  onLayout,
  onSuggestion,
  className = "",
}: FundamentalsChartProps) {
  const reduced = useReducedMotion();
  const [tableShown, setTableShown] = useState(view === "table");
  useEffect(() => {
    // The URL is the source of truth: `v=table` forces the table; switching
    // to a chart view shows the chart again.
    setTableShown(view === "table");
  }, [view]);

  const layout = useMemo(() => computeLayout(series, { view, narrow, metricLabels, metricOrder }), [series, view, narrow, metricLabels, metricOrder]);
  useEffect(() => {
    onLayout?.(layout);
  }, [layout, onLayout]);

  const tickers = useMemo(() => {
    const seen = Array.from(new Set(series.map((s) => s.ticker)));
    const ordered = (tickerOrder ?? []).filter((t) => seen.includes(t));
    for (const t of seen) if (!ordered.includes(t)) ordered.push(t);
    return ordered;
  }, [series, tickerOrder]);
  const metrics = useMemo(() => {
    const seen = Array.from(new Set(series.map((s) => s.metric)));
    const ordered = (metricOrder ?? []).filter((m) => seen.includes(m));
    for (const m of seen) if (!ordered.includes(m)) ordered.push(m);
    return ordered;
  }, [series, metricOrder]);

  const multi = layout.panels.length > 1;
  const nothing = layout.panels.length === 0;

  return (
    <div className={`space-y-3 ${className}`} data-resolved-mode={layout.resolved} data-testid="fundamentals-chart">
      {(layout.fallback || layout.warnings.length > 0) && (
        <div role="status" className="text-xs text-slate-300 space-y-1">
          {layout.fallback && (
            <p data-testid="layout-fallback">
              The <strong>{layout.fallback.from}</strong> view is not available here ({layout.fallback.reason}); showing{" "}
              {layout.resolved === "shared" ? "a shared axis" : layout.resolved.replace("-", " ")} instead.
            </p>
          )}
          {layout.warnings.map((w) => (
            <p key={w} data-testid="layout-warning">
              {w}
            </p>
          ))}
        </div>
      )}

      {layout.suggestion && !tableShown && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300 border border-ink-700 rounded-lg px-3 py-2" data-testid="layout-suggestion">
          <span>{layout.suggestion.reason}</span>
          {onSuggestion && (
            <button type="button" className={`btn-ghost !py-1 !px-2 text-xs ${FOCUS}`} onClick={() => onSuggestion(layout.suggestion!.mode)}>
              Switch to {layout.suggestion.mode === "indexed" ? "indexed" : "small multiples"}
            </button>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2">
        <SeriesLegend series={series} metricLabels={metricLabels} excluded={layout.excluded} />
        {!nothing && (
          <button
            type="button"
            aria-pressed={tableShown}
            onClick={() => setTableShown((v) => !v)}
            className={`btn-ghost !py-1 !px-2 text-xs ${FOCUS}`}
            data-testid="table-toggle"
          >
            {tableShown ? "View as chart" : "View as table"}
          </button>
        )}
      </div>

      {nothing ? (
        <p className="text-sm text-slate-400" data-testid="chart-empty">
          No observed points to draw. Each series above says why.
        </p>
      ) : tableShown ? (
        <SeriesTable series={layout.drawn} metricLabels={metricLabels} indexBasePeriod={layout.indexBasePeriod} />
      ) : (
        <div className={multi ? "grid gap-3 grid-cols-1 md:grid-cols-2 xl:grid-cols-3" : ""}>
          {layout.panels.map((panel) => (
            <ChartPanel
              key={panel.id}
              panel={panel}
              series={layout.drawn}
              tickers={tickers}
              metrics={metrics}
              metricLabels={metricLabels}
              animate={!reduced}
              width={width}
              height={height}
              onToggleTable={() => setTableShown(true)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
