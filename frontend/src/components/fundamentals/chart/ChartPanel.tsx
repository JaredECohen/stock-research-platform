import React, { useMemo } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MetricSeries } from "@/types/fundamentals";
import { panelLabel, seriesName } from "@/lib/fundamentals/a11y";
import { axisTickFormatter, formatValue, unitLabel } from "@/lib/fundamentals/format";
import type { AxisSpec, PanelSpec } from "@/lib/fundamentals/layout";
import { buildRows, seriesId, type ChartRow } from "@/lib/fundamentals/transform";
import { AXIS_STROKE, CHART_SURFACE, GRID_STROKE, STALE_OPACITY, colorForTicker, dashForMetric } from "./palette";

/**
 * One recharts panel for a `PanelSpec` from the layout engine: a shared or
 * dual axis, one line per series, gaps where a point is null
 * (`connectNulls={false}`), hollow markers on estimated points, reduced
 * opacity on stale series. The container is `role="img"` with a generated
 * sentence as its label; pressing T on it switches to the data table.
 */
export interface ChartPanelProps {
  panel: PanelSpec;
  series: MetricSeries[];
  /** Colour/dash assignment order — all tickers/metrics in the chart, not
   *  just this panel, so a company keeps its colour across panels. */
  tickers: string[];
  metrics: string[];
  metricLabels?: Record<string, string>;
  animate: boolean;
  /** Fixed size (tests, print); otherwise the panel fills its container. */
  width?: number;
  height?: number;
  onToggleTable?: () => void;
}

const FOCUS = "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500";

interface DotProps {
  key?: React.Key;
  cx?: number;
  cy?: number;
  value?: number | null;
  payload?: ChartRow;
}

function axisUnitOf(axes: AxisSpec[], id: string): AxisSpec {
  return axes.find((a) => a.seriesIds.includes(id)) ?? axes[0];
}

export default function ChartPanel({ panel, series, tickers, metrics, metricLabels, animate, width, height = 280, onToggleTable }: ChartPanelProps) {
  const inPanel = useMemo(() => {
    const byId = new Map(series.map((s) => [seriesId(s), s]));
    return panel.seriesIds.map((id) => byId.get(id)).filter((s): s is MetricSeries => !!s);
  }, [panel, series]);
  const rows = useMemo(() => buildRows(inPanel), [inPanel]);
  const label = useMemo(() => panelLabel(panel, inPanel, metricLabels), [panel, inPanel, metricLabels]);
  const left = panel.axes.find((a) => a.id === "left") ?? panel.axes[0];
  const right = panel.axes.find((a) => a.id === "right");

  const chart = (
    <LineChart data={rows} width={width} height={width ? height : undefined} margin={{ top: 12, right: right ? 8 : 16, left: 4, bottom: 4 }}>
      <CartesianGrid stroke={GRID_STROKE} vertical={false} />
      <XAxis dataKey="period" stroke={AXIS_STROKE} tick={{ fontSize: 11 }} tickLine={false} />
      <YAxis
        yAxisId="left"
        stroke={AXIS_STROKE}
        tick={{ fontSize: 11 }}
        tickLine={false}
        width={64}
        tickFormatter={axisTickFormatter(left.unit_type, left.currency)}
        label={{ value: unitLabel(left.unit_type, left.currency), angle: -90, position: "insideLeft", fill: AXIS_STROKE, fontSize: 10 }}
      />
      {right && (
        <YAxis
          yAxisId="right"
          orientation="right"
          stroke={AXIS_STROKE}
          tick={{ fontSize: 11 }}
          tickLine={false}
          width={56}
          tickFormatter={axisTickFormatter(right.unit_type, right.currency)}
          label={{ value: unitLabel(right.unit_type, right.currency), angle: 90, position: "insideRight", fill: AXIS_STROKE, fontSize: 10 }}
        />
      )}
      <Tooltip
        contentStyle={{ background: "#0E1525", border: `1px solid ${GRID_STROKE}`, borderRadius: 8, fontSize: 12 }}
        labelStyle={{ color: "#e2e8f0" }}
        itemStyle={{ color: "#cbd5e1" }}
        formatter={(value: unknown, _name: unknown, item: { dataKey?: unknown; payload?: ChartRow }) => {
          // Recharts hands back the resolved value; the row carries the
          // reason and estimate flag for this series id.
          const id = typeof item?.dataKey === "function" ? (item.dataKey as { seriesId?: string }).seriesId ?? "" : String(item?.dataKey ?? "");
          const s = inPanel.find((x) => seriesId(x) === id);
          if (!s) return [typeof value === "number" ? String(value) : "n/a", ""];
          const row = item?.payload;
          const unit = panel.indexed ? "index" : s.unit_type;
          const text = formatValue(typeof value === "number" ? value : null, unit, { currency: s.currency, reason: row?.reasons[id] });
          return [row?.estimated[id] ? `≈ ${text} (estimated)` : text, seriesName(s, metricLabels)];
        }}
      />
      {inPanel.map((s) => {
        const id = seriesId(s);
        const axis = axisUnitOf(panel.axes, id);
        const color = colorForTicker(s.ticker, tickers);
        const stale = !!s.provenance?.stale;
        // Function dataKey: series ids contain ":" and tickers may contain
        // ".", which recharts would otherwise parse as a lodash path.
        const dataKey = Object.assign((row: ChartRow) => row.values[id], { seriesId: id });
        return (
          <Line
            key={id}
            yAxisId={axis.id}
            dataKey={dataKey}
            name={seriesName(s, metricLabels)}
            stroke={color}
            strokeWidth={2}
            strokeDasharray={dashForMetric(s.metric, metrics) || undefined}
            strokeOpacity={stale ? STALE_OPACITY : 1}
            connectNulls={false}
            isAnimationActive={animate}
            activeDot={{ r: 5, strokeWidth: 0 }}
            dot={(p: DotProps) => {
              // Only estimated points get a marker: hollow, so the eye reads
              // "computed with a fallback" without a second colour.
              if (!p.payload?.estimated[id] || typeof p.cx !== "number" || typeof p.cy !== "number") return null as unknown as React.ReactElement;
              return (
                <circle
                  key={`${id}-${p.payload.period}`}
                  cx={p.cx}
                  cy={p.cy}
                  r={4}
                  fill={CHART_SURFACE}
                  stroke={color}
                  strokeWidth={2}
                  strokeOpacity={stale ? STALE_OPACITY : 1}
                  data-estimated="true"
                />
              );
            }}
          />
        );
      })}
    </LineChart>
  );

  return (
    <figure className="card-tight min-w-0" data-panel-id={panel.id}>
      <figcaption className="text-sm font-medium text-slate-200 mb-1">{panel.title}</figcaption>
      <div
        role="img"
        aria-label={label}
        tabIndex={0}
        data-animate={animate ? "on" : "off"}
        className={`rounded-md ${FOCUS}`}
        style={{ height: width ? undefined : height }}
        onKeyDown={(e) => {
          if ((e.key === "t" || e.key === "T") && !e.metaKey && !e.ctrlKey && !e.altKey && onToggleTable) {
            e.preventDefault();
            onToggleTable();
          }
        }}
      >
        {width ? chart : <ResponsiveContainer width="100%" height="100%">{chart}</ResponsiveContainer>}
      </div>
      {onToggleTable && <p className="sr-only">Press T while the chart is focused to view the data as a table.</p>}
    </figure>
  );
}
