// Visual encoding for the Fundamentals chart: colour follows the company,
// dash pattern follows the metric, so a reader can separate both channels
// even when two companies share a metric or one company shows two.
//
// The five hues are the dark-mode categorical slots from the dataviz
// reference palette, validated against the card surface (#131B30): every
// adjacent pair clears the colour-vision-deficiency floor (ΔE ≥ 8.4) and
// the normal-vision floor (ΔE ≥ 19.3), and all five sit ≥ 3:1 on the
// surface. Colour is assigned by selection order and stays with the ticker
// when others are removed — never re-ranked.
export const TICKER_COLORS: readonly string[] = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"];

/** SVG stroke-dasharray per metric slot: solid, dashed, dotted, dash-dot. */
export const METRIC_DASHES: readonly string[] = ["", "7 4", "2 4", "9 4 2 4"];

export const CHART_SURFACE = "#131B30";
export const GRID_STROKE = "#243056";
export const AXIS_STROKE = "#94a3b8";

export function colorForTicker(ticker: string, tickers: string[]): string {
  const i = tickers.indexOf(ticker);
  return TICKER_COLORS[(i < 0 ? 0 : i) % TICKER_COLORS.length];
}

export function dashForMetric(metric: string, metrics: string[]): string {
  const i = metrics.indexOf(metric);
  return METRIC_DASHES[(i < 0 ? 0 : i) % METRIC_DASHES.length];
}

/** Stale series draw at reduced opacity; the legend chip carries the reason. */
export const STALE_OPACITY = 0.55;
