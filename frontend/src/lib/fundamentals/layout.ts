// Deterministic mixed-unit layout rules for the Fundamentals Explorer.
// Given the displayed series and the requested view, decide how many panels
// to draw, which series share an axis, and which view modes are available
// (with a reason when one is not). Pure and table-tested; the chart just
// renders what comes back.
//
// Rules (orchestrator decisions, FEAT-001):
//   1. Series group by unit type. Percent, multiple, ratio, count and
//      currency never share an axis with one another.
//   2. auto: one unit group → shared axis. Two unit groups with ≤ 4 series
//      → dual axis (left = first selected metric's unit). Otherwise small
//      multiples, one panel per metric.
//   3. Currency across companies is compared only when currencies match;
//      otherwise the currency panel splits per company (no FX is attempted).
//   4. Indexed is offered only for currency/count series that all have a
//      positive base at a common observed period ("indexing a ratio is
//      misleading").
//   5. When latest values in one currency panel differ by ≥ 20×, suggest
//      (never auto-apply) the indexed view, or small multiples when indexing
//      is not possible.
//   6. Dual axis collapses to small multiples on narrow screens.
import type { MetricSeries, UnitType, ViewMode } from "@/types/fundamentals";
import { humanizeMetric } from "./format";
import { indexSeries, isDrawable, latestValueRatio, seriesId } from "./transform";

export const DUAL_AXIS_MAX_SERIES = 4;
export const INDEX_ELIGIBLE_UNITS: readonly UnitType[] = ["currency", "count"];
export const SCALE_RATIO_SUGGESTION = 20;

export type ResolvedMode = "shared" | "dual-axis" | "small-multiples" | "indexed";

export type AxisUnit = UnitType | "index";

export interface AxisSpec {
  id: "left" | "right";
  unit_type: AxisUnit;
  /** Currency the axis is denominated in (currency axes only). */
  currency: string | null;
  seriesIds: string[];
}

export interface PanelSpec {
  id: string;
  title: string;
  seriesIds: string[];
  axes: AxisSpec[];
  indexed: boolean;
}

export interface ModeAvailability {
  enabled: boolean;
  /** Why the mode is disabled; null when enabled. */
  reason: string | null;
}

export type SelectableMode = Exclude<ViewMode, "auto" | "table">;

export interface LayoutSuggestion {
  mode: "indexed" | "small-multiples";
  reason: string;
}

export interface LayoutResult {
  requested: ViewMode;
  resolved: ResolvedMode;
  /** Set when the requested mode was unavailable and auto was used instead. */
  fallback: { from: ViewMode; reason: string } | null;
  panels: PanelSpec[];
  availability: Record<SelectableMode, ModeAvailability>;
  warnings: string[];
  suggestion: LayoutSuggestion | null;
  /** Series not drawn, with the reason (listed by the legend, never silently dropped). */
  excluded: Array<{ id: string; reason: string }>;
  /** Period the indexed view rebases to (indexed mode only). */
  indexBasePeriod: string | null;
  /** The series as drawn — rebased copies in indexed mode, otherwise the input. */
  drawn: MetricSeries[];
}

export interface LayoutOptions {
  view: ViewMode;
  /** Viewport below `md`: dual axis is unreadable, so it collapses. */
  narrow?: boolean;
  metricLabels?: Record<string, string>;
  /** Selection order of metrics; defaults to first appearance in `series`. */
  metricOrder?: string[];
}

function uniq<T>(xs: T[]): T[] {
  return Array.from(new Set(xs));
}

function labelFor(metric: string, labels?: Record<string, string>): string {
  return labels?.[metric] ?? humanizeMetric(metric);
}

function orderedMetrics(series: MetricSeries[], order?: string[]): string[] {
  const seen = uniq(series.map((s) => s.metric));
  if (!order || order.length === 0) return seen;
  const ordered = order.filter((m) => seen.includes(m));
  for (const m of seen) if (!ordered.includes(m)) ordered.push(m);
  return ordered;
}

/** Known currencies among currency-unit series, in first-seen order. */
function currenciesOf(series: MetricSeries[]): string[] {
  return uniq(series.filter((s) => s.unit_type === "currency" && s.currency).map((s) => (s.currency as string).toUpperCase()));
}

function axisFor(id: "left" | "right", unit: AxisUnit, series: MetricSeries[]): AxisSpec {
  const currencies = unit === "currency" ? currenciesOf(series) : [];
  return { id, unit_type: unit, currency: currencies.length === 1 ? currencies[0] : null, seriesIds: series.map(seriesId) };
}

function sharedPanel(id: string, title: string, series: MetricSeries[], unit: AxisUnit): PanelSpec {
  return { id, title, seriesIds: series.map(seriesId), axes: [axisFor("left", unit, series)], indexed: false };
}

/** One panel per metric; a currency metric with mismatched currencies
 *  splits further into one panel per company. */
function smallMultiplePanels(series: MetricSeries[], metrics: string[], labels?: Record<string, string>): PanelSpec[] {
  const panels: PanelSpec[] = [];
  for (const m of metrics) {
    const ofMetric = series.filter((s) => s.metric === m);
    if (ofMetric.length === 0) continue;
    const unit = ofMetric[0].unit_type;
    if (unit === "currency" && currenciesOf(ofMetric).length > 1) {
      for (const s of ofMetric) {
        const cur = s.currency ? ` (${s.currency.toUpperCase()})` : "";
        panels.push(sharedPanel(`${m}:${s.ticker}`, `${labelFor(m, labels)} — ${s.ticker}${cur}`, [s], unit));
      }
    } else {
      panels.push(sharedPanel(m, labelFor(m, labels), ofMetric, unit));
    }
  }
  return panels;
}

export function computeLayout(input: MetricSeries[], opts: LayoutOptions): LayoutResult {
  const requested = opts.view;
  const labels = opts.metricLabels;
  const warnings: string[] = [];
  const excluded: LayoutResult["excluded"] = [];

  const drawable: MetricSeries[] = [];
  for (const s of input) {
    if (isDrawable(s)) drawable.push(s);
    else excluded.push({ id: seriesId(s), reason: "no observed points" });
  }

  const metrics = orderedMetrics(drawable, opts.metricOrder);
  const unitGroups = uniq(metrics.map((m) => drawable.find((s) => s.metric === m)!.unit_type));
  const currencies = currenciesOf(drawable);
  const currencyMismatch = currencies.length > 1;

  // --- availability -------------------------------------------------------
  const dualReason = (() => {
    if (drawable.length === 0) return "nothing to draw";
    if (unitGroups.length !== 2) return `dual axis needs exactly two unit types (have ${unitGroups.length})`;
    if (drawable.length > DUAL_AXIS_MAX_SERIES) return `dual axis is limited to ${DUAL_AXIS_MAX_SERIES} series (have ${drawable.length})`;
    if (currencyMismatch) return "currencies differ across companies";
    if (opts.narrow) return "dual axis is unreadable on narrow screens";
    return null;
  })();

  const ineligibleUnits = unitGroups.filter((u) => !INDEX_ELIGIBLE_UNITS.includes(u));
  const indexAttempt = ineligibleUnits.length === 0 && drawable.length > 0 ? indexSeries(drawable) : null;
  const indexedReason = (() => {
    if (drawable.length === 0) return "nothing to draw";
    if (ineligibleUnits.length > 0) return `indexing a ${ineligibleUnits.join("/")} series is misleading`;
    if (!indexAttempt || !indexAttempt.basePeriod) return "no period where every series is observed and positive";
    return null;
  })();

  const availability: LayoutResult["availability"] = {
    "dual-axis": { enabled: dualReason === null, reason: dualReason },
    "small-multiples": { enabled: drawable.length > 0, reason: drawable.length > 0 ? null : "nothing to draw" },
    indexed: { enabled: indexedReason === null, reason: indexedReason },
  };

  // --- resolve ------------------------------------------------------------
  let fallback: LayoutResult["fallback"] = null;
  let mode: ResolvedMode;
  const autoMode = (): ResolvedMode => {
    if (unitGroups.length <= 1) return currencyMismatch ? "small-multiples" : "shared";
    if (availability["dual-axis"].enabled) return "dual-axis";
    return "small-multiples";
  };

  switch (requested) {
    case "dual-axis":
      if (availability["dual-axis"].enabled) mode = "dual-axis";
      else {
        fallback = { from: requested, reason: dualReason ?? "" };
        mode = autoMode();
      }
      break;
    case "indexed":
      if (availability.indexed.enabled) mode = "indexed";
      else {
        fallback = { from: requested, reason: indexedReason ?? "" };
        mode = autoMode();
      }
      break;
    case "small-multiples":
      mode = "small-multiples";
      break;
    case "table":
    case "auto":
    default:
      mode = autoMode();
  }

  // --- panels -------------------------------------------------------------
  let panels: PanelSpec[] = [];
  let drawn: MetricSeries[] = drawable;
  let indexBasePeriod: string | null = null;

  if (drawable.length === 0) {
    panels = [];
  } else if (mode === "shared") {
    const title = metrics.map((m) => labelFor(m, labels)).join(", ");
    panels = [sharedPanel("shared", title, drawable, unitGroups[0])];
  } else if (mode === "dual-axis") {
    const leftUnit = unitGroups[0];
    const rightUnit = unitGroups[1];
    const left = drawable.filter((s) => s.unit_type === leftUnit);
    const right = drawable.filter((s) => s.unit_type === rightUnit);
    const title = metrics.map((m) => labelFor(m, labels)).join(" and ");
    panels = [
      {
        id: "dual",
        title,
        seriesIds: drawable.map(seriesId),
        axes: [axisFor("left", leftUnit, left), axisFor("right", rightUnit, right)],
        indexed: false,
      },
    ];
  } else if (mode === "indexed" && indexAttempt && indexAttempt.basePeriod) {
    indexBasePeriod = indexAttempt.basePeriod;
    drawn = indexAttempt.series;
    for (const s of indexAttempt.skipped) excluded.push(s);
    const title = `${metrics.map((m) => labelFor(m, labels)).join(", ")} — indexed to 100 at ${indexBasePeriod}`;
    panels = [{ ...sharedPanel("indexed", title, drawn, "index"), indexed: true }];
  } else {
    mode = "small-multiples";
    panels = smallMultiplePanels(drawable, metrics, labels);
  }

  // --- warnings -----------------------------------------------------------
  if (currencyMismatch) {
    // The indexed view is unit-less, so mixed currencies share one panel
    // there; the warning must say what the reader is actually looking at.
    warnings.push(
      mode === "indexed"
        ? `Currencies differ (${currencies.join(", ")}); the indexed view compares trajectories in each company's reporting currency, not magnitudes, and no FX conversion is attempted.`
        : `Currencies differ (${currencies.join(", ")}); currency series are shown per company and no FX conversion is attempted.`,
    );
  }

  // --- suggestion (rule 5) -----------------------------------------------
  let suggestion: LayoutSuggestion | null = null;
  if (mode === "shared" || mode === "dual-axis") {
    for (const m of metrics) {
      const ofMetric = drawable.filter((s) => s.metric === m && s.unit_type === "currency");
      if (ofMetric.length < 2) continue;
      const ratio = latestValueRatio(ofMetric);
      if (ratio !== null && ratio >= SCALE_RATIO_SUGGESTION) {
        const how = availability.indexed.enabled ? "indexed" : "small-multiples";
        suggestion = {
          mode: how,
          reason: `Latest ${labelFor(m, labels)} values differ by ${Math.round(ratio)}×; the smaller company is hard to read on a shared axis. Consider the ${how === "indexed" ? "indexed" : "small multiples"} view.`,
        };
        break;
      }
    }
  }

  return { requested, resolved: mode, fallback, panels, availability, warnings, suggestion, excluded, indexBasePeriod, drawn };
}
