// Fixtures for the Fundamentals Explorer (FEAT-001), shaped exactly as the
// series / commentary routes serve them (see types/fundamentals.ts). Values
// are hand-picked so tests can assert on exact strings.
import type {
  CommentaryResponse,
  MetricPoint,
  MetricSeries,
  MetricSpec,
  MissingReason,
  SeriesResponse,
  UnitType,
} from "@/types/fundamentals";

export const PERIODS = ["FY2020", "FY2021", "FY2022", "FY2023", "FY2024"] as const;

export function periodEnd(period: string): string {
  const year = period.replace(/^FY/, "");
  return `${year}-12-31`;
}

/** Catalog labels the chart tests pass in; mirrors the ids the backend
 *  catalog must include (orchestrator decision). */
export const METRIC_LABELS: Record<string, string> = {
  revenue: "Revenue",
  revenue_growth_yoy: "Revenue growth (YoY)",
  gross_margin: "Gross margin",
  operating_margin: "Operating margin",
  net_margin: "Net margin",
  net_income: "Net income",
  eps_diluted: "Diluted EPS",
  operating_cash_flow: "Operating cash flow",
  capex: "Capital expenditure",
  free_cash_flow: "Free cash flow",
  fcf_margin: "FCF margin",
  fcf_after_sbc: "FCF after SBC",
  roic: "ROIC",
  net_debt: "Net debt",
  shares_diluted: "Diluted shares",
  pe_ttm: "P/E (period end)",
  ev_ebitda: "EV / EBITDA",
  ev_revenue: "EV / Revenue",
  p_fcf: "P / FCF",
  fcf_yield: "FCF yield",
};

export const METRIC_UNITS: Record<string, UnitType> = {
  revenue: "currency",
  revenue_growth_yoy: "percent",
  gross_margin: "percent",
  operating_margin: "percent",
  net_margin: "percent",
  net_income: "currency",
  eps_diluted: "currency",
  operating_cash_flow: "currency",
  capex: "currency",
  free_cash_flow: "currency",
  fcf_margin: "percent",
  fcf_after_sbc: "currency",
  roic: "percent",
  net_debt: "currency",
  shares_diluted: "count",
  pe_ttm: "multiple",
  ev_ebitda: "multiple",
  ev_revenue: "multiple",
  p_fcf: "multiple",
  fcf_yield: "percent",
};

export function makeSpec(id: string, over: Partial<MetricSpec> = {}): MetricSpec {
  return {
    id,
    label: METRIC_LABELS[id] ?? id,
    family: "income",
    unit_type: METRIC_UNITS[id] ?? "currency",
    kind: "reported",
    formula_text: `Reported ${id}`,
    inputs: [id],
    sign_note: null,
    frequency: "annual",
    ...over,
  };
}

/** A point per period. `null` → missing with `reason` (default missing_line);
 *  wrap a number in `{ v, estimated: true }` to mark an estimate. */
export type PointInput = number | null | { v: number | null; estimated?: boolean; reason?: MissingReason };

export function makePoints(values: PointInput[], periods: readonly string[] = PERIODS, reason: MissingReason = "missing_line"): MetricPoint[] {
  return values.map((raw, i) => {
    const period = periods[i] ?? `FY${2020 + i}`;
    const obj = raw !== null && typeof raw === "object" ? raw : { v: raw };
    const missing = obj.v === null || obj.v === undefined;
    return {
      period,
      period_end: periodEnd(period),
      value: missing ? null : obj.v,
      reason: missing ? (obj.reason ?? reason) : null,
      estimated: !missing && !!obj.estimated,
    };
  });
}

export interface SeriesOverrides extends Partial<Omit<MetricSeries, "points">> {
  stale?: boolean;
  stale_reason?: string | null;
  source?: string;
}

export function makeSeries(ticker: string, metric: string, values: PointInput[], over: SeriesOverrides = {}): MetricSeries {
  const points = makePoints(values);
  const observed = points.filter((p) => typeof p.value === "number");
  const { stale, stale_reason, source, ...rest } = over;
  return {
    ticker,
    metric,
    unit_type: METRIC_UNITS[metric] ?? "currency",
    currency: "USD",
    points,
    coverage: {
      first: observed[0]?.period ?? null,
      last: observed[observed.length - 1]?.period ?? null,
      n: observed.length,
      expected: points.length,
    },
    provenance: {
      source: source ?? "fmp",
      fetched_at: "2026-09-06T03:16:40Z",
      stale: !!stale,
      stale_reason: stale ? (stale_reason ?? "last fiscal period FY2022 is older than 15 months") : null,
    },
    ...rest,
  };
}

/** Two companies × two metrics, one gap, one estimate, one stale series. */
export function makeSeriesSet(): MetricSeries[] {
  return [
    makeSeries("AAPL", "revenue", [274.5e9, 365.8e9, 394.3e9, 383.3e9, 391.0e9]),
    makeSeries("MSFT", "revenue", [143.0e9, 168.1e9, null, 211.9e9, 245.1e9]),
    makeSeries("AAPL", "gross_margin", [0.382, 0.418, { v: 0.433, estimated: true }, 0.441, 0.462]),
    makeSeries("MSFT", "gross_margin", [0.679, 0.689, 0.684, 0.688, 0.697], { stale: true }),
  ];
}

export function makeSeriesResponse(over: Partial<SeriesResponse> = {}): SeriesResponse {
  return {
    catalog_version: "2026.09.1",
    as_of: "2026-09-08T12:00:00Z",
    series: makeSeriesSet(),
    limits: { applied: { max_companies: 5, max_metrics: 4, max_years: null }, capped_by_plan: false },
    unavailable: [],
    ...over,
  };
}

export function makeCommentary(over: Partial<CommentaryResponse> = {}): CommentaryResponse {
  return {
    commentary_id: 42,
    fingerprint: "9d1cabc",
    cache_hit: false,
    degraded: false,
    degraded_reason: null,
    observed: [
      {
        text: "AAPL revenue grew from $274.5B (FY2020) to $391.0B (FY2024).",
        refs: [
          { ticker: "AAPL", metric: "revenue", period: "FY2020" },
          { ticker: "AAPL", metric: "revenue", period: "FY2024" },
        ],
      },
    ],
    memo_view: [
      {
        text: "The stored memo sees services mix as the margin lever.",
        ticker: "AAPL",
        memo_version: 14,
        memo_generated_at: "2026-08-30T02:11:00Z",
        memo_stale: false,
        memo_stale_reason: null,
      },
    ],
    caveats: ["MSFT FY2022 revenue missing (line item not reported)"],
    generated_at: "2026-09-08T12:00:05Z",
    model: "cheap",
    ...over,
  };
}
