import { describe, expect, it } from "vitest";
import { computeLayout, DUAL_AXIS_MAX_SERIES, SCALE_RATIO_SUGGESTION } from "@/lib/fundamentals/layout";
import { makeSeries, makeSeriesSet, METRIC_LABELS } from "@/test/fixtures/fundamentals";
import type { MetricSeries, ViewMode } from "@/types/fundamentals";

const rev = (t: string, scale = 1, over = {}) => makeSeries(t, "revenue", [100e9 * scale, 110e9 * scale, 120e9 * scale, 130e9 * scale, 140e9 * scale], over);
const gm = (t: string) => makeSeries(t, "gross_margin", [0.4, 0.41, 0.42, 0.43, 0.44]);
const pe = (t: string) => makeSeries(t, "pe_ttm", [20, 22, 25, 24, 28]);
const shares = (t: string) => makeSeries(t, "shares_diluted", [16e9, 16.2e9, 16.4e9, 16.6e9, 16.8e9]);

describe("computeLayout — auto mode rules", () => {
  const cases: Array<{ name: string; series: MetricSeries[]; narrow?: boolean; resolved: string; panels: number; axes?: number }> = [
    { name: "one unit group → shared axis, one panel", series: [rev("AAPL"), rev("MSFT")], resolved: "shared", panels: 1, axes: 1 },
    { name: "two unit groups with 4 series → dual axis", series: makeSeriesSet(), resolved: "dual-axis", panels: 1, axes: 2 },
    { name: "two unit groups with 5 series → small multiples per metric", series: [...makeSeriesSet(), rev("NVDA")], resolved: "small-multiples", panels: 2 },
    { name: "three unit groups → small multiples per metric", series: [rev("AAPL"), gm("AAPL"), pe("AAPL")], resolved: "small-multiples", panels: 3 },
    { name: "two groups on a narrow viewport → small multiples", series: makeSeriesSet(), narrow: true, resolved: "small-multiples", panels: 2 },
    { name: "currency and count are different groups", series: [rev("AAPL"), shares("AAPL")], resolved: "dual-axis", panels: 1, axes: 2 },
    { name: "percent and multiple never share an axis", series: [gm("AAPL"), pe("AAPL")], resolved: "dual-axis", panels: 1, axes: 2 },
  ];
  for (const c of cases) {
    it(c.name, () => {
      const out = computeLayout(c.series, { view: "auto", narrow: c.narrow, metricLabels: METRIC_LABELS });
      expect(out.resolved).toBe(c.resolved);
      expect(out.panels).toHaveLength(c.panels);
      if (c.axes) expect(out.panels[0].axes).toHaveLength(c.axes);
      expect(out.fallback).toBeNull();
    });
  }

  it("puts the first selected metric's unit on the left axis", () => {
    const out = computeLayout(makeSeriesSet(), { view: "auto", metricOrder: ["gross_margin", "revenue"] });
    expect(out.panels[0].axes[0]).toMatchObject({ id: "left", unit_type: "percent" });
    expect(out.panels[0].axes[1]).toMatchObject({ id: "right", unit_type: "currency", currency: "USD" });
    expect(out.panels[0].axes[1].seriesIds).toEqual(["AAPL:revenue", "MSFT:revenue"]);
  });

  it("names the metrics in the panel title", () => {
    const out = computeLayout(makeSeriesSet(), { view: "auto", metricLabels: METRIC_LABELS });
    expect(out.panels[0].title).toBe("Revenue and Gross margin");
  });
});

describe("computeLayout — availability and fallbacks", () => {
  it(`dual axis is refused above ${DUAL_AXIS_MAX_SERIES} series and falls back to auto with the reason`, () => {
    const out = computeLayout([...makeSeriesSet(), rev("NVDA")], { view: "dual-axis" });
    expect(out.availability["dual-axis"]).toEqual({ enabled: false, reason: `dual axis is limited to ${DUAL_AXIS_MAX_SERIES} series (have 5)` });
    expect(out.fallback).toEqual({ from: "dual-axis", reason: `dual axis is limited to ${DUAL_AXIS_MAX_SERIES} series (have 5)` });
    expect(out.resolved).toBe("small-multiples");
  });

  it("dual axis needs exactly two unit types", () => {
    const one = computeLayout([rev("AAPL"), rev("MSFT")], { view: "dual-axis" });
    expect(one.availability["dual-axis"].reason).toBe("dual axis needs exactly two unit types (have 1)");
    expect(one.resolved).toBe("shared");
    const three = computeLayout([rev("AAPL"), gm("AAPL"), pe("AAPL")], { view: "dual-axis" });
    expect(three.availability["dual-axis"].reason).toBe("dual axis needs exactly two unit types (have 3)");
  });

  it("dual axis is disabled on narrow screens with a reason", () => {
    const out = computeLayout(makeSeriesSet(), { view: "dual-axis", narrow: true });
    expect(out.availability["dual-axis"].reason).toBe("dual axis is unreadable on narrow screens");
    expect(out.resolved).toBe("small-multiples");
  });

  it("small multiples is always honoured", () => {
    const out = computeLayout(makeSeriesSet(), { view: "small-multiples" });
    expect(out.resolved).toBe("small-multiples");
    expect(out.panels.map((p) => p.id)).toEqual(["revenue", "gross_margin"]);
    expect(out.panels[0].seriesIds).toEqual(["AAPL:revenue", "MSFT:revenue"]);
    expect(out.suggestion).toBeNull();
  });

  it("table view resolves the same panels as auto so switching back is instant", () => {
    const auto = computeLayout(makeSeriesSet(), { view: "auto" });
    const table = computeLayout(makeSeriesSet(), { view: "table" });
    expect(table.resolved).toBe(auto.resolved);
    expect(table.panels).toEqual(auto.panels);
  });
});

describe("computeLayout — currency mismatch", () => {
  it("splits a currency panel per company and warns; no FX is attempted", () => {
    const out = computeLayout([rev("AAPL"), rev("SAP", 1, { currency: "EUR" })], { view: "auto", metricLabels: METRIC_LABELS });
    expect(out.resolved).toBe("small-multiples");
    expect(out.panels.map((p) => p.id)).toEqual(["revenue:AAPL", "revenue:SAP"]);
    expect(out.panels[1].title).toBe("Revenue — SAP (EUR)");
    expect(out.panels[1].axes[0].currency).toBe("EUR");
    expect(out.warnings).toEqual(["Currencies differ (USD, EUR); currency series are shown per company and no FX conversion is attempted."]);
  });

  it("disables dual axis when the currency axis would mix currencies", () => {
    const out = computeLayout([rev("AAPL"), rev("SAP", 1, { currency: "EUR" }), gm("AAPL"), gm("SAP")], { view: "auto" });
    expect(out.availability["dual-axis"].reason).toBe("currencies differ across companies");
    expect(out.resolved).toBe("small-multiples");
    expect(out.panels.map((p) => p.id)).toEqual(["revenue:AAPL", "revenue:SAP", "gross_margin"]);
  });

  it("keeps mixed currencies in one indexed panel and says the view compares trajectories", () => {
    // Indexing is unit-less, so the per-company warning would misdescribe
    // what is on screen.
    const out = computeLayout([rev("AAPL"), rev("SAP", 1, { currency: "EUR" })], { view: "indexed", metricLabels: METRIC_LABELS });
    expect(out.resolved).toBe("indexed");
    expect(out.panels.map((p) => p.id)).toEqual(["indexed"]);
    expect(out.warnings).toEqual([
      "Currencies differ (USD, EUR); the indexed view compares trajectories in each company's reporting currency, not magnitudes, and no FX conversion is attempted.",
    ]);
  });

  it("treats an unknown currency as not comparable-mismatched (no split)", () => {
    const out = computeLayout([rev("AAPL"), rev("XYZ", 1, { currency: null })], { view: "auto" });
    expect(out.resolved).toBe("shared");
    expect(out.warnings).toEqual([]);
  });
});

describe("computeLayout — indexed eligibility", () => {
  it("indexes currency and count series to 100 at the first common positive period", () => {
    const out = computeLayout([rev("AAPL"), shares("AAPL")], { view: "indexed" });
    expect(out.availability.indexed).toEqual({ enabled: true, reason: null });
    expect(out.resolved).toBe("indexed");
    expect(out.indexBasePeriod).toBe("FY2020");
    expect(out.panels).toHaveLength(1);
    expect(out.panels[0].indexed).toBe(true);
    expect(out.panels[0].axes[0].unit_type).toBe("index");
    expect(out.drawn[0].points[0].value).toBe(100);
    expect(out.drawn[0].points[4].value).toBeCloseTo(140);
  });

  it("refuses to index percent, ratio or multiple series and falls back to auto", () => {
    const out = computeLayout(makeSeriesSet(), { view: "indexed" });
    expect(out.availability.indexed).toEqual({ enabled: false, reason: "indexing a percent series is misleading" });
    expect(out.fallback).toEqual({ from: "indexed", reason: "indexing a percent series is misleading" });
    expect(out.resolved).toBe("dual-axis");
  });

  it("refuses to index when no period has every series observed and positive", () => {
    const neg = makeSeries("LOSS", "net_income", [-5e9, -3e9, -1e9, -2e9, -4e9]);
    const out = computeLayout([neg, makeSeries("AAPL", "net_income", [1e9, 2e9, 3e9, 4e9, 5e9])], { view: "indexed" });
    expect(out.availability.indexed.reason).toBe("no period where every series is observed and positive");
    expect(out.resolved).toBe("shared");
  });

  it("lists an all-missing series as excluded rather than dropping it silently", () => {
    const empty = makeSeries("NEW", "revenue", [null, null, null, null, null], { currency: "USD" });
    const out = computeLayout([rev("AAPL"), empty], { view: "indexed" });
    expect(out.excluded).toEqual([{ id: "NEW:revenue", reason: "no observed points" }]);
    expect(out.resolved).toBe("indexed");
    expect(out.panels[0].seriesIds).toEqual(["AAPL:revenue"]);
  });
});

describe("computeLayout — scale suggestion", () => {
  it(`suggests indexed when latest values in a currency panel differ by ≥ ${SCALE_RATIO_SUGGESTION}×`, () => {
    const out = computeLayout([rev("AAPL"), rev("TINY", 1 / 50)], { view: "auto", metricLabels: METRIC_LABELS });
    expect(out.resolved).toBe("shared");
    expect(out.suggestion?.mode).toBe("indexed");
    expect(out.suggestion?.reason).toContain("differ by 50×");
    expect(out.suggestion?.reason).toContain("Revenue");
  });

  it("suggests small multiples instead when indexing is not possible", () => {
    // A percent series on the other axis makes the whole selection ineligible
    // for indexing; the currency panel still differs 50×.
    const out = computeLayout([rev("AAPL"), rev("TINY", 1 / 50), gm("AAPL")], { view: "auto" });
    expect(out.resolved).toBe("dual-axis");
    expect(out.availability.indexed.enabled).toBe(false);
    expect(out.suggestion?.mode).toBe("small-multiples");
    expect(out.suggestion?.reason).toContain("small multiples view");
  });

  it("stays quiet below the threshold and in small-multiples mode", () => {
    expect(computeLayout([rev("AAPL"), rev("MSFT", 1 / 10)], { view: "auto" }).suggestion).toBeNull();
    expect(computeLayout([rev("AAPL"), rev("TINY", 1 / 50)], { view: "small-multiples" }).suggestion).toBeNull();
  });
});

describe("computeLayout — nothing to draw", () => {
  it("returns no panels and disables every mode with a reason", () => {
    const out = computeLayout([makeSeries("NEW", "revenue", [null, null, null, null, null])], { view: "auto" });
    expect(out.panels).toEqual([]);
    expect(out.excluded).toEqual([{ id: "NEW:revenue", reason: "no observed points" }]);
    for (const mode of ["dual-axis", "small-multiples", "indexed"] as const) {
      expect(out.availability[mode].enabled).toBe(false);
    }
  });

  it("handles an empty selection", () => {
    const out = computeLayout([], { view: "auto" as ViewMode });
    expect(out.panels).toEqual([]);
    expect(out.warnings).toEqual([]);
  });
});
