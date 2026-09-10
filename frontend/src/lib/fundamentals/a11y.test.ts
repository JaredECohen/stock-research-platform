import { describe, expect, it } from "vitest";
import { describeChange, joinNames, panelLabel, seriesName, tableCaption } from "@/lib/fundamentals/a11y";
import { computeLayout } from "@/lib/fundamentals/layout";
import { makeSeries, makeSeriesSet, METRIC_LABELS } from "@/test/fixtures/fundamentals";

describe("joinNames", () => {
  it("joins with commas and a final and", () => {
    expect(joinNames([])).toBe("");
    expect(joinNames(["AAPL"])).toBe("AAPL");
    expect(joinNames(["AAPL", "MSFT"])).toBe("AAPL and MSFT");
    expect(joinNames(["AAPL", "MSFT", "NVDA"])).toBe("AAPL, MSFT and NVDA");
  });
});

describe("describeChange", () => {
  it("describes a rise in percent from first to last observed point", () => {
    expect(describeChange(makeSeriesSet()[0])).toBe("rose 42% from FY2020 to FY2024");
  });

  it("uses one decimal under 10% and says flat for no change", () => {
    expect(describeChange(makeSeries("X", "revenue", [100, 101, 102, 103, 104.5]))).toBe("rose 4.5% from FY2020 to FY2024");
    expect(describeChange(makeSeries("X", "revenue", [100, 90, 100, 100, 100]))).toBe("was flat from FY2020 to FY2024");
    expect(describeChange(makeSeries("X", "revenue", [100, null, null, null, 80]))).toBe("fell 20% from FY2020 to FY2024");
  });

  it("falls back to from/to values when the first observed value is not positive", () => {
    expect(describeChange(makeSeries("X", "net_income", [-1.2e9, null, 2e9, 3e9, 3.4e9]))).toBe("moved from -$1.2B (FY2020) to $3.4B (FY2024)");
  });

  it("uses from/to values whenever the sign changes, never a relative change across zero", () => {
    // +$1.0B → -$2.0B used to read "fell 300%", which is meaningless.
    expect(describeChange(makeSeries("X", "free_cash_flow", [1e9, 2e9, 3e9, 4e9, -2e9]))).toBe("moved from $1.0B (FY2020) to -$2.0B (FY2024)");
    expect(describeChange(makeSeries("X", "free_cash_flow", [1e9, 2e9, 3e9, 4e9, 0]))).toBe("moved from $1.0B (FY2020) to $0 (FY2024)");
  });

  it("describes percent series in percentage points, not as a relative change", () => {
    // 38.2% → 46.2% is an 8-point rise; "rose 21%" would be heard as 21 points.
    expect(describeChange(makeSeries("X", "gross_margin", [0.382, 0.418, 0.433, 0.441, 0.462]))).toBe("rose 8.0 points from 38.2% (FY2020) to 46.2% (FY2024)");
    expect(describeChange(makeSeries("X", "operating_margin", [0.25, null, 0.2, 0.18, 0.212]))).toBe("fell 3.8 points from 25.0% (FY2020) to 21.2% (FY2024)");
    expect(describeChange(makeSeries("X", "fcf_margin", [-0.05, 0.0, 0.02, 0.05, 0.1]))).toBe("rose 15.0 points from -5.0% (FY2020) to 10.0% (FY2024)");
    expect(describeChange(makeSeries("X", "net_margin", [0.2, 0.25, 0.1, 0.15, 0.2002]))).toBe("was flat from FY2020 to FY2024");
  });

  it("describes ratios and multiples as a plain difference in their own unit", () => {
    expect(describeChange(makeSeries("X", "pe_ttm", [14.0, 16.0, 12.0, 13.5, 12.8]))).toBe("fell 1.2x from 14.0x (FY2020) to 12.8x (FY2024)");
    expect(describeChange(makeSeries("X", "ev_ebitda", [10.0, 11.0, 12.0, 13.0, 10.02]))).toBe("was flat from FY2020 to FY2024");
    const ratio = makeSeries("X", "current_ratio", [1.2, 1.3, 1.4, 1.45, 1.5], { unit_type: "ratio" });
    expect(describeChange(ratio)).toBe("rose 0.30 from 1.20 (FY2020) to 1.50 (FY2024)");
  });

  it("names the no-data case instead of calling it flat", () => {
    expect(describeChange(makeSeries("X", "revenue", [null, null, null, null, null]))).toBe("has no observed points");
    expect(describeChange(makeSeries("X", "revenue", [null, null, 5e9, null, null]))).toBe("has one observed point ($5.0B in FY2022)");
  });
});

describe("panelLabel", () => {
  it("names metrics, span, tickers, each series' change and the status counts", () => {
    const series = makeSeriesSet();
    const layout = computeLayout(series, { view: "auto", metricLabels: METRIC_LABELS });
    const label = panelLabel(layout.panels[0], series, METRIC_LABELS);
    expect(label).toBe(
      "Revenue and Gross margin, FY2020 to FY2024, AAPL and MSFT; " +
        "AAPL Revenue rose 42% from FY2020 to FY2024, MSFT Revenue rose 71% from FY2020 to FY2024, " +
        "AAPL Gross margin rose 8.0 points from 38.2% (FY2020) to 46.2% (FY2024), MSFT Gross margin rose 1.8 points from 67.9% (FY2020) to 69.7% (FY2024); " +
        "1 missing point, 1 estimated point, 1 stale series",
    );
  });

  it("uses bare tickers when the panel has a single metric", () => {
    const series = makeSeriesSet().slice(0, 2);
    const layout = computeLayout(series, { view: "auto", metricLabels: METRIC_LABELS });
    expect(panelLabel(layout.panels[0], series, METRIC_LABELS)).toBe(
      "Revenue, FY2020 to FY2024, AAPL and MSFT; AAPL rose 42% from FY2020 to FY2024, MSFT rose 71% from FY2020 to FY2024; 1 missing point",
    );
  });

  it("reads in index points for an indexed panel", () => {
    const series = makeSeriesSet().slice(0, 2);
    const layout = computeLayout(series, { view: "indexed", metricLabels: METRIC_LABELS });
    const label = panelLabel(layout.panels[0], layout.drawn, METRIC_LABELS);
    expect(label.startsWith("Revenue (indexed to 100), FY2020 to FY2024, AAPL and MSFT; AAPL rose 42%")).toBe(true);
  });
});

describe("seriesName / tableCaption", () => {
  it("names a series ticker-first and falls back to a humanized id", () => {
    expect(seriesName({ ticker: "AAPL", metric: "revenue" }, METRIC_LABELS)).toBe("AAPL Revenue");
    expect(seriesName({ ticker: "AAPL", metric: "fcf_after_sbc" })).toBe("AAPL Fcf after sbc");
  });

  it("writes a caption that explains n/a and ≈", () => {
    expect(tableCaption(makeSeriesSet(), METRIC_LABELS)).toBe(
      "Revenue and Gross margin for AAPL and MSFT, FY2020 to FY2024, annual. n/a means the value was not obtained; the reason follows in parentheses. ≈ marks an estimated value.",
    );
    expect(tableCaption(makeSeriesSet().slice(0, 1), METRIC_LABELS, "FY2021")).toContain("indexed to 100 at FY2021");
  });
});
