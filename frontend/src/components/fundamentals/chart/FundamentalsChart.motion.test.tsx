import { afterEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { render } from "@testing-library/react";

// Recharts is mocked here to capture the props handed to each <Line> and
// <Tooltip>, so the reduced-motion contract (isAnimationActive={false}) and
// the tooltip's null handling are asserted on the props themselves rather
// than inferred from the DOM (jsdom cannot hover a recharts chart).
const lineProps: Array<Record<string, unknown>> = [];
const tooltipProps: Array<Record<string, unknown>> = [];
vi.mock("recharts", () => {
  const Noop = ({ children }: { children?: React.ReactNode }) => <>{children}</>;
  const Line = (props: Record<string, unknown>) => {
    lineProps.push(props);
    return null;
  };
  const Tooltip = (props: Record<string, unknown>) => {
    tooltipProps.push(props);
    return null;
  };
  return { CartesianGrid: Noop, Line, LineChart: Noop, ResponsiveContainer: Noop, Tooltip, XAxis: Noop, YAxis: Noop };
});

import { FundamentalsChart } from "@/components/fundamentals/chart";
import { makeSeriesSet } from "@/test/fixtures/fundamentals";

function stubReducedMotion(matches: boolean) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn((query: string) => ({
      matches: query.includes("prefers-reduced-motion") ? matches : false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  lineProps.length = 0;
  tooltipProps.length = 0;
});

describe("FundamentalsChart reduced motion", () => {
  it("passes isAnimationActive=false to every line when motion is reduced", () => {
    stubReducedMotion(true);
    render(<FundamentalsChart series={makeSeriesSet()} view="auto" />);
    expect(lineProps).toHaveLength(4);
    for (const p of lineProps) {
      expect(p.isAnimationActive).toBe(false);
      expect(p.connectNulls).toBe(false);
    }
  });

  it("passes isAnimationActive=true otherwise", () => {
    stubReducedMotion(false);
    render(<FundamentalsChart series={makeSeriesSet()} view="auto" />);
    expect(lineProps).toHaveLength(4);
    for (const p of lineProps) expect(p.isAnimationActive).toBe(true);
  });

  it("uses function dataKeys so tickers with dots resolve", () => {
    stubReducedMotion(false);
    const series = makeSeriesSet();
    series[0].ticker = "BRK.B";
    render(<FundamentalsChart series={series} view="auto" />);
    const key = lineProps[0].dataKey as ((row: { values: Record<string, number | null> }) => number | null) & { seriesId: string };
    expect(typeof key).toBe("function");
    expect(key.seriesId).toBe("BRK.B:revenue");
    expect(key({ values: { "BRK.B:revenue": 7 } })).toBe(7);
  });
});

describe("FundamentalsChart tooltip", () => {
  type Formatter = (value: unknown, name: unknown, item: { dataKey?: unknown; payload?: unknown }) => [string, string];

  it("keeps null entries so a missing point reads n/a with its reason", () => {
    stubReducedMotion(true);
    render(<FundamentalsChart series={makeSeriesSet()} view="auto" metricLabels={{ revenue: "Revenue", gross_margin: "Gross margin" }} />);
    expect(tooltipProps).toHaveLength(1);
    // Recharts defaults filterNull to true, which would drop the entry.
    expect(tooltipProps[0].filterNull).toBe(false);
    const formatter = tooltipProps[0].formatter as Formatter;
    const msft = lineProps[1].dataKey as { seriesId: string };
    expect(msft.seriesId).toBe("MSFT:revenue");
    const fy22 = { period: "FY2022", period_end: "2022-12-31", values: { "MSFT:revenue": null }, estimated: { "MSFT:revenue": false }, reasons: { "MSFT:revenue": "missing_line" } };
    expect(formatter(null, "MSFT Revenue", { dataKey: msft, payload: fy22 })).toEqual(["n/a (line item not reported)", "MSFT Revenue"]);
    // An observed value formats in the series' unit and currency.
    const fy21 = { period: "FY2021", period_end: "2021-12-31", values: { "MSFT:revenue": 168.1e9 }, estimated: { "MSFT:revenue": false }, reasons: { "MSFT:revenue": null } };
    expect(formatter(168.1e9, "MSFT Revenue", { dataKey: msft, payload: fy21 })).toEqual(["$168.1B", "MSFT Revenue"]);
  });
});
