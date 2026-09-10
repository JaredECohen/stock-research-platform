import { afterEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { render } from "@testing-library/react";

// Recharts is mocked here to capture the props handed to each <Line>, so
// the reduced-motion contract (isAnimationActive={false}) is asserted on
// the prop itself rather than inferred from the DOM.
const lineProps: Array<Record<string, unknown>> = [];
vi.mock("recharts", () => {
  const Noop = ({ children }: { children?: React.ReactNode }) => <>{children}</>;
  const Line = (props: Record<string, unknown>) => {
    lineProps.push(props);
    return null;
  };
  return { CartesianGrid: Noop, Line, LineChart: Noop, ResponsiveContainer: Noop, Tooltip: Noop, XAxis: Noop, YAxis: Noop };
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
