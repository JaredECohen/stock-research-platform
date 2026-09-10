import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { FundamentalsChart } from "@/components/fundamentals/chart";
import { makeSeries, makeSeriesSet, METRIC_LABELS } from "@/test/fixtures/fundamentals";

// Real recharts at a fixed size: jsdom cannot lay out a ResponsiveContainer,
// but a LineChart with explicit width/height computes its scales and emits
// real SVG paths, which is what the gap assertion needs.
const SIZE = { width: 640, height: 300 };

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
});

describe("FundamentalsChart", () => {
  it("renders a role=img panel whose label names the tickers and metrics", () => {
    render(<FundamentalsChart series={makeSeriesSet()} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    const img = screen.getByRole("img");
    const label = img.getAttribute("aria-label") ?? "";
    for (const needle of ["AAPL", "MSFT", "Revenue", "Gross margin", "FY2020 to FY2024", "1 missing point", "1 estimated point", "1 stale series"]) {
      expect(label).toContain(needle);
    }
    expect(img).toHaveAttribute("tabindex", "0");
    expect(screen.getByTestId("fundamentals-chart")).toHaveAttribute("data-resolved-mode", "dual-axis");
  });

  it("breaks the line at a missing point instead of interpolating across it", () => {
    render(<FundamentalsChart series={makeSeriesSet().slice(0, 2)} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    const paths = Array.from(document.querySelectorAll("path.recharts-line-curve"));
    expect(paths).toHaveLength(2);
    const segments = (d: string | null) => (d?.match(/M/g) ?? []).length;
    // AAPL revenue is complete: one continuous sub-path.
    expect(segments(paths[0].getAttribute("d"))).toBe(1);
    // MSFT revenue is missing FY2022: the curve is two sub-paths with a gap.
    expect(segments(paths[1].getAttribute("d"))).toBe(2);
  });

  // Recharts withholds dots and the real dash pattern until its draw
  // animation finishes, which never happens in jsdom; reduced motion turns
  // the animation off so the static rendering is what gets asserted.
  it("draws a hollow marker on an estimated point only", () => {
    stubReducedMotion(true);
    render(<FundamentalsChart series={makeSeriesSet()} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    const markers = document.querySelectorAll('circle[data-estimated="true"]');
    expect(markers).toHaveLength(1);
    expect(markers[0]).toHaveAttribute("fill", "#131B30");
    // Every observed point in the fixture set has an observed neighbour, so
    // no isolated-point markers appear and the lines stay clean.
    expect(document.querySelectorAll('circle[data-isolated="true"]')).toHaveLength(0);
  });

  it("draws a filled marker for an observed point with a gap on both sides so it cannot vanish", () => {
    stubReducedMotion(true);
    // Three observed values, each with no observed neighbour: the line path
    // is three zero-length sub-paths, so the markers are the only pixels.
    const alternating = makeSeries("AAPL", "revenue", [100e9, null, 120e9, null, 140e9]);
    render(<FundamentalsChart series={[alternating]} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    const path = document.querySelector("path.recharts-line-curve");
    expect((path?.getAttribute("d")?.match(/M/g) ?? []).length).toBe(3);
    const markers = document.querySelectorAll('circle[data-isolated="true"]');
    expect(markers).toHaveLength(3);
    for (const m of Array.from(markers)) {
      expect(m).toHaveAttribute("fill", path?.getAttribute("stroke") ?? "");
      expect(m).not.toHaveAttribute("data-estimated");
    }
    // The label still reports every observed point.
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain("2 missing points");
  });

  it("draws a single observed point in an otherwise empty series", () => {
    stubReducedMotion(true);
    const lone = makeSeries("NEW", "revenue", [null, null, 5e9, null, null]);
    render(<FundamentalsChart series={[lone]} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    expect(document.querySelectorAll('circle[data-isolated="true"]')).toHaveLength(1);
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain("has one observed point ($5.0B in FY2022)");
  });

  it("marks an isolated estimated point as both estimated (hollow) and isolated", () => {
    stubReducedMotion(true);
    const s = makeSeries("AAPL", "revenue", [100e9, 110e9, null, { v: 130e9, estimated: true }, null]);
    render(<FundamentalsChart series={[s]} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    const markers = document.querySelectorAll("circle[data-isolated]");
    expect(markers).toHaveLength(1);
    expect(markers[0]).toHaveAttribute("data-estimated", "true");
    expect(markers[0]).toHaveAttribute("fill", "#131B30");
  });

  it("renders stale series at reduced opacity and dashes the second metric", () => {
    stubReducedMotion(true);
    render(<FundamentalsChart series={makeSeriesSet()} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    const paths = Array.from(document.querySelectorAll("path.recharts-line-curve"));
    expect(paths).toHaveLength(4);
    expect(paths[3]).toHaveAttribute("stroke-opacity", "0.55"); // MSFT gross margin is stale
    expect(paths[0]).not.toHaveAttribute("stroke-dasharray"); // revenue = first metric = solid
    expect(paths[2]).toHaveAttribute("stroke-dasharray", "7 4"); // gross margin = second metric = dashed
    // Colour follows the company across metrics.
    expect(paths[0].getAttribute("stroke")).toBe(paths[2].getAttribute("stroke"));
    expect(paths[0].getAttribute("stroke")).not.toBe(paths[1].getAttribute("stroke"));
  });

  it("toggles to a data table with a caption and sortable headers", () => {
    render(<FundamentalsChart series={makeSeriesSet()} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    const toggle = screen.getByRole("button", { name: "View as table" });
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(toggle);
    const table = screen.getByRole("table");
    expect(within(table).getByText(/Revenue and Gross margin for AAPL and MSFT/)).toBeInTheDocument();
    expect(table.querySelector("caption")).not.toBeNull();
    const headers = within(table).getAllByRole("columnheader");
    expect(headers).toHaveLength(5);
    expect(headers[0]).toHaveAttribute("aria-sort", "ascending");
    for (const h of headers.slice(1)) expect(h).toHaveAttribute("aria-sort", "none");
    expect(within(table).getByText("n/a (line item not reported)")).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "View as chart" })).toHaveAttribute("aria-pressed", "true");
  });

  it("switches to the table when T is pressed on the focused chart", () => {
    render(<FundamentalsChart series={makeSeriesSet()} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    fireEvent.keyDown(screen.getByRole("img"), { key: "t" });
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("forces the table for view=table and returns to the chart when the view changes", () => {
    const view = render(<FundamentalsChart series={makeSeriesSet()} view="table" metricLabels={METRIC_LABELS} {...SIZE} />);
    expect(screen.getByRole("table")).toBeInTheDocument();
    view.rerender(<FundamentalsChart series={makeSeriesSet()} view="small-multiples" metricLabels={METRIC_LABELS} {...SIZE} />);
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(screen.getAllByRole("img")).toHaveLength(2);
  });

  it("shows legend chips for missing, estimated and stale series", () => {
    render(<FundamentalsChart series={makeSeriesSet()} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    const legend = screen.getByRole("list", { name: "Series" });
    const items = within(legend).getAllByRole("listitem");
    expect(items).toHaveLength(4);
    expect(items[1]).toHaveTextContent("MSFT Revenue");
    expect(items[1]).toHaveTextContent("1 missing");
    expect(items[2]).toHaveTextContent("1 estimated");
    expect(items[3]).toHaveTextContent("stale");
    expect(items[0]).not.toHaveTextContent(/missing|estimated|stale/);
  });

  it("disables the draw animation under prefers-reduced-motion", () => {
    stubReducedMotion(true);
    render(<FundamentalsChart series={makeSeriesSet()} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    expect(screen.getByRole("img")).toHaveAttribute("data-animate", "off");
  });

  it("animates when motion is not reduced", () => {
    stubReducedMotion(false);
    render(<FundamentalsChart series={makeSeriesSet()} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    expect(screen.getByRole("img")).toHaveAttribute("data-animate", "on");
  });

  it("explains a fallback when the requested view is unavailable and reports the layout", () => {
    const onLayout = vi.fn();
    render(<FundamentalsChart series={makeSeriesSet()} view="indexed" metricLabels={METRIC_LABELS} onLayout={onLayout} {...SIZE} />);
    expect(screen.getByTestId("layout-fallback")).toHaveTextContent("indexing a percent series is misleading");
    expect(onLayout).toHaveBeenCalled();
    expect(onLayout.mock.calls[0][0].availability.indexed.enabled).toBe(false);
  });

  it("offers the suggestion as an action when scales differ 20×", () => {
    const onSuggestion = vi.fn();
    const series = [makeSeries("AAPL", "revenue", [100e9, 110e9, 120e9, 130e9, 140e9]), makeSeries("TINY", "revenue", [1e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9])];
    render(<FundamentalsChart series={series} view="auto" metricLabels={METRIC_LABELS} onSuggestion={onSuggestion} {...SIZE} />);
    expect(screen.getByTestId("layout-suggestion")).toHaveTextContent("differ by 100×");
    fireEvent.click(screen.getByRole("button", { name: "Switch to indexed" }));
    expect(onSuggestion).toHaveBeenCalledWith("indexed");
  });

  it("lists a series with no observed points as not drawn, shows the empty state, and still offers the table with the reasons", () => {
    const empty = makeSeries("NEW", "revenue", [null, null, null, null, null], { currency: "USD" });
    empty.points = empty.points.map((p) => ({ ...p, reason: "not_backfilled" }));
    render(<FundamentalsChart series={[empty]} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    expect(screen.getByTestId("chart-empty")).toHaveTextContent("No observed points to draw");
    expect(screen.getByRole("listitem")).toHaveTextContent("not drawn: no observed points");
    // The per-period reason code lives only in the table, so it must be reachable.
    fireEvent.click(screen.getByRole("button", { name: "View as table" }));
    const table = screen.getByRole("table");
    expect(within(table).getAllByText("n/a (history not loaded)")).toHaveLength(5);
    expect(screen.queryByTestId("chart-empty")).not.toBeInTheDocument();
  });

  it("keeps an all-missing series in the data table with its per-period reasons", () => {
    const missing = makeSeries("NEW", "revenue", Array(5).fill({ v: null, reason: "not_backfilled" }));
    render(<FundamentalsChart series={[makeSeriesSet()[0], missing]} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    // The chart draws only AAPL; the legend says NEW is not drawn.
    expect(screen.getAllByRole("img")).toHaveLength(1);
    expect(screen.getByRole("list", { name: "Series" })).toHaveTextContent("not drawn: no observed points");
    fireEvent.click(screen.getByRole("button", { name: "View as table" }));
    const table = screen.getByRole("table");
    const headers = within(table).getAllByRole("columnheader").map((h) => h.textContent ?? "");
    expect(headers).toEqual([expect.stringContaining("Period"), expect.stringContaining("AAPL Revenue"), expect.stringContaining("NEW Revenue")]);
    expect(within(table).getAllByText("n/a (history not loaded)")).toHaveLength(5);
    expect(within(table).getByText(/AAPL Revenue/).closest("th")).not.toBeNull();
  });

  it("uses the rebased series in the indexed table but keeps an all-missing series' reasons", () => {
    const missing = makeSeries("NEW", "revenue", Array(5).fill({ v: null, reason: "not_backfilled" }));
    render(<FundamentalsChart series={[makeSeriesSet()[0], makeSeriesSet()[1], missing]} view="indexed" metricLabels={METRIC_LABELS} {...SIZE} />);
    fireEvent.click(screen.getByRole("button", { name: "View as table" }));
    const table = screen.getByRole("table");
    expect(table.querySelector("caption")).toHaveTextContent("indexed to 100 at FY2020");
    const headers = within(table).getAllByRole("columnheader");
    expect(headers).toHaveLength(4);
    expect(headers[3]).toHaveTextContent("NEW Revenue");
    expect(within(table).getAllByText("n/a (history not loaded)")).toHaveLength(5);
    expect(within(table).getAllByText("100.0").length).toBeGreaterThanOrEqual(2);
  });

  it("says so when no series are selected", () => {
    render(<FundamentalsChart series={[]} view="auto" metricLabels={METRIC_LABELS} {...SIZE} />);
    expect(screen.getByTestId("chart-empty")).toHaveTextContent("No series selected.");
    expect(screen.queryByRole("button", { name: /View as/ })).not.toBeInTheDocument();
  });
});
