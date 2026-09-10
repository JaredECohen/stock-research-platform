import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import SeriesLegend from "@/components/fundamentals/SeriesLegend";
import { TICKER_COLORS } from "@/components/fundamentals/chart";
import { makeSeries, makeSeriesSet, METRIC_LABELS } from "@/test/fixtures/fundamentals";

describe("SeriesLegend", () => {
  it("shows one chip per series with missing / estimated / stale counts as text", () => {
    render(<SeriesLegend series={makeSeriesSet()} metricLabels={METRIC_LABELS} />);
    const items = within(screen.getByRole("list", { name: "Series" })).getAllByRole("listitem");
    expect(items.map((i) => i.getAttribute("data-series-id"))).toEqual(["AAPL:revenue", "MSFT:revenue", "AAPL:gross_margin", "MSFT:gross_margin"]);
    expect(items[0]).toHaveTextContent("AAPL Revenue");
    expect(items[0]).not.toHaveTextContent(/missing|estimated|stale/);
    expect(items[1]).toHaveTextContent("1 missing");
    expect(items[2]).toHaveTextContent("1 estimated");
    expect(items[3]).toHaveTextContent("stale");
    expect(items[3].querySelector(".sr-only")).toHaveTextContent("last fiscal period FY2022 is older than 15 months");
  });

  it("keys colour to the company and dash to the metric", () => {
    render(<SeriesLegend series={makeSeriesSet()} metricLabels={METRIC_LABELS} />);
    const lines = Array.from(document.querySelectorAll("li svg line"));
    expect(lines[0]).toHaveAttribute("stroke", TICKER_COLORS[0]);
    expect(lines[1]).toHaveAttribute("stroke", TICKER_COLORS[1]);
    expect(lines[2]).toHaveAttribute("stroke", TICKER_COLORS[0]);
    expect(lines[0]).not.toHaveAttribute("stroke-dasharray");
    expect(lines[2]).toHaveAttribute("stroke-dasharray", "7 4");
    expect(lines[3]).toHaveAttribute("stroke-opacity", "0.55");
  });

  it("marks excluded series as not drawn with the reason", () => {
    const empty = makeSeries("NEW", "revenue", [null, null]);
    render(<SeriesLegend series={[empty]} excluded={[{ id: "NEW:revenue", reason: "no observed points" }]} />);
    const item = screen.getByRole("listitem");
    expect(item).toHaveTextContent("not drawn: no observed points");
    expect(item).toHaveTextContent("2 missing");
  });
});
