import { describe, expect, it } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import SeriesTable from "@/components/fundamentals/SeriesTable";
import { indexSeries } from "@/lib/fundamentals/transform";
import { makeSeries, makeSeriesSet, METRIC_LABELS } from "@/test/fixtures/fundamentals";

function bodyRows() {
  const table = screen.getByRole("table");
  return within(table.querySelector("tbody") as HTMLElement).getAllByRole("row");
}

describe("SeriesTable", () => {
  it("renders a caption, one column per series with units, and one row per period", () => {
    render(<SeriesTable series={makeSeriesSet()} metricLabels={METRIC_LABELS} />);
    const table = screen.getByRole("table");
    expect(table.querySelector("caption")).toHaveTextContent("Revenue and Gross margin for AAPL and MSFT, FY2020 to FY2024, annual.");
    const headers = within(table).getAllByRole("columnheader");
    expect(headers.map((h) => h.textContent)).toEqual([
      expect.stringContaining("Period"),
      expect.stringContaining("AAPL Revenue"),
      expect.stringContaining("MSFT Revenue"),
      expect.stringContaining("AAPL Gross margin"),
      expect.stringContaining("MSFT Gross margin"),
    ]);
    expect(headers[1]).toHaveTextContent("USD");
    expect(headers[3]).toHaveTextContent("%");
    expect(bodyRows()).toHaveLength(5);
    expect(within(bodyRows()[0]).getByRole("rowheader")).toHaveTextContent("FY2020");
  });

  it("renders missing as n/a with the reason and estimated with a marker, never as 0", () => {
    render(<SeriesTable series={makeSeriesSet()} metricLabels={METRIC_LABELS} />);
    const fy22 = bodyRows()[2];
    const cells = within(fy22).getAllByRole("cell");
    expect(cells[1]).toHaveTextContent("n/a (line item not reported)");
    expect(cells[1]).toHaveAttribute("data-missing", "true");
    expect(cells[2]).toHaveTextContent("≈ estimated 43.3%");
    expect(cells[2].querySelector(".sr-only")).toHaveTextContent("estimated");
    expect(fy22.textContent).not.toMatch(/\$0\b/);
  });

  it("sorts by a series column with aria-sort, keeping missing values last in both directions", () => {
    render(<SeriesTable series={makeSeriesSet()} metricLabels={METRIC_LABELS} />);
    const msft = screen.getByRole("columnheader", { name: /MSFT Revenue/ });
    fireEvent.click(within(msft).getByRole("button"));
    expect(msft).toHaveAttribute("aria-sort", "ascending");
    expect(screen.getByRole("columnheader", { name: /Period/ })).toHaveAttribute("aria-sort", "none");
    let periods = bodyRows().map((r) => within(r).getByRole("rowheader").textContent?.trim().split(" ")[0]);
    expect(periods).toEqual(["FY2020", "FY2021", "FY2023", "FY2024", "FY2022"]);

    fireEvent.click(within(msft).getByRole("button"));
    expect(msft).toHaveAttribute("aria-sort", "descending");
    periods = bodyRows().map((r) => within(r).getByRole("rowheader").textContent?.trim().split(" ")[0]);
    expect(periods).toEqual(["FY2024", "FY2023", "FY2021", "FY2020", "FY2022"]);
  });

  it("sorts by period descending on a second click of the period header", () => {
    render(<SeriesTable series={makeSeriesSet()} metricLabels={METRIC_LABELS} />);
    const period = screen.getByRole("columnheader", { name: /Period/ });
    fireEvent.click(within(period).getByRole("button"));
    expect(period).toHaveAttribute("aria-sort", "descending");
    expect(within(bodyRows()[0]).getByRole("rowheader")).toHaveTextContent("FY2024");
  });

  it("shows index units and the base period for an indexed view", () => {
    const idx = indexSeries(makeSeriesSet().slice(0, 2));
    render(<SeriesTable series={idx.series} metricLabels={METRIC_LABELS} indexBasePeriod={idx.basePeriod} />);
    expect(screen.getByRole("table").querySelector("caption")).toHaveTextContent("indexed to 100 at FY2020");
    expect(screen.getByRole("columnheader", { name: /AAPL Revenue/ })).toHaveTextContent("index (base = 100)");
    expect(within(bodyRows()[0]).getAllByRole("cell")[0]).toHaveTextContent("100.0");
  });

  it("says so when there are no periods", () => {
    render(<SeriesTable series={[makeSeries("X", "revenue", [])]} />);
    expect(screen.getByText("No periods to show.")).toBeInTheDocument();
  });
});
