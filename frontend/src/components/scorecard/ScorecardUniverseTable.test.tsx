import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import ScorecardUniverseTable from "@/components/scorecard/ScorecardUniverseTable";
import { exportHref, makeUniverse } from "@/test/fixtures/scorecard";

function bodyRows() {
  const table = screen.getByTestId("universe-table");
  return within(table.querySelector("tbody") as HTMLElement).getAllByRole("row");
}

function tickers() {
  return bodyRows().map((r) => r.getAttribute("data-ticker"));
}

function header(name: RegExp) {
  return screen.getByRole("columnheader", { name });
}

describe("ScorecardUniverseTable", () => {
  it("renders a caption, sortable headers with aria-sort, and one row per ticker ranked first", () => {
    render(<ScorecardUniverseTable universe={makeUniverse(5)} />);
    const table = screen.getByTestId("universe-table");
    expect(table.querySelector("caption")).toHaveTextContent("Scorecard universe, fs-v1 as of 2026-08-31: 8 names ranked by overall score.");
    expect(header(/Rank/)).toHaveAttribute("aria-sort", "ascending");
    expect(header(/^Score/)).toHaveAttribute("aria-sort", "none");
    expect(bodyRows()).toHaveLength(8);
    expect(tickers().slice(0, 2)).toEqual(["COST", "JPM"]);
    // Unranked (null overall) sorts last even on rank ascending.
    expect(tickers()[7]).toBe("NEWCO");
    expect(screen.getByTestId("universe-count")).toHaveTextContent("8 of 8 names · fs-v1 · as of 2026-08-31");
  });

  it("renders unscored values as n/a with a reason, never as 0", () => {
    render(<ScorecardUniverseTable universe={makeUniverse(0)} />);
    const newco = screen.getByTestId("row-NEWCO");
    expect(newco).toHaveAttribute("data-unscored", "true");
    const cells = within(newco).getAllByRole("cell");
    expect(cells[0]).toHaveTextContent("n/a"); // rank
    expect(cells[2]).toHaveTextContent("n/a (insufficient coverage)");
    expect(cells[2]).toHaveAttribute("data-missing", "true");
    expect(cells[3]).toHaveTextContent("n/a (unranked)");
    expect(cells[5]).toHaveTextContent("35%");
    // Masked family for a Financials name.
    const jpm = screen.getByTestId("row-JPM");
    const leverage = jpm.querySelector('[data-family="leverage"]') as HTMLElement;
    expect(leverage).toHaveTextContent("n/a");
    expect(leverage.querySelector("[data-missing]")).not.toBeNull();
    expect(newco.textContent).not.toMatch(/\b0\.0\b/);
  });

  it("sorts by score descending first, then ascending, announcing through aria-sort and keeping unscored last", () => {
    render(<ScorecardUniverseTable universe={makeUniverse(5)} />);
    const score = header(/^Score/);
    fireEvent.click(within(score).getByRole("button"));
    expect(score).toHaveAttribute("aria-sort", "descending");
    expect(header(/Rank/)).toHaveAttribute("aria-sort", "none");
    let order = tickers();
    expect(order[order.length - 1]).toBe("NEWCO");
    const scoresDesc = bodyRows()
      .slice(0, -1)
      .map((r) => parseFloat(within(r).getAllByRole("cell")[2].textContent ?? ""));
    expect([...scoresDesc].sort((a, b) => b - a)).toEqual(scoresDesc);

    fireEvent.click(within(score).getByRole("button"));
    expect(score).toHaveAttribute("aria-sort", "ascending");
    order = tickers();
    expect(order[order.length - 1]).toBe("NEWCO");
    const scoresAsc = bodyRows()
      .slice(0, -1)
      .map((r) => parseFloat(within(r).getAllByRole("cell")[2].textContent ?? ""));
    expect([...scoresAsc].sort((a, b) => a - b)).toEqual(scoresAsc);
  });

  it("sorts by ticker and by a family column", () => {
    render(<ScorecardUniverseTable universe={makeUniverse(5)} />);
    fireEvent.click(within(header(/Ticker/)).getByRole("button"));
    expect(header(/Ticker/)).toHaveAttribute("aria-sort", "ascending");
    expect(tickers()).toEqual(["COST", "JPM", "NEWCO", "T01", "T02", "T03", "T04", "T05"]);

    const growth = header(/^Growth/);
    fireEvent.click(within(growth).getByRole("button"));
    expect(growth).toHaveAttribute("aria-sort", "descending");
    const growthValues = bodyRows().map((r) => (r.querySelector('[data-family="growth"]') as HTMLElement).textContent?.trim() ?? "");
    const nums = growthValues.filter((v) => v !== "n/a").map(Number);
    expect([...nums].sort((a, b) => b - a)).toEqual(nums);
    // n/a family scores sit at the bottom (JPM has growth, NEWCO's growth is 61 so it sorts by it).
    expect(growthValues.indexOf("n/a")).toBe(-1);
  });

  it("filters by sector and by minimum coverage", () => {
    render(<ScorecardUniverseTable universe={makeUniverse(8)} />);
    fireEvent.change(screen.getByTestId("sector-filter"), { target: { value: "Financial Services" } });
    expect(tickers().every((t) => ["JPM", "T02", "T06"].includes(t ?? ""))).toBe(true);
    expect(tickers()).toHaveLength(3);
    fireEvent.change(screen.getByTestId("sector-filter"), { target: { value: "" } });
    expect(bodyRows()).toHaveLength(11);

    fireEvent.change(screen.getByTestId("coverage-filter"), { target: { value: "0.6" } });
    expect(tickers()).not.toContain("NEWCO");
    expect(bodyRows().every((r) => parseInt(within(r).getAllByRole("cell")[5].textContent ?? "0", 10) >= 60)).toBe(true);
    expect(screen.getByTestId("universe-count")).toHaveTextContent(/of 11 names/);
  });

  it("shows an in-table status when the filters leave nothing", () => {
    render(<ScorecardUniverseTable universe={makeUniverse(0)} initialMinCoverage={0.99} />);
    expect(screen.getByRole("status")).toHaveTextContent("No names match the filters.");
  });

  it("calls onSelect with the ticker when a row's ticker is activated", () => {
    const onSelect = vi.fn();
    render(<ScorecardUniverseTable universe={makeUniverse(0)} onSelect={onSelect} />);
    fireEvent.click(within(screen.getByTestId("row-JPM")).getByRole("button", { name: "JPM" }));
    expect(onSelect).toHaveBeenCalledWith("JPM");
  });

  it("renders the export link only when the href carries contract=v1", () => {
    const href = exportHref("csv");
    render(<ScorecardUniverseTable universe={makeUniverse(0)} exportHref={href} />);
    const link = screen.getByTestId("export-link");
    expect(link).toHaveAttribute("href", href);
    expect(link.getAttribute("href")).toContain("contract=v1");
    expect(link).toHaveTextContent("Export CSV (contract v1)");

    render(<ScorecardUniverseTable universe={makeUniverse(0)} exportHref="/api/scorecard/export?format=csv" />);
    expect(screen.getAllByTestId("export-link")).toHaveLength(1);
  });

  it("renders a status message when there is no run yet", () => {
    render(<ScorecardUniverseTable universe={null} />);
    expect(screen.getByRole("status")).toHaveTextContent("n/a (no scorecard run yet)");
    expect(screen.queryByTestId("universe-table")).not.toBeInTheDocument();
  });

  it("shows the first positive and negative contributor per row", () => {
    render(<ScorecardUniverseTable universe={makeUniverse(0)} />);
    const cost = within(screen.getByTestId("row-COST")).getAllByRole("cell");
    expect(cost[cost.length - 2]).toHaveTextContent("Accruals ratio +1.40");
    expect(cost[cost.length - 1]).toHaveTextContent("Ebitda ev yield -1.30");
    const newco = within(screen.getByTestId("row-NEWCO")).getAllByRole("cell");
    expect(newco[newco.length - 2]).toHaveTextContent("n/a");
  });
});
