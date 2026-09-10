import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import ScorecardHistoryChart, { historyLabel } from "@/components/scorecard/ScorecardHistoryChart";
import { makeHistory } from "@/test/fixtures/scorecard";

// Real recharts at a fixed size: jsdom cannot lay out a ResponsiveContainer,
// but a LineChart with explicit width/height emits real SVG paths, which
// the gap assertion needs.
const SIZE = { width: 640, height: 240 };

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

describe("ScorecardHistoryChart", () => {
  it("renders a role=img chart whose label names the ticker, metric, range, count and missing points", () => {
    render(<ScorecardHistoryChart history={makeHistory()} {...SIZE} />);
    const img = screen.getByRole("img");
    const label = img.getAttribute("aria-label") ?? "";
    expect(label).toBe("COST universe percentile (fs-v1), 2023-09-30 to 2026-08-31, 36 month-ends, 2 missing points; latest 76th (2026-08-31).");
    expect(img).toHaveAttribute("tabindex", "0");
    expect(screen.getByText("COST · universe percentile · month-ends")).toBeInTheDocument();
  });

  it("breaks the line at a month with no rank instead of interpolating across it", () => {
    stubReducedMotion(true);
    render(<ScorecardHistoryChart history={makeHistory()} {...SIZE} />);
    const path = document.querySelector("path.recharts-line-curve");
    expect(path).not.toBeNull();
    // Two gaps split the curve into three sub-paths.
    expect((path?.getAttribute("d")?.match(/M/g) ?? []).length).toBe(3);
  });

  it("switches metric and reflects it in the label", () => {
    render(<ScorecardHistoryChart history={makeHistory()} metric="overall_score" {...SIZE} />);
    const label = screen.getByRole("img").getAttribute("aria-label") ?? "";
    expect(label).toContain("COST overall score (fs-v1)");
    expect(label).toContain("latest 61.0 (2026-08-31)");
  });

  it("offers the data table as the accessible equivalent, with n/a for unranked months", () => {
    render(<ScorecardHistoryChart history={makeHistory()} {...SIZE} />);
    const toggle = screen.getByTestId("history-table-toggle");
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    const table = screen.getByTestId("history-table");
    const rows = within(table.querySelector("tbody") as HTMLElement).getAllByRole("row");
    expect(rows).toHaveLength(36);
    const gap = rows[5];
    expect(within(gap).getByRole("rowheader")).toHaveTextContent("2024-02-29");
    const cells = within(gap).getAllByRole("cell");
    expect(cells[0]).toHaveTextContent("n/a (not ranked)");
    expect(cells[0]).toHaveAttribute("data-missing", "true");
    expect(cells[1]).toHaveTextContent("40%");
    expect(within(rows[0]).getAllByRole("cell")[0]).toHaveTextContent("35th");
  });

  it("renders a status message, not an empty chart, when there is no history", () => {
    render(<ScorecardHistoryChart history={null} />);
    expect(screen.getByRole("status")).toHaveTextContent("n/a (no month-end history yet)");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    render(<ScorecardHistoryChart history={makeHistory({ points: [] })} />);
    expect(screen.getAllByRole("status")[1]).toHaveTextContent("COST: n/a (no month-end history yet)");
  });

  it("honours reduced motion", () => {
    stubReducedMotion(true);
    render(<ScorecardHistoryChart history={makeHistory()} {...SIZE} />);
    expect(screen.getByRole("img")).toHaveAttribute("data-animate", "off");
  });
});

describe("historyLabel", () => {
  it("describes an all-missing series without inventing a latest value", () => {
    const h = makeHistory();
    h.points = h.points.slice(0, 2).map((p) => ({ ...p, universe_percentile: null }));
    expect(historyLabel(h, "universe_percentile")).toBe("COST universe percentile (fs-v1), 2023-09-30 to 2023-10-31, 2 month-ends, 2 missing points; no ranked month.");
  });
  it("uses the singular for one point", () => {
    const h = makeHistory();
    h.points = h.points.slice(0, 1);
    expect(historyLabel(h, "sector_percentile")).toBe("COST sector percentile (fs-v1), 2023-09-30 to 2023-09-30, 1 month-end; latest 27th (2023-09-30).");
  });
});
