import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import ContributionBars, { fromContributions, fromFeatures } from "@/components/scorecard/ContributionBars";
import { makeFeatures, TOP_NEGATIVE, TOP_POSITIVE } from "@/test/fixtures/scorecard";

describe("ContributionBars", () => {
  it("renders one item per contribution with the number in text and a bar sized by magnitude", () => {
    render(<ContributionBars title="Top positive" items={fromContributions(TOP_POSITIVE)} />);
    expect(screen.getByText("Top positive")).toBeInTheDocument();
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveTextContent("Accruals ratio · Earnings quality");
    expect(items[0]).toHaveTextContent("z +1.40 · contribution +0.058");
    const big = screen.getByTestId("bar-accruals_ratio");
    const small = screen.getByTestId("bar-gross_margin");
    expect(big.style.width).toBe("50%");
    expect(parseFloat(small.style.width)).toBeLessThan(50);
    // Bars are decorative; the text carries the value.
    expect(big.parentElement).toHaveAttribute("aria-hidden", "true");
  });

  it("colours negative contributions differently and anchors their bar on the other side", () => {
    render(<ContributionBars items={fromContributions(TOP_NEGATIVE)} />);
    const bar = screen.getByTestId("bar-ebitda_ev_yield");
    expect(bar.className).toContain("bg-danger-500");
    expect(bar.className).toContain("right-1/2");
    expect(screen.getByTestId("contribution-ebitda_ev_yield")).toHaveTextContent("z -1.30 · contribution -0.041");
  });

  it("renders a null contribution as n/a with its reason and draws no bar", () => {
    render(<ContributionBars items={fromFeatures(makeFeatures())} />);
    const missing = screen.getByTestId("contribution-revenue_cagr_3y");
    expect(missing).toHaveAttribute("data-missing", "true");
    expect(missing).toHaveTextContent("n/a (fewer than 3 annual points)");
    expect(screen.queryByTestId("bar-revenue_cagr_3y")).not.toBeInTheDocument();
    const masked = screen.getByTestId("contribution-net_debt_to_ebitda");
    expect(masked).toHaveTextContent("n/a (not applicable to sector)");
    expect(masked.textContent).not.toMatch(/0\.000/);
  });

  it("shows the sum of listed contributions beside the overall z when asked", () => {
    const items = fromContributions([
      { feature: "a", family: "quality", z: 1, contribution: 0.25 },
      { feature: "b", family: "growth", z: -1, contribution: -0.1 },
    ]);
    render(<ContributionBars items={items} overallZ={0.15} />);
    expect(screen.getByTestId("contribution-sum")).toHaveTextContent("Σ listed contributions +0.15 · overall z +0.15");
  });

  it("says when the overall z is missing rather than printing zero", () => {
    render(<ContributionBars items={fromContributions(TOP_POSITIVE)} overallZ={null} />);
    expect(screen.getByTestId("contribution-sum")).toHaveTextContent("overall z n/a (insufficient coverage)");
  });

  it("renders the empty text as a status when there is nothing to show", () => {
    render(<ContributionBars items={[]} emptyText="Nothing computed." overallZ={0.5} />);
    expect(screen.getByRole("status")).toHaveTextContent("Nothing computed.");
    expect(screen.queryByTestId("contribution-sum")).not.toBeInTheDocument();
  });

  it("accepts a custom test id so two lists on one page stay distinguishable", () => {
    render(
      <div>
        <ContributionBars items={fromContributions(TOP_POSITIVE)} data-testid="pos" />
        <ContributionBars items={fromContributions(TOP_NEGATIVE)} data-testid="neg" />
      </div>,
    );
    expect(within(screen.getByTestId("pos")).getAllByRole("listitem")).toHaveLength(3);
    expect(within(screen.getByTestId("neg")).getAllByRole("listitem")).toHaveLength(3);
  });
});
