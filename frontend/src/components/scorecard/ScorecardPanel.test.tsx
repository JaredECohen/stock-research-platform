import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import ScorecardPanel, { profileText } from "@/components/scorecard/ScorecardPanel";
import { makeDetail, makeDisagreementSummary, makeInsufficientSummary, makeSummary, partialCategories } from "@/test/fixtures/scorecard";

describe("ScorecardPanel", () => {
  it("renders nothing when there is no scorecard", () => {
    const { container: a } = render(<ScorecardPanel scorecard={null} />);
    expect(a).toBeEmptyDOMElement();
    const { container: b } = render(<ScorecardPanel scorecard={undefined} />);
    expect(b).toBeEmptyDOMElement();
  });

  it("renders the headline, version, and all eight families in canonical order", () => {
    render(<ScorecardPanel scorecard={makeSummary()} />);
    expect(screen.getByTestId("overall-score")).toHaveTextContent("62.4");
    expect(screen.getByTestId("universe-percentile")).toHaveTextContent("71st");
    expect(screen.getByTestId("sector-percentile")).toHaveTextContent("58th");
    expect(screen.getByTestId("coverage")).toHaveTextContent("93%");
    expect(screen.getByText("fs-v1")).toBeInTheDocument();
    expect(screen.getByText(/as of 2026-08-31/)).toBeInTheDocument();
    expect(screen.queryByTestId("stale-badge")).not.toBeInTheDocument();

    const list = screen.getByTestId("category-list");
    const ids = within(list)
      .getAllByRole("listitem")
      .map((li) => li.getAttribute("data-testid"));
    expect(ids).toEqual(["valuation", "quality", "growth", "profitability", "efficiency", "leverage", "capital_allocation", "earnings_quality"].map((f) => `category-${f}`));
    expect(screen.getByTestId("category-capital_allocation")).toHaveTextContent("Capital allocation");
    expect(screen.getByTestId("category-valuation")).toHaveTextContent("28.0");
    expect(screen.getByTestId("category-valuation")).toHaveTextContent("z -1.10");
    expect(screen.getByTestId("category-valuation")).toHaveTextContent("14th pct");
    expect(screen.getByTestId("category-leverage")).toHaveTextContent("3/4");
  });

  it("is null-safe with partial categories: missing families read n/a with a reason, never 0", () => {
    render(<ScorecardPanel scorecard={makeSummary({ categories: partialCategories() })} />);
    // Masked-out families are absent from the payload.
    const leverage = screen.getByTestId("category-leverage");
    expect(leverage).toHaveTextContent("n/a (not scored)");
    expect(leverage).toHaveAttribute("data-missing", "true");
    expect(screen.getByTestId("category-efficiency")).toHaveTextContent("n/a (not scored)");
    // Present but unscored: the input count explains why.
    const growth = screen.getByTestId("category-growth");
    expect(growth).toHaveTextContent("n/a (1 of 5 inputs)");
    expect(growth).toHaveAttribute("data-missing", "true");
    expect(growth.textContent).not.toMatch(/\b0\.0\b/);
    // Scored families still render.
    expect(screen.getByTestId("category-quality")).toHaveTextContent("69.0");
  });

  it("says why an overall score is missing instead of printing a neutral number", () => {
    render(<ScorecardPanel scorecard={makeInsufficientSummary()} />);
    expect(screen.getByTestId("overall-score")).toHaveTextContent("n/a (insufficient coverage: fewer than 5 families scored)");
    expect(screen.getByTestId("universe-percentile")).toHaveTextContent("n/a (unranked)");
    expect(screen.getByTestId("sector-percentile")).toHaveTextContent("n/a (unranked)");
    expect(screen.getByTestId("coverage")).toHaveTextContent("42%");
    expect(screen.getByTestId("stale-badge")).toBeInTheDocument();
    expect(screen.getByTestId("profile-line")).toHaveTextContent("n/a (profile not computed)");
    expect(screen.getByTestId("top-positive")).toHaveTextContent("No positive contributors (overall not scored).");
    expect(screen.getByTestId("overall-score").textContent).not.toMatch(/50/);
  });

  it("labels the model read separately from observed figures", () => {
    render(<ScorecardPanel scorecard={makeDetail()} />);
    const profile = screen.getByTestId("profile-line");
    expect(profile).toHaveTextContent("Model read");
    expect(profile).toHaveTextContent("reads as a compounder");
    expect(profile).toHaveTextContent("not a recommendation");

    // Feature table: two explicitly labelled column groups.
    expect(screen.getByTestId("observed-header")).toHaveTextContent("Observed");
    expect(screen.getByTestId("model-read-header")).toHaveTextContent("Model read (fs-v1)");
    const table = screen.getByTestId("feature-table");
    expect(table.querySelector("caption")).toHaveTextContent("Observed inputs (as of FY2025, available 2026-02-12) beside the fs-v1 model read");

    const roic = screen.getByTestId("feature-roic");
    const cells = within(roic).getAllByRole("cell");
    expect(cells[1]).toHaveTextContent("18.7%"); // observed, in its unit
    expect(cells[2]).toHaveTextContent("nopat_ttm / invested_capital");
    expect(cells[3]).toHaveTextContent("+1.30"); // model read z
    expect(cells[4]).toHaveTextContent("+0.041");

    const footnote = screen.getByTestId("scorecard-footnote");
    expect(footnote).toHaveTextContent("Observed figures come from reported statements (latest period FY2025, available 2026-02-12)");
    expect(footnote).toHaveTextContent("model read is spec fs-v1");
  });

  it("renders a missing feature input as n/a with its reason and a masked feature as not applicable", () => {
    render(<ScorecardPanel scorecard={makeDetail()} />);
    const cagr = within(screen.getByTestId("feature-revenue_cagr_3y")).getAllByRole("cell");
    expect(cagr[1]).toHaveTextContent("n/a (fewer than 3 annual points)");
    expect(cagr[1]).toHaveAttribute("data-missing", "true");
    expect(cagr[3]).toHaveTextContent("n/a (fewer than 3 annual points)");

    const masked = screen.getByTestId("feature-net_debt_to_ebitda");
    expect(masked).toHaveAttribute("data-applicable", "false");
    expect(within(masked).getAllByRole("cell")[1]).toHaveTextContent("n/a (not applicable to sector)");
    expect(within(masked).getByRole("rowheader")).toHaveTextContent("(lower is better)");

    const noReason = within(screen.getByTestId("feature-shareholder_yield")).getAllByRole("cell");
    expect(noReason[1]).toHaveTextContent("n/a (input not available)");
    // Nothing missing ever prints as a zero.
    for (const id of ["feature-revenue_cagr_3y", "feature-net_debt_to_ebitda", "feature-shareholder_yield"]) {
      expect(screen.getByTestId(id).textContent).not.toMatch(/\b0\.0+\b/);
    }
  });

  it("hides the feature table in compact mode and for summaries without features", () => {
    render(<ScorecardPanel scorecard={makeDetail()} compact />);
    expect(screen.queryByTestId("feature-table")).not.toBeInTheDocument();
    render(<ScorecardPanel scorecard={makeSummary()} />);
    expect(screen.queryByTestId("feature-table")).not.toBeInTheDocument();
  });

  it("lists the top contributors with family, z and contribution", () => {
    render(<ScorecardPanel scorecard={makeSummary()} />);
    const pos = screen.getByTestId("top-positive");
    expect(within(pos).getAllByRole("listitem")).toHaveLength(3);
    expect(within(pos).getByTestId("contribution-accruals_ratio")).toHaveTextContent("Accruals ratio · Earnings quality");
    expect(within(pos).getByTestId("contribution-accruals_ratio")).toHaveTextContent("z +1.40 · contribution +0.058");
    const neg = screen.getByTestId("top-negative");
    expect(within(neg).getByTestId("contribution-ebitda_ev_yield")).toHaveTextContent("z -1.30 · contribution -0.041");
  });

  it("shows the disagreement callout with the PM reconciliation, and n/a when none was written", () => {
    render(<ScorecardPanel scorecard={makeDisagreementSummary()} />);
    const note = screen.getByTestId("disagreement");
    expect(note).toHaveAttribute("role", "note");
    expect(note).toHaveTextContent("Memo / scorecard disagreement · material · overall");
    expect(note).toHaveTextContent("Memo rating Bullish (70) vs universe percentile 18: gap 52 points.");
    expect(note).toHaveTextContent("the memo is more positive than the quant read; gap +52 points");
    expect(note).toHaveTextContent("PM reconciliation: The PM attributes the gap");

    render(<ScorecardPanel scorecard={makeDisagreementSummary({ reconciliation: null })} />);
    expect(screen.getAllByTestId("disagreement")[1]).toHaveTextContent("PM reconciliation: n/a (not yet written).");
  });

  it("does not show a disagreement callout when none is present", () => {
    render(<ScorecardPanel scorecard={makeSummary()} />);
    expect(screen.queryByTestId("disagreement")).not.toBeInTheDocument();
  });

  it("keeps an unknown family from the backend rather than dropping it", () => {
    const s = makeSummary();
    s.categories = { ...s.categories, resilience: { z: 0.2, score: 54, percentile: 55, weight: 0.1, n_features: 2, n_available: 2 } };
    render(<ScorecardPanel scorecard={s} />);
    expect(screen.getByTestId("category-resilience")).toHaveTextContent("Resilience");
  });
});

describe("profileText", () => {
  it("maps the sub-composites to a model-read label", () => {
    expect(profileText({ compounder: 1.0, inflection: 0.1 })).toBe("reads as a compounder");
    expect(profileText({ compounder: 0.1, inflection: 0.9 })).toBe("reads as an early inflection");
    expect(profileText({ compounder: 0.6, inflection: 0.6 })).toBe("reads as a compounder with an early inflection");
    expect(profileText({ compounder: -0.2, inflection: 0.0 })).toBe("reads as neither a compounder nor an inflection");
    expect(profileText({ compounder: null, inflection: null })).toBe("n/a (profile not computed)");
    expect(profileText(undefined)).toBe("n/a (profile not computed)");
  });
});
