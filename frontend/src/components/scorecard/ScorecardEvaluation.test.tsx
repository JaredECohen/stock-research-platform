import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import ScorecardEvaluation from "@/components/scorecard/ScorecardEvaluation";
import { SCORECARD_CLIENT_RULES, evalParam, type ScorecardEvaluation as EvaluationRow } from "@/types/scorecard";
import { ALL_CAVEATS, CAVEATS, makeEvaluation, makeInsufficientEvaluation, makeLassoResult } from "@/test/fixtures/scorecard";

const SIZE = { width: 640, height: 220 };

describe("ScorecardEvaluation", () => {
  it("renders all three evaluation cards with their statistics", () => {
    render(<ScorecardEvaluation evaluation={makeEvaluation()} {...SIZE} />);
    const q = screen.getByTestId("eval-quintile_ls");
    expect(q).toHaveAttribute("data-state", "ok");
    expect(q).toHaveTextContent("Mean monthly spread");
    expect(q).toHaveTextContent("+0.41%");
    expect(q).toHaveTextContent("Sharpe (ann.)");
    expect(q).toHaveTextContent("1.00");
    expect(q).toHaveTextContent("Top 20% minus bottom 20%");
    expect(q).toHaveTextContent("at least 15 names");
    expect(q).toHaveTextContent("Monotonic quintiles");
    expect(q).toHaveTextContent("yes");
    const qt = within(q).getByTestId("quintile-table");
    expect(within(qt.querySelector("tbody") as HTMLElement).getAllByRole("row")).toHaveLength(5);
    expect(within(q).getByTestId("skipped-months")).toHaveTextContent("2024-01-31 — long leg had 12 names (minimum 15)");

    const ff = screen.getByTestId("eval-ff6_regression");
    expect(ff).toHaveAttribute("data-state", "ok");
    expect(ff).toHaveTextContent("Alpha (monthly)");
    expect(ff).toHaveTextContent("+0.28%");
    expect(ff).toHaveTextContent("Alpha t-stat");
    expect(ff).toHaveTextContent("2.14");
    const rows = within(within(ff).getByTestId("ff6-table").querySelector("tbody") as HTMLElement).getAllByRole("row");
    expect(rows).toHaveLength(6);
    expect(rows[3]).toHaveTextContent("RMW");
    expect(rows[3]).toHaveTextContent("0.44");
    expect(rows[3]).toHaveTextContent("3.10");

    const lasso = screen.getByTestId("eval-double_lasso");
    expect(lasso).toHaveAttribute("data-verdict", "independent");
    expect(within(lasso).getByTestId("lasso-verdict")).toHaveTextContent("Independent information");
    expect(lasso).toHaveTextContent("0.0031");
    expect(within(lasso).getByTestId("lasso-selected-y")).toHaveTextContent("log_mktcap, book_to_market, momentum_12_1");
    expect(lasso).toHaveTextContent("4,680");
    expect(lasso).toHaveTextContent("30 · 14");
  });

  it("points each card at the list above when the page rendered the caveats first", () => {
    render(<ScorecardEvaluation evaluation={makeEvaluation()} caveatsRenderedAbove width={640} />);
    for (const kind of ["quintile_ls", "ff6_regression", "double_lasso"]) {
      const el = screen.getByTestId(`caveats-${kind}`);
      expect(el).toHaveAttribute("data-rendered-above", "true");
      expect(el).toHaveTextContent("listed above, verbatim");
      expect(within(el).queryAllByRole("listitem")).toHaveLength(0);
    }
  });

  it("renders the caveats verbatim on every card", () => {
    render(<ScorecardEvaluation evaluation={makeEvaluation()} {...SIZE} />);
    const q = within(screen.getByTestId("caveats-quintile_ls")).getAllByRole("listitem");
    expect(q.map((li) => li.textContent)).toEqual(ALL_CAVEATS);
    const ff = within(screen.getByTestId("caveats-ff6_regression")).getAllByRole("listitem");
    expect(ff.map((li) => li.textContent)).toEqual([CAVEATS.unadjusted, CAVEATS.constituents]);
    const l = within(screen.getByTestId("caveats-double_lasso")).getAllByRole("listitem");
    expect(l.map((li) => li.textContent)).toEqual(ALL_CAVEATS);
    // Exact sentence, unparaphrased.
    expect(screen.getAllByText("Prices are FMP close values, unadjusted for splits; a split month can produce a spurious return.")).toHaveLength(3);
  });

  it("draws a role=img cumulative spread chart with a descriptive label", () => {
    render(<ScorecardEvaluation evaluation={makeEvaluation()} {...SIZE} />);
    const img = screen.getByRole("img");
    const label = img.getAttribute("aria-label") ?? "";
    expect(label).toContain("Cumulative top-minus-bottom quintile spread, 2024-02-29 to 2026-07-31, 30 months");
    expect(label).toContain("1 skipped month.");
  });

  it("shows the insufficient_data state on every card without inventing numbers", () => {
    render(<ScorecardEvaluation evaluation={makeInsufficientEvaluation()} {...SIZE} />);
    const q = screen.getByTestId("eval-quintile_ls");
    expect(q).toHaveAttribute("data-state", "insufficient");
    expect(within(q).getByTestId("quintile-insufficient")).toHaveTextContent("insufficient: 8 of 24 months");
    expect(q).toHaveTextContent("n/a (insufficient months)");
    expect(q).toHaveTextContent("n/a (insufficient months to bucket)");
    expect(q).toHaveTextContent("Monotonic quintiles");
    expect(q).toHaveTextContent("n/a (not computed)");

    const ff = screen.getByTestId("eval-ff6_regression");
    expect(ff).toHaveAttribute("data-state", "insufficient");
    expect(within(ff).getByTestId("ff6-insufficient")).toHaveTextContent("insufficient: 8 of 24 months");
    expect(ff).toHaveTextContent("n/a (insufficient: 8 of 24 months)");

    const lasso = screen.getByTestId("eval-double_lasso");
    expect(lasso).toHaveAttribute("data-verdict", "insufficient_data");
    expect(within(lasso).getByTestId("lasso-verdict")).toHaveTextContent("Insufficient data");
    expect(lasso).toHaveTextContent("Fewer than 24 month-ends (8) and fewer than 2,000 observations (1,360): no verdict.");
    expect(lasso).toHaveTextContent("n/a (insufficient data)");
    expect(within(lasso).getByTestId("lasso-selected-y")).toHaveTextContent("n/a (insufficient data)");
    expect(lasso.textContent).not.toMatch(/0\.0000/);
  });

  it("renders the subsumed verdict", () => {
    const ev = makeEvaluation();
    ev.evaluations = ev.evaluations.map((e) => (e.kind === "double_lasso" ? { ...e, result: makeLassoResult({ verdict: "subsumed", t_stat: 0.8, p_value: 0.42, interpretation: "Subsumed by size and value." }) } : e));
    render(<ScorecardEvaluation evaluation={ev} {...SIZE} />);
    const lasso = screen.getByTestId("eval-double_lasso");
    expect(within(lasso).getByTestId("lasso-verdict")).toHaveTextContent("Subsumed by known characteristics");
    expect(lasso).toHaveTextContent("Subsumed by size and value.");
    expect(lasso).toHaveTextContent("|t| < 2");
  });

  it("says a kind has not run yet instead of hiding it", () => {
    const ev = makeEvaluation();
    ev.evaluations = ev.evaluations.filter((e) => e.kind !== "double_lasso");
    render(<ScorecardEvaluation evaluation={ev} {...SIZE} />);
    const lasso = screen.getByTestId("eval-double_lasso");
    expect(lasso).toHaveAttribute("data-state", "not-run");
    expect(within(lasso).getByRole("status")).toHaveTextContent("n/a (not run yet)");
  });

  it("renders a status message when there is no evaluation at all", () => {
    render(<ScorecardEvaluation evaluation={null} />);
    expect(screen.getByRole("status")).toHaveTextContent("n/a (no evaluation yet)");
  });

  it("takes the minimum leg, months and observations from the row's params when the worker recorded them", () => {
    const ev = makeInsufficientEvaluation();
    const params = { min_leg: 20, min_months: 36, min_obs: 3000 };
    ev.evaluations = ev.evaluations.map((e) => ({ ...e, params }) as EvaluationRow);
    render(<ScorecardEvaluation evaluation={ev} {...SIZE} />);
    const q = screen.getByTestId("eval-quintile_ls");
    expect(q).toHaveTextContent("legs need at least 20 names");
    expect(within(q).getByTestId("quintile-insufficient")).toHaveTextContent("insufficient: 8 of 36 months");
    expect(q.textContent).not.toMatch(/at least 15 names/);
    const ff = screen.getByTestId("eval-ff6_regression");
    expect(within(ff).getByTestId("ff6-insufficient")).toHaveTextContent("insufficient: 8 of 36 months");
    expect(ff).toHaveTextContent("n/a (insufficient: 8 of 36 months)");
    expect(within(screen.getByTestId("eval-double_lasso")).getByTestId("lasso-gloss")).toHaveTextContent("Fewer than 36 month-ends or 3,000 observations: no verdict is drawn.");
  });

  it("falls back to the documented fs-v1 client rules when params omit or malform the minimums", () => {
    const ev = makeInsufficientEvaluation();
    ev.evaluations = ev.evaluations.map((e) => ({ ...e, params: { min_leg: "fifteen", min_months: NaN } }) as EvaluationRow);
    render(<ScorecardEvaluation evaluation={ev} {...SIZE} />);
    const q = screen.getByTestId("eval-quintile_ls");
    expect(q).toHaveTextContent(`legs need at least ${SCORECARD_CLIENT_RULES.evalMinLeg} names`);
    expect(within(q).getByTestId("quintile-insufficient")).toHaveTextContent(`insufficient: 8 of ${SCORECARD_CLIENT_RULES.evalMinMonths} months`);
    expect(q.textContent).not.toMatch(/NaN/);
    expect(within(screen.getByTestId("eval-double_lasso")).getByTestId("lasso-gloss")).toHaveTextContent(
      `Fewer than ${SCORECARD_CLIENT_RULES.evalMinMonths} month-ends or ${SCORECARD_CLIENT_RULES.lassoMinObs.toLocaleString("en-US")} observations`,
    );
  });

  it("gives a reason when a sample bound was not recorded", () => {
    const ev = makeEvaluation();
    ev.evaluations = ev.evaluations.map((e) => ({ ...e, sample_start: null, sample_end: null }) as EvaluationRow);
    render(<ScorecardEvaluation evaluation={ev} {...SIZE} />);
    const q = screen.getByTestId("eval-quintile_ls");
    expect(q).toHaveTextContent("Sample n/a (no start recorded) to n/a (no end recorded)");
    expect(q.textContent?.replace(/n\/a \([^)]+\)/g, "")).not.toMatch(/n\/a/);
  });

  it("labels the results as model outputs, not recommendations", () => {
    render(<ScorecardEvaluation evaluation={makeEvaluation()} {...SIZE} />);
    expect(screen.getByTestId("scorecard-evaluation")).toHaveTextContent("model outputs for research and education, not a forecast or a recommendation");
  });
});

describe("evalParam", () => {
  it("prefers a finite numeric param and otherwise returns the fallback", () => {
    expect(evalParam({ min_leg: 20 }, "min_leg", 15)).toBe(20);
    expect(evalParam({ min_leg: "20" }, "min_leg", 15)).toBe(15);
    expect(evalParam({ min_leg: Number.POSITIVE_INFINITY }, "min_leg", 15)).toBe(15);
    expect(evalParam({}, "min_months", 24)).toBe(24);
    expect(evalParam(null, "min_obs", 2000)).toBe(2000);
    expect(evalParam(undefined, "min_obs", 2000)).toBe(2000);
  });
});
