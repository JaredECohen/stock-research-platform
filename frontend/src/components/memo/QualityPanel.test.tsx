import { describe, expect, it } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import QualityPanel from "@/components/memo/QualityPanel";
import { capText, sourceLabel } from "@/lib/memoQuality";
import { makeMemo } from "@/test/fixtures/memo";
import { presentedMemo, PRESENTED_MEMO_NAMES } from "@/test/fixtures/memoSections";
import { QUALITY_EXPECT, qualityMemo, qualityMemoConfidenceHidden } from "@/test/fixtures/memoQuality";
import type { StockMemoOut } from "@/types";

function panel(memo: StockMemoOut) {
  render(<QualityPanel memo={memo} />);
  return screen.getByTestId("research-checks");
}

// Every piece of machine vocabulary the record carries. None may reach the
// page (owner decision 8: human labels, no codes).
const CAP_CODES = [
  "pm_template", "template_sections", "critic_not_live", "no_transcript", "no_filing_review",
  "divergence_unreviewed", "figures_unchecked", "no_primary_trace", "single_primary_kind",
  "untraceable_figures", "last_full_run", "quality_check_failed",
];

describe("QualityPanel", () => {
  it("renders nothing for a memo without quality (every memo stored before W2b)", () => {
    const { container } = render(<QualityPanel memo={makeMemo()} />);
    expect(container.innerHTML).toBe("");
    const nulled = render(<QualityPanel memo={makeMemo({ quality: null })} />);
    expect(nulled.container.innerHTML).toBe("");
    for (const name of PRESENTED_MEMO_NAMES) {
      expect(render(<QualityPanel memo={presentedMemo(name)} />).container.innerHTML).toBe("");
    }
  });

  it("states the figure counts the number check recorded", () => {
    const memo = qualityMemo();
    const c = memo.quality!.number_check!.counts;
    const counts = within(panel(memo)).getByTestId("research-checks-counts");
    expect(counts.textContent).toBe(
      `${c.claims_total} checked · ${c.traced} traced to source data · ${c.weak} matched by value only · ` +
        `${c.untraceable} not found in source data · ${c.assumption} PM assumption`,
    );
  });

  it("lists the untraceable figure with where it appears, and labels the PM assumption", () => {
    const p = panel(qualityMemo());
    const flagged = within(p).getByTestId("research-checks-flagged");
    expect(flagged.textContent).toContain(`${QUALITY_EXPECT.fabricated_pm_figure} in PM view — not found in the data`);
    const assumptions = within(p).getByTestId("research-checks-assumptions");
    expect(assumptions.textContent).toBe(
      `${QUALITY_EXPECT.declared_assumption} in PM view — PM assumption, FY2027, based on financial statements`,
    );
  });

  it("keeps withheld points behind a disclosure, verbatim", () => {
    const p = panel(qualityMemo());
    expect(p.textContent).not.toContain(QUALITY_EXPECT.withheld_point);
    const button = within(p).getByRole("button", { name: "Show 1 withheld point" });
    expect(button).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(button);
    expect(within(p).getByText(QUALITY_EXPECT.withheld_point)).toBeInTheDocument();
    expect(within(p).getByTestId("research-checks-withheld").textContent).toContain(
      `Valuation analyst: ${QUALITY_EXPECT.withheld_point}`,
    );
    expect(within(p).getByRole("button", { name: "Hide 1 withheld point" })).toBeInTheDocument();
  });

  it("shows the rating check's note", () => {
    const memo = qualityMemo();
    const rec = memo.quality!.rating_reconciliation!;
    expect(rec.outcome).toBe("downgraded");
    expect(within(panel(memo)).getByTestId("research-checks-rating").textContent).toContain(rec.note);
  });

  it("shows raw -> final confidence and every cap as a sentence, the binding one first", () => {
    const memo = qualityMemo();
    const conf = memo.quality!.confidence!;
    const block = within(panel(memo)).getByTestId("research-checks-confidence");
    expect(within(block).getByTestId("research-checks-confidence-line").textContent).toBe(
      `PM confidence ${Math.round(conf.raw)} → ${Math.round(conf.final)} after the research checks.`,
    );
    const items = within(block).getAllByRole("listitem");
    expect(items).toHaveLength(conf.caps.length);
    expect(items[0]).toHaveAttribute("data-binding", "true");
    expect(items[0].textContent).toBe(
      `≤45 — 3 core analyst sections were template-filled (sector, earnings, filing). (binding)`,
    );
    expect(items[1].textContent).toBe("≤60 — No live risk-committee review was completed.");
  });

  it("hides raw -> final confidence and the caps when section availability hides confidence", () => {
    const memo = qualityMemoConfidenceHidden();
    const p = panel(memo);
    const block = within(p).getByTestId("research-checks-confidence");
    expect(block.textContent).toBe("ConfidenceConfidence is unavailable in this version.");
    expect(within(block).queryByRole("listitem")).toBeNull();
    // Neither number the placeholder withholds is anywhere in the panel.
    const conf = memo.quality!.confidence!;
    expect(p.textContent).not.toMatch(new RegExp(`\\b${Math.round(conf.raw)}\\b`));
    expect(p.textContent).not.toMatch(new RegExp(`(≤|\\b)${Math.round(conf.final)}\\b`));
    // The figures and the rating check are not about confidence; they stay.
    expect(within(p).getByTestId("research-checks-counts")).toBeInTheDocument();
    expect(within(p).getByTestId("research-checks-rating")).toBeInTheDocument();
  });

  it("names sources as labels, and prints no code, field path or industry code", () => {
    const memo = qualityMemo();
    const nc = memo.quality!.number_check!;
    // Probes for what a routed memo cites: an industry-group slug and a
    // ref carrying a GICS-shaped code, which must never be printed.
    nc.sources_cited = [...nc.sources_cited, "industry_group:semiconductors-equipment", "industry_group:4530", "made_up:2550"];
    nc.lists_not_withheld = ["extra_agent_views.industry_group.key_points"];
    nc.unchecked_fields = ["bull_case.key_points[0]"];
    memo.quality!.confidence!.caps.push({
      code: "single_primary_kind",
      cap: 65,
      detail: "Figures trace to one primary source kind only (financials).",
    });
    const text = panel(memo).textContent ?? "";
    const sources = screen.getByTestId("research-checks-sources").textContent ?? "";
    expect(sources).toContain("Financial statements");
    expect(sources).toContain("DCF model (consensus-anchored)");
    expect(sources).toContain("Industry group analysis");
    expect(sources).toContain("Other data given to the analysts");
    for (const ref of nc.sources_cited) expect(text).not.toContain(ref);
    for (const code of CAP_CODES) expect(text).not.toContain(code);
    expect(text).not.toMatch(/\b4530\b|\b2550\b|semiconductors-equipment/);
    expect(text).not.toMatch(/_agent_view|extra_agent_views|key_points|rule_based|\w+:\w+/);
    expect(text).toContain("Kept with flags, because most of the list was flagged: Industry group analyst.");
    expect(text).toContain("Changed by a news update after the check, so not checked: Bull case.");
    expect(text).toContain("Figures trace to one kind of primary source only (financial statements).");
  });

  it("says the figures were not checked when the check did not run", () => {
    const memo = qualityMemo();
    memo.quality!.number_check = { ...memo.quality!.number_check!, checked: false, claims: [], withheld: [] };
    const p = panel(memo);
    expect(within(p).getByTestId("research-checks-unchecked").textContent).toBe(
      "Figures were not source-checked for this version.",
    );
    expect(within(p).queryByTestId("research-checks-counts")).toBeNull();
    expect(within(p).queryByTestId("research-checks-sources")).toBeNull();
  });

  it("words every known cap without its code, and an unknown one generically", () => {
    for (const code of CAP_CODES) {
      const t = capText({ code, cap: 50, detail: "x_y_z sector_agent_view (review mode: rule_based)" });
      expect(t).not.toMatch(/_|rule_based/);
      expect(t.length).toBeGreaterThan(10);
    }
    expect(capText({ code: "brand_new_cap", cap: 50, detail: "raw_detail" })).toBe(
      "Another research check limited confidence.",
    );
    expect(sourceLabel("brand_new:thing")).toBe("Other data given to the analysts");
  });
});
