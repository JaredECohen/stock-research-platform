import { describe, expect, it } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import QualityPanel from "@/components/memo/QualityPanel";
import { capText, declaredAssumptionFor, sourceLabel } from "@/lib/memoQuality";
import { makeMemo } from "@/test/fixtures/memo";
import { presentedMemo, PRESENTED_MEMO_NAMES } from "@/test/fixtures/memoSections";
import {
  QUALITY_EXPECT,
  qualityMemo,
  qualityMemoConfidenceHidden,
  qualityMemoPmTemplate,
  qualityVariant,
  VARIANT_REASONS,
} from "@/test/fixtures/memoQuality";
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
    expect(flagged.textContent).toContain(
      `${QUALITY_EXPECT.fabricated_view_figure} in Where we differ from consensus — not found in the data`,
    );
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

  // Every note shape `reconcile_rating` / `enforce_after_patch` writes, from
  // the captured variants. The panel words each from the record's fields;
  // the backend note (which names the reason checks and the config switch)
  // is never printed.
  const RATING_TEXT: Array<[string, () => StockMemoOut, string[]]> = [
    ["no reason (the captured run)", qualityMemo, [
      "The blended rating was Bullish, but the valuation evidence reads overvalued, and no reason was given. The rating was set to Neutral.",
    ]],
    ["a reason that fails the checks", () => qualityVariant("failed_reason"), [
      "The blended rating was Bullish, but the valuation evidence reads overvalued, and the stated reason was too short or generic to count as a reason, did not name the valuation signal it goes against and did not quote that signal's value. The rating was set to Neutral.",
      `PM's reason: \u201c${VARIANT_REASONS.thin}\u201d`,
      "No live risk-committee review of the reason.",
    ]],
    ["a reason the live critic rejected", () => qualityVariant("critic_unsupported"), [
      "The blended rating was Bullish, but the valuation evidence reads overvalued, and the risk committee judged the stated reason unsupported. The rating was set to Neutral.",
      `PM's reason: \u201c${VARIANT_REASONS.quoted}\u201d`,
      "The risk committee reviewed the reason and did not support it.",
    ]],
    ["an accepted, unreviewed reason", () => qualityVariant("accepted_unreviewed"), [
      "Rated Bullish although the valuation evidence reads overvalued, on the PM's stated reason.",
      `PM's reason: \u201c${VARIANT_REASONS.quoted}\u201d`,
      "No live risk-committee review of the reason.",
    ]],
    ["an accepted, supported reason", () => qualityVariant("accepted_supported"), [
      "Rated Bullish although the valuation evidence reads overvalued, on the PM's stated reason.",
      `PM's reason: \u201c${VARIANT_REASONS.quoted}\u201d`,
      "The risk committee reviewed the reason and supported it.",
    ]],
    ["record mode", () => qualityVariant("record_mode"), [
      "The blended rating was Bullish, but the valuation evidence reads overvalued, and no reason was given. Recorded only: the rating check is not changing ratings, so it stays Bullish.",
    ]],
    ["the news-patch guard", () => qualityVariant("patch_guard"), [
      "A news update moved the rating to Very Bullish against the valuation evidence (overvalued) with no valuation reason (a news update cannot state one). The rating was set to Neutral.",
    ]],
    ["the news-patch guard in record mode", () => qualityVariant("patch_guard_record_mode"), [
      "A news update moved the rating to Very Bullish against the valuation evidence (overvalued) with no valuation reason (a news update cannot state one). Recorded only: the rating check is not changing ratings, so it stays Very Bullish.",
    ]],
    ["a patch that left the full run's record", () => qualityVariant("patch_kept_record"), [
      "The blended rating was Bullish, but the valuation evidence reads overvalued, and no reason was given. The rating was set to Neutral.",
      "A news update has since moved the rating to Bearish; this check describes the last full run.",
    ]],
    ["a patch after an accepted divergence", () => qualityVariant("patch_after_accepted"), [
      "Rated Bullish although the valuation evidence reads overvalued, on the PM's stated reason.",
      `PM's reason: \u201c${VARIANT_REASONS.quoted}\u201d`,
      "No live risk-committee review of the reason.",
      "A news update has since moved the rating to Very Bullish; this check describes the last full run.",
    ]],
  ];

  it.each(RATING_TEXT)("words the rating check from its fields: %s", (_name, make, lines) => {
    const memo = make();
    const block = within(panel(memo)).getByTestId("research-checks-rating-text");
    expect(Array.from(block.querySelectorAll("p")).map((p) => p.textContent)).toEqual(lines);
    const text = block.textContent ?? "";
    // The backend note carries machine terms; none may reach the page.
    expect(text).not.toContain(memo.quality!.rating_reconciliation!.note);
    expect(text).not.toMatch(/failed:|rating_reconciliation_mode|substantive|names_signal|quotes_value/);
    expect(text).not.toMatch(/\b[a-z]+_[a-z_]+\b/);
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
  it("prints only the earned confidence when the PM synthesis was template-filled", () => {
    // The real template-PM run: the presenter hides the PM view but shows
    // the confidence, and `raw` is the template's number.
    const memo = qualityMemoPmTemplate();
    const conf = memo.quality!.confidence!;
    expect(conf.binding).toBe("pm_template");
    expect(memo.section_availability!.confidence_score.status).toBe("available");
    const block = within(panel(memo)).getByTestId("research-checks-confidence");
    expect(within(block).getByTestId("research-checks-confidence-line").textContent).toBe(
      `Confidence ${Math.round(conf.final)} after the research checks. The PM synthesis was template-filled, so it has no confidence of its own to show.`,
    );
    expect(block.textContent).not.toMatch(new RegExp(`\\b${Math.round(conf.raw)}\\b`));
    expect(block.textContent).not.toContain("→");
    expect(within(block).getAllByRole("listitem")[0].textContent).toBe(
      "≤40 — The PM synthesis was template-filled. (binding)",
    );
  });

  it("finds a declared assumption the way the check matched it: by unit, within printed precision", () => {
    const memo = qualityMemo();
    const nc = memo.quality!.number_check!;
    const claim = nc.claims.find((c) => c.status === "assumption")!;
    expect(claim.raw).toBe("18.5%");
    // The prose rounds the declared 18.47; a same-valued declaration in
    // another unit is not the one the check matched.
    nc.assumptions = [
      { value: 18.5, unit: "usd", basis_ref: "price:NVDA", horizon: "FY2030", status: "assumption" },
      { value: 18.47, unit: "pct", basis_ref: "financials:NVDA", horizon: "FY2027", status: "assumption" },
    ];
    expect(declaredAssumptionFor(nc, claim)?.horizon).toBe("FY2027");
    expect(within(panel(memo)).getByTestId("research-checks-assumptions").textContent).toBe(
      `${QUALITY_EXPECT.declared_assumption} in PM view — PM assumption, FY2027, based on financial statements`,
    );
    // Outside the printed precision ("18.5%" is 18.45..18.55) it is not.
    nc.assumptions = [{ value: 18.6, unit: "pct", basis_ref: "financials:NVDA", horizon: "FY2027" }];
    expect(declaredAssumptionFor(nc, claim)).toBeNull();
  });

  it("prints withheld points as a list, with no toggle, in the paper (PDF) variant", () => {
    render(<QualityPanel memo={qualityMemo()} variant="paper" />);
    const p = screen.getByTestId("research-checks");
    expect(within(p).queryByRole("button")).toBeNull();
    expect(within(p).getByTestId("research-checks-withheld").textContent).toContain(
      `1 withheld point:`,
    );
    expect(p.textContent).toContain(`Valuation analyst: ${QUALITY_EXPECT.withheld_point}`);
  });

  it("says the counts and sources cover hidden sections when the presenter hid any", () => {
    const memo = qualityMemo();
    // The captured memo hides template sections (bull case, sector analyst, …).
    const p = panel(memo);
    expect(within(p).getByTestId("research-checks-scope").textContent).toContain(
      "including sections not shown in this version",
    );
    expect(within(p).getByTestId("research-checks-sources").textContent).toMatch(
      /^Figures across the whole memo, including sections not shown, trace to: /,
    );
  });

  it("drops the scope note when nothing is hidden", () => {
    const memo = qualityMemo();
    for (const av of Object.values(memo.section_availability!)) {
      av.status = "available";
      av.reason = null;
      av.hidden_items = 0;
    }
    const p = panel(memo);
    expect(within(p).queryByTestId("research-checks-scope")).toBeNull();
    expect(within(p).getByTestId("research-checks-sources").textContent).toMatch(/^Figures trace to: /);
  });
});
