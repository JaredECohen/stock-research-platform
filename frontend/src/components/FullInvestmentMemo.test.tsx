import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import FullInvestmentMemo from "@/components/FullInvestmentMemo";
import { REASON_TEXT, UNAVAILABLE_TEXT } from "@/lib/memoSections";
import { PM_TEMPLATE_TAIL, presentedMemo } from "@/test/fixtures/memoSections";
import {
  BLANK_MISPRICING,
  CLAMPED_DCF_SUMMARY,
  PRICED_DCF_SUMMARY,
  UNPRICED_DCF_SUMMARY,
  makeMemo,
} from "@/test/fixtures/memo";
import { makeDisagreementSummary, makeInsufficientSummary, makeSummary } from "@/test/fixtures/scorecard";
import type { StockMemoOut } from "@/types";

function renderMemo(memo: StockMemoOut) {
  return render(<FullInvestmentMemo memo={memo} open onClose={() => {}} />);
}

describe("FullInvestmentMemo", () => {
  it("renders the cover identity from the fixture", () => {
    renderMemo(makeMemo());
    expect(screen.getByText("Costco Wholesale")).toBeInTheDocument();
    expect(screen.getByText(/COST · Consumer Staples/)).toBeInTheDocument();
  });

  describe("degraded_agents banner", () => {
    it("shows the partial-coverage banner with agent names when non-empty", () => {
      renderMemo(makeMemo({ degraded_agents: ["earnings analyst", "comps analyst"] }));
      expect(screen.getByText("Partial coverage:")).toBeInTheDocument();
      expect(screen.getByText(/earnings analyst, comps analyst/)).toBeInTheDocument();
    });

    it("hides the banner when degraded_agents is empty", () => {
      renderMemo(makeMemo({ degraded_agents: [] }));
      expect(screen.queryByText("Partial coverage:")).not.toBeInTheDocument();
    });

    it("hides the banner when degraded_agents is absent (older memos)", () => {
      renderMemo(makeMemo({ degraded_agents: undefined }));
      expect(screen.queryByText("Partial coverage:")).not.toBeInTheDocument();
    });
  });

  describe("mispricing thesis section", () => {
    it("renders consensus / our view / gap / falsifiers when populated", () => {
      renderMemo(makeMemo());
      expect(screen.getByText("Where We Differ From Consensus")).toBeInTheDocument();
      expect(screen.getByText(/Consensus sees membership growth decelerating/)).toBeInTheDocument();
      expect(screen.getByText(/renewal rates holding above 90%/)).toBeInTheDocument();
      expect(screen.getByText(/underprices membership stickiness/)).toBeInTheDocument();
      expect(screen.getByText(/US renewal rate prints below 88%/)).toBeInTheDocument();
    });

    it("hides the section when every field is blank", () => {
      renderMemo(makeMemo({ mispricing_thesis: BLANK_MISPRICING }));
      expect(screen.queryByText("Where We Differ From Consensus")).not.toBeInTheDocument();
    });

    it("hides the section when the field is absent (older memos)", () => {
      renderMemo(makeMemo({ mispricing_thesis: undefined }));
      expect(screen.queryByText("Where We Differ From Consensus")).not.toBeInTheDocument();
    });
  });

  describe("valuation verdict", () => {
    it("renders the reconciled verdict summary in the Valuation section", () => {
      renderMemo(makeMemo());
      expect(
        screen.getByText("Fairly priced: DCF base case lands within 5% of spot."),
      ).toBeInTheDocument();
    });

    it("omits the verdict line when valuation_verdict is absent", () => {
      renderMemo(makeMemo({ valuation_verdict: undefined }));
      expect(screen.queryByText(/Fairly priced:/)).not.toBeInTheDocument();
    });
  });

  describe("generation_mode label", () => {
    it("labels demo memos as illustrative", () => {
      renderMemo(makeMemo({ generation_mode: "demo" }));
      expect(
        screen.getByText(/Demo dataset — figures are illustrative, not live market data/),
      ).toBeInTheDocument();
      expect(screen.queryByText("Live market data")).not.toBeInTheDocument();
    });

    it("labels live memos as live market data", () => {
      renderMemo(makeMemo({ generation_mode: "live" }));
      expect(screen.getByText("Live market data")).toBeInTheDocument();
      expect(screen.queryByText(/Demo dataset/)).not.toBeInTheDocument();
    });
  });

  describe("empty-array sections", () => {
    it("renders Catalysts when present", () => {
      renderMemo(makeMemo());
      expect(screen.getByText("Catalysts")).toBeInTheDocument();
      expect(screen.getByText("Membership fee increase")).toBeInTheDocument();
    });

    it("hides Catalysts when the array is empty", () => {
      renderMemo(makeMemo({ catalysts: [] }));
      expect(screen.queryByText("Catalysts")).not.toBeInTheDocument();
    });

    it("renders Risks & Thesis Breakers when key_risks is present", () => {
      renderMemo(makeMemo());
      expect(screen.getByText("Risks & Thesis Breakers")).toBeInTheDocument();
      expect(screen.getByText("Multiple compression")).toBeInTheDocument();
    });

    it("hides Risks & Thesis Breakers when both risk arrays are empty", () => {
      renderMemo(makeMemo({ key_risks: [], thesis_breakers: [] }));
      expect(screen.queryByText("Risks & Thesis Breakers")).not.toBeInTheDocument();
    });
  });

  describe("DCF summary", () => {
    it("renders the graph's base_implied_price / base_upside on the cover and in the table", () => {
      renderMemo(makeMemo({ dcf_summary: PRICED_DCF_SUMMARY }));
      expect(screen.getByText(/Fair value: \$918\.00 \(\+2\.0%\)/)).toBeInTheDocument();
      // KvTable rows
      expect(screen.getByText("Fair value / share")).toBeInTheDocument();
      expect(screen.getAllByText("$918.00").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("+2.0%")).toBeInTheDocument();
      expect(screen.queryByText("Terminal value clamped")).not.toBeInTheDocument();
    });

    it("renders n/a — never $0.00 or +0.0% — when the DCF could not price the shares", () => {
      renderMemo(makeMemo({ dcf_summary: UNPRICED_DCF_SUMMARY }));
      expect(screen.getByText(/Fair value: n\/a \(n\/a\)/)).toBeInTheDocument();
      // Two KvTable cells (fair value / share, implied upside).
      expect(screen.getAllByText("n/a")).toHaveLength(2);
      expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
      expect(screen.queryByText(/\+0\.0%/)).not.toBeInTheDocument();
    });

    it("shows dashes, not n/a, when the memo carries no DCF at all", () => {
      renderMemo(makeMemo({ dcf_summary: {} }));
      expect(screen.queryByText(/Fair value:/)).not.toBeInTheDocument();
      expect(screen.queryByText("n/a")).not.toBeInTheDocument();
    });

    it("shows the terminal-clamp badge with an explanatory tooltip when tv_clamped is set", () => {
      renderMemo(makeMemo({ dcf_summary: CLAMPED_DCF_SUMMARY }));
      const badge = screen.getByText("Terminal value clamped").closest("[title]");
      expect(badge).not.toBeNull();
      expect(badge?.getAttribute("title")).toMatch(/0\.5% floor/);
    });
  });

  describe("Fundamental Factor Scorecard section (Phase 6)", () => {
    it("renders the observed inputs and the model read in separate blocks", () => {
      renderMemo(makeMemo({ scorecard: makeSummary() }));
      expect(screen.getByText("Fundamental Factor Scorecard")).toBeInTheDocument();
      const observed = screen.getByTestId("memo-scorecard-observed");
      expect(observed).toHaveTextContent("Observed — reported inputs");
      expect(observed).toHaveTextContent("FY2025");
      expect(observed).toHaveTextContent("2026-02-12");
      expect(observed).toHaveTextContent("93%");
      // Nothing from the model read leaks into the observed block.
      expect(observed).not.toHaveTextContent("62.4");
      expect(observed).not.toHaveTextContent("71st");
      const model = screen.getByTestId("memo-scorecard-model");
      expect(model).toHaveTextContent("Model read — fs-v1");
      expect(model).toHaveTextContent("62.4");
      expect(model).toHaveTextContent("+0.62");
      expect(model).toHaveTextContent("71st");
      expect(model).toHaveTextContent("58th");
      expect(model).toHaveTextContent("50 = z of 0");
      expect(screen.getByTestId("memo-scorecard-profile")).toHaveTextContent("reads as a compounder");
      expect(screen.getByTestId("memo-scorecard-profile")).toHaveTextContent("not a recommendation");
      expect(screen.getByTestId("memo-scorecard")).toHaveTextContent("did not move the rating");
      const quality = screen.getByTestId("memo-category-quality");
      expect(quality).toHaveTextContent("78.5");
      expect(quality).toHaveTextContent("4/4");
      expect(screen.getByTestId("memo-scorecard-top-positive")).toHaveTextContent("Accruals ratio");
      expect(screen.getByTestId("memo-scorecard-top-negative")).toHaveTextContent("Fcf yield");
      expect(screen.queryByTestId("memo-scorecard-disagreement")).toBeNull();
    });

    it("renders the disagreement callout with the PM reconciliation when the memo contradicts the quant read", () => {
      renderMemo(makeMemo({ scorecard: makeDisagreementSummary() }));
      const note = screen.getByTestId("memo-scorecard-disagreement");
      expect(note).toHaveAttribute("role", "note");
      expect(note).toHaveTextContent("Memo / scorecard disagreement · material · overall");
      expect(note).toHaveTextContent("Memo rating Bullish (70) vs universe percentile 18: gap 52 points.");
      expect(note).toHaveTextContent("the memo is more positive than the quant read; gap +52 points.");
      expect(note).toHaveTextContent("PM reconciliation: The PM attributes the gap to a one-off restructuring charge");
    });

    it("says the reconciliation is not yet written rather than hiding the disagreement", () => {
      renderMemo(makeMemo({ scorecard: makeDisagreementSummary({ reconciliation: null }) }));
      expect(screen.getByTestId("memo-scorecard-disagreement")).toHaveTextContent("PM reconciliation: n/a (not yet written)");
    });

    it("renders n/a with a reason — never 0 — when the run could not score the name", () => {
      renderMemo(makeMemo({ scorecard: makeInsufficientSummary() }));
      const model = screen.getByTestId("memo-scorecard-model");
      expect(model).toHaveTextContent("n/a (insufficient coverage: fewer than 5 families scored)");
      expect(model).toHaveTextContent("n/a (unranked)");
      expect(model).not.toHaveTextContent(/\b0\.0\b/);
      expect(screen.getByTestId("memo-scorecard")).toHaveTextContent("stale (the latest run is older than 45 days)");
      const growth = screen.getByTestId("memo-category-growth");
      expect(growth).toHaveAttribute("data-missing", "true");
      expect(growth).toHaveTextContent("n/a (not scored)");
      const quality = screen.getByTestId("memo-category-quality");
      expect(quality).toHaveTextContent("n/a (1 of 4 inputs)");
      expect(within(screen.getByTestId("memo-scorecard-top-positive")).getByText("n/a (overall not scored)")).toBeInTheDocument();
    });

    it("hides the section on memos that pre-date the field or carry null", () => {
      renderMemo(makeMemo({ scorecard: undefined }));
      expect(screen.queryByText("Fundamental Factor Scorecard")).not.toBeInTheDocument();
      expect(screen.queryByTestId("memo-scorecard")).toBeNull();
      renderMemo(makeMemo({ scorecard: null }));
      expect(screen.queryByTestId("memo-scorecard")).toBeNull();
    });
  });
});

// W2a — placeholders from the captured presenter output. The PDF is the
// modal's DOM printed from a popup, so the test captures what
// `downloadPdf` writes into that popup.
describe("FullInvestmentMemo W2a placeholders (captured presenter output)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  function placeholderSections(): string[] {
    return screen
      .queryAllByTestId("unavailable-section")
      .map((el) => el.getAttribute("data-section") ?? "")
      .sort();
  }

  function printedHtml(): string {
    let written = "";
    const popup = {
      document: {
        open: () => {},
        write: (html: string) => {
          written += html;
        },
        close: () => {},
      },
      focus: () => {},
      print: () => {},
      onload: null as null | (() => void),
    };
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window);
    fireEvent.click(screen.getByText("Download PDF"));
    return written;
  }

  it("GOOGL: the PDF carries placeholders and reasons, never the PM template tail or Portfolio Fit", () => {
    renderMemo(presentedMemo("googl_live_prepflag"));
    const html = printedHtml();
    expect(html).toContain(UNAVAILABLE_TEXT);
    expect(html).toContain(REASON_TEXT.template_fallback);
    expect(html).toContain(REASON_TEXT.critic_not_run);
    expect(html).toContain("Conviction: unavailable in this version");
    expect(html).not.toContain(PM_TEMPLATE_TAIL);
    expect(html).not.toContain("Portfolio Fit");
    expect(html).not.toMatch(/Conviction: \d+\/100/);
  });

  it("GOOGL: hidden sections render as placeholders in the modal", () => {
    renderMemo(presentedMemo("googl_live_prepflag"));
    expect(placeholderSections()).toEqual([
      "bear_case",
      "bull_case",
      "final_pm_view",
      "final_verdict",
      "mispricing_thesis",
      "one_sentence_thesis",
      "risk_committee_challenge",
      "sector_agent_view",
      "technical_agent_view",
    ]);
    expect(screen.getByText(REASON_TEXT.pm_view_unavailable)).toBeInTheDocument();
    expect(screen.getByText(/Synthetic intake rationale 40 for the googl_live_prepflag fixture\./)).toBeInTheDocument();
    // Banner count excludes Portfolio Fit (template_always) and technical
    // (intake skip).
    expect(screen.getByTestId("unavailable-count")).toHaveTextContent("8 sections unavailable in this version.");
  });

  it("META: only technical and the critic are placeholders; Portfolio Fit is omitted", () => {
    renderMemo(presentedMemo("meta_v1"));
    expect(placeholderSections()).toEqual(["risk_committee_challenge", "technical_agent_view"]);
    expect(screen.queryByText("Portfolio Fit")).not.toBeInTheDocument();
    expect(screen.getByText("Synthetic thesis 96 for the meta_v1 fixture.")).toBeInTheDocument();
    expect(screen.getByText("Conviction: 71/100")).toBeInTheDocument();
  });

  it("still prints Portfolio Fit for a memo without a map (pre-W2a body)", () => {
    renderMemo(makeMemo({ section_availability: undefined }));
    expect(screen.getByText("Portfolio Fit")).toBeInTheDocument();
    expect(screen.getByText("Core defensive holding.")).toBeInTheDocument();
    expect(screen.queryAllByTestId("unavailable-section")).toHaveLength(0);
  });
});
