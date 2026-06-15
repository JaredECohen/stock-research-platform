import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import FullInvestmentMemo from "@/components/FullInvestmentMemo";
import { BLANK_MISPRICING, makeMemo } from "@/test/fixtures/memo";
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
});
