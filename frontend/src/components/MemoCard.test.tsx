import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import MemoCard from "@/components/MemoCard";
import {
  BLANK_MISPRICING,
  CLAMPED_DCF_SUMMARY,
  PRICED_DCF_SUMMARY,
  UNPRICED_DCF_SUMMARY,
  makeMemo,
} from "@/test/fixtures/memo";
import type { StockMemoOut } from "@/types";

// MemoryRouter because CrossSectorChips renders react-router <Link>s when
// the sector agent flags cross-sector tickers.
function renderCard(memo: StockMemoOut) {
  return render(
    <MemoryRouter>
      <MemoCard memo={memo} />
    </MemoryRouter>,
  );
}

describe("MemoCard", () => {
  it("renders the identity row and thesis from the fixture", () => {
    renderCard(makeMemo());
    expect(screen.getByText("Costco Wholesale")).toBeInTheDocument();
    expect(
      screen.getByText("COST is fairly priced for best-in-class execution."),
    ).toBeInTheDocument();
  });

  describe("degraded_agents banner", () => {
    it("shows the partial-result banner with agent names when non-empty", () => {
      renderCard(makeMemo({ degraded_agents: ["earnings analyst", "comps analyst"] }));
      expect(screen.getByText("Partial result:")).toBeInTheDocument();
      expect(screen.getByText("earnings analyst, comps analyst")).toBeInTheDocument();
    });

    it("hides the banner when degraded_agents is empty", () => {
      renderCard(makeMemo({ degraded_agents: [] }));
      expect(screen.queryByText("Partial result:")).not.toBeInTheDocument();
    });

    it("hides the banner when degraded_agents is absent (older memos)", () => {
      renderCard(makeMemo({ degraded_agents: undefined }));
      expect(screen.queryByText("Partial result:")).not.toBeInTheDocument();
    });
  });

  describe("mispricing thesis card", () => {
    it("renders consensus / our view / gap / falsifiers when populated", () => {
      renderCard(makeMemo());
      expect(screen.getByText("Where We Differ From Consensus")).toBeInTheDocument();
      expect(screen.getByText(/Consensus sees membership growth decelerating/)).toBeInTheDocument();
      expect(screen.getByText(/renewal rates holding above 90%/)).toBeInTheDocument();
      expect(screen.getByText(/underprices membership stickiness/)).toBeInTheDocument();
      expect(screen.getByText(/US renewal rate prints below 88%/)).toBeInTheDocument();
    });

    it("hides the card when every field is blank", () => {
      renderCard(makeMemo({ mispricing_thesis: BLANK_MISPRICING }));
      expect(screen.queryByText("Where We Differ From Consensus")).not.toBeInTheDocument();
    });

    it("hides the card when the field is absent (older memos)", () => {
      renderCard(makeMemo({ mispricing_thesis: undefined }));
      expect(screen.queryByText("Where We Differ From Consensus")).not.toBeInTheDocument();
    });
  });

  describe("valuation verdict", () => {
    it("renders the verdict summary under the headline thesis", () => {
      renderCard(makeMemo());
      expect(screen.getByText("Valuation verdict")).toBeInTheDocument();
      expect(
        screen.getByText(/Fairly priced: DCF base case lands within 5% of spot\./),
      ).toBeInTheDocument();
    });

    it("omits the verdict block when valuation_verdict is absent", () => {
      renderCard(makeMemo({ valuation_verdict: undefined }));
      expect(screen.queryByText("Valuation verdict")).not.toBeInTheDocument();
    });
  });

  describe("generation_mode label", () => {
    it("labels demo memos as demo data", () => {
      renderCard(makeMemo({ generation_mode: "demo" }));
      expect(screen.getByText("demo data")).toBeInTheDocument();
      expect(screen.queryByText("live data")).not.toBeInTheDocument();
    });

    it("labels live memos as live data", () => {
      renderCard(makeMemo({ generation_mode: "live" }));
      expect(screen.getByText("live data")).toBeInTheDocument();
      expect(screen.queryByText("demo data")).not.toBeInTheDocument();
    });
  });

  describe("empty-array sections", () => {
    it("renders Catalysts and Key Risks when present", () => {
      renderCard(makeMemo());
      expect(screen.getByText("Catalysts")).toBeInTheDocument();
      expect(screen.getByText("Membership fee increase")).toBeInTheDocument();
      expect(screen.getByText("Key Risks & Thesis Breakers")).toBeInTheDocument();
      expect(screen.getByText("Multiple compression")).toBeInTheDocument();
    });

    it("hides the Catalysts card when the array is empty", () => {
      renderCard(makeMemo({ catalysts: [] }));
      expect(screen.queryByText("Catalysts")).not.toBeInTheDocument();
      expect(screen.getByText("Key Risks & Thesis Breakers")).toBeInTheDocument();
    });

    it("hides the Key Risks card when the array is empty", () => {
      renderCard(makeMemo({ key_risks: [] }));
      expect(screen.queryByText("Key Risks & Thesis Breakers")).not.toBeInTheDocument();
      expect(screen.getByText("Catalysts")).toBeInTheDocument();
    });

    it("hides the whole row when both arrays are empty", () => {
      renderCard(makeMemo({ catalysts: [], key_risks: [] }));
      expect(screen.queryByText("Catalysts")).not.toBeInTheDocument();
      expect(screen.queryByText("Key Risks & Thesis Breakers")).not.toBeInTheDocument();
    });
  });

  describe("DCF snapshot", () => {
    it("renders priced scenarios with signed upside", () => {
      renderCard(makeMemo({ dcf_summary: PRICED_DCF_SUMMARY }));
      expect(screen.getByText("$918.00")).toBeInTheDocument();
      expect(screen.getByText("+2.0%")).toBeInTheDocument();
      expect(screen.getByText("-20.0%")).toBeInTheDocument();
      expect(screen.queryByText("Terminal value clamped")).not.toBeInTheDocument();
    });

    it("renders n/a — never $0.00 or +0.0% — when the DCF could not price the shares", () => {
      renderCard(makeMemo({ dcf_summary: UNPRICED_DCF_SUMMARY }));
      // Current + three implied prices + three upsides.
      expect(screen.getAllByText("n/a")).toHaveLength(7);
      expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
      expect(screen.queryByText(/\+0\.0%/)).not.toBeInTheDocument();
      expect(screen.queryByText("DCF unavailable.")).not.toBeInTheDocument();
    });

    it("does not recompute an upside from a null current price on older memos", () => {
      // Older memos lack per-scenario upside keys; the fallback recompute
      // must yield n/a, not -100%, when the quote is missing.
      const legacy: Record<string, unknown> = { ...PRICED_DCF_SUMMARY, current_price: null };
      delete legacy.base_upside;
      delete legacy.bull_upside;
      delete legacy.bear_upside;
      renderCard(makeMemo({ dcf_summary: legacy }));
      expect(screen.getByText("$918.00")).toBeInTheDocument();
      expect(screen.getAllByText("n/a")).toHaveLength(4);
      expect(screen.queryByText("-100.0%")).not.toBeInTheDocument();
    });

    it("shows the terminal-clamp badge with an explanatory tooltip when tv_clamped is set", () => {
      renderCard(makeMemo({ dcf_summary: CLAMPED_DCF_SUMMARY }));
      const badge = screen.getByText("Terminal value clamped").closest("[title]");
      expect(badge).not.toBeNull();
      expect(badge?.getAttribute("title")).toMatch(/0\.5% floor/);
    });
  });
});
