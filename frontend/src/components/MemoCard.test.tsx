import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import MemoCard from "@/components/MemoCard";
import { REASON_TEXT, SECTION_REASON_TEXT, UNAVAILABLE_TEXT } from "@/lib/memoSections";
import {
  PM_TEMPLATE_TAIL,
  PRESENTED_MEMO_NAMES,
  PROBES,
  THESIS_BUILDER_CLAUSE,
  countedPlaceholders,
  presentedMemo,
  withRawTemplateProse,
} from "@/test/fixtures/memoSections";
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

    it("labels the stored DCF price as the memo's, never as 'Current' (W5b)", () => {
      renderCard(makeMemo({ dcf_summary: PRICED_DCF_SUMMARY, generated_at: "2026-09-03T14:00:00" }));
      expect(screen.getByText("Price used in DCF")).toBeInTheDocument();
      expect(screen.queryByText("Current")).not.toBeInTheDocument();
      expect(screen.getByText("as of memo, 2026-09-03")).toBeInTheDocument();
    });

    it("dates the DCF price in ET, so an evening memo is not shown on the next day (W5b)", () => {
      // 01:30 UTC on Sep 23 is 9:30 PM ET on Sep 22, the date the live chip shows.
      renderCard(makeMemo({ dcf_summary: PRICED_DCF_SUMMARY, generated_at: "2026-09-23T01:30:00" }));
      expect(screen.getByText("as of memo, 2026-09-22")).toBeInTheDocument();
    });

    it("shows the terminal-clamp badge with an explanatory tooltip when tv_clamped is set", () => {
      renderCard(makeMemo({ dcf_summary: CLAMPED_DCF_SUMMARY }));
      const badge = screen.getByText("Terminal value clamped").closest("[title]");
      expect(badge).not.toBeNull();
      expect(badge?.getAttribute("title")).toMatch(/0\.5% floor/);
    });
  });
});

// W2a — the card renders the presenter's verdicts from the captured wire
// fixture (the server's real output, pinned by
// `test_memo_sections_fixture_contract.py`), never from hand-built maps.
describe("MemoCard W2a placeholders (captured presenter output)", () => {
  function placeholderSections(): string[] {
    return screen
      .queryAllByTestId("unavailable-section")
      .map((el) => el.getAttribute("data-section") ?? "")
      .sort();
  }
  function placeholder(section: string): HTMLElement {
    const el = document.querySelector(`[data-testid="unavailable-section"][data-section="${section}"]`);
    if (!el) throw new Error(`no placeholder for ${section}`);
    return el as HTMLElement;
  }

  it("GOOGL: the template PM view and everything built on it read as unavailable, with reasons", () => {
    const { container } = renderCard(presentedMemo("googl_live_prepflag"));
    expect(placeholderSections()).toEqual([
      "bear_case",
      "bull_case",
      "confidence_score",
      "final_pm_view",
      "final_verdict",
      "mispricing_thesis",
      "one_sentence_thesis",
      "risk_committee_challenge",
      "sector_agent_view",
      "sector_synthesis",
      "technical_agent_view",
    ]);
    for (const section of ["one_sentence_thesis", "final_pm_view", "confidence_score", "mispricing_thesis", "sector_agent_view"]) {
      expect(within(placeholder(section)).getByText(UNAVAILABLE_TEXT)).toBeInTheDocument();
      expect(within(placeholder(section)).getByText(REASON_TEXT.template_fallback)).toBeInTheDocument();
    }
    expect(within(placeholder("final_verdict")).getByText(REASON_TEXT.derived_from_hidden)).toBeInTheDocument();
    expect(within(placeholder("sector_synthesis")).getByText(REASON_TEXT.derived_from_hidden)).toBeInTheDocument();
    expect(within(placeholder("risk_committee_challenge")).getByText(REASON_TEXT.critic_not_run)).toBeInTheDocument();
    // Nothing reads the placeholder twice (the stored field already holds it).
    expect(within(placeholder("final_pm_view")).getAllByText(UNAVAILABLE_TEXT)).toHaveLength(1);
    expect(container.textContent).not.toContain(PM_TEMPLATE_TAIL);
  });

  it("GOOGL: the confidence number is withheld and the rating says what it rests on", () => {
    renderCard(presentedMemo("googl_live_prepflag"));
    const card = screen.getByText("Confidence").closest(".card-tight") as HTMLElement;
    expect(within(card).getByText(UNAVAILABLE_TEXT)).toBeInTheDocument();
    expect(within(card).queryByText("59")).not.toBeInTheDocument();
    expect(screen.getByText(REASON_TEXT.pm_view_unavailable)).toBeInTheDocument();
  });

  it("GOOGL: the case headlines are placeholders and removed template items are counted", () => {
    renderCard(presentedMemo("googl_live_prepflag"));
    const bull = screen.getByTestId("case-bull");
    expect(within(bull).getByText(UNAVAILABLE_TEXT)).toBeInTheDocument();
    expect(within(bull).getByText(/1 template item not shown/)).toBeInTheDocument();
    // The analyst's own DCF-driver items survive.
    expect(within(bull).getByText(/Synthetic driver 67/)).toBeInTheDocument();
    expect(screen.getByText(/Key Risks & Thesis Breakers/).closest(".card-tight")).toHaveTextContent(
      "1 template item not shown",
    );
  });

  it("shows the PM's intake rationale in a skipped analyst's placeholder", () => {
    renderCard(presentedMemo("googl_live_prepflag"));
    const tech = placeholder("technical_agent_view");
    expect(within(tech).getByText("Technical Analyst")).toBeInTheDocument();
    expect(within(tech).getByText(REASON_TEXT.skipped_by_intake)).toBeInTheDocument();
    expect(within(tech).getByText(/Synthetic intake rationale 40 for the googl_live_prepflag fixture\./)).toBeInTheDocument();
  });

  it("banner counts hidden sections but not template_always or intake skips", () => {
    renderCard(presentedMemo("googl_live_prepflag"));
    // No degraded agents on this memo: the banner shows for the hidden
    // sections alone. 8 = thesis, PM view, confidence, mispricing, sector
    // view, sector synthesis, critic, final verdict; not portfolio fit
    // (template_always) and not technical (intake skip).
    expect(screen.getByText("Partial result:")).toBeInTheDocument();
    expect(screen.getByTestId("unavailable-count")).toHaveTextContent(
      "8 sections unavailable in this version.",
    );
  });

  it("AAPL: a demo memo's template sections are placeholders while computed numbers stay", () => {
    renderCard(presentedMemo("aapl_demo"));
    const bull = screen.getByTestId("case-bull");
    expect(within(bull).getByText(REASON_TEXT.template_fallback)).toBeInTheDocument();
    expect(within(bull).getByText("2 template items not shown")).toBeInTheDocument();
    expect(within(bull).getByText("DCF bull case implies $150.20 (-55%).")).toBeInTheDocument();
    const bear = screen.getByTestId("case-bear");
    expect(within(bear).getByText(UNAVAILABLE_TEXT)).toBeInTheDocument();
    expect(within(bear).getByText(/2 template items not shown/)).toBeInTheDocument();
    expect(within(bear).getByText("Risk lens: DCF bear scenario maps explicit downside")).toBeInTheDocument();
    // Every key risk was template-derived: the card stays, as a placeholder.
    const risks = placeholder("key_risks");
    expect(within(risks).getByText("Key Risks & Thesis Breakers")).toBeInTheDocument();
    expect(within(risks).getByText("2 template items not shown")).toBeInTheDocument();
    expect(within(placeholder("earnings_agent_view")).getByText(REASON_TEXT.no_source_data)).toBeInTheDocument();
    // The comps card is computed and shown; only its template drill-down is gone.
    expect(screen.getByTestId("drilldown-unavailable")).toBeInTheDocument();
    expect(screen.getByText("Next earnings: 2026-10-29")).toBeInTheDocument();
  });

  it("META: the agentic memo shows placeholders only for technical and the critic", () => {
    // Portfolio Fit is the third hidden section, but the card never
    // renders it.
    renderCard(presentedMemo("meta_v1"));
    expect(placeholderSections()).toEqual(["risk_committee_challenge", "technical_agent_view"]);
    expect(screen.getByText("Synthetic thesis 96 for the meta_v1 fixture.")).toBeInTheDocument();
    expect(screen.getByText("Synthetic PM view 95 for the meta_v1 fixture.")).toBeInTheDocument();
    expect(screen.getByText("71")).toBeInTheDocument();
    expect(screen.queryByText(REASON_TEXT.pm_view_unavailable)).not.toBeInTheDocument();
    expect(
      within(placeholder("technical_agent_view")).getByText(/Synthetic intake rationale 64 for the meta_v1 fixture\./),
    ).toBeInTheDocument();
    expect(screen.getByTestId("unavailable-count")).toHaveTextContent("· 1 section unavailable in this version.");
  });

  it("renders no placeholder for a memo without a map (pre-W2a body)", () => {
    renderCard(makeMemo({ section_availability: undefined }));
    expect(screen.queryAllByTestId("unavailable-section")).toHaveLength(0);
    expect(screen.getByText("62")).toBeInTheDocument();
  });
});

describe("MemoCard W2a banner and notes (review fixes)", () => {
  function bannerNumber(): number {
    const el = screen.queryByTestId("unavailable-count");
    if (!el) return 0;
    const m = /(\d+) sections? unavailable/.exec(el.textContent ?? "");
    if (!m) throw new Error(`unparseable banner: ${el.textContent}`);
    return Number(m[1]);
  }

  const bodies: [string, () => StockMemoOut][] = [
    ...PRESENTED_MEMO_NAMES.map((n): [string, () => StockMemoOut] => [n, () => presentedMemo(n)]),
    ...Object.entries(PROBES).map(([n, f]): [string, () => StockMemoOut] => [`probe:${n}`, f]),
  ];

  it.each(bodies)("%s: the banner counts exactly the placeholders the card shows", (_name, build) => {
    const memo = build();
    const { container, unmount } = renderCard(memo);
    expect(bannerNumber()).toBe(countedPlaceholders(memo, container).length);
    unmount();
  });

  it("does not count a hidden industry finding the card never shows", () => {
    // FEAT-003 routing (owner decision 10) gives every unmapped ticker one.
    renderCard(PROBES.industry());
    expect(screen.getByTestId("unavailable-count")).toHaveTextContent("· 1 section unavailable in this version.");
  });

  it("shows why a failed valuation verdict and DCF are missing", () => {
    renderCard(PROBES.valuationVerdict());
    const vv = document.querySelector('[data-section="valuation_verdict"]') as HTMLElement;
    expect(within(vv).getByText("Valuation verdict")).toBeInTheDocument();
    expect(within(vv).getByText(REASON_TEXT.agent_failed)).toBeInTheDocument();
  });

  it("notes a degraded thesis instead of letting the builder sentence read as analysis", () => {
    renderCard(PROBES.degradedThesis());
    // The thesis stays visible (critique delta 11) ...
    expect(screen.getByText(new RegExp(THESIS_BUILDER_CLAUSE))).toBeInTheDocument();
    // ... with the thesis-specific note under it.
    const note = document.querySelector('[data-testid="degraded-note"][data-section="one_sentence_thesis"]');
    expect(note).toHaveTextContent(SECTION_REASON_TEXT.one_sentence_thesis.partial_template!);
    expect(note).not.toHaveTextContent(REASON_TEXT.partial_template);
  });

  it("follows the map, not the text: raw template prose under a hidden verdict never prints", () => {
    const { container } = renderCard(withRawTemplateProse("googl_live_prepflag"));
    expect(container.textContent).not.toContain(PM_TEMPLATE_TAIL);
    expect(document.querySelector('[data-section="final_pm_view"]')).toHaveTextContent(UNAVAILABLE_TEXT);
  });

  it("ABBV: a not-produced mispricing view gets a placeholder instead of vanishing", () => {
    renderCard(presentedMemo("abbv_v7_patch"));
    const misp = document.querySelector(
      '[data-testid="unavailable-section"][data-section="mispricing_thesis"]',
    ) as HTMLElement;
    expect(within(misp).getByText("Where We Differ From Consensus")).toBeInTheDocument();
    expect(within(misp).getByText(REASON_TEXT.not_produced)).toBeInTheDocument();
  });
});
