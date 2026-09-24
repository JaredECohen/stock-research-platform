import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import CommentaryBlock from "@/components/public/CommentaryBlock";
import DataFreshness from "@/components/public/DataFreshness";
import SafeSection from "@/components/public/SafeSection";
import SampleCompsTable from "@/components/public/SampleCompsTable";
import SampleDCFCard from "@/components/public/SampleDCFCard";
import SampleFundamentalsChart, { metricLabel } from "@/components/public/SampleFundamentalsChart";
import SampleMemoSummary from "@/components/public/SampleMemoSummary";
import SamplePriceChart from "@/components/public/SamplePriceChart";
import SampleScreenerRow from "@/components/public/SampleScreenerRow";
import { REASON_TEXT, UNAVAILABLE_TEXT } from "@/lib/memoSections";
import { makeMemo } from "@/test/fixtures/memo";
import { PM_TEMPLATE_TAIL, presentedMemo } from "@/test/fixtures/memoSections";
import { makeComps, makeDCF, makeSample } from "@/test/fixtures/sample";

vi.mock("recharts", () => {
  const Noop = () => null;
  return { Bar: Noop, BarChart: Noop, CartesianGrid: Noop, Line: Noop, LineChart: Noop, ResponsiveContainer: Noop, Tooltip: Noop, XAxis: Noop, YAxis: Noop };
});

describe("SampleDCFCard", () => {
  it("shows base / bull / bear with constant discount rate and terminal growth", () => {
    render(<SampleDCFCard dcf={makeDCF()} />);
    expect(screen.getByText("Scenarios, not confidence intervals")).toBeInTheDocument();
    const base = document.querySelector('[data-scenario="base"]') as HTMLElement;
    expect(within(base).getByText("$918.00")).toBeInTheDocument();
    expect(within(base).getByText("+2.0%")).toBeInTheDocument();
    const bear = document.querySelector('[data-scenario="bear"]') as HTMLElement;
    expect(within(bear).getByText("-20.0%")).toBeInTheDocument();
    expect(screen.getByText(/held constant across scenarios/)).toHaveTextContent("8.0%");
    expect(screen.getByText(/held constant across scenarios/)).toHaveTextContent("2.5%");
  });

  it("prints n/a, never $0.00, when the engine could not price the shares", () => {
    const dcf = makeDCF();
    dcf.base = { ...dcf.base, implied_share_price: null, upside_pct: null, tv_clamped: true };
    dcf.current_price = null;
    dcf.guardrails = [{ severity: "warn", message: "Terminal value is 90% of EV", metric: "tv_share" }];
    render(<SampleDCFCard dcf={dcf} />);
    const base = document.querySelector('[data-scenario="base"]') as HTMLElement;
    expect(within(base).getAllByText("n/a")).toHaveLength(2);
    expect(base.textContent).not.toContain("$0.00");
    expect(within(base).getByText("terminal value capped")).toBeInTheDocument();
    expect(screen.getByText(/Price at model:/)).toHaveTextContent("n/a");
    expect(screen.getByText(/Caution: Terminal value is 90% of EV/)).toBeInTheDocument();
  });

  it("says when the rates vary by scenario instead of claiming they are constant", () => {
    const dcf = makeDCF();
    dcf.bull = { ...dcf.bull, assumptions: { ...dcf.bull.assumptions, wacc: 0.07 } };
    render(<SampleDCFCard dcf={dcf} />);
    expect(screen.getByText(/vary by scenario/)).toBeInTheDocument();
    expect(screen.queryByText(/held constant/)).not.toBeInTheDocument();
  });
});

describe("SampleCompsTable", () => {
  it("renders target, peers and the median with dashes for missing metrics", () => {
    render(<SampleCompsTable comps={makeComps()} />);
    const rows = screen.getAllByRole("row");
    expect(rows).toHaveLength(5); // header + target + 2 peers + median
    expect(document.querySelector('tr[data-kind="target"]')).toHaveTextContent("COST");
    expect(document.querySelector('tr[data-kind="median"]')).toHaveTextContent("Median");
    const bj = screen.getByRole("row", { name: /BJ/ });
    expect(within(bj).getAllByText("—")).toHaveLength(2);
    expect(within(bj).getByText("22.4x")).toBeInTheDocument();
    expect(screen.getByText(/premium to the peer median/)).toBeInTheDocument();
  });
});

describe("SampleFundamentalsChart", () => {
  it("switches metrics and prints the numbers, with n/a (not obtained) for blanks", () => {
    render(<SampleFundamentalsChart fundamentals={makeSample().fundamentals!} />);
    const revenue = screen.getByRole("button", { name: "Revenue" });
    expect(revenue).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("$242.3B")).toBeInTheDocument();
    expect(screen.getByText("n/a (not obtained)")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Free cash flow" }));
    expect(screen.getByRole("button", { name: "Free cash flow" })).toHaveAttribute("aria-pressed", "true");
    expect(revenue).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText("$6.7B")).toBeInTheDocument();
    expect(screen.queryByText("n/a (not obtained)")).not.toBeInTheDocument();
    expect(metricLabel("operating_income")).toBe("Operating income");
    expect(metricLabel("weird_line")).toBe("Weird line");
  });

  it("says so when there is no series", () => {
    render(<SampleFundamentalsChart fundamentals={{ series: [] }} />);
    expect(screen.getByText("No fundamentals series stored for this sample.")).toBeInTheDocument();
  });
});

describe("SamplePriceChart", () => {
  it("summarises the series in text and renders nothing for an empty list", () => {
    const { unmount } = render(<SamplePriceChart prices={makeSample().prices!} />);
    expect(screen.getByText(/3 daily closes, 2025-09-08 to 2026-09-04/)).toHaveTextContent("$880.12");
    expect(screen.getByText(/not a live quote/)).toBeInTheDocument();
    unmount();
    const { container } = render(<SamplePriceChart prices={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("SampleScreenerRow", () => {
  it("shows the rank, the factor meters and the thesis lines", () => {
    render(<SampleScreenerRow row={makeSample().screener_row!} />);
    expect(screen.getByText("#12")).toBeInTheDocument();
    expect(screen.getByText("of 240")).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "Quality" })).toHaveAttribute("aria-valuenow", "80");
    expect(screen.getByText("Membership economics fully priced.")).toBeInTheDocument();
    expect(screen.getByText(/A ranking, not a forecast/)).toHaveTextContent("Sep 5, 2026");
  });

  it("marks a missing factor as not available rather than zero", () => {
    const row = { ...makeSample().screener_row!, quality: undefined as unknown as number };
    render(<SampleScreenerRow row={row} />);
    expect(screen.getByRole("meter", { name: "Quality" })).toHaveAttribute("aria-valuetext", "not available");
  });
});

describe("CommentaryBlock", () => {
  it("carries the model and date provenance", () => {
    render(<CommentaryBlock commentary={makeSample().commentary!} />);
    expect(screen.getByText(/Written by model/)).toHaveTextContent("openai:cheap on Sep 6, 2026");
    expect(screen.getByText(/not a recommendation/)).toBeInTheDocument();
  });
});

describe("DataFreshness", () => {
  it("states the build time and lists what is missing", () => {
    render(<DataFreshness builtAt="2026-09-06T07:00:00" degraded={["comps: not built", "commentary: skipped (no LLM configured)"]} />);
    expect(screen.getByText(/Built from stored research on/)).toHaveTextContent("September 6, 2026 at 07:00 UTC");
    expect(screen.getByText("What is missing from this sample (2)")).toBeInTheDocument();
    expect(screen.getByText("commentary: skipped (no LLM configured)")).toBeInTheDocument();
  });
  it("says when nothing is built", () => {
    render(<DataFreshness builtAt={null} />);
    expect(screen.getByText("This sample has not been built yet.")).toBeInTheDocument();
  });
});

describe("SampleMemoSummary", () => {
  it("renders the verdict layer read-only", () => {
    render(<SampleMemoSummary memo={makeMemo()} />);
    expect(screen.getByRole("heading", { name: /Costco Wholesale/ })).toBeInTheDocument();
    expect(screen.getByText("Neutral")).toBeInTheDocument();
    expect(screen.getByText("Bull case headline")).toBeInTheDocument();
    expect(screen.getByText("Bear case headline")).toBeInTheDocument();
    expect(screen.getByText("fairly priced")).toBeInTheDocument();
    expect(screen.getByText(/Model output, not a recommendation/)).toHaveTextContent("Demo-mode run");
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
  it("prints n/a when the verdict has no DCF upside", () => {
    render(<SampleMemoSummary memo={makeMemo({ valuation_verdict: { verdict: "overvalued", summary: "", dcf_base_upside: null } })} />);
    expect(screen.getByText("DCF base-case upside n/a")).toBeInTheDocument();
  });

  // W2a — the served sample memo is presented server-side; these bodies are
  // the presenter's captured output.
  describe("W2a placeholders (captured presenter output)", () => {
    function placeholderSections(): string[] {
      return screen
        .queryAllByTestId("unavailable-section")
        .map((el) => el.getAttribute("data-section") ?? "")
        .sort();
    }

    it("GOOGL: template thesis, PM view, confidence and case headlines read as unavailable", () => {
      const { container } = render(<SampleMemoSummary memo={presentedMemo("googl_live_prepflag")} />);
      expect(placeholderSections()).toEqual(["bear_case", "bull_case", "final_pm_view", "one_sentence_thesis"]);
      expect(screen.getAllByText(REASON_TEXT.template_fallback)).toHaveLength(2);
      expect(screen.getByTestId("sample-confidence")).toHaveTextContent("Confidence unavailable in this version");
      expect(screen.getByTestId("sample-confidence")).not.toHaveTextContent("59");
      expect(screen.getByText(REASON_TEXT.pm_view_unavailable)).toBeInTheDocument();
      expect(within(screen.getByTestId("case-bull")).getByText(/1 template item not shown/)).toBeInTheDocument();
      expect(screen.getByText(/Model output, not a recommendation/)).toHaveTextContent(
        "8 sections unavailable in this version.",
      );
      expect(container.textContent).not.toContain(PM_TEMPLATE_TAIL);
    });

    it("AAPL: the emptied key-risks list is a placeholder, not a vanished block", () => {
      render(<SampleMemoSummary memo={presentedMemo("aapl_demo")} />);
      const risks = document.querySelector('[data-section="key_risks"]') as HTMLElement;
      expect(within(risks).getByText(UNAVAILABLE_TEXT)).toBeInTheDocument();
      expect(within(risks).getByText("2 template items not shown")).toBeInTheDocument();
    });

    it("META: the agentic memo shows no placeholder", () => {
      render(<SampleMemoSummary memo={presentedMemo("meta_v1")} />);
      expect(placeholderSections()).toEqual([]);
      expect(screen.getByText("Synthetic thesis 96 for the meta_v1 fixture.")).toBeInTheDocument();
      expect(screen.getByTestId("sample-confidence")).toHaveTextContent("Confidence 71/100");
      expect(screen.getByText(/Model output, not a recommendation/)).toHaveTextContent(
        "1 section unavailable in this version.",
      );
    });
  });
});

describe("SafeSection", () => {
  it("replaces a throwing section with a labelled placeholder", () => {
    const Boom = () => {
      throw new Error("bad payload");
    };
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    render(
      <SafeSection label="The DCF scenarios">
        <Boom />
      </SafeSection>,
    );
    expect(screen.getByTestId("section-failed")).toHaveTextContent("The DCF scenarios could not be displayed for this sample.");
    spy.mockRestore();
  });
});
