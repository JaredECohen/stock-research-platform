import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import DCFLab from "@/pages/DCFLab";
import type { DCFAssumptions, DCFResult, DCFScenario, DCFSensitivity } from "@/types";
import type { QuotesOut } from "@/types/quotes";
import { eodClose, liveOpen, staleQuote } from "@/test/fixtures/quotes";

// The lab loads assumptions and runs the engine through `api` on mount;
// the whole surface is driven by those responses, so the mock IS the
// fixture. `vi.hoisted` because `vi.mock` factories run before imports.
const apiMock = vi.hoisted(() => ({
  listStocks: vi.fn(),
  dcfConsensus: vi.fn(),
  dcfSaved: vi.fn(),
  dcfDefaults: vi.fn(),
  runDCF: vi.fn(),
  getQuotes: vi.fn(),
}));

vi.mock("@/api/client", () => ({ api: apiMock }));

// recharts' ResponsiveContainer needs ResizeObserver, which jsdom lacks,
// and the charts are not what these tests assert on. Stub every chart
// primitive the page imports as an empty component.
vi.mock("recharts", () => {
  const Noop = () => null;
  return {
    Bar: Noop, BarChart: Noop, CartesianGrid: Noop, Cell: Noop, Line: Noop,
    LineChart: Noop, ResponsiveContainer: Noop, Tooltip: Noop, XAxis: Noop, YAxis: Noop,
  };
});

const ASSUMPTIONS: DCFAssumptions = {
  revenue_growth: [0.1, 0.09, 0.08, 0.07, 0.06],
  operating_margin: [0.25, 0.26, 0.27, 0.27, 0.27],
  tax_rate: 0.21,
  da_pct_revenue: 0.04,
  capex_pct_revenue: 0.05,
  nwc_pct_revenue: 0.02,
  terminal_growth: 0.025,
  exit_ebitda_multiple: 15,
  wacc: 0.085,
  base_revenue: 100,
  net_debt: 10,
  diluted_shares: 1,
  current_price: 120,
};

function scenario(
  name: DCFScenario["name"],
  implied: number | null,
  upside: number | null,
  extra: Partial<DCFScenario> = {},
): DCFScenario {
  return {
    name,
    label: `${name[0].toUpperCase()}${name.slice(1)} case`,
    assumptions: ASSUMPTIONS,
    projections: [1, 2, 3, 4, 5].map((year) => ({
      year, revenue: 100, ebit: 25, nopat: 20, da: 4, capex: 5, change_nwc: 1,
      fcff: 18, discount_factor: 0.9, pv_fcff: 16,
    })),
    pv_explicit: 80,
    terminal_value_gordon: 300,
    terminal_value_exit_multiple: 320,
    pv_terminal_gordon: 200,
    pv_terminal_exit: 210,
    enterprise_value_gordon: 280,
    enterprise_value_exit: 290,
    enterprise_value_blended: 280,
    equity_value: 270,
    implied_share_price: implied,
    upside_pct: upside,
    ...extra,
  };
}

function sensitivity(value: number | null): DCFSensitivity {
  return {
    name: "WACC vs Terminal Growth",
    row_axis: "WACC",
    col_axis: "Terminal Growth",
    rows: [0.08],
    cols: [0.02],
    cells: [{ row_label: "8.00%", col_label: "2.00%", value }],
  };
}

function exitCrossCheck(value: number | null): DCFSensitivity {
  return {
    name: "Exit Multiple Sensitivity (cross-check vs Gordon headline)",
    row_axis: "Exit EBITDA",
    col_axis: "Scenario",
    rows: [12],
    cols: [0, 1, 2],
    cells: ["bear", "base", "bull"].map((col_label) => ({ row_label: "12.0x", col_label, value })),
  };
}

function result(overrides: Partial<DCFResult> = {}): DCFResult {
  return {
    ticker: "MSFT",
    current_price: 120,
    base: scenario("base", 132, 0.1),
    bull: scenario("bull", 150, 0.25),
    bear: scenario("bear", 96, -0.2),
    sensitivities: [sensitivity(132), exitCrossCheck(128)],
    summary: "Base case implied price $132.00 vs current $120.00 (+10.0%).",
    guardrails: [],
    generated_at: "2026-06-01T12:00:00Z",
    ...overrides,
  };
}

function primeApi(res: DCFResult) {
  apiMock.listStocks.mockResolvedValue([]);
  apiMock.dcfConsensus.mockResolvedValue({
    consensus_revenue_growth: null, trailing_op_margin: null, has_consensus: false,
  });
  apiMock.dcfSaved.mockResolvedValue({ has_saved: false });
  apiMock.dcfDefaults.mockResolvedValue(ASSUMPTIONS);
  apiMock.runDCF.mockResolvedValue(res);
  // No quote by default: the lab must render without one.
  apiMock.getQuotes.mockRejectedValue(new Error("no quote in this test"));
}

// The captured live body (test/fixtures/quotes.ts), addressed to the lab's ticker.
const LIVE_MSFT: QuotesOut = { ...liveOpen, quotes: [{ ...liveOpen.quotes[0], ticker: "MSFT" }] };

describe("DCFLab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders priced scenarios with signed upside and no clamp badge", async () => {
    primeApi(result());
    render(<DCFLab />);
    // Base card + the sensitivity cell both print $132.00.
    await waitFor(() => expect(screen.getAllByText("$132.00")).toHaveLength(2));
    expect(screen.getByText("+10.0% vs model price")).toBeInTheDocument();
    expect(screen.getByText("-20.0% vs model price")).toBeInTheDocument();
    expect(screen.queryByText("Terminal value clamped")).not.toBeInTheDocument();
    expect(screen.queryByText("n/a")).not.toBeInTheDocument();
  });

  it("renders n/a for scenarios and sensitivity cells when the engine could not price the shares", async () => {
    primeApi(result({
      current_price: null,
      base: scenario("base", null, null),
      bull: scenario("bull", null, null),
      bear: scenario("bear", null, null),
      sensitivities: [sensitivity(null), exitCrossCheck(null)],
      summary: "Base case implied price n/a vs current n/a (n/a).",
    }));
    render(<DCFLab />);
    await waitFor(() => expect(screen.getAllByText("n/a vs model price")).toHaveLength(3));
    // Three scenario prices + one 5x5-style cell + three cross-check cells
    // + the "Price used in model" line.
    expect(screen.getAllByText("n/a")).toHaveLength(8);
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
    expect(screen.queryByText(/\+0\.0%/)).not.toBeInTheDocument();
  });

  it("shows the terminal-clamp badge with a tooltip when any scenario is clamped", async () => {
    primeApi(result({
      bull: scenario("bull", 400, 2.3, { tv_clamped: true }),
    }));
    render(<DCFLab />);
    const badge = await screen.findByText("Terminal value clamped");
    const holder = badge.closest("[title]");
    expect(holder).not.toBeNull();
    expect(holder?.getAttribute("title")).toMatch(/0\.5% floor/);
  });

  it("keeps saved assumptions verbatim and shows the live upside for display only (W5b)", async () => {
    primeApi(result({ current_price: 100 }));
    const saved = { ...ASSUMPTIONS, current_price: 100 };
    apiMock.dcfSaved.mockResolvedValue({
      has_saved: true, version: 3, trigger: "memo_rebuild", generated_at: "2026-08-01T12:00:00",
      assumption_changes: [], assumptions: saved,
    });
    apiMock.getQuotes.mockResolvedValue(LIVE_MSFT);
    render(<DCFLab />);
    // Base implied $132 against the live $123.45: +6.9%, computed in the browser.
    await waitFor(() => expect(screen.getByTestId("vs-live-base")).toHaveTextContent("+6.9% vs live price"));
    expect(screen.getByTestId("vs-live-bear")).toHaveTextContent("-22.2% vs live price");
    // What was POSTed is the saved set, save-date price included: nothing re-priced.
    expect(apiMock.runDCF).toHaveBeenCalledTimes(1);
    expect(apiMock.runDCF.mock.calls[0][1]).toEqual(saved);
    expect(apiMock.dcfDefaults).not.toHaveBeenCalled();
    expect(screen.getByText(/saved v3, 2026-08-01/)).toBeInTheDocument();
    expect(screen.getByTestId("live-quote")).toHaveTextContent("$123.45");
  });

  it("never labels the upside against a stored close or a stale quote as 'live'", async () => {
    // The captured eod_close / stale bodies, addressed to the lab's ticker.
    const asMsft = (body: QuotesOut): QuotesOut => ({ ...body, quotes: [{ ...body.quotes[0], ticker: "MSFT" }] });
    primeApi(result());
    apiMock.getQuotes.mockResolvedValue(asMsft(eodClose));
    const view = render(<DCFLab />);
    // Base $132 against the stored close $45.67.
    await waitFor(() => expect(screen.getByTestId("vs-live-base")).toHaveTextContent("+189.0% vs last close"));
    expect(screen.queryByText(/vs live price/)).not.toBeInTheDocument();
    expect(screen.getByText("Latest price")).toBeInTheDocument();
    expect(screen.queryByText("Live price")).not.toBeInTheDocument();
    view.unmount();

    primeApi(result());
    apiMock.getQuotes.mockResolvedValue(asMsft(staleQuote));
    render(<DCFLab />);
    await waitFor(() => expect(screen.getByTestId("vs-live-base")).toHaveTextContent("vs last quote"));
    expect(screen.queryByText(/vs live price/)).not.toBeInTheDocument();
  });

  it("calls the summary's price the model price, not 'current' (a saved run's is the save-date price)", async () => {
    primeApi(result({ current_price: 100, summary: "Base case implied price $132.00 vs current $100.00 (+32.0%). Bull $150.00" }));
    apiMock.dcfSaved.mockResolvedValue({
      has_saved: true, version: 3, trigger: "memo_rebuild", generated_at: "2026-08-02T01:30:00",
      assumption_changes: [], assumptions: { ...ASSUMPTIONS, current_price: 100 },
    });
    render(<DCFLab />);
    await waitFor(() =>
      expect(screen.getByText("Base case implied price $132.00 vs model price $100.00 (+32.0%). Bull $150.00")).toBeInTheDocument(),
    );
    expect(screen.queryByText(/vs current/)).not.toBeInTheDocument();
    // Saved at 01:30 UTC on Aug 2 = 9:30 PM ET on Aug 1: the ET date, as the chip dates things.
    expect(screen.getByText(/saved v3, 2026-08-01/)).toBeInTheDocument();
  });

  it("runs the engine defaults unchanged on the Defaults source", async () => {
    primeApi(result());
    apiMock.getQuotes.mockResolvedValue(LIVE_MSFT);
    render(<DCFLab />);
    await waitFor(() => expect(apiMock.runDCF).toHaveBeenCalledTimes(1));
    // No saved DCF: the lab falls through to the defaults and posts them as served.
    expect(apiMock.runDCF.mock.calls[0][1]).toEqual(ASSUMPTIONS);
  });

  it("still renders the DCF when the quote request fails, without a live upside", async () => {
    primeApi(result());
    render(<DCFLab />);
    await waitFor(() => expect(screen.getByText("+10.0% vs model price")).toBeInTheDocument());
    expect(screen.queryByText(/vs live price/)).not.toBeInTheDocument();
    expect(screen.queryByTestId("live-quote")).not.toBeInTheDocument();
  });
});
