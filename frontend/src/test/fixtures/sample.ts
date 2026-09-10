// Fixtures for the public sample payloads (FEAT-002 S6), shaped exactly
// as backend/app/services/public_samples.py serves them. `sampleRoutes()`
// wires them into `stubFetch` so a page test can mount against a full
// mocked `/api/public/*` surface in one line.
import type { CompsResult, DCFResult, DCFScenario } from "@/types";
import type { ExpectationsLedger, SamplePayload, SampleSummary } from "@/types/public";
import type { Matcher, Responder } from "@/test/providers";
import { errJson, okJson } from "@/test/providers";
import { makeMemo } from "./memo";

export const OBSERVED_NOTE =
  "Observed cells quote stored data (prices, filings, transcripts). Interpretation cells are the research committee's view. A blank means the evidence was not obtained, not that it is zero.";

export function makeLedger(over: Partial<ExpectationsLedger> = {}): ExpectationsLedger {
  return {
    columns: ["reported_consensus", "management_guidance", "price_implied", "our_forecast"],
    note: OBSERVED_NOTE,
    reported_consensus: {
      status: "available",
      reason: null,
      items: [
        { label: "Consensus view", value: "Consensus sees membership growth decelerating.", basis: "interpretation", source: "mispricing_thesis.consensus_view" },
      ],
    },
    management_guidance: {
      status: "available",
      reason: null,
      items: [
        {
          label: "FY revenue growth",
          value: { prior: "6-7%", current: "7-8%", direction: "raised", rationale: "Renewal rates ran ahead of plan." },
          basis: "observed",
          source: "earnings_agent_view.data.structured.guidance_changes",
        },
      ],
    },
    price_implied: {
      status: "available",
      reason: null,
      items: [
        { label: "Price at memo", value: 900, basis: "observed", source: "price_at_memo", as_of: "2026-06-01T12:00:00" },
        { label: "DCF base-case upside vs. price", value: 0.02, basis: "interpretation", source: "valuation_verdict.dcf_base_upside" },
      ],
    },
    our_forecast: {
      status: "available",
      reason: null,
      items: [
        { label: "Our view", value: "We see renewal rates holding above 90% through the cycle.", basis: "interpretation", source: "mispricing_thesis.our_view" },
        { label: "What would prove us wrong", value: ["US renewal rate prints below 88% for two quarters"], basis: "interpretation", source: "mispricing_thesis.falsifiers" },
      ],
    },
    ...over,
  };
}

/** The ledger the backend emits for a listed-but-unbuilt ticker. */
export function emptyLedger(): ExpectationsLedger {
  return makeLedger({
    reported_consensus: { status: "n/a", items: [], reason: "no stored memo" },
    management_guidance: { status: "not_captured", items: [], reason: "no stored memo" },
    price_implied: { status: "n/a", items: [], reason: "no stored memo" },
    our_forecast: { status: "n/a", items: [], reason: "no stored memo" },
  });
}

function scenario(name: DCFScenario["name"], implied: number | null, upside: number | null, over: Partial<DCFScenario> = {}): DCFScenario {
  return {
    name,
    label: name === "base" ? "Base" : name === "bull" ? "Bull" : "Bear",
    assumptions: {
      revenue_growth: [0.08, 0.07, 0.06, 0.05, 0.04],
      operating_margin: [0.04, 0.041, 0.042, 0.043, 0.044],
      tax_rate: 0.24,
      da_pct_revenue: 0.01,
      capex_pct_revenue: 0.02,
      nwc_pct_revenue: 0.0,
      terminal_growth: 0.025,
      exit_ebitda_multiple: 18,
      wacc: 0.08,
      base_revenue: 250_000_000_000,
      net_debt: 0,
      diluted_shares: 443_000_000,
      current_price: 900,
    },
    projections: [],
    pv_explicit: 1,
    terminal_value_gordon: 1,
    terminal_value_exit_multiple: 1,
    pv_terminal_gordon: 1,
    pv_terminal_exit: 1,
    enterprise_value_gordon: 1,
    enterprise_value_exit: 1,
    enterprise_value_blended: 1,
    equity_value: 1,
    implied_share_price: implied,
    upside_pct: upside,
    ...over,
  };
}

export function makeDCF(over: Partial<DCFResult> = {}): DCFResult {
  return {
    ticker: "COST",
    current_price: 900,
    base: scenario("base", 918, 0.02),
    bull: scenario("bull", 1080, 0.2),
    bear: scenario("bear", 720, -0.2),
    sensitivities: [],
    summary: "Base case implied price $918.00 vs current $900.00 (+2.0%).",
    guardrails: [],
    generated_at: "2026-06-01T12:00:00Z",
    ...over,
  };
}

export function makeComps(over: Partial<CompsResult> = {}): CompsResult {
  return {
    target: { ticker: "COST", company_name: "Costco Wholesale", market_cap: 4.0e11, revenue_growth: 0.07, gross_margin: 0.125, operating_margin: 0.037, ev_ebitda: 32.1, pe: 48.2, fcf_yield: 0.019 },
    peers: [
      { ticker: "WMT", company_name: "Walmart", market_cap: 6.5e11, revenue_growth: 0.05, gross_margin: 0.24, operating_margin: 0.042, ev_ebitda: 18.5, pe: 30.1, fcf_yield: 0.028 },
      { ticker: "BJ", company_name: "BJ's Wholesale", market_cap: 1.1e10, revenue_growth: null, gross_margin: 0.18, operating_margin: 0.036, ev_ebitda: null, pe: 22.4, fcf_yield: 0.04 },
    ],
    median: { ticker: "MEDIAN", company_name: "", market_cap: 3.3e11, revenue_growth: 0.05, gross_margin: 0.21, operating_margin: 0.039, ev_ebitda: 18.5, pe: 26.3, fcf_yield: 0.034 },
    target_percentiles: {},
    premium_discount: { ev_ebitda: 0.7 },
    interpretation: "COST trades at a premium to the peer median on every multiple.",
    history: null,
    ...over,
  };
}

export function makeSample(over: Partial<SamplePayload> = {}): SamplePayload {
  return {
    ticker: "COST",
    company_name: "Costco Wholesale",
    sector: "Consumer Staples",
    built_at: "2026-09-06T07:00:00",
    memo: makeMemo(),
    dcf: makeDCF(),
    comps: makeComps(),
    fundamentals: {
      series: [
        {
          metric: "revenue",
          statement: "income",
          cadence: "annual",
          points: [
            { period: "FY2023", period_end: "2023-09-03", value: 242_290_000_000 },
            { period: "FY2024", period_end: "2024-09-01", value: 254_453_000_000 },
            { period: "FY2025", period_end: "2025-08-31", value: null },
          ],
        },
        {
          metric: "free_cash_flow",
          statement: "cash",
          cadence: "annual",
          points: [
            { period: "FY2023", period_end: "2023-09-03", value: 6_744_000_000 },
            { period: "FY2024", period_end: "2024-09-01", value: 6_627_000_000 },
          ],
        },
      ],
    },
    prices: [
      { date: "2025-09-08", close: 880.12 },
      { date: "2025-09-09", close: 884.5 },
      { date: "2026-09-04", close: 901.33 },
    ],
    screener_row: {
      rank: 12,
      ticker: "COST",
      company_name: "Costco Wholesale",
      sector: "Consumer Staples",
      pm_score: 68,
      quality: 80,
      growth: 55,
      valuation: 30,
      earnings_momentum: 62,
      risk: 75,
      macro_fit: 58,
      one_line_thesis: "Membership economics fully priced.",
      main_catalyst: "Fee increase",
      main_risk: "Multiple compression",
      theme: null,
      universe_size: 240,
      scored_at: "2026-09-05T06:00:00",
    },
    commentary: {
      text: "The committee concluded COST is fairly priced for best-in-class execution; a renewal-rate print below 88% would change the view.",
      generated_at: "2026-09-06T07:00:00",
      model: "openai:cheap",
    },
    expectations_ledger: makeLedger(),
    kinds_built: ["memo", "dcf", "comps", "fundamentals", "prices", "screener_row", "commentary"],
    degraded: [],
    disclosures: [
      "MarketMosaic is for investment research and education only. Sample pages are model outputs from stored research runs, not recommendations, and not personalized financial, investment, legal, or tax advice.",
      OBSERVED_NOTE,
      "Built from stored research on 2026-09-06T07:00:00Z; not updated in real time.",
    ],
    ...over,
  };
}

/** A listed ticker the worker has not built yet: every kind null. */
export function unbuiltSample(ticker = "JPM"): SamplePayload {
  return makeSample({
    ticker,
    company_name: "JPMorgan Chase",
    sector: "Financials",
    built_at: null,
    memo: null,
    dcf: null,
    comps: null,
    fundamentals: null,
    prices: null,
    screener_row: null,
    commentary: null,
    expectations_ledger: emptyLedger(),
    kinds_built: [],
    degraded: ["memo: not built", "dcf: not built", "comps: not built", "fundamentals: not built", "prices: not built", "screener_row: not built", "commentary: not built"],
    disclosures: ["MarketMosaic is for investment research and education only.", OBSERVED_NOTE, "This sample has not been built yet."],
  });
}

export const NVDA_SAMPLE: SamplePayload = makeSample({
  ticker: "NVDA",
  company_name: "NVIDIA",
  sector: "Information Technology",
  memo: makeMemo({ ticker: "NVDA", company_name: "NVIDIA", sector: "Information Technology", rating_label: "Bullish", one_sentence_thesis: "NVDA's data-center demand is not yet in the price." }),
  dcf: makeDCF({ ticker: "NVDA" }),
});

export const SAMPLE_LIST: SampleSummary[] = [
  { ticker: "NVDA", company_name: "NVIDIA", sector: "Information Technology", built_at: "2026-09-06T07:00:00", kinds: ["memo", "dcf", "comps", "fundamentals", "prices", "screener_row", "commentary"] },
  { ticker: "COST", company_name: "Costco Wholesale", sector: "Consumer Staples", built_at: "2026-09-06T07:00:00", kinds: ["memo", "dcf", "comps", "fundamentals", "prices", "screener_row", "commentary"] },
  { ticker: "JPM", company_name: "JPMorgan Chase", sector: "Financials", built_at: null, kinds: [] },
];

export const SAMPLE_TICKERS = SAMPLE_LIST.map((s) => s.ticker);

/** `stubFetch` routes for the whole public sample surface. Detail
 *  matchers come first: the list path is a prefix of the detail path. */
export function sampleRoutes(details: Record<string, SamplePayload> = { NVDA: NVDA_SAMPLE, COST: makeSample(), JPM: unbuiltSample() }): Array<[Matcher, Responder]> {
  const detail: Responder = (url) => {
    const m = /\/api\/public\/samples\/([A-Z0-9.-]+)/.exec(url);
    const t = m ? decodeURIComponent(m[1]) : "";
    const payload = details[t];
    if (!payload) return errJson(404, { code: "not_found", message: `${t} is not a public sample`, sample_tickers: SAMPLE_TICKERS });
    return okJson(payload);
  };
  return [
    [/\/api\/public\/samples\/[^/?]+/, detail],
    [/\/api\/public\/samples(\?|$)/, () => okJson(SAMPLE_LIST)],
  ];
}
