// Fixture StockMemoOut for component render tests. The baseline memo is
// fully populated (every optional section present) so tests subtract via
// overrides to exercise the hidden/empty paths.
import type { AgentFinding, MispricingThesis, StockMemoOut } from "@/types";

export function makeFinding(
  agent: string,
  overrides: Partial<AgentFinding> = {},
): AgentFinding {
  return {
    agent,
    headline: `${agent} headline`,
    summary: `${agent} summary`,
    key_points: [`${agent} point one`, `${agent} point two`],
    confidence: 70,
    sources: ["yahoo_finance"],
    ...overrides,
  };
}

// All-blank mispricing thesis — the shape the backend emits when the PM
// declined to take a differentiated view. UI must hide the section.
export const BLANK_MISPRICING: MispricingThesis = {
  consensus_view: "",
  our_view: "",
  gap: "",
  falsifiers: [],
};

export function makeMemo(overrides: Partial<StockMemoOut> = {}): StockMemoOut {
  return {
    ticker: "COST",
    company_name: "Costco Wholesale",
    sector: "Consumer Staples",
    final_pm_view: "Durable compounder, but the quality is fully priced in.",
    rating_label: "Neutral",
    confidence_score: 62,
    one_sentence_thesis: "COST is fairly priced for best-in-class execution.",
    mispricing_thesis: {
      consensus_view: "Consensus sees membership growth decelerating.",
      our_view: "We see renewal rates holding above 90% through the cycle.",
      gap: "The market underprices membership stickiness.",
      falsifiers: ["US renewal rate prints below 88% for two quarters"],
    },
    valuation_verdict: {
      verdict: "fairly_priced",
      dcf_base_upside: 0.02,
      comps_ev_ebitda_premium: 0.4,
      factor_valuation: 35,
      summary: "Fairly priced: DCF base case lands within 5% of spot.",
    },
    business_summary: "Costco operates membership-only warehouse clubs.",
    sector_agent_view: makeFinding("sector"),
    earnings_agent_view: makeFinding("earnings"),
    filing_agent_view: makeFinding("filing"),
    valuation_agent_view: makeFinding("valuation"),
    comps_agent_view: makeFinding("comps"),
    macro_sensitivity: makeFinding("macro"),
    technical_agent_view: null,
    bull_case: { headline: "Bull case headline", key_points: ["Bull point one"] },
    bear_case: { headline: "Bear case headline", key_points: ["Bear point one"] },
    catalysts: [
      {
        title: "Membership fee increase",
        detail: "Periodic fee hike drops straight to operating income.",
        horizon: "near_term",
        impact: "high",
      },
    ],
    key_risks: [
      {
        title: "Multiple compression",
        detail: "Premium multiple de-rates toward staples peers.",
        severity: "medium",
        type: "valuation",
      },
    ],
    thesis_breakers: [],
    dcf_summary: {},
    portfolio_fit: "Core defensive holding.",
    risk_committee_challenge: {
      overall_assessment: "Thesis is coherent but offers little edge vs consensus.",
      challenges: ["What is differentiated about the renewal-rate view?"],
      underweighted_risks: [],
      suggested_revisions: [],
      advice_compliance_check: "ok",
    },
    final_verdict: "Hold at current levels.",
    scores: { factor_pm_score: 68, factor_quality: 80 },
    sources_used: ["yahoo_finance", "sec_edgar"],
    generated_at: "2026-06-01T12:00:00Z",
    generation_mode: "demo",
    degraded_agents: [],
    disclaimer: "Research and education only. Not investment advice.",
    ...overrides,
  };
}
