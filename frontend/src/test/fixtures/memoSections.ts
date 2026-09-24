// W2a presented-memo fixtures.
//
// `memo-sections.wire.json` is NOT hand-written: it is
// `memo_sections.present_memo` over the committed, minimized memo fixtures
// in `backend/app/tests/fixtures/memo_sections/`, serialized the way the
// API serializes a memo (`backend/app/scripts/capture_memo_sections_fixture.py`).
// `test_memo_sections_fixture_contract.py` fails when it drifts from the
// presenter, so these renderer tests exercise the server's real output.
//
// The five memos are the design's scenarios (W2a §7): `meta_v1` is the
// fully agentic false-positive control (only technical — a PM intake skip —
// the rule-based critic and Portfolio Fit are unavailable); `googl_live_prepflag`
// a template PM view with no flag; `aapl_demo` a demo-mode memo whose
// LLM-intended sections are all template; `msft_live` an LLM PM beside a
// template sector view, with a blanked follow-up answer in round 1;
// `abbv_v7_patch` the legacy patched shape. The repo is public, so
// non-template prose in them is synthetic.
import wire from "./memo-sections.wire.json";
import { isCounted } from "@/lib/memoSections";
import type { AgentFinding, SectionAvailability, StockMemoOut } from "@/types";

export type PresentedMemoName =
  | "aapl_demo"
  | "abbv_v7_patch"
  | "googl_live_prepflag"
  | "meta_v1"
  | "msft_live";

const MEMOS = wire.memos as unknown as Record<PresentedMemoName, StockMemoOut>;

/** A fresh deep copy of one captured presented memo. */
export function presentedMemo(name: PresentedMemoName): StockMemoOut {
  return structuredClone(MEMOS[name]);
}

export const PRESENTED_MEMO_NAMES = Object.keys(MEMOS) as PresentedMemoName[];

// The deterministic PM fallback tail, verbatim: `memo_sections.SIG["pm_view_tail"].text`
// (`graph._pm_synthesis`). `test_memo_sections_fixture_contract.py` fails if
// this copy drifts from the backend signature. A renderer, the PDF
// included, must never print it.
export const PM_TEMPLATE_TAIL =
  "Sector framing supports the cohort thesis; valuation-relative read is the main swing factor. The risk committee flagged the dominant downside scenarios; portfolio fit depends on macro view.";

/** The presented memo with its hidden prose fields put back to raw template
 * text (each carrying `PM_TEMPLATE_TAIL`) and the map left as the presenter
 * wrote it. The presenter already replaces hidden prose with the
 * placeholder, so "the PDF never prints the tail" can only fail against a
 * body like this: it proves a renderer follows the map, not the text. */
export function withRawTemplateProse(name: PresentedMemoName): StockMemoOut {
  const memo = presentedMemo(name);
  memo.final_pm_view = `Research view: Bullish. ${PM_TEMPLATE_TAIL}`;
  memo.one_sentence_thesis = `${memo.ticker} is undervalued. ${PM_TEMPLATE_TAIL}`;
  memo.final_verdict = `PM final view: Bullish (confidence 59). ${PM_TEMPLATE_TAIL}`;
  return memo;
}

// Presenter outputs the five captured bodies do not reach. Each is `meta_v1`
// with one change and the map entry `memo_sections.present_memo` returned
// for exactly that change (run on 2026-09-24 against the S11 presenter):
//  - scorecard: `scorecard: null` plus the graph's soft event
//    ("Fundamental Scorecard", "DataUnavailable") (graph.py:1824-1828);
//  - industry: `extra_agent_views.industry_group` =
//    `industry_analysts._unmapped_finding` (FEAT-003 routing, owner decision 10);
//  - valuationVerdict: `ValuationVerdict()` plus a "Valuation Verdict" hard event;
//  - dcf: `dcf_summary: {}` plus a "DCF Engine" hard event;
//  - degradedThesis: a builder rewrite (`section_provenance.thesis = "rewrite"`)
//    of an available analyst's headline plus the builder's lever sentence.
const unavailable = (
  reason: NonNullable<SectionAvailability["reason"]>,
  basis: string[],
): SectionAvailability => ({ status: "unavailable", reason, hidden_items: 0, headline_hidden: false, basis });

export const THESIS_BUILDER_CLAUSE =
  "At this price the return comes from steady compounding, not a re-rating — own the floor, not the multiple.";

export const PROBES = {
  scorecard(): StockMemoOut {
    const memo = presentedMemo("meta_v1");
    memo.scorecard = null;
    memo.section_availability!.scorecard = unavailable("no_source_data", [
      "event:Fundamental Scorecard/DataUnavailable",
    ]);
    return memo;
  },
  industry(): StockMemoOut {
    const memo = presentedMemo("meta_v1");
    memo.extra_agent_views = {
      industry_group: {
        agent: "Industry Group Analyst",
        headline: "Industry group read unavailable: no mapping.",
        summary: "",
        key_points: [],
        confidence: 0,
        sources: [],
        data: { no_mapping: true } as AgentFinding["data"],
      },
    };
    memo.section_availability!["extra_agent_views.industry_group"] = unavailable("no_source_data", [
      "signature:industry_no_mapping",
    ]);
    return memo;
  },
  valuationVerdict(): StockMemoOut {
    const memo = presentedMemo("meta_v1");
    memo.valuation_verdict = {
      verdict: "fairly_priced",
      summary: "",
      dcf_base_upside: null,
    } as StockMemoOut["valuation_verdict"];
    memo.section_availability!.valuation_verdict = unavailable("agent_failed", [
      "event:Valuation Verdict/RuntimeError",
    ]);
    return memo;
  },
  dcf(): StockMemoOut {
    const memo = presentedMemo("meta_v1");
    memo.dcf_summary = {};
    memo.section_availability!.dcf_summary = unavailable("agent_failed", ["event:DCF Engine/RuntimeError"]);
    return memo;
  },
  degradedThesis(): StockMemoOut {
    const memo = presentedMemo("meta_v1");
    memo.one_sentence_thesis = `META is fairly priced — ${memo.valuation_agent_view.headline} ${THESIS_BUILDER_CLAUSE}`;
    memo.section_availability!.one_sentence_thesis = {
      status: "degraded",
      reason: "partial_template",
      hidden_items: 0,
      headline_hidden: false,
      basis: ["provenance:thesis=rewrite", "claim:llm_headline"],
    };
    return memo;
  },
} as const;

/** Every counted section a rendered memo shows a placeholder for: the
 * distinct `data-section` keys in `root` that `isCounted` says the banner
 * counts. A renderer's banner number must equal this list's length. */
export function countedPlaceholders(memo: StockMemoOut, root: ParentNode): string[] {
  const keys = new Set(
    Array.from(root.querySelectorAll("[data-section]")).map((el) => el.getAttribute("data-section") ?? ""),
  );
  return [...keys].filter((k) => isCounted(memo, k)).sort();
}
