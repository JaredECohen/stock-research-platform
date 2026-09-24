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
import type { StockMemoOut } from "@/types";

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

// The first sentence of the deterministic PM fallback tail
// (`memo_sections` signature `pm_view_tail`, `graph.py:1275-1276`). A
// renderer, the PDF included, must never print it.
export const PM_TEMPLATE_TAIL =
  "Sector framing supports the cohort thesis; valuation-relative read is the main swing factor.";
