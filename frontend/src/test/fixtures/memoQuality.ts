// W2b research-checks fixture.
//
// `memo_quality.wire.json` is NOT hand-written: it is
// `graph.run_stock_memo("NVDA")` on the demo dataset with two scripted model
// answers (the PM synthesis and the valuation analyst), passed through the
// presenter and serialized as the memo route serializes it
// (`backend/app/scripts/capture_memo_quality_fixture.py`, whose docstring
// and `meta.scripted` say exactly what was scripted).
// `test_memo_quality_fixture_contract.py` re-validates it against the
// backend models and pins its key sets, so these renderer tests exercise
// the real producer's output.
//
// What it carries: one fabricated PM figure (untraceable, marked in the PM
// view), one declared PM assumption, one withheld valuation key point, a
// Bullish call downgraded to Neutral by the rating check, and confidence
// capped by the template-section and critic caps.
import wire from "./memo_quality.wire.json";
import type { SectionAvailability, StockMemoOut } from "@/types";

const MEMO = wire.memo as unknown as StockMemoOut;

/** What the capture script planted, for assertions. */
export const QUALITY_EXPECT = wire.meta.expect as {
  fabricated_pm_figure: string;
  declared_assumption: string;
  withheld_point: string;
};

/** A fresh deep copy of the captured memo. */
export function qualityMemo(): StockMemoOut {
  return structuredClone(MEMO);
}

/** The captured memo as the presenter serves it when the PM input behind
 * the confidence was a template: the map marks `confidence_score`
 * unavailable (the entry `memo_sections` writes for that case), the number
 * stays in the body, and every renderer must keep it off the page. */
export function qualityMemoConfidenceHidden(): StockMemoOut {
  const memo = qualityMemo();
  const hidden: SectionAvailability = {
    status: "unavailable",
    reason: "template_fallback",
    hidden_items: 0,
    headline_hidden: false,
    basis: ["event:PM Synthesis/DeterministicFallback"],
  };
  memo.section_availability = { ...(memo.section_availability ?? {}), confidence_score: hidden };
  return memo;
}
