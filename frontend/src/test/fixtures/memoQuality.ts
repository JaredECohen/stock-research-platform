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
// What it carries: two fabricated figures (untraceable; one in the PM view,
// one in "Our view"), one declared PM assumption, one withheld valuation
// key point, a Bullish call downgraded to Neutral by the rating check, and
// confidence capped by the template-section and critic caps.
//
// `variants` are rating-check states the one run cannot reach, each written
// by the real `reconcile_rating` / `enforce_after_patch` on the captured
// memo's own verdict (record mode, a reason that fails the checks, accepted
// reasons, a critic that rejects one, and news patches). A variant replaces
// the rating, the confidence and their two quality records on the captured
// body; nothing else changes.
//
// `memo_quality_pm_template.wire.json` is the same run with the PM's answer
// left out: the template PM ships and the `pm_template` cap binds.
import wire from "./memo_quality.wire.json";
import pmTemplateWire from "./memo_quality_pm_template.wire.json";
import type { ConfidenceAssessment, RatingReconciliation, SectionAvailability, StockMemoOut } from "@/types";

const MEMO = wire.memo as unknown as StockMemoOut;
const PM_TEMPLATE_MEMO = pmTemplateWire.memo as unknown as StockMemoOut;

interface Variant {
  rating_label: StockMemoOut["rating_label"];
  confidence_score: number;
  rating_reconciliation: RatingReconciliation;
  confidence: ConfidenceAssessment | null;
}
const VARIANTS = wire.variants as unknown as Record<string, Variant>;

export type QualityVariant =
  | "record_mode"
  | "failed_reason"
  | "accepted_unreviewed"
  | "accepted_supported"
  | "critic_unsupported"
  | "patch_kept_record"
  | "patch_guard"
  | "patch_guard_record_mode"
  | "patch_after_accepted";

/** What the capture script planted, for assertions. */
export const QUALITY_EXPECT = wire.meta.expect as {
  fabricated_pm_figure: string;
  fabricated_view_figure: string;
  declared_assumption: string;
  withheld_point: string;
};

/** The PM reasons the variants were built with. */
export const VARIANT_REASONS = wire.meta.scripted.variant_reasons as { quoted: string; thin: string };

/** A fresh deep copy of the captured memo. */
export function qualityMemo(): StockMemoOut {
  return structuredClone(MEMO);
}

/** The captured memo with one rating-check variant applied. */
export function qualityVariant(name: QualityVariant): StockMemoOut {
  const memo = qualityMemo();
  const v = structuredClone(VARIANTS[name]);
  if (!v) throw new Error(`no captured variant ${name}`);
  memo.rating_label = v.rating_label;
  memo.confidence_score = v.confidence_score;
  memo.quality = { ...memo.quality!, rating_reconciliation: v.rating_reconciliation, confidence: v.confidence };
  return memo;
}

/** The captured PM-template run (see the header). */
export function qualityMemoPmTemplate(): StockMemoOut {
  return structuredClone(PM_TEMPLATE_MEMO);
}

/** DEFENSIVE ONLY — a state the producer does not write today: the quality
 * stage stamps confidence `earned`, so the presenter keeps
 * `confidence_score` available on every memo that carries `quality` (the
 * template-PM case is `qualityMemoPmTemplate`). This hand-set map entry
 * (the one `memo_sections` writes for a template confidence on an older
 * memo) pins that the renderers still honour the map if that changes. */
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

/** The captured memo with exactly one stored claim, re-pointed at `path`:
 * the field at `path` is set to a sentence carrying the captured
 * fabricated figure, and the claim is the captured untraceable claim
 * object with offsets taken from that real text. Every section is shown
 * (no presenter map), so the test exercises the renderer's call site for
 * `path` and nothing else. The path formats are the ones
 * `test_memo_quality_fixture_contract.test_field_paths_the_renderers_build`
 * pins against the backend's `number_check.iter_fields`. */
export function qualityMemoWithClaimAt(path: string): { memo: StockMemoOut; raw: string } {
  const memo = qualityMemo();
  memo.section_availability = undefined;
  const nc = memo.quality!.number_check!;
  const template = nc.claims.find((c) => c.status === "untraceable" && c.raw === QUALITY_EXPECT.fabricated_pm_figure);
  if (!template) throw new Error("the captured fixture lost its untraceable PM figure");
  const raw = template.raw;
  const text = `Revenue of ${raw} next year.`;
  setPath(memo as unknown as Record<string, unknown>, path, text);
  const start = text.indexOf(raw);
  nc.claims = [{ ...template, field: path, start, end: start + raw.length }];
  return { memo, raw };
}

// Paths are `a.b`, `a[0].b`, `a.b[0]`. A missing list item is created with
// the model's defaults, so a list the captured memo leaves empty still
// renders one item.
const ITEM_DEFAULTS: Record<string, Record<string, unknown>> = {
  catalysts: { title: "", detail: "", horizon: "medium_term", impact: "medium" },
  key_risks: { title: "", detail: "", severity: "medium", type: "company" },
  thesis_breakers: { title: "", detail: "", severity: "high", type: "thesis_breaker" },
};

function setPath(root: Record<string, unknown>, path: string, value: string): void {
  const tokens = path.match(/[^.[\]]+|\[\d+\]/g) ?? [];
  let obj: Record<string, unknown> | unknown[] = root;
  let listName = "";
  tokens.forEach((tok, i) => {
    const last = i === tokens.length - 1;
    const idx = /^\[(\d+)\]$/.exec(tok);
    const key: string | number = idx ? Number(idx[1]) : tok;
    const container = obj as Record<string | number, unknown>;
    if (last) {
      container[key] = value;
      return;
    }
    if (container[key] == null) {
      container[key] = /^\[/.test(tokens[i + 1]) ? [] : { ...(ITEM_DEFAULTS[listName] ?? {}) };
    }
    if (!idx) listName = tok;
    obj = container[key] as Record<string, unknown> | unknown[];
  });
}
