// Wave 10 — surfaces the PM's adjustments to the consensus-anchored DCF.
// Renders nothing when no adjustments fired (PM agreed with consensus).

import type { StockMemoOut } from "../types";

const FIELD_LABEL: Record<string, string> = {
  revenue_growth: "Revenue growth",
  operating_margin: "Operating margin",
  da_pct_revenue: "D&A % revenue",
  capex_pct_revenue: "Capex % revenue",
  nwc_pct_revenue: "NWC % revenue",
  terminal_growth: "Terminal growth",
  wacc: "WACC",
  exit_ebitda_multiple: "Exit EV/EBITDA",
};

function fmtNumber(v: number | string | null | undefined): string {
  if (v == null) return "—";
  if (typeof v === "string") return v;
  // Heuristic: values <1 with decimal → render as %; else as multiple/raw.
  if (Math.abs(v) < 1 && v !== 0) {
    return `${(v * 100).toFixed(2)}%`;
  }
  return v.toFixed(2);
}

function labelFor(field: string): string {
  // Strip array suffix, e.g. "revenue_growth[2]" → "Revenue growth (Y3)".
  const m = field.match(/^([a-z_]+)\[(\d+)\]$/);
  if (m) {
    const base = FIELD_LABEL[m[1]] ?? m[1];
    return `${base} (Y${parseInt(m[2], 10) + 1})`;
  }
  return FIELD_LABEL[field] ?? field;
}

export default function PMDCFAdjustments({ memo }: { memo: StockMemoOut }) {
  const adjustments = memo.dcf_pm_adjustments ?? [];
  const headline = memo.dcf_pm_adjustment_headline ?? "";
  const initial = memo.dcf_initial_summary ?? {};

  if (!adjustments || adjustments.length === 0) {
    if (headline) {
      // PM ran but made no changes — surface the rationale so the
      // user sees the adjustment step actually fired.
      return (
        <div className="card-tight border-ink-700 text-xs text-slate-400">
          <span className="font-semibold text-slate-300">DCF — PM review:</span>{" "}
          {headline}
        </div>
      );
    }
    return null;
  }

  const initialBaseUpside = initial.base_upside as number | undefined;
  const adjustedBaseUpside = (memo.dcf_summary as Record<string, number>)
    ?.base_upside as number | undefined;
  const upsideDelta =
    typeof initialBaseUpside === "number" && typeof adjustedBaseUpside === "number"
      ? adjustedBaseUpside - initialBaseUpside
      : null;

  return (
    <div className="card-tight border-accent-600/30 bg-accent-600/[0.04]">
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="section-title">DCF — PM adjustments</div>
        <div className="text-[10px] uppercase tracking-widest text-slate-500">
          {adjustments.length} change{adjustments.length === 1 ? "" : "s"}
          {upsideDelta !== null && (
            <>
              {" · "}base upside{" "}
              <span
                className={
                  upsideDelta > 0 ? "text-emerald-400" : "text-rose-400"
                }
              >
                {upsideDelta >= 0 ? "+" : ""}
                {(upsideDelta * 100).toFixed(1)}pp
              </span>
            </>
          )}
        </div>
      </div>
      {headline && (
        <p className="text-xs text-slate-400 leading-relaxed mb-3 italic">
          {headline}
        </p>
      )}
      <div className="space-y-2">
        {adjustments.map((adj, i) => (
          <div
            key={i}
            className="rounded border border-ink-700 bg-ink-900/40 px-3 py-2"
          >
            <div className="flex items-baseline justify-between gap-2">
              <div className="text-sm font-medium text-slate-100">
                {labelFor(adj.field)}
              </div>
              <div className="text-xs font-mono text-slate-300">
                <span className="text-slate-500">{fmtNumber(adj.from)}</span>
                {" → "}
                <span className="text-accent-400">{fmtNumber(adj.to)}</span>
              </div>
            </div>
            {adj.rationale && (
              <div className="text-xs text-slate-400 mt-1 leading-snug">
                {adj.rationale}
              </div>
            )}
          </div>
        ))}
      </div>
      <p className="text-[10px] text-slate-500 mt-3 leading-relaxed">
        The PM updated the consensus-anchored DCF after reading the
        specialists' research. The adjusted DCF drives the memo's rating,
        bull/bear case, and factor scores. ±20% per-cycle clamp prevents
        a single change from yanking valuation.
      </p>
    </div>
  );
}
