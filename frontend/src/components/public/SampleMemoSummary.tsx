import React from "react";
import { fmtUpside, ratingBadgeClass } from "@/lib/format";
import { formatShortUtc } from "@/lib/entitlements";
import type { StockMemoOut } from "@/types";

/**
 * The verdict layer of a stored memo, read-only: rating, thesis, the
 * PM's view, bull and bear headlines, the valuation verdict, top
 * catalysts and risks. The full committee card (`MemoCard`) lives on the
 * sample detail page; the landing page shows this so a visitor sees the
 * shape of the output without a page of reading.
 */
export default function SampleMemoSummary({ memo, headingLevel = 2 }: { memo: StockMemoOut; headingLevel?: 2 | 3 }) {
  const H = `h${headingLevel}` as "h2" | "h3";
  const generated = formatShortUtc(memo.generated_at || null);
  const verdict = memo.valuation_verdict;
  const degraded = memo.degraded_agents || [];
  return (
    <section aria-labelledby="sample-memo-heading" className="card">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-widest text-slate-400">Research memo</div>
          <H id="sample-memo-heading" className="text-xl font-semibold mt-0.5">
            {memo.company_name} <span className="font-mono text-slate-400 text-base">{memo.ticker}</span>
          </H>
          {memo.sector ? <div className="text-xs text-slate-400 mt-0.5">{memo.sector}</div> : null}
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className={ratingBadgeClass(memo.rating_label)}>{memo.rating_label}</span>
          <span className="text-[11px] text-slate-400">
            Confidence <span className="font-mono text-slate-200">{Math.round(memo.confidence_score)}</span>/100
          </span>
        </div>
      </div>

      {memo.one_sentence_thesis ? <p className="text-base text-slate-100 mt-4 leading-relaxed">{memo.one_sentence_thesis}</p> : null}
      {memo.final_pm_view ? <p className="text-sm text-slate-300 mt-2 leading-relaxed">{memo.final_pm_view}</p> : null}

      <div className="grid gap-3 sm:grid-cols-2 mt-4">
        <div className="card-tight border-accent-600/30">
          <div className="text-xs uppercase tracking-wider text-accent-500 font-semibold">Bull case</div>
          <div className="text-sm font-medium mt-1">{memo.bull_case?.headline}</div>
          <ul className="list-disc pl-4 mt-1 space-y-0.5 text-sm text-slate-300">
            {(memo.bull_case?.key_points || []).slice(0, 3).map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </div>
        <div className="card-tight border-danger-500/30">
          <div className="text-xs uppercase tracking-wider text-danger-500 font-semibold">Bear case</div>
          <div className="text-sm font-medium mt-1">{memo.bear_case?.headline}</div>
          <ul className="list-disc pl-4 mt-1 space-y-0.5 text-sm text-slate-300">
            {(memo.bear_case?.key_points || []).slice(0, 3).map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </div>
      </div>

      {verdict ? (
        <div className="mt-4 text-sm">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold mr-2">Valuation verdict</span>
          <span className="badge-neutral capitalize">{verdict.verdict.replace(/_/g, " ")}</span>
          {typeof verdict.dcf_base_upside === "number" ? (
            <span className="ml-2 text-slate-300">
              DCF base case <span className="font-mono">{fmtUpside(verdict.dcf_base_upside)}</span> vs. price at memo
            </span>
          ) : (
            <span className="ml-2 text-slate-400">DCF base-case upside n/a</span>
          )}
          {verdict.summary ? <p className="text-slate-300 mt-1">{verdict.summary}</p> : null}
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2 mt-4 text-sm">
        {memo.catalysts && memo.catalysts.length > 0 ? (
          <div>
            <div className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Catalysts</div>
            <ul className="mt-1 space-y-1">
              {memo.catalysts.slice(0, 3).map((c, i) => (
                <li key={i}>
                  <span className="text-slate-100">{c.title}</span>
                  <span className="text-slate-400"> — {c.horizon.replace(/_/g, " ")}, {c.impact} impact</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {memo.key_risks && memo.key_risks.length > 0 ? (
          <div>
            <div className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Key risks</div>
            <ul className="mt-1 space-y-1">
              {memo.key_risks.slice(0, 3).map((r, i) => (
                <li key={i}>
                  <span className="text-slate-100">{r.title}</span>
                  <span className="text-slate-400"> — {r.severity} severity, {r.type.replace(/_/g, " ")}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>

      <p className="text-[11px] text-slate-400 mt-4">
        {generated ? `Memo generated ${generated}. ` : ""}
        {memo.generation_mode === "demo" ? "Demo-mode run. " : ""}
        {degraded.length > 0 ? `Specialists unavailable during this run: ${degraded.join(", ")}. ` : ""}
        Model output, not a recommendation.
      </p>
    </section>
  );
}
