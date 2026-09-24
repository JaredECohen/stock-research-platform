import React from "react";
import { fmtUpside, ratingBadgeClass } from "@/lib/format";
import { formatShortUtc } from "@/lib/entitlements";
import {
  availability,
  bannerCount,
  bannerText,
  isHidden,
  isUnavailable,
  SAMPLE_SUMMARY_SECTIONS,
} from "@/lib/memoSections";
import type { BullBearCase, StockMemoOut } from "@/types";
import UnavailableSection, { DegradedNote } from "@/components/UnavailableSection";

/**
 * The verdict layer of a stored memo, read-only: rating, thesis, the
 * PM's view, bull and bear headlines, the valuation verdict, top
 * catalysts and risks. The full committee card (`MemoCard`) lives on the
 * sample detail page; the landing page shows this so a visitor sees the
 * shape of the output without a page of reading.
 *
 * W2a: the served sample memo is presented server-side (template sections
 * hidden, `section_availability` filled). Hidden sections read "Unavailable
 * in this version." with a reason, and the footnote says how many there are.
 */
export default function SampleMemoSummary({ memo, headingLevel = 2 }: { memo: StockMemoOut; headingLevel?: 2 | 3 }) {
  const H = `h${headingLevel}` as "h2" | "h3";
  const generated = formatShortUtc(memo.generated_at || null);
  const verdict = memo.valuation_verdict;
  const degraded = memo.degraded_agents || [];
  const hiddenCount = bannerCount(memo, SAMPLE_SUMMARY_SECTIONS);
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
          <DegradedNote availability={availability(memo, "rating_label")} className="max-w-[16rem] text-right" />
          <span
            className="text-[11px] text-slate-400"
            data-testid="sample-confidence"
            data-section={isHidden(memo, "confidence_score") ? "confidence_score" : undefined}
          >
            {isHidden(memo, "confidence_score") ? (
              "Confidence unavailable in this version"
            ) : (
              <>
                Confidence <span className="font-mono text-slate-200">{Math.round(memo.confidence_score)}</span>/100
              </>
            )}
          </span>
        </div>
      </div>

      {isHidden(memo, "one_sentence_thesis") ? (
        <UnavailableSection
          variant="inline"
          title="Thesis"
          section="one_sentence_thesis"
          availability={availability(memo, "one_sentence_thesis")}
          className="mt-4"
        />
      ) : memo.one_sentence_thesis ? (
        <>
          <p className="text-base text-slate-100 mt-4 leading-relaxed">{memo.one_sentence_thesis}</p>
          {/* A degraded thesis is a builder rewrite around the analyst's
              claim; the note keeps its canned sentence from reading as
              analysis. */}
          <DegradedNote
            availability={availability(memo, "one_sentence_thesis")}
            section="one_sentence_thesis"
            className="mt-1"
          />
        </>
      ) : null}
      {isHidden(memo, "final_pm_view") ? (
        <UnavailableSection
          variant="inline"
          title="PM view"
          section="final_pm_view"
          availability={availability(memo, "final_pm_view")}
          className="mt-2"
        />
      ) : memo.final_pm_view ? (
        <p className="text-sm text-slate-300 mt-2 leading-relaxed">{memo.final_pm_view}</p>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2 mt-4">
        <SummaryCase label="Bull case" tone="bull" data={memo.bull_case} memo={memo} />
        <SummaryCase label="Bear case" tone="bear" data={memo.bear_case} memo={memo} />
      </div>

      {/* An empty summary means the verdict was not produced (the
          `ValuationVerdict` contract, W2a §4.3), and an unavailable one
          means its step failed: either way `verdict` is the model default,
          and printing it would state a "fairly priced" call nobody made. */}
      {isHidden(memo, "valuation_verdict") ? (
        <UnavailableSection
          variant="inline"
          title="Valuation verdict"
          section="valuation_verdict"
          availability={availability(memo, "valuation_verdict")}
          className="mt-4"
        />
      ) : verdict && verdict.summary && !isUnavailable(memo, "valuation_verdict") ? (
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
          <p className="text-slate-300 mt-1">{verdict.summary}</p>
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2 mt-4 text-sm">
        {isHidden(memo, "catalysts") ? (
          <UnavailableSection
            variant="inline"
            title="Catalysts"
            section="catalysts"
            availability={availability(memo, "catalysts")}
          />
        ) : memo.catalysts && memo.catalysts.length > 0 ? (
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
        {isHidden(memo, "key_risks") ? (
          <UnavailableSection
            variant="inline"
            title="Key risks"
            section="key_risks"
            availability={availability(memo, "key_risks")}
          />
        ) : memo.key_risks && memo.key_risks.length > 0 ? (
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
        {hiddenCount > 0 ? `${bannerText(hiddenCount).replace(/^./, (c) => c.toUpperCase())}. ` : ""}
        Model output, not a recommendation.
      </p>
    </section>
  );
}

function SummaryCase({
  label,
  tone,
  data,
  memo,
}: {
  label: string;
  tone: "bull" | "bear";
  data: BullBearCase | undefined;
  memo: StockMemoOut;
}) {
  const key = tone === "bull" ? "bull_case" : "bear_case";
  const av = availability(memo, key);
  const hidden = isHidden(memo, key);
  const headlineHidden = hidden || !!av?.headline_hidden;
  return (
    <div
      className={`card-tight ${tone === "bull" ? "border-accent-600/30" : "border-danger-500/30"}`}
      data-testid={`case-${tone}`}
    >
      <div
        className={`text-xs uppercase tracking-wider font-semibold ${
          tone === "bull" ? "text-accent-500" : "text-danger-500"
        }`}
      >
        {label}
      </div>
      {headlineHidden ? (
        <UnavailableSection variant="inline" section={key} availability={hidden ? av : undefined} className="mt-1" />
      ) : (
        <div className="text-sm font-medium mt-1">{data?.headline}</div>
      )}
      <ul className="list-disc pl-4 mt-1 space-y-0.5 text-sm text-slate-300">
        {(data?.key_points || []).slice(0, 3).map((p, i) => (
          <li key={i}>{p}</li>
        ))}
      </ul>
      {/* An unavailable case's placeholder already carries its item count. */}
      {!hidden && <DegradedNote availability={av} className="mt-1" />}
    </div>
  );
}
