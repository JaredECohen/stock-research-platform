import React, { useEffect, useRef } from "react";
import type { AgentFinding, RiskItem, ScorecardContribution, ScorecardSummary, StockMemoOut } from "@/types";
import { SCORECARD_FAMILIES } from "@/types/scorecard";
import { fmtPct, fmtPrice, fmtUpside, ratingBadgeClass } from "@/lib/format";
import { profileText } from "./scorecard/ScorecardPanel";
import { familyLabel, fmtCoverage, fmtPercentile, fmtScore, fmtZ, humanize, isNum, na } from "./scorecard/format";
import { Markdown } from "./Markdown";
import TerminalClampBadge from "./TerminalClampBadge";

/**
 * Professional investment-memo layout (sell-side / buy-side IC style):
 *
 *   Cover         — ticker, company, rating, target, sector, date, author
 *   Executive     — one-sentence thesis, key takeaways
 *   Bull / Bear   — full structured bull case, full structured bear case
 *   Sector        — sector view + KPI placements + cohort
 *   Earnings      — most recent print + guidance changes
 *   Filings       — 10-K / 10-Q risk + MD&A highlights
 *   Valuation     — DCF summary, comps, sensitivity
 *   Scorecard     — observed inputs vs the fs-v1 model read (Phase 6)
 *   Macro & risk  — macro overlay, key risks, thesis breakers, critic
 *   Catalysts     — calendar / timeline
 *   Appendix      — sources, disclaimer
 *
 * Rendered as a print-friendly modal. The "Download PDF" button opens
 * the memo in an isolated window and triggers the system print dialog
 * with the print stylesheet active; the user picks "Save as PDF" for a
 * proper text PDF. Zero dependencies.
 */

interface Props {
  memo: StockMemoOut;
  open: boolean;
  onClose: () => void;
}

export default function FullInvestmentMemo({ memo, open, onClose }: Props) {
  const modalRef = useRef<HTMLDivElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open, onClose]);

  if (!open) return null;

  const generatedDate = formatDate(memo.generated_at);
  const misp = memo.mispricing_thesis;
  const hasMispricing = Boolean(
    misp && (misp.consensus_view || misp.our_view || misp.gap),
  );
  const degraded = memo.degraded_agents || [];
  const dcf = memo.dcf_summary || {};
  // `hasDcf` separates "the model ran" (numbers may still be null → "n/a")
  // from "no DCF on this memo" (the section shows "—").
  const hasDcf = Object.keys(dcf).length > 0;
  // `base_implied_price` / `base_upside` are what the graph writes; the
  // older aliases are kept for memos that pre-date that summary shape.
  const dcfFair = pickNumber(dcf, ["base_implied_price", "target_price", "fair_value", "implied_per_share", "fair_value_per_share"]);
  const dcfImpliedUpside = pickNumber(dcf, ["base_upside", "implied_upside", "upside", "upside_pct"]);
  const dcfWacc = pickNumber(dcf, ["wacc", "discount_rate"]);
  const dcfGrowth = pickNumber(dcf, ["terminal_growth", "g_terminal"]);
  const tvClamped = dcf.tv_clamped === true;

  return (
    <div
      ref={modalRef}
      className="fixed inset-0 z-50 bg-black/70 flex items-start justify-center p-4 md:p-8 overflow-y-auto print:bg-white print:p-0"
      onClick={(e) => {
        if (e.target === modalRef.current) onClose();
      }}
    >
      <div className="w-full max-w-[920px] bg-ink-900 border border-ink-700 rounded-lg shadow-2xl overflow-hidden print:bg-white print:border-0 print:shadow-none print:max-w-none print:rounded-none">
        {/* Toolbar — hidden on print */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-ink-700 bg-ink-950/60 print:hidden">
          <div className="text-sm text-slate-300">
            Full Investment Memo — {memo.ticker}
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="px-3 py-1.5 text-xs rounded-md bg-accent-600 hover:bg-accent-500 text-white font-medium"
              onClick={() => downloadPdf(memo, contentRef.current)}
            >
              Download PDF
            </button>
            <button
              type="button"
              className="px-3 py-1.5 text-xs rounded-md bg-ink-800 hover:bg-ink-700 border border-ink-700 text-slate-200"
              onClick={onClose}
            >
              Close
            </button>
          </div>
        </div>

        <div
          ref={contentRef}
          className="memo-paper px-8 md:px-14 py-10 text-slate-100 print:text-slate-900 print:bg-white"
          style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
        >
          {/* COVER */}
          <header className="border-b border-slate-700 print:border-slate-300 pb-6">
            <div className="flex items-baseline justify-between gap-4">
              <div>
                <div className="text-xs uppercase tracking-[0.3em] text-slate-400 print:text-slate-500">
                  Equity Research · Investment Memo
                </div>
                <h1 className="mt-2 text-4xl font-bold leading-tight" style={{ fontFamily: "Georgia, serif" }}>
                  {memo.company_name || memo.ticker}
                </h1>
                <div className="mt-1 text-sm text-slate-400 print:text-slate-600">
                  {memo.ticker} · {memo.sector || "Unclassified"}
                </div>
              </div>
              <div className="text-right">
                <div className={`inline-block px-3 py-1 rounded ${ratingBadgeClass(memo.rating_label)}`}>
                  {memo.rating_label}
                </div>
                <div className="mt-2 text-xs text-slate-400 print:text-slate-600">
                  Conviction: {Math.round(memo.confidence_score)}/100
                </div>
                {hasDcf && (
                  <div className="mt-1 text-xs text-slate-400 print:text-slate-600">
                    Fair value: {fmtPrice(dcfFair)} ({fmtUpside(dcfImpliedUpside)})
                  </div>
                )}
              </div>
            </div>
            <div className="mt-4 text-xs text-slate-500 print:text-slate-600 flex flex-wrap gap-x-6 gap-y-1">
              <span>Report date: {generatedDate}</span>
              <span>Prepared by: MarketMosaic AI Research</span>
              <span>
                {memo.generation_mode === "demo"
                  ? "Demo dataset — figures are illustrative, not live market data"
                  : "Live market data"}
              </span>
            </div>
          </header>

          {/* DEGRADED AGENTS — must survive into the PDF so a partial memo
              never presents as complete. */}
          {degraded.length > 0 && (
            <div className="mt-5 border border-amber-500/50 print:border-amber-700 rounded p-3 text-sm text-amber-400 print:text-amber-800">
              <span className="font-semibold">Partial coverage:</span>{" "}
              {degraded.join(", ")}{" "}
              {degraded.length === 1 ? "was" : "were"} unavailable or fell back
              to deterministic output during this run. Treat the affected
              section{degraded.length === 1 ? "" : "s"} as thinner evidence.
            </div>
          )}

          {/* EXECUTIVE SUMMARY */}
          <Section title="Executive Summary">
            <p className="text-base leading-relaxed italic text-slate-200 print:text-slate-800 mb-3">
              {memo.one_sentence_thesis || memo.final_verdict}
            </p>
            {memo.final_pm_view && (
              <Markdown text={memo.final_pm_view} />
            )}
          </Section>

          {/* MISPRICING — consensus vs our view, and what would prove us wrong */}
          {hasMispricing && misp && (
            <Section title="Where We Differ From Consensus">
              <div className="space-y-2 text-sm text-slate-200 print:text-slate-800">
                {misp.consensus_view && (
                  <p>
                    <span className="font-semibold">Consensus:</span>{" "}
                    {misp.consensus_view}
                  </p>
                )}
                {misp.our_view && (
                  <p>
                    <span className="font-semibold">Our view:</span>{" "}
                    {misp.our_view}
                  </p>
                )}
                {misp.gap && (
                  <p>
                    <span className="font-semibold">The gap:</span> {misp.gap}
                  </p>
                )}
                {misp.falsifiers && misp.falsifiers.length > 0 && (
                  <div>
                    <div className="font-semibold">What would prove us wrong:</div>
                    <ul className="list-disc pl-5 mt-1 space-y-0.5">
                      {misp.falsifiers.map((f, i) => (
                        <li key={i}>{f}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </Section>
          )}

          {/* BUSINESS OVERVIEW */}
          {memo.business_summary && (
            <Section title="Business Overview">
              <Markdown text={memo.business_summary} />
            </Section>
          )}

          {/* BULL / BEAR */}
          <Section title="Investment Thesis — Bull vs Bear">
            <div className="grid md:grid-cols-2 gap-6 print:grid-cols-2">
              <CaseBlock label="Bull case" tone="bull" headline={memo.bull_case?.headline} points={memo.bull_case?.key_points} />
              <CaseBlock label="Bear case" tone="bear" headline={memo.bear_case?.headline} points={memo.bear_case?.key_points} />
            </div>
          </Section>

          {/* SECTOR */}
          {memo.sector_agent_view && (
            <Section title="Sector & Industry Context">
              <AgentBlock finding={memo.sector_agent_view} />
            </Section>
          )}

          {/* EARNINGS */}
          {memo.earnings_agent_view && (
            <Section title="Recent Earnings & Guidance">
              <AgentBlock finding={memo.earnings_agent_view} />
            </Section>
          )}

          {/* FILINGS */}
          {memo.filing_agent_view && (
            <Section title="Filings — Risk Factors & MD&A">
              <AgentBlock finding={memo.filing_agent_view} />
            </Section>
          )}

          {/* VALUATION */}
          <Section title="Valuation">
            {memo.valuation_verdict?.summary && (
              <p className="text-sm font-semibold text-slate-100 print:text-slate-900 mb-3">
                {memo.valuation_verdict.summary}
              </p>
            )}
            {tvClamped && <TerminalClampBadge className="mb-3" />}
            <div className="grid md:grid-cols-2 gap-4 mb-4">
              <KvTable
                title="DCF Summary"
                rows={[
                  ["Fair value / share", hasDcf ? fmtPrice(dcfFair) : "—"],
                  ["Implied upside", hasDcf ? fmtUpside(dcfImpliedUpside) : "—"],
                  ["WACC", dcfWacc !== null ? fmtPctLoose(dcfWacc) : "—"],
                  ["Terminal growth", dcfGrowth !== null ? fmtPctLoose(dcfGrowth) : "—"],
                ]}
              />
              {memo.dcf_pm_adjustment_headline && (
                <div className="border border-slate-700 print:border-slate-300 rounded p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400 print:text-slate-600">
                    PM DCF Adjustment
                  </div>
                  <div className="mt-1 text-sm">{memo.dcf_pm_adjustment_headline}</div>
                </div>
              )}
            </div>
            {memo.valuation_agent_view && <AgentBlock finding={memo.valuation_agent_view} />}
            {memo.comps_agent_view && (
              <div className="mt-4">
                <AgentBlock finding={memo.comps_agent_view} subhead="Comparable companies" />
              </div>
            )}
          </Section>

          {/* SCORECARD — hidden on memos that pre-date the field (undefined)
              and on runs with no row for the ticker (null). Informs the
              memo; the rating blend does not read it. */}
          {memo.scorecard && (
            <Section title="Fundamental Factor Scorecard">
              <ScorecardMemoSection scorecard={memo.scorecard} />
            </Section>
          )}

          {/* MACRO + RISK */}
          {(memo.macro_sensitivity || memo.technical_agent_view) && (
            <Section title="Macro & Positioning">
              {memo.macro_sensitivity && <AgentBlock finding={memo.macro_sensitivity} subhead="Macro sensitivity" />}
              {memo.technical_agent_view && (
                <div className="mt-3">
                  <AgentBlock finding={memo.technical_agent_view} subhead="Technical positioning" />
                </div>
              )}
            </Section>
          )}

          {/* CATALYSTS */}
          {memo.catalysts && memo.catalysts.length > 0 && (
            <Section title="Catalysts">
              <ul className="space-y-2">
                {memo.catalysts.map((cat, i) => (
                  <li key={i} className="text-sm">
                    <div className="flex items-baseline gap-2">
                      <span className="font-semibold">{cat.title}</span>
                      <span className="text-xs uppercase tracking-wider text-slate-400 print:text-slate-600">
                        {cat.horizon.replace("_", " ")} · {cat.impact} impact
                      </span>
                    </div>
                    <div className="text-slate-300 print:text-slate-700">{cat.detail}</div>
                  </li>
                ))}
              </ul>
            </Section>
          )}

          {/* RISKS */}
          {(memo.key_risks?.length || memo.thesis_breakers?.length) ? (
            <Section title="Risks & Thesis Breakers">
              {memo.key_risks && memo.key_risks.length > 0 && (
                <>
                  <h4 className="text-sm font-semibold mt-1 mb-2 text-slate-300 print:text-slate-700">
                    Key risks
                  </h4>
                  <RiskList items={memo.key_risks} />
                </>
              )}
              {memo.thesis_breakers && memo.thesis_breakers.length > 0 && (
                <>
                  <h4 className="text-sm font-semibold mt-4 mb-2 text-slate-300 print:text-slate-700">
                    Thesis breakers
                  </h4>
                  <RiskList items={memo.thesis_breakers} />
                </>
              )}
            </Section>
          ) : null}

          {/* CRITIC */}
          {memo.risk_committee_challenge && (
            <Section title="Risk Committee Challenge">
              <div className="text-sm text-slate-300 print:text-slate-700">
                <p className="italic mb-2">{memo.risk_committee_challenge.overall_assessment}</p>
                {memo.risk_committee_challenge.challenges?.length > 0 && (
                  <>
                    <div className="font-semibold mt-2">Challenges:</div>
                    <ul className="list-disc pl-5">
                      {memo.risk_committee_challenge.challenges.map((c, i) => (
                        <li key={i}>{c}</li>
                      ))}
                    </ul>
                  </>
                )}
                {memo.risk_committee_challenge.underweighted_risks?.length > 0 && (
                  <>
                    <div className="font-semibold mt-2">Underweighted risks:</div>
                    <ul className="list-disc pl-5">
                      {memo.risk_committee_challenge.underweighted_risks.map((c, i) => (
                        <li key={i}>{c}</li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
            </Section>
          )}

          {/* PORTFOLIO FIT */}
          {memo.portfolio_fit && (
            <Section title="Portfolio Fit">
              <p className="text-sm text-slate-200 print:text-slate-800">{memo.portfolio_fit}</p>
            </Section>
          )}

          {/* FINAL VERDICT */}
          {memo.final_verdict && (
            <Section title="Final Verdict">
              <p className="text-base font-semibold text-slate-100 print:text-slate-900">
                {memo.final_verdict}
              </p>
            </Section>
          )}

          {/* APPENDIX */}
          <Section title="Sources & Disclaimer">
            {memo.sources_used && memo.sources_used.length > 0 && (
              <div className="text-xs text-slate-400 print:text-slate-600 mb-3">
                <span className="font-semibold">Sources consulted:</span>{" "}
                {memo.sources_used.slice(0, 30).join(" · ")}
              </div>
            )}
            <div className="text-[10px] leading-relaxed text-slate-500 print:text-slate-600 border-t border-slate-700 print:border-slate-300 pt-3">
              {memo.disclaimer ||
                "Research and education only. Not personalized financial / investment / legal / tax advice. Conduct your own diligence or consult a qualified advisor before acting."}
            </div>
          </Section>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-7 print:mt-6 print:break-inside-avoid">
      <h2 className="text-lg font-bold border-b border-slate-700 print:border-slate-400 pb-1 mb-3 tracking-wide">
        {title}
      </h2>
      {children}
    </section>
  );
}

function CaseBlock({
  label,
  tone,
  headline,
  points,
}: {
  label: string;
  tone: "bull" | "bear";
  headline?: string;
  points?: string[];
}) {
  const accent =
    tone === "bull"
      ? "border-emerald-500/40 print:border-emerald-700"
      : "border-rose-500/40 print:border-rose-700";
  return (
    <div className={`border-l-2 ${accent} pl-3`}>
      <div className="text-xs uppercase tracking-widest text-slate-400 print:text-slate-600 font-semibold">
        {label}
      </div>
      {headline && (
        <div className="mt-1 text-sm font-semibold">{headline}</div>
      )}
      {points && points.length > 0 && (
        <ul className="mt-2 list-disc pl-4 space-y-1 text-sm text-slate-200 print:text-slate-800">
          {points.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function AgentBlock({ finding, subhead }: { finding: AgentFinding; subhead?: string }) {
  if (!finding) return null;
  return (
    <div>
      {subhead && (
        <div className="text-xs uppercase tracking-widest text-slate-400 print:text-slate-600 mb-1">
          {subhead}
        </div>
      )}
      {finding.headline && (
        <div className="text-sm font-semibold mb-1">{finding.headline}</div>
      )}
      {finding.summary && (
        <p className="text-sm leading-relaxed text-slate-200 print:text-slate-800">
          {finding.summary}
        </p>
      )}
      {finding.key_points && finding.key_points.length > 0 && (
        <ul className="mt-2 list-disc pl-5 space-y-1 text-sm text-slate-300 print:text-slate-700">
          {finding.key_points.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      )}
      {finding.long_form_report && (
        <details className="mt-2 text-sm">
          <summary className="cursor-pointer text-xs uppercase tracking-wider text-slate-400 print:text-slate-600 hover:text-slate-200">
            Full {finding.agent || "analyst"} report
          </summary>
          <div className="mt-2 border-l border-slate-700 print:border-slate-300 pl-3 text-slate-300 print:text-slate-700">
            <Markdown text={finding.long_form_report} />
          </div>
        </details>
      )}
    </div>
  );
}

// Phase 6 — the memo's Scorecard section. Two explicitly labelled blocks:
// **Observed** (the reported inputs: fiscal period, availability date,
// coverage, the as-of date) and **Model read** (spec version, 0–100
// score, z, percentiles, family composites, contributors). The two never
// share a table so a reader cannot mistake a model number for a reported
// one. Anything the run could not compute prints as "n/a (reason)" —
// never 0, never "neutral". Print-safe like the other sections.
const SCORECARD_OVERALL_REASON = "insufficient coverage: fewer than 5 families scored";

function directionText(d: NonNullable<ScorecardSummary["disagreement"]>["direction"]): string {
  return d === "narrative_above_quant" ? "the memo is more positive than the quant read" : "the memo is more negative than the quant read";
}

function ContribList({ label, items, emptyReason, testId }: { label: string; items: ScorecardContribution[]; emptyReason: string; testId: string }) {
  return (
    <div data-testid={testId}>
      <div className="text-xs uppercase tracking-widest text-slate-400 print:text-slate-600 font-semibold">{label}</div>
      {items.length === 0 ? (
        <p className="mt-1 text-sm text-slate-400 print:text-slate-600">{na(emptyReason)}</p>
      ) : (
        <ul className="mt-1 list-disc pl-4 space-y-0.5 text-sm text-slate-200 print:text-slate-800">
          {items.map((c) => (
            <li key={c.feature}>
              {humanize(c.feature)} <span className="text-slate-400 print:text-slate-600">({familyLabel(c.family)})</span>{" "}
              <span className="font-mono">z {fmtZ(c.z)}</span>
              <span className="text-slate-400 print:text-slate-600 font-mono"> · {c.contribution > 0 ? "+" : ""}{c.contribution.toFixed(3)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ScorecardMemoSection({ scorecard }: { scorecard: ScorecardSummary }) {
  const known = SCORECARD_FAMILIES as readonly string[];
  // Canonical order first, then any family the backend added that the
  // client does not know yet — shown rather than dropped.
  const extra = Object.keys(scorecard.categories ?? {}).filter((k) => !known.includes(k) && !k.startsWith("_"));
  const families = [...known, ...extra];
  const overallMissing = !isNum(scorecard.overall_score);
  const disagreement = scorecard.disagreement ?? null;
  const contributorEmpty = overallMissing ? "overall not scored" : "none listed";

  return (
    <div data-testid="memo-scorecard">
      <p className="text-xs text-slate-400 print:text-slate-600 mb-3">
        Spec {scorecard.version_key} · scored as of {scorecard.as_of}
        {scorecard.stale ? " · stale (the latest run is older than 45 days)" : ""}. Observed inputs and the model&apos;s read are kept apart below. The
        scorecard informed this memo; it did not move the rating.
      </p>
      <div className="grid md:grid-cols-2 gap-4 print:grid-cols-2">
        <div data-testid="memo-scorecard-observed">
          <KvTable
            title="Observed — reported inputs"
            rows={[
              ["Latest fiscal period", scorecard.latest_period || na("period unknown")],
              ["Data available", scorecard.data_available_at || na("availability unknown")],
              ["Input coverage", fmtCoverage(scorecard.coverage)],
              ["Scored as of", scorecard.as_of],
            ]}
          />
        </div>
        <div data-testid="memo-scorecard-model">
          <KvTable
            title={`Model read — ${scorecard.version_key}`}
            rows={[
              ["Overall score (0–100, 50 = z of 0)", fmtScore(scorecard.overall_score, SCORECARD_OVERALL_REASON)],
              ["Overall z", fmtZ(scorecard.overall_z, SCORECARD_OVERALL_REASON)],
              ["Universe percentile", fmtPercentile(scorecard.universe_percentile, overallMissing ? "unranked" : "not ranked")],
              ["Sector percentile", fmtPercentile(scorecard.sector_percentile, overallMissing ? "unranked" : "sector too small")],
            ]}
          />
        </div>
      </div>

      <p className="mt-3 text-sm text-slate-200 print:text-slate-800" data-testid="memo-scorecard-profile">
        <span className="text-xs uppercase tracking-widest text-slate-400 print:text-slate-600 mr-2">Model read</span>
        {profileText(scorecard.profiles)}
        <span className="text-slate-400 print:text-slate-600"> — a scenario label from the sub-composites, not a recommendation.</span>
      </p>

      <table className="w-full text-sm mt-3" data-testid="memo-scorecard-categories">
        <thead>
          <tr className="text-xs uppercase tracking-wider text-slate-400 print:text-slate-600">
            <th scope="col" className="text-left px-3 py-1.5 font-normal">Family</th>
            <th scope="col" className="text-right px-3 py-1.5 font-normal">Score</th>
            <th scope="col" className="text-right px-3 py-1.5 font-normal">z</th>
            <th scope="col" className="text-right px-3 py-1.5 font-normal">Percentile</th>
            <th scope="col" className="text-right px-3 py-1.5 font-normal">Inputs</th>
          </tr>
        </thead>
        <tbody>
          {families.map((fam) => {
            const cat = scorecard.categories?.[fam];
            const reason = !cat ? "not scored" : cat.n_available === 0 ? "no applicable inputs" : `${cat.n_available} of ${cat.n_features} inputs`;
            return (
              <tr key={fam} className="border-b border-slate-800 print:border-slate-200 last:border-0" data-testid={`memo-category-${fam}`} data-missing={cat && isNum(cat.score) ? undefined : "true"}>
                <td className="px-3 py-1.5 text-slate-300 print:text-slate-700">{familyLabel(fam)}</td>
                <td className="px-3 py-1.5 text-right font-mono">{fmtScore(cat?.score, reason)}</td>
                <td className="px-3 py-1.5 text-right font-mono">{fmtZ(cat?.z, reason)}</td>
                <td className="px-3 py-1.5 text-right font-mono">{fmtPercentile(cat?.percentile, reason)}</td>
                <td className="px-3 py-1.5 text-right font-mono text-slate-400 print:text-slate-600">{cat ? `${cat.n_available}/${cat.n_features}` : na("not scored")}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="grid md:grid-cols-2 gap-4 mt-3 print:grid-cols-2">
        <ContribList label="Top positive contributors" items={scorecard.top_positive ?? []} emptyReason={contributorEmpty} testId="memo-scorecard-top-positive" />
        <ContribList label="Top negative contributors" items={scorecard.top_negative ?? []} emptyReason={contributorEmpty} testId="memo-scorecard-top-negative" />
      </div>

      {disagreement && (
        <div role="note" className="mt-4 border border-amber-500/50 print:border-amber-700 rounded p-3 text-sm" data-testid="memo-scorecard-disagreement">
          <div className="font-semibold text-amber-400 print:text-amber-800">
            Memo / scorecard disagreement · {disagreement.severity} · {humanize(disagreement.dimension).toLowerCase()}
          </div>
          <p className="mt-1 text-slate-200 print:text-slate-800">
            {disagreement.note} ({directionText(disagreement.direction)}; gap {disagreement.gap > 0 ? "+" : ""}
            {disagreement.gap.toFixed(0)} points.)
          </p>
          <p className="mt-1 text-slate-300 print:text-slate-700">
            <span className="font-semibold">PM reconciliation:</span> {scorecard.reconciliation || na("not yet written")}
          </p>
        </div>
      )}

      <p className="mt-3 text-[10px] leading-relaxed text-slate-500 print:text-slate-600">
        Observed figures come from reported statements (latest period {scorecard.latest_period || na("unknown")}, available {scorecard.data_available_at || na("unknown")}).
        The model read is spec {scorecard.version_key} scored against the universe on {scorecard.as_of}: sector-neutral z-scores, equal family weights,
        contributions summing to the overall z. A scenario output for research and education, not a recommendation.
      </p>
    </div>
  );
}

function KvTable({ title, rows }: { title: string; rows: [string, string][] }) {
  return (
    <div className="border border-slate-700 print:border-slate-300 rounded">
      <div className="text-xs uppercase tracking-wider px-3 py-1.5 border-b border-slate-700 print:border-slate-300 text-slate-400 print:text-slate-600">
        {title}
      </div>
      <table className="w-full text-sm">
        <tbody>
          {rows.map(([k, v], i) => (
            <tr key={i} className="border-b border-slate-800 print:border-slate-200 last:border-0">
              <td className="px-3 py-1.5 text-slate-400 print:text-slate-600">{k}</td>
              <td className="px-3 py-1.5 text-right font-mono">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RiskList({ items }: { items: RiskItem[] }) {
  return (
    <ul className="space-y-2">
      {items.map((r, i) => (
        <li key={i} className="text-sm">
          <div className="flex items-baseline gap-2">
            <span className="font-semibold">{r.title}</span>
            <span className="text-[10px] uppercase tracking-wider text-slate-400 print:text-slate-600">
              {r.type} · {r.severity}
            </span>
          </div>
          <div className="text-slate-300 print:text-slate-700">{r.detail}</div>
        </li>
      ))}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function pickNumber(obj: Record<string, unknown>, keys: string[]): number | null {
  for (const k of keys) {
    const v = obj?.[k];
    if (typeof v === "number" && Number.isFinite(v)) return v;
  }
  return null;
}

function fmtPctLoose(v: number): string {
  // Accepts either 0.12 or 12 — heuristically formats either as a %.
  if (Math.abs(v) <= 1.5) return fmtPct(v);
  return `${v.toFixed(1)}%`;
}

function formatDate(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, {
    year: "numeric", month: "long", day: "numeric",
  });
}

/**
 * Open the memo body in a popup window with a print-only stylesheet,
 * then trigger window.print(). The user picks "Save as PDF" in the
 * print dialog for a proper text-PDF. Zero deps; works in every modern
 * browser; produces selectable + searchable text (not a screenshot).
 */
function downloadPdf(memo: StockMemoOut, container: HTMLDivElement | null) {
  if (!container) return;
  const html = container.outerHTML;
  const popup = window.open("", "_blank", "width=900,height=1200");
  if (!popup) {
    // Popup blocked — fall back to printing the current window. The user
    // can still pick "Save as PDF" from the dialog.
    window.print();
    return;
  }
  const filename = `${memo.ticker}-investment-memo-${memo.generated_at?.slice(0, 10) || ""}.pdf`;
  popup.document.open();
  popup.document.write(`<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>${filename}</title>
  <style>
    @page { size: Letter; margin: 0.6in; }
    html, body {
      background: #ffffff; color: #111827; margin: 0; padding: 0;
      font-family: Georgia, 'Times New Roman', serif;
      -webkit-print-color-adjust: exact; print-color-adjust: exact;
    }
    body { font-size: 11pt; line-height: 1.45; }
    h1 { font-size: 26pt; margin: 0 0 4pt; }
    h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #94a3b8; padding-bottom: 2pt; }
    h3, h4 { font-size: 11pt; margin: 10pt 0 4pt; }
    p { margin: 0 0 6pt; }
    ul, ol { margin: 4pt 0 8pt 18pt; padding: 0; }
    li { margin: 0 0 2pt; }
    table { width: 100%; border-collapse: collapse; }
    table, th, td { border-color: #cbd5e1; }
    section { break-inside: avoid; page-break-inside: avoid; }
    .memo-paper { padding: 0 !important; max-width: none !important; }
    /* Hide the dark-mode utility classes when printing */
    [class*="text-slate-100"], [class*="text-slate-200"], [class*="text-slate-300"] { color: #111827 !important; }
    [class*="text-slate-400"], [class*="text-slate-500"], [class*="text-slate-600"] { color: #4b5563 !important; }
    [class*="bg-ink-"], [class*="bg-black"] { background: transparent !important; }
    .border-ink-700, .border-slate-700, .border-ink-800 { border-color: #cbd5e1 !important; }
    /* Bull / bear accents stay visible in print */
    .border-emerald-500\\/40 { border-color: #047857 !important; }
    .border-rose-500\\/40 { border-color: #be123c !important; }
  </style>
</head>
<body>
  ${html}
</body>
</html>`);
  popup.document.close();
  // Wait for the new window to lay out, then open the print dialog.
  popup.onload = () => {
    popup.focus();
    popup.print();
  };
  // Fallback for browsers that fire load before script runs
  setTimeout(() => {
    try { popup.focus(); popup.print(); } catch { /* noop */ }
  }, 400);
}
