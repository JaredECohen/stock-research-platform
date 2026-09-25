import React from "react";
import type {
  AgentFinding,
  BullBearAnalysis,
  BullBearCase,
  SectionAvailability,
  StockMemoOut,
} from "@/types";
import { fmtPct, fmtPrice, fmtUpside, numOrNull, ratingBadgeClass } from "@/lib/format";
import {
  availability,
  bannerCount,
  bannerText,
  intakeRationale,
  isHidden,
  isUnavailable,
  MEMO_CARD_SECTIONS,
} from "@/lib/memoSections";
import UnavailableSection, { DegradedNote } from "./UnavailableSection";
import CrossSectorChips from "./CrossSectorChips";
import TerminalClampBadge from "./TerminalClampBadge";
import DiligenceDialog from "./DiligenceDialog";
import EarningsBreakdown from "./EarningsBreakdown";
import MacroRegimeBanner from "./MacroRegimeBanner";
import { Markdown } from "./Markdown";
import PMDCFAdjustments from "./PMDCFAdjustments";
import { fmtEtDate } from "./LiveQuote";
import type { EarningsStructured } from "@/types";

/**
 * Wave 8N — explicit two-card scorecard so users can't conflate the
 * agent's *conviction* in its rating call (Confidence) with the
 * *quantitative ranking* of the company's fundamentals (Stock Score).
 *
 * Each card has:
 *  - A distinct icon + accent color
 *  - A bold heading (the metric name)
 *  - The 0-100 number with a horizontal bar
 *  - A one-line plain-English description (NOT a tooltip — always visible)
 *  - The components feeding into it
 */
function ScorecardRow({
  confidence,
  confidenceAvailability,
  factorPmScore,
  rating,
}: {
  confidence: number;
  // W2a: when the presenter hid the confidence (its PM input was a
  // template), the card shows the placeholder instead of the number and
  // bar. The number is never rewritten server-side, so the card must not
  // print it.
  confidenceAvailability?: SectionAvailability;
  factorPmScore?: number;
  rating: string;
}) {
  const confidenceHidden =
    confidenceAvailability?.status === "unavailable" &&
    confidenceAvailability.reason !== "not_produced";
  const tone = (v: number) =>
    v >= 70 ? "text-accent-500"
    : v >= 50 ? "text-slate-100"
    : v >= 30 ? "text-warn-500"
    : "text-danger-500";
  const bar = (v: number) =>
    v >= 70 ? "bg-accent-500"
    : v >= 50 ? "bg-slate-400"
    : v >= 30 ? "bg-warn-500"
    : "bg-danger-500";
  const conf = Math.round(confidence);
  return (
    <div className="grid md:grid-cols-[2fr_1fr] gap-3 mt-4 pt-4 border-t border-ink-700">
      {/* STOCK SCORE — primary, big, prominent */}
      <div className="card-tight !p-4 border-accent-600/40 bg-accent-600/[0.06]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-sm bg-accent-500" />
            <div className="text-xs uppercase tracking-widest text-accent-400 font-semibold">
              Stock Score
            </div>
          </div>
          <div className="text-[10px] uppercase tracking-widest text-slate-500">
            quant factor blend
          </div>
        </div>
        {typeof factorPmScore === "number" ? (
          <>
            <div className="mt-2 flex items-baseline gap-2">
              <span className={`text-5xl font-mono font-bold ${tone(factorPmScore)}`}>
                {Math.round(factorPmScore)}
              </span>
              <span className="text-sm text-slate-500">/ 100</span>
            </div>
            <div className="h-1.5 mt-3 rounded bg-ink-800 overflow-hidden">
              <div
                className={`h-full ${bar(factorPmScore)}`}
                style={{ width: `${Math.max(0, Math.min(100, factorPmScore))}%` }}
              />
            </div>
            <div className="mt-3 text-xs text-slate-200 leading-relaxed">
              How the <strong className="text-slate-100">company's fundamentals</strong>{" "}
              rank versus the universe.
            </div>
            <div className="mt-1 text-[11px] text-slate-500 leading-relaxed">
              Quality 25% · growth 20% · valuation 15% · macro fit 15% ·
              momentum 10% · risk 10% · catalyst 5%.
            </div>
          </>
        ) : (
          <div className="mt-2 text-xs text-slate-500">
            Stock score unavailable for this memo.
          </div>
        )}
      </div>

      {/* CONFIDENCE — secondary, compact, smaller numerals */}
      <div
        className="card-tight !p-3 border-ink-700"
        title={`How sure the PM is that "${rating}" is the right call. From signal counts across all 8 specialist findings, dampened by source-evidence quality.`}
      >
        <div className="flex items-center justify-between">
          <div className="text-[10px] uppercase tracking-widest text-slate-500">
            Confidence
          </div>
          <div className="text-[9px] text-slate-600">agent certainty</div>
        </div>
        {confidenceHidden ? (
          <UnavailableSection
            variant="inline"
            section="confidence_score"
            availability={confidenceAvailability}
            className="mt-1.5"
          />
        ) : (
          <>
            <div className="mt-1.5 flex items-baseline gap-1">
              <span className={`text-2xl font-mono ${tone(conf)}`}>{conf}</span>
              <span className="text-[10px] text-slate-500">/ 100</span>
            </div>
            <div className="h-1 mt-2 rounded bg-ink-800 overflow-hidden">
              <div
                className={`h-full ${bar(conf)}`}
                style={{ width: `${Math.max(0, Math.min(100, conf))}%` }}
              />
            </div>
          </>
        )}
        <div className="mt-2 text-[11px] text-slate-400 leading-snug">
          PM's certainty in the <em>"{rating}"</em> call. Not a quality
          score — see Stock Score for fundamentals ranking.
        </div>
      </div>
    </div>
  );
}


function FactorScorePanel({ scores }: { scores?: Record<string, number> }) {
  if (!scores) return null;
  const items: Array<{ key: string; label: string; weight: string; tooltip: string }> = [
    {
      key: "factor_quality",
      label: "Quality",
      weight: "25%",
      tooltip: "ROIC + operating margin + gross margin (linear ramps above floors).",
    },
    {
      key: "factor_growth",
      label: "Growth",
      weight: "20%",
      tooltip: "Revenue growth: 0% → 0, 30%+ → 100.",
    },
    {
      key: "factor_valuation",
      label: "Valuation",
      weight: "15%",
      tooltip: "EV/EBITDA + P/FCF + FCF yield (inverted — cheap scores high).",
    },
    {
      key: "factor_macro_fit",
      label: "Macro fit",
      weight: "15%",
      tooltip: "Sector × theme bias (60 baseline when no theme is selected).",
    },
    {
      key: "factor_earnings_momentum",
      label: "Earnings momentum",
      weight: "10%",
      tooltip: "Recent earnings-surprise history (50 baseline when no surprises on file).",
    },
    {
      key: "factor_risk",
      label: "Risk",
      weight: "10%",
      tooltip: "Higher = LOWER risk. Penalizes beta distance from 1, debt/EBITDA, drawdown.",
    },
    {
      key: "factor_catalyst",
      label: "Catalyst",
      weight: "5%",
      tooltip: "AI-keyword + theme bias (50/65 baseline today).",
    },
  ];
  const have = items.filter((i) => typeof scores[i.key] === "number");
  if (have.length === 0) return null;
  const tone = (v: number) =>
    v >= 70
      ? "text-accent-500"
      : v >= 50
      ? "text-slate-200"
      : v >= 30
      ? "text-warn-500"
      : "text-danger-500";
  return (
    <div className="border-t border-ink-700 mt-4 pt-3">
      <div className="flex items-center justify-between mb-2">
        <div className="section-title">Quant factor scores</div>
        <div
          className="text-[10px] uppercase tracking-widest text-slate-500"
          title="Same factor scoring the screener uses. PM score = weighted blend per the % column."
        >
          0–100 · screener-aligned
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {have.map((f) => {
          const v = scores[f.key];
          const pct = Math.max(0, Math.min(100, v));
          return (
            <div
              key={f.key}
              className="card-tight !p-2 space-y-1"
              title={f.tooltip}
            >
              <div className="flex items-center justify-between text-[10px] uppercase tracking-widest text-slate-500">
                <span>{f.label}</span>
                <span>{f.weight}</span>
              </div>
              <div className={`text-base font-mono ${tone(v)}`}>
                {Math.round(v)}
              </div>
              <div className="h-1 rounded bg-ink-800 overflow-hidden">
                <div
                  className={`h-full ${
                    v >= 70
                      ? "bg-accent-500"
                      : v >= 50
                      ? "bg-slate-400"
                      : v >= 30
                      ? "bg-warn-500"
                      : "bg-danger-500"
                  }`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


function BullBearAnalysisBlock({ analysis }: { analysis: BullBearAnalysis }) {
  const leanBadge =
    analysis.sector_lean === "bull"
      ? "bg-accent-500/15 text-accent-500 border-accent-500/30"
      : analysis.sector_lean === "bear"
      ? "bg-danger-500/15 text-danger-500 border-danger-500/30"
      : "bg-slate-500/15 text-slate-300 border-slate-500/30";
  return (
    <div className="card-tight border-ink-700/80 space-y-3">
      <div className="flex items-center justify-between">
        <div className="section-title">Sector synthesis · key disagreement</div>
        <span
          className={`text-[10px] uppercase tracking-widest px-2 py-0.5 rounded border ${leanBadge}`}
          title="Sector analyst's lean — PM may diverge"
        >
          Sector lean: {analysis.sector_lean}
        </span>
      </div>
      {analysis.sector_synthesis && (
        <p className="text-sm text-slate-200">{analysis.sector_synthesis}</p>
      )}
      {analysis.key_disagreement && (
        <div className="text-sm">
          <span className="text-slate-400">Where bulls and bears disagree: </span>
          <span className="text-slate-100">{analysis.key_disagreement}</span>
        </div>
      )}
      {analysis.falsifiable_tests?.length > 0 && (
        <div>
          <div className="section-title mb-1">Falsifiable tests</div>
          <ul className="text-xs text-slate-300 space-y-1">
            {analysis.falsifiable_tests.map((t, i) => (
              <li key={i}>
                <span
                  className={
                    t.invalidates_side === "bull"
                      ? "text-accent-500 font-medium"
                      : "text-danger-500 font-medium"
                  }
                >
                  Invalidates {t.invalidates_side}:
                </span>{" "}
                {t.statement}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function FindingBlock({
  title,
  body,
  footer,
  extra,
  section,
  memo,
}: {
  title: string;
  body: AgentFinding | { headline: string; summary: string; key_points?: string[] };
  footer?: React.ReactNode;
  extra?: React.ReactNode;
  // W2a: the finding's map key; with `memo`, it decides whether the card
  // shows the analyst's view or the placeholder with its reason.
  section?: string;
  memo?: StockMemoOut;
}) {
  const [showFull, setShowFull] = React.useState(false);
  // Long-form report only present on AgentFinding shape; the structurally-typed
  // alternative has no `long_form_report` field, so the cast is a no-op there.
  const longForm = (body as AgentFinding).long_form_report;
  const av = section ? availability(memo, section) : undefined;
  if (section && memo && isHidden(memo, section)) {
    // A skipped analyst still gets its card, so the reader sees that the PM
    // chose to skip it and why. `extra`/`footer` read the finding's data,
    // which the presenter reduced to its allowlist — nothing left to show.
    return (
      <UnavailableSection
        title={title}
        section={section}
        availability={av}
        detail={
          av?.reason === "skipped_by_intake"
            ? intakeRationale(memo, body as AgentFinding) || undefined
            : undefined
        }
      />
    );
  }
  // The presenter hides a template drill-down on its own (only the
  // analyst's expansion is ever shown); say so rather than dropping the
  // button silently.
  const drilldownHidden =
    !!section && !!memo && isHidden(memo, `${section}.long_form_report`);
  return (
    <div className="card-tight">
      <div className="section-title mb-1">{title}</div>
      <DegradedNote availability={av} className="mb-1" />
      <div className="text-sm font-medium text-slate-100">{body.headline}</div>
      <div className="text-sm text-slate-300 mt-1">{body.summary}</div>
      {body.key_points && body.key_points.length > 0 && (
        <ul className="text-xs text-slate-400 mt-2 list-disc pl-5 space-y-0.5">
          {body.key_points.slice(0, 4).map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      )}
      {extra}
      {longForm && (
        <div className="mt-2">
          <button
            type="button"
            onClick={() => setShowFull((v) => !v)}
            className="text-xs text-accent-500 hover:text-accent-400 inline-flex items-center gap-1"
          >
            {showFull ? "Hide full report ▴" : "Read full report ▾"}
          </button>
          {showFull && (
            <div className="mt-3 border-t border-ink-700 pt-3 text-slate-200">
              <Markdown text={longForm} />
            </div>
          )}
        </div>
      )}
      {drilldownHidden && !longForm && (
        <div className="mt-2 text-[11px] text-slate-500" data-testid="drilldown-unavailable">
          Full report unavailable in this version.
        </div>
      )}
      {footer && <div className="mt-3 border-t border-ink-700 pt-2">{footer}</div>}
    </div>
  );
}

/**
 * Bull or Bear card. The presenter removes template items and replaces a
 * template headline with the placeholder; this renders its verdict: the
 * placeholder and reason for an unavailable case (plus any computed item the
 * presenter kept, such as "DCF bull case implies …" — numbers are never
 * hidden), or the surviving items under a headline placeholder with an
 * "N template items not shown" note for a partially templated case.
 */
function CaseCard({
  title,
  tone,
  data,
  section,
  memo,
}: {
  title: string;
  tone: "bull" | "bear";
  data: BullBearCase;
  section: "bull_case" | "bear_case";
  memo: StockMemoOut;
}) {
  const av = availability(memo, section);
  const hidden = isHidden(memo, section);
  const headlineHidden = hidden || !!av?.headline_hidden;
  const color = tone === "bull" ? "text-accent-500" : "text-danger-500";
  return (
    <div className="card-tight" data-testid={`case-${tone}`}>
      <div className="section-title mb-1 flex items-center gap-2">{title}</div>
      {headlineHidden ? (
        // The reason line belongs to the whole card only when the whole
        // card is unavailable; a hidden headline on a partial case is
        // explained by the degraded note below.
        <UnavailableSection variant="inline" section={section} availability={hidden ? av : undefined} />
      ) : (
        <div className={`text-sm font-medium ${color}`}>{data.headline}</div>
      )}
      {data.key_points.length > 0 && (
        <ul className="text-sm text-slate-300 mt-2 list-disc pl-5 space-y-1">
          {data.key_points.map((p, i) => <li key={i}>{p}</li>)}
        </ul>
      )}
      {/* An unavailable case's placeholder already carries its item count. */}
      {!hidden && <DegradedNote availability={av} className="mt-2" />}
    </div>
  );
}

export default function MemoCard({ memo }: { memo: StockMemoOut }) {
  const dcf = memo.dcf_summary as Record<string, unknown>;
  const degraded = memo.degraded_agents ?? [];
  // Phase 6 sector-finding fields ride on `sector_agent_view.data`.
  const sectorData = memo.sector_agent_view.data;
  const crossSector = sectorData?.cross_sector_relevance ?? [];
  const macroBroadcast = sectorData?.macro_broadcast;
  const macroAlignment = sectorData?.macro_alignment;
  // W2a: sections the presenter hid. A memo can have hidden sections and no
  // degraded agents (a pre-flag template PM view carries no event), so the
  // banner shows for either.
  const hiddenCount = bannerCount(memo, MEMO_CARD_SECTIONS);
  return (
    <div className="space-y-4">
      {(degraded.length > 0 || hiddenCount > 0) && (
        <div
          className="card-tight border-warn-500/40 bg-warn-500/5 text-warn-500 text-sm"
          data-testid="partial-result-banner"
        >
          <span className="font-semibold">Partial result:</span>{" "}
          {degraded.length > 0 && (
            <>
              {degraded.length} agent{degraded.length === 1 ? "" : "s"} degraded —{" "}
              <span className="text-slate-200">{degraded.join(", ")}</span>. These
              either failed or fell back to deterministic output; treat their
              sections as thinner evidence.
            </>
          )}
          {hiddenCount > 0 && (
            <span data-testid="unavailable-count">
              {degraded.length > 0 ? " · " : ""}
              {bannerText(hiddenCount)}.
            </span>
          )}
        </div>
      )}
      <div className="card">
        {/* Identity row: ticker + name + rating badge. Compact. */}
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="flex-1 min-w-0">
            <div className="text-xs uppercase tracking-widest text-slate-500">
              {memo.sector}
            </div>
            <div className="text-2xl font-semibold mt-1">
              {memo.ticker} ·{" "}
              <span className="text-slate-300 font-normal">{memo.company_name}</span>
            </div>
          </div>
          <div className="flex flex-col items-end gap-1 shrink-0">
            <span className={ratingBadgeClass(memo.rating_label)}>
              {memo.rating_label}
            </span>
            {/* The rating is never hidden (60% of it is the quant factor
                blend), but when its PM input was a template the reader is
                told what it rests on. */}
            <DegradedNote
              availability={availability(memo, "rating_label")}
              className="max-w-[16rem] text-right"
            />
            <div
              className="text-[10px] uppercase tracking-widest text-slate-600"
              title={
                memo.generation_mode === "demo"
                  ? "Generated from the demo dataset — figures are illustrative, not live market data."
                  : "Generated from live market data."
              }
            >
              {memo.generation_mode === "demo" ? "demo data" : "live data"}
            </div>
          </div>
        </div>

        {/* HEADLINE THESIS — biggest above-the-fold takeaway. */}
        <div className="mt-4 rounded-lg bg-ink-800/60 border border-ink-700 px-4 py-3">
          <div className="text-[10px] uppercase tracking-widest text-accent-500 mb-1">
            One-sentence thesis
          </div>
          {isHidden(memo, "one_sentence_thesis") ? (
            <UnavailableSection
              variant="inline"
              section="one_sentence_thesis"
              availability={availability(memo, "one_sentence_thesis")}
            />
          ) : (
            <>
              <div className="text-base md:text-lg text-slate-100 leading-snug">
                {memo.one_sentence_thesis}
              </div>
              {/* A degraded thesis is a builder rewrite around the analyst's
                  claim; it stays visible only with this note (critique
                  delta 11), or the canned sentence reads as analysis. */}
              <DegradedNote
                availability={availability(memo, "one_sentence_thesis")}
                section="one_sentence_thesis"
                className="mt-1"
              />
            </>
          )}
          {isHidden(memo, "valuation_verdict") ? (
            // The verdict step failed: its `ValuationVerdict()` default
            // would otherwise read as a "fairly priced" call.
            <UnavailableSection
              variant="inline"
              title="Valuation verdict"
              section="valuation_verdict"
              availability={availability(memo, "valuation_verdict")}
              className="mt-2 pt-2 border-t border-ink-700"
            />
          ) : memo.valuation_verdict?.summary && (
            <div className="mt-2 pt-2 border-t border-ink-700 text-xs text-slate-300">
              <span className="text-[10px] uppercase tracking-widest text-slate-500 mr-2">
                Valuation verdict
              </span>
              {memo.valuation_verdict.summary}
            </div>
          )}
        </div>

        <ScorecardRow
          confidence={memo.confidence_score}
          confidenceAvailability={availability(memo, "confidence_score")}
          factorPmScore={memo.scores?.factor_pm_score}
          rating={memo.rating_label}
        />

        <FactorScorePanel scores={memo.scores} />
        <div className="border-t border-ink-700 mt-4 pt-3 text-sm text-slate-200">
          <div className="section-title mb-1">PM Final View</div>
          {isHidden(memo, "final_pm_view") ? (
            <UnavailableSection
              variant="inline"
              section="final_pm_view"
              availability={availability(memo, "final_pm_view")}
            />
          ) : (
            <p>{memo.final_pm_view}</p>
          )}
          {crossSector.length > 0 && <CrossSectorChips tickers={crossSector} className="mt-3" />}
        </div>
      </div>

      {/* The one section whose `not_produced` also gets a placeholder: the
          card used to vanish, which read as "no view" rather than "none
          was produced". */}
      {isUnavailable(memo, "mispricing_thesis") ? (
        <UnavailableSection
          title="Where We Differ From Consensus"
          section="mispricing_thesis"
          availability={availability(memo, "mispricing_thesis")}
          className="border-accent-600/30"
        />
      ) : memo.mispricing_thesis &&
        (memo.mispricing_thesis.consensus_view ||
          memo.mispricing_thesis.our_view ||
          memo.mispricing_thesis.gap) && (
          <div className="card-tight border-accent-600/30">
            <div className="section-title mb-2">Where We Differ From Consensus</div>
            <div className="space-y-1.5 text-sm text-slate-200">
              {memo.mispricing_thesis.consensus_view && (
                <p>
                  <span className="text-slate-400">Consensus:</span>{" "}
                  {memo.mispricing_thesis.consensus_view}
                </p>
              )}
              {memo.mispricing_thesis.our_view && (
                <p>
                  <span className="text-slate-400">Our view:</span>{" "}
                  {memo.mispricing_thesis.our_view}
                </p>
              )}
              {memo.mispricing_thesis.gap && (
                <p>
                  <span className="text-slate-400">The gap:</span>{" "}
                  {memo.mispricing_thesis.gap}
                </p>
              )}
              {memo.mispricing_thesis.falsifiers?.length > 0 && (
                <div className="pt-1">
                  <span className="text-xs text-slate-400">
                    What would prove us wrong:
                  </span>
                  <ul className="text-xs text-slate-300 list-disc pl-5 mt-1 space-y-0.5">
                    {memo.mispricing_thesis.falsifiers.map((f, i) => (
                      <li key={i}>{f}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}

      {macroBroadcast && (
        <MacroRegimeBanner
          broadcast={macroBroadcast}
          alignment={macroAlignment}
          sector={memo.sector}
        />
      )}

      <div className="grid md:grid-cols-2 gap-4">
        <FindingBlock
          title="Sector Analyst"
          body={memo.sector_agent_view}
          section="sector_agent_view"
          memo={memo}
          footer={
            crossSector.length > 0 ? <CrossSectorChips tickers={crossSector} /> : undefined
          }
        />
        <FindingBlock
          title="Earnings Analyst"
          body={memo.earnings_agent_view}
          section="earnings_agent_view"
          memo={memo}
          extra={
            memo.earnings_agent_view.data &&
            (memo.earnings_agent_view.data as { structured?: EarningsStructured }).structured ? (
              <EarningsBreakdown
                structured={
                  (memo.earnings_agent_view.data as { structured: EarningsStructured }).structured
                }
              />
            ) : undefined
          }
        />
        <FindingBlock title="Filing Analyst" body={memo.filing_agent_view} section="filing_agent_view" memo={memo} />
        <FindingBlock title="Valuation Analyst" body={memo.valuation_agent_view} section="valuation_agent_view" memo={memo} />
        <FindingBlock title="Comps Analyst" body={memo.comps_agent_view} section="comps_agent_view" memo={memo} />
        <FindingBlock title="Macro Analyst" body={memo.macro_sensitivity} section="macro_sensitivity" memo={memo} />
        {/* A PM intake skip keeps its (blanked) finding, so this card still
            renders and says why the analyst did not run. */}
        {memo.technical_agent_view && (
          <FindingBlock
            title="Technical Analyst"
            body={memo.technical_agent_view}
            section="technical_agent_view"
            memo={memo}
          />
        )}
      </div>

      {memo.round_findings && memo.round_findings.length > 0 && (
        <DiligenceDialog rounds={memo.round_findings} />
      )}

      <PMDCFAdjustments memo={memo} />


      <div className="grid md:grid-cols-2 gap-4">
        <CaseCard title="Bull Case" tone="bull" data={memo.bull_case} section="bull_case" memo={memo} />
        <CaseCard title="Bear Case" tone="bear" data={memo.bear_case} section="bear_case" memo={memo} />
      </div>

      {/* The presenter drops a template synthesis block from the sector
          finding's data; the placeholder says it existed and why it is gone. */}
      {isHidden(memo, "sector_synthesis") ? (
        <UnavailableSection
          title="Sector synthesis · key disagreement"
          section="sector_synthesis"
          availability={availability(memo, "sector_synthesis")}
        />
      ) : (
        sectorData?.bull_bear_analysis && (
          <BullBearAnalysisBlock analysis={sectorData.bull_bear_analysis} />
        )
      )}

      {/* Empty-extraction memos (B4) must not render bare section headers.
          A list the presenter emptied is not an empty extraction: it gets
          the placeholder. */}
      {(memo.catalysts.length > 0 ||
        memo.key_risks.length > 0 ||
        isHidden(memo, "catalysts") ||
        isHidden(memo, "key_risks")) && (
        <div className="grid md:grid-cols-2 gap-4">
          {isHidden(memo, "catalysts") ? (
            <UnavailableSection
              title="Catalysts"
              section="catalysts"
              availability={availability(memo, "catalysts")}
            />
          ) : memo.catalysts.length > 0 && (
            <div className="card-tight">
              <div className="section-title mb-1">Catalysts</div>
              <ul className="text-sm text-slate-300 space-y-1">
                {memo.catalysts.map((c, i) => (
                  <li key={i}>
                    <span className="font-medium text-slate-100">{c.title}</span>
                    <span className="text-xs text-slate-400 ml-2">[{c.horizon} · {c.impact}]</span>
                    <div className="text-xs text-slate-400">{c.detail}</div>
                  </li>
                ))}
              </ul>
              <DegradedNote availability={availability(memo, "catalysts")} className="mt-2" />
            </div>
          )}
          {isHidden(memo, "key_risks") ? (
            <UnavailableSection
              title="Key Risks & Thesis Breakers"
              section="key_risks"
              availability={availability(memo, "key_risks")}
            />
          ) : memo.key_risks.length > 0 && (
            <div className="card-tight">
              <div className="section-title mb-1">Key Risks & Thesis Breakers</div>
              <ul className="text-sm text-slate-300 space-y-1">
                {memo.key_risks.map((r, i) => (
                  <li key={i}>
                    <span className="font-medium text-slate-100">{r.title}</span>
                    <span className="text-xs text-slate-400 ml-2">[{r.severity} · {r.type}]</span>
                  </li>
                ))}
              </ul>
              <DegradedNote availability={availability(memo, "key_risks")} className="mt-2" />
            </div>
          )}
        </div>
      )}

      <div className="card-tight">
        <div className="section-title mb-1">DCF Snapshot</div>
        <DegradedNote availability={availability(memo, "dcf_summary")} className="mb-2" />
        {isHidden(memo, "dcf_summary") ? (
          // The DCF engine failed on this run; say why rather than the bare
          // "DCF unavailable." an empty summary gets.
          <UnavailableSection
            variant="inline"
            section="dcf_summary"
            availability={availability(memo, "dcf_summary")}
          />
        ) : dcf && Object.keys(dcf).length > 0 ? (
          (() => {
            // null = the engine could not compute the number (no share
            // count / no quote) → rendered "n/a". `Number(null)` is 0,
            // which is exactly the "+0.0%" lie this block used to print.
            const current = numOrNull(dcf.current_price);
            const base = numOrNull(dcf.base_implied_price);
            const bull = numOrNull(dcf.bull_implied_price);
            const bear = numOrNull(dcf.bear_implied_price);
            // Prefer per-scenario upside fields when present (Wave 8L);
            // recompute from current_price only for memos that pre-date
            // them (key absent) — never when the field is present-but-null.
            const upside = (key: string, implied: number | null): number | null => {
              if (key in dcf) return numOrNull(dcf[key]);
              return implied != null && current != null && current > 0
                ? (implied - current) / current
                : null;
            };
            const baseUp = upside("base_upside", base);
            const bullUp = upside("bull_upside", bull);
            const bearUp = upside("bear_upside", bear);
            const tone = (v: number | null) =>
              v == null
                ? "text-slate-500"
                : v > 0.005 ? "text-accent-500" : v < -0.005 ? "text-danger-500" : "text-slate-400";
            return (
              <div className="space-y-3 text-sm">
                <div className="flex items-baseline gap-2 flex-wrap">
                  {/* W5b: this is the quote the memo's DCF ran on, frozen
                      in the stored memo; it was labelled "Current" and could
                      be months old. Today's price is the live chip above. */}
                  <span className="text-xs text-slate-500">Price used in DCF</span>
                  <span className="font-mono text-base text-slate-100">
                    {fmtPrice(current)}
                  </span>
                  {memo.generated_at && (
                    <span className="text-[10px] text-slate-500">
                      {/* ET, like the chip's "memo price …" date: the UTC
                          slice put evening-ET memos on the next day. */}
                      as of memo, {fmtEtDate(memo.generated_at)}
                    </span>
                  )}
                  {dcf.tv_clamped === true && <TerminalClampBadge className="ml-auto" />}
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <div className="text-xs text-slate-500">Bear</div>
                    <div className="font-mono text-base">{fmtPrice(bear)}</div>
                    <div className={`text-xs ${tone(bearUp)}`}>
                      {fmtUpside(bearUp)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500">Base</div>
                    <div className="font-mono text-base">{fmtPrice(base)}</div>
                    <div className={`text-xs ${tone(baseUp)}`}>
                      {fmtUpside(baseUp)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500">Bull</div>
                    <div className="font-mono text-base">{fmtPrice(bull)}</div>
                    <div className={`text-xs ${tone(bullUp)}`}>
                      {fmtUpside(bullUp)}
                    </div>
                  </div>
                </div>
                <div className="text-xs text-slate-500 pt-2 border-t border-ink-700">
                  WACC{" "}
                  <span className="text-slate-300 font-mono">
                    {fmtPct(Number(dcf.wacc), 2)}
                  </span>{" "}
                  · Terminal growth{" "}
                  <span className="text-slate-300 font-mono">
                    {fmtPct(Number(dcf.terminal_growth), 1)}
                  </span>
                </div>
              </div>
            );
          })()
        ) : (
          <div className="text-sm text-slate-400">DCF unavailable.</div>
        )}
      </div>

      {isHidden(memo, "risk_committee_challenge") ? (
        <UnavailableSection
          title="Risk Committee Challenge"
          section="risk_committee_challenge"
          availability={availability(memo, "risk_committee_challenge")}
        />
      ) : (
        <div className="card-tight border-warn-500/30 bg-warn-500/5">
          <div className="section-title mb-1 text-warn-500">Risk Committee Challenge</div>
          <div className="text-sm text-slate-200">{memo.risk_committee_challenge.overall_assessment}</div>
          {memo.risk_committee_challenge.challenges.length > 0 && (
            <>
              <div className="text-xs text-slate-400 mt-2">Challenges raised:</div>
              <ul className="text-sm text-slate-300 list-disc pl-5 space-y-0.5">
                {memo.risk_committee_challenge.challenges.map((c, i) => <li key={i}>{c}</li>)}
              </ul>
            </>
          )}
          {memo.risk_committee_challenge.suggested_revisions.length > 0 && (
            <>
              <div className="text-xs text-slate-400 mt-2">Suggested revisions:</div>
              <ul className="text-sm text-slate-300 list-disc pl-5 space-y-0.5">
                {memo.risk_committee_challenge.suggested_revisions.slice(0, 4).map((c, i) => <li key={i}>{c}</li>)}
              </ul>
            </>
          )}
        </div>
      )}

      {isHidden(memo, "final_verdict") ? (
        <UnavailableSection
          title="Final Verdict"
          section="final_verdict"
          availability={availability(memo, "final_verdict")}
        />
      ) : (
        <div className="card-tight">
          <div className="section-title mb-1">Final Verdict</div>
          <DegradedNote availability={availability(memo, "final_verdict")} className="mb-1" />
          <div className="text-sm text-slate-200">{memo.final_verdict}</div>
        </div>
      )}

      <div className="text-[11px] text-slate-500 leading-snug">
        Sources: {memo.sources_used.slice(0, 8).join(" · ")}
        <br />
        {memo.disclaimer}
      </div>
    </div>
  );
}
