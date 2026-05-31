import React from "react";
import type { AgentFinding, BullBearAnalysis, StockMemoOut } from "@/types";
import { fmtCurrency, fmtPct, ratingBadgeClass } from "@/lib/format";
import CrossSectorChips from "./CrossSectorChips";
import DiligenceDialog from "./DiligenceDialog";
import EarningsBreakdown from "./EarningsBreakdown";
import MacroRegimeBanner from "./MacroRegimeBanner";
import { Markdown } from "./Markdown";
import PMDCFAdjustments from "./PMDCFAdjustments";
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
  factorPmScore,
  rating,
}: {
  confidence: number;
  factorPmScore?: number;
  rating: string;
}) {
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
}: {
  title: string;
  body: AgentFinding | { headline: string; summary: string; key_points?: string[] };
  footer?: React.ReactNode;
  extra?: React.ReactNode;
}) {
  const [showFull, setShowFull] = React.useState(false);
  // Long-form report only present on AgentFinding shape; the structurally-typed
  // alternative has no `long_form_report` field, so the cast is a no-op there.
  const longForm = (body as AgentFinding).long_form_report;
  return (
    <div className="card-tight">
      <div className="section-title mb-1">{title}</div>
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
      {footer && <div className="mt-3 border-t border-ink-700 pt-2">{footer}</div>}
    </div>
  );
}

export default function MemoCard({ memo }: { memo: StockMemoOut }) {
  const dcf = memo.dcf_summary as Record<string, number | string | undefined>;
  const degraded = memo.degraded_agents ?? [];
  // Phase 6 sector-finding fields ride on `sector_agent_view.data`.
  const sectorData = memo.sector_agent_view.data;
  const crossSector = sectorData?.cross_sector_relevance ?? [];
  const macroBroadcast = sectorData?.macro_broadcast;
  const macroAlignment = sectorData?.macro_alignment;
  return (
    <div className="space-y-4">
      {degraded.length > 0 && (
        <div className="card-tight border-warn-500/40 bg-warn-500/5 text-warn-500 text-sm">
          <span className="font-semibold">Partial result:</span>{" "}
          {degraded.length} agent{degraded.length === 1 ? "" : "s"} unavailable —{" "}
          <span className="text-slate-200">{degraded.join(", ")}</span>. The memo
          was generated from the remaining specialists.
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
            <div className="text-[10px] uppercase tracking-widest text-slate-600">
              mode: {memo.generation_mode}
            </div>
          </div>
        </div>

        {/* HEADLINE THESIS — biggest above-the-fold takeaway. */}
        <div className="mt-4 rounded-lg bg-ink-800/60 border border-ink-700 px-4 py-3">
          <div className="text-[10px] uppercase tracking-widest text-accent-500 mb-1">
            One-sentence thesis
          </div>
          <div className="text-base md:text-lg text-slate-100 leading-snug">
            {memo.one_sentence_thesis}
          </div>
        </div>

        <ScorecardRow
          confidence={memo.confidence_score}
          factorPmScore={memo.scores?.factor_pm_score}
          rating={memo.rating_label}
        />

        <FactorScorePanel scores={memo.scores} />
        <div className="border-t border-ink-700 mt-4 pt-3 text-sm text-slate-200">
          <div className="section-title mb-1">PM Final View</div>
          <p>{memo.final_pm_view}</p>
          {crossSector.length > 0 && <CrossSectorChips tickers={crossSector} className="mt-3" />}
        </div>
      </div>

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
          footer={
            crossSector.length > 0 ? <CrossSectorChips tickers={crossSector} /> : undefined
          }
        />
        <FindingBlock
          title="Earnings Analyst"
          body={memo.earnings_agent_view}
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
        <FindingBlock title="Filing Analyst" body={memo.filing_agent_view} />
        <FindingBlock title="Valuation Analyst" body={memo.valuation_agent_view} />
        <FindingBlock title="Comps Analyst" body={memo.comps_agent_view} />
        <FindingBlock title="Macro Analyst" body={memo.macro_sensitivity} />
        {memo.technical_agent_view && (
          <FindingBlock title="Technical Analyst" body={memo.technical_agent_view} />
        )}
      </div>

      {memo.round_findings && memo.round_findings.length > 0 && (
        <DiligenceDialog rounds={memo.round_findings} />
      )}

      <PMDCFAdjustments memo={memo} />


      <div className="grid md:grid-cols-2 gap-4">
        <div className="card-tight">
          <div className="section-title mb-1 flex items-center gap-2">Bull Case</div>
          <div className="text-sm font-medium text-accent-500">{memo.bull_case.headline}</div>
          <ul className="text-sm text-slate-300 mt-2 list-disc pl-5 space-y-1">
            {memo.bull_case.key_points.map((p, i) => <li key={i}>{p}</li>)}
          </ul>
        </div>
        <div className="card-tight">
          <div className="section-title mb-1 flex items-center gap-2">Bear Case</div>
          <div className="text-sm font-medium text-danger-500">{memo.bear_case.headline}</div>
          <ul className="text-sm text-slate-300 mt-2 list-disc pl-5 space-y-1">
            {memo.bear_case.key_points.map((p, i) => <li key={i}>{p}</li>)}
          </ul>
        </div>
      </div>

      {sectorData?.bull_bear_analysis && (
        <BullBearAnalysisBlock analysis={sectorData.bull_bear_analysis} />
      )}

      <div className="grid md:grid-cols-2 gap-4">
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
        </div>
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
        </div>
      </div>

      <div className="card-tight">
        <div className="section-title mb-1">DCF Snapshot</div>
        {dcf && Object.keys(dcf).length > 0 ? (
          (() => {
            const current = Number(dcf.current_price) || 0;
            const base = Number(dcf.base_implied_price) || 0;
            const bull = Number(dcf.bull_implied_price) || 0;
            const bear = Number(dcf.bear_implied_price) || 0;
            // Prefer per-scenario upside fields when present (Wave 8L);
            // fall back to recomputing from current_price for older memos.
            const baseUp = dcf.base_upside !== undefined
              ? Number(dcf.base_upside)
              : current ? (base - current) / current : 0;
            const bullUp = dcf.bull_upside !== undefined
              ? Number(dcf.bull_upside)
              : current ? (bull - current) / current : 0;
            const bearUp = dcf.bear_upside !== undefined
              ? Number(dcf.bear_upside)
              : current ? (bear - current) / current : 0;
            const tone = (v: number) =>
              v > 0.005 ? "text-accent-500" : v < -0.005 ? "text-danger-500" : "text-slate-400";
            return (
              <div className="space-y-3 text-sm">
                <div className="flex items-baseline gap-2">
                  <span className="text-xs text-slate-500">Current</span>
                  <span className="font-mono text-base text-slate-100">
                    {fmtCurrency(current)}
                  </span>
                  <span className="text-[10px] text-slate-500">
                    Δ vs DCF below
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <div className="text-xs text-slate-500">Bear</div>
                    <div className="font-mono text-base">{fmtCurrency(bear)}</div>
                    <div className={`text-xs ${tone(bearUp)}`}>
                      {fmtPct(bearUp)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500">Base</div>
                    <div className="font-mono text-base">{fmtCurrency(base)}</div>
                    <div className={`text-xs ${tone(baseUp)}`}>
                      {fmtPct(baseUp)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500">Bull</div>
                    <div className="font-mono text-base">{fmtCurrency(bull)}</div>
                    <div className={`text-xs ${tone(bullUp)}`}>
                      {fmtPct(bullUp)}
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

      <div className="card-tight">
        <div className="section-title mb-1">Final Verdict</div>
        <div className="text-sm text-slate-200">{memo.final_verdict}</div>
      </div>

      <div className="text-[11px] text-slate-500 leading-snug">
        Sources: {memo.sources_used.slice(0, 8).join(" · ")}
        <br />
        {memo.disclaimer}
      </div>
    </div>
  );
}
