import React from "react";
import type { AgentFinding, StockMemoOut } from "@/types";
import { fmtCurrency, fmtPct, ratingBadgeClass } from "@/lib/format";
import CrossSectorChips from "./CrossSectorChips";
import MacroRegimeBanner from "./MacroRegimeBanner";

function FindingBlock({
  title,
  body,
  footer,
}: {
  title: string;
  body: AgentFinding | { headline: string; summary: string; key_points?: string[] };
  footer?: React.ReactNode;
}) {
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
        <div className="flex items-start justify-between">
          <div>
            <div className="text-xs uppercase tracking-widest text-slate-500">{memo.sector}</div>
            <div className="text-2xl font-semibold mt-1">
              {memo.ticker} · <span className="text-slate-300 font-normal">{memo.company_name}</span>
            </div>
            <div className="text-sm text-slate-300 mt-2 max-w-3xl">{memo.one_sentence_thesis}</div>
          </div>
          <div className="text-right space-y-2">
            <span className={ratingBadgeClass(memo.rating_label)}>{memo.rating_label}</span>
            <div className="text-xs text-slate-400">Confidence: <span className="text-slate-200 font-medium">{Math.round(memo.confidence_score)}/100</span></div>
            <div className="text-[10px] uppercase tracking-widest text-slate-600">mode: {memo.generation_mode}</div>
          </div>
        </div>
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
        <FindingBlock title="Earnings Analyst" body={memo.earnings_agent_view} />
        <FindingBlock title="Filing Analyst" body={memo.filing_agent_view} />
        <FindingBlock title="Valuation Analyst" body={memo.valuation_agent_view} />
        <FindingBlock title="Comps Analyst" body={memo.comps_agent_view} />
        <FindingBlock title="Macro Analyst" body={memo.macro_sensitivity} />
      </div>

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
          <div className="grid md:grid-cols-3 gap-3 text-sm">
            <div>
              <div className="text-xs text-slate-500">Base Implied</div>
              <div className="font-mono text-base">{fmtCurrency(Number(dcf.base_implied_price))}</div>
              <div className="text-xs text-slate-400">{fmtPct(Number(dcf.base_upside))}</div>
            </div>
            <div>
              <div className="text-xs text-slate-500">Bull / Bear</div>
              <div className="font-mono text-base">
                {fmtCurrency(Number(dcf.bull_implied_price))} <span className="text-slate-500">/</span>{" "}
                {fmtCurrency(Number(dcf.bear_implied_price))}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-500">WACC / Term Growth</div>
              <div className="font-mono text-base">
                {fmtPct(Number(dcf.wacc), 2)} / {fmtPct(Number(dcf.terminal_growth), 1)}
              </div>
            </div>
          </div>
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
