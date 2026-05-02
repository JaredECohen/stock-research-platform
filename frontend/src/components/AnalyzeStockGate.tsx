import React from "react";
import type { CompanyOut } from "@/types";

/**
 * Gate UI for `data_only` tier tickers (Wave 1B / Phase J).
 *
 * Backend's `/api/stocks/{t}/memo` returns 409 when a ticker is in the
 * `data_only` tier and the caller didn't pass `ondemand=true` — this
 * component renders the friendly "we haven't analyzed this stock yet"
 * affordance, with one button that retries with `ondemand=true`.
 *
 * After a successful first run the backend promotes the ticker to
 * `analyzed_on_demand`; subsequent visits skip the gate and use the
 * cached memo.
 */
interface Props {
  company: CompanyOut;
  loading: boolean;
  onAnalyze: () => void;
}

export default function AnalyzeStockGate({ company, loading, onAnalyze }: Props) {
  return (
    <div className="card max-w-2xl">
      <div className="flex items-center gap-3 mb-3">
        <span className="badge bg-slate-500/15 border-slate-500/40 text-slate-300 uppercase tracking-widest text-[10px]">
          Data only
        </span>
        <span className="text-xs text-slate-500">{company.sector}</span>
      </div>
      <h2 className="text-xl font-semibold mb-1">
        {company.ticker} · <span className="text-slate-300 font-normal">{company.company_name}</span>
      </h2>
      <p className="text-sm text-slate-300 mt-3 leading-relaxed">
        MarketMosaic hasn't analyzed this stock yet. Generating a memo runs every
        specialist agent (sector, earnings, filing, valuation, comps, macro,
        technical, risk) plus the cross-family critic — typically 20-40 seconds and
        a few cents in API spend.
      </p>
      <p className="text-xs text-slate-500 mt-2">
        After the first analysis, the memo is cached and re-rendered instantly on
        future visits. Use "Refresh memo" to force a fresh run.
      </p>
      <button
        type="button"
        disabled={loading}
        className="btn-primary mt-4"
        onClick={onAnalyze}
      >
        {loading ? "Analyzing… (this can take ~30s)" : "Analyze this stock"}
      </button>
      <p className="text-[11px] text-slate-500 mt-4 leading-snug">
        Research / education only. Not personalized financial advice.
      </p>
    </div>
  );
}


/** Tier badge for ticker list items. Stable string colors per tier. */
export function TierBadge({ tier }: { tier?: string }) {
  const t = tier || "data_only";
  const cfg: Record<string, { label: string; cls: string }> = {
    auto_analysis: {
      label: "auto",
      cls: "bg-accent-700/15 border-accent-600/40 text-accent-500",
    },
    analyzed_on_demand: {
      label: "cached",
      cls: "bg-warn-500/10 border-warn-500/30 text-warn-500",
    },
    data_only: {
      label: "data-only",
      cls: "bg-slate-500/15 border-slate-500/40 text-slate-400",
    },
  };
  const c = cfg[t] || cfg.data_only;
  return (
    <span className={`badge text-[10px] uppercase tracking-widest ${c.cls}`}>
      {c.label}
    </span>
  );
}
