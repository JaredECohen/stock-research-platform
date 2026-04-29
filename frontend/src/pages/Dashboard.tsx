import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Briefcase, MessageCircle, Newspaper, Search, TrendingUp } from "lucide-react";
import { api } from "@/api/client";
import type { ScreenerRow, ProvidersStatusResponse } from "@/types";
import { ratingBadgeClass } from "@/lib/format";

const QUICK_PROMPTS = [
  "Analyze NVDA as a long-term investment.",
  "Compare MSFT and GOOGL from a portfolio manager's perspective.",
  "Find 5 high-quality stocks that could benefit from falling rates.",
  "Build a 10-stock portfolio for a soft landing with continued AI infrastructure spending.",
  "What sectors benefit if inflation stays sticky?",
];

export default function Dashboard() {
  const [topRows, setTopRows] = useState<ScreenerRow[]>([]);
  const [providers, setProviders] = useState<ProvidersStatusResponse | null>(null);

  useEffect(() => {
    api.screener({ limit: 8 }).then((r) => setTopRows(r.rows)).catch(() => {});
    api.providersStatus().then(setProviders).catch(() => {});
  }, []);

  return (
    <div className="space-y-6">
      <header>
        <div className="text-xs uppercase tracking-widest text-accent-500 font-semibold">Your AI Investment Committee</div>
        <h1 className="text-3xl font-semibold mt-1">MarketMosaic</h1>
        <p className="text-slate-400 max-w-2xl mt-2 leading-relaxed">
          A virtual investment research team. Specialist agents — sector analysts, earnings analysts, filings, valuation,
          macro, risk — collaborate under a Portfolio Manager orchestrator to produce structured stock memos, ranked
          ideas, and scenario-based model portfolios. Research and education only; not personalized financial advice.
        </p>
      </header>

      <section className="grid md:grid-cols-2 lg:grid-cols-4 gap-3">
        <Link to="/chat" className="card hover:border-accent-600 transition-colors">
          <MessageCircle className="text-accent-500 mb-2" size={18} />
          <div className="font-medium">Ask the PM</div>
          <div className="text-xs text-slate-400 mt-1">Chat with the orchestrator and watch the agent trace.</div>
        </Link>
        <Link to="/research" className="card hover:border-accent-600 transition-colors">
          <Newspaper className="text-accent-500 mb-2" size={18} />
          <div className="font-medium">Stock Research</div>
          <div className="text-xs text-slate-400 mt-1">Generate a full investment memo for any supported ticker.</div>
        </Link>
        <Link to="/screener" className="card hover:border-accent-600 transition-colors">
          <Search className="text-accent-500 mb-2" size={18} />
          <div className="font-medium">Screener</div>
          <div className="text-xs text-slate-400 mt-1">Agent-ranked ideas across themes — falling rates, AI infra, defense.</div>
        </Link>
        <Link to="/portfolio" className="card hover:border-accent-600 transition-colors">
          <Briefcase className="text-accent-500 mb-2" size={18} />
          <div className="font-medium">Portfolio Builder</div>
          <div className="text-xs text-slate-400 mt-1">Translate a market view into a diversified scenario portfolio.</div>
        </Link>
      </section>

      <section className="card">
        <div className="flex items-center justify-between">
          <div className="section-title">Demo prompts</div>
          <Link to="/chat" className="text-xs text-accent-500 inline-flex items-center gap-1">
            Open chat <ArrowRight size={12} />
          </Link>
        </div>
        <div className="grid md:grid-cols-2 gap-2 mt-3">
          {QUICK_PROMPTS.map((p) => (
            <Link
              key={p}
              to={`/chat?q=${encodeURIComponent(p)}`}
              className="card-tight hover:border-accent-600 transition-colors"
            >
              <div className="text-sm">{p}</div>
            </Link>
          ))}
        </div>
      </section>

      <section className="grid md:grid-cols-3 gap-4">
        <div className="card md:col-span-2">
          <div className="section-title mb-2 flex items-center justify-between">
            <span>Top-ranked ideas</span>
            <Link to="/screener" className="text-xs text-accent-500 inline-flex items-center gap-1">
              Full screener <ArrowRight size={12} />
            </Link>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-xs text-slate-500 border-b border-ink-700">
                <tr>
                  <th className="text-left font-medium py-2">#</th>
                  <th className="text-left font-medium">Ticker</th>
                  <th className="text-left font-medium">Sector</th>
                  <th className="text-right font-medium">PM Score</th>
                  <th className="text-left font-medium hidden lg:table-cell">Thesis</th>
                </tr>
              </thead>
              <tbody>
                {topRows.map((r) => (
                  <tr key={r.ticker} className="border-b border-ink-800 table-row-hover">
                    <td className="py-2 text-slate-500">{r.rank}</td>
                    <td className="font-mono">
                      <Link to={`/research?ticker=${r.ticker}`} className="text-accent-500 hover:underline">
                        {r.ticker}
                      </Link>
                    </td>
                    <td className="text-slate-300">{r.sector}</td>
                    <td className="text-right font-mono">{r.pm_score.toFixed(0)}</td>
                    <td className="text-slate-400 hidden lg:table-cell text-xs">{r.one_line_thesis}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <div className="section-title mb-2 flex items-center gap-2">
            <TrendingUp size={14} /> Mode & providers
          </div>
          {providers ? (
            <div className="space-y-2 text-sm">
              <div>
                Mode: <span className={`font-medium ${providers.mode === "demo" ? "text-warn-500" : "text-accent-500"}`}>{providers.mode}</span>
              </div>
              <div>
                LLM: <span className={providers.llm_configured ? "text-accent-500" : "text-slate-400"}>{providers.llm_configured ? "configured" : "demo agents only"}</span>
              </div>
              <div className="text-xs text-slate-400 mt-2">Providers configured:</div>
              <div className="flex flex-wrap gap-1.5 mt-1">
                {Object.values(providers.providers).map((p) => (
                  <span
                    key={p.name}
                    className={`badge ${p.configured ? "border-accent-600/40 text-accent-500" : "border-ink-700 text-slate-500"}`}
                  >
                    {p.name}
                  </span>
                ))}
              </div>
              {providers.missing_api_keys.length > 0 && (
                <div className="text-xs text-slate-500 mt-3">
                  Missing keys: {providers.missing_api_keys.join(", ")}
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-slate-400">Loading provider status…</div>
          )}
        </div>
      </section>

      <footer className="text-[11px] text-slate-500 leading-snug pt-2">
        MarketMosaic is for investment research and education only. It does not provide personalized financial,
        investment, legal, or tax advice. Model portfolios and stock analyses are illustrative and scenario-based.
      </footer>
    </div>
  );
}
