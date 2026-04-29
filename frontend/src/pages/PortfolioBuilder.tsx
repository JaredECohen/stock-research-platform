import React, { useState } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { api } from "@/api/client";
import type { ModelPortfolio } from "@/types";
import { fmtPct } from "@/lib/format";

const COLORS = ["#52E0C4", "#F2B045", "#94A3B8", "#EF6F6F", "#7FBDFF", "#B985FF", "#FFD27A", "#7AE0A1"];

export default function PortfolioBuilder() {
  const [marketView, setMarketView] = useState(
    "Soft landing with continued AI infrastructure spending",
  );
  const [riskLevel, setRiskLevel] = useState<"conservative" | "balanced" | "aggressive">("balanced");
  const [numHoldings, setNumHoldings] = useState(10);
  const [maxPos, setMaxPos] = useState(0.15);
  const [excludedSectors, setExcludedSectors] = useState("");
  const [excludedTickers, setExcludedTickers] = useState("");
  const [portfolio, setPortfolio] = useState<ModelPortfolio | null>(null);
  const [loading, setLoading] = useState(false);

  const build = () => {
    setLoading(true);
    api
      .buildPortfolio({
        market_view: marketView,
        risk_level: riskLevel,
        num_holdings: numHoldings,
        max_position_size: maxPos,
        excluded_sectors: excludedSectors.split(",").map((s) => s.trim()).filter(Boolean),
        excluded_tickers: excludedTickers.split(",").map((s) => s.trim().toUpperCase()).filter(Boolean),
      })
      .then(setPortfolio)
      .finally(() => setLoading(false));
  };

  const sectorData = portfolio
    ? Object.entries(portfolio.sector_allocation).map(([name, w]) => ({ name, value: w }))
    : [];

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Portfolio Builder</h1>
        <p className="text-slate-400 text-sm mt-1">
          Translate a market view into a diversified scenario portfolio. Educational, not advice.
        </p>
      </div>

      <div className="grid lg:grid-cols-[360px_1fr] gap-4">
        <div className="card">
          <div className="space-y-3">
            <div>
              <div className="text-xs text-slate-400 mb-1">Market view</div>
              <textarea
                className="input min-h-[80px]"
                value={marketView}
                onChange={(e) => setMarketView(e.target.value)}
              />
            </div>
            <div>
              <div className="text-xs text-slate-400 mb-1">Risk level</div>
              <div className="flex gap-2">
                {(["conservative", "balanced", "aggressive"] as const).map((r) => (
                  <button
                    key={r}
                    onClick={() => setRiskLevel(r)}
                    className={`badge ${riskLevel === r ? "border-accent-600 text-accent-500 bg-accent-600/15" : "border-ink-700 text-slate-300"}`}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <label className="block">
                <div className="text-xs text-slate-400 mb-1"># holdings</div>
                <input type="number" className="input w-full" min={3} max={25} value={numHoldings} onChange={(e) => setNumHoldings(parseInt(e.target.value) || 10)} />
              </label>
              <label className="block">
                <div className="text-xs text-slate-400 mb-1">Max position</div>
                <input type="number" step="0.01" className="input w-full" value={maxPos} onChange={(e) => setMaxPos(parseFloat(e.target.value) || 0.15)} />
              </label>
            </div>
            <label className="block">
              <div className="text-xs text-slate-400 mb-1">Excluded sectors (comma sep.)</div>
              <input className="input w-full" value={excludedSectors} onChange={(e) => setExcludedSectors(e.target.value)} />
            </label>
            <label className="block">
              <div className="text-xs text-slate-400 mb-1">Excluded tickers (comma sep.)</div>
              <input className="input w-full" value={excludedTickers} onChange={(e) => setExcludedTickers(e.target.value)} />
            </label>
            <button className="btn-primary w-full" onClick={build} disabled={loading}>
              {loading ? "Building…" : "Build portfolio"}
            </button>
          </div>
        </div>

        <div className="space-y-4">
          {portfolio && (
            <>
              <div className="card">
                <div className="section-title mb-2">{portfolio.name}</div>
                <div className="text-sm text-slate-300">{portfolio.market_view}</div>
                <div className="text-xs text-slate-500 mt-1">
                  Risk level: {portfolio.risk_level} · Expected vol proxy: {fmtPct(portfolio.expected_volatility)}
                </div>
              </div>

              <div className="grid lg:grid-cols-[1fr_280px] gap-4">
                <div className="card overflow-x-auto">
                  <div className="section-title mb-2">Holdings</div>
                  <table className="w-full text-sm">
                    <thead className="text-xs text-slate-500 border-b border-ink-700">
                      <tr>
                        <th className="text-left py-2">Ticker</th>
                        <th className="text-left">Sector</th>
                        <th className="text-right">Weight</th>
                        <th className="text-right">PM Conv.</th>
                        <th className="text-left">Rationale</th>
                      </tr>
                    </thead>
                    <tbody>
                      {portfolio.holdings.map((h) => (
                        <tr key={h.ticker} className="border-b border-ink-800">
                          <td className="py-2 font-mono text-accent-500">{h.ticker}</td>
                          <td className="text-slate-300">{h.sector}</td>
                          <td className="text-right font-mono">{(h.weight * 100).toFixed(1)}%</td>
                          <td className="text-right font-mono text-slate-300">{h.pm_conviction.toFixed(0)}</td>
                          <td className="text-slate-400 text-xs">{h.rationale}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="card">
                  <div className="section-title mb-2">Sector allocation</div>
                  <div style={{ height: 240 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={sectorData} dataKey="value" nameKey="name" outerRadius={80} stroke="#0E1525" strokeWidth={2}>
                          {sectorData.map((_, i) => (
                            <Cell key={i} fill={COLORS[i % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{ background: "#0E1525", border: "1px solid #243056" }}
                          formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <ul className="text-xs space-y-1 mt-2">
                    {sectorData.map((s, i) => (
                      <li key={s.name} className="flex justify-between">
                        <span className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full inline-block" style={{ background: COLORS[i % COLORS.length] }} />
                          {s.name}
                        </span>
                        <span className="font-mono">{(s.value * 100).toFixed(1)}%</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="grid md:grid-cols-3 gap-3">
                <div className="card-tight">
                  <div className="section-title mb-2">Top thesis drivers</div>
                  <ul className="text-sm text-slate-300 list-disc pl-5 space-y-1">
                    {portfolio.top_thesis_drivers.map((t, i) => <li key={i}>{t}</li>)}
                  </ul>
                </div>
                <div className="card-tight">
                  <div className="section-title mb-2">Risk notes</div>
                  <ul className="text-sm text-slate-300 list-disc pl-5 space-y-1">
                    {portfolio.risk_notes.map((t, i) => <li key={i}>{t}</li>)}
                  </ul>
                </div>
                <div className="card-tight">
                  <div className="section-title mb-2">Watch items</div>
                  <ul className="text-sm text-slate-300 list-disc pl-5 space-y-1">
                    {portfolio.watch_items.map((t, i) => <li key={i}>{t}</li>)}
                  </ul>
                </div>
              </div>

              <div className="card-tight">
                <div className="section-title mb-2">What could invalidate this portfolio</div>
                <ul className="text-sm text-slate-300 list-disc pl-5 space-y-1">
                  {portfolio.what_could_invalidate.map((t, i) => <li key={i}>{t}</li>)}
                </ul>
              </div>

              <div className="text-[11px] text-slate-500">{portfolio.disclaimer}</div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
