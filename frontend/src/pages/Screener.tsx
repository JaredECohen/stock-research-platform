import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { RefreshCw } from "lucide-react";
import { api } from "@/api/client";
import type { ScreenerResult } from "@/types";

const THEMES: Array<{ key: string; label: string }> = [
  { key: "", label: "All" },
  { key: "ai_infrastructure", label: "AI Infrastructure" },
  { key: "falling_rates", label: "Falling Rates" },
  { key: "sticky_inflation", label: "Sticky Inflation" },
  { key: "recession_defense", label: "Recession Defense" },
  { key: "high_quality_compounders", label: "High Quality Compounders" },
  { key: "margin_expansion", label: "Margin Expansion" },
  { key: "reasonable_valuation_growth", label: "Reasonable Valuation Growth" },
];

export default function Screener() {
  const [theme, setTheme] = useState<string>("");
  const [sector, setSector] = useState<string>("");
  const [search, setSearch] = useState<string>("");
  const [data, setData] = useState<ScreenerResult | null>(null);
  const [loading, setLoading] = useState(false);

  const load = () => {
    setLoading(true);
    api
      .screener({ theme: theme || undefined, sector: sector || undefined, limit: 100 })
      .then(setData)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme, sector]);

  const sectors = useMemo(() => {
    if (!data) return [];
    return Array.from(new Set(data.rows.map((r) => r.sector))).sort();
  }, [data]);

  const filtered = useMemo(() => {
    if (!data) return [];
    if (!search) return data.rows;
    const q = search.toLowerCase();
    return data.rows.filter(
      (r) => r.ticker.toLowerCase().includes(q) || r.company_name.toLowerCase().includes(q),
    );
  }, [data, search]);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Screener</h1>
        <p className="text-slate-400 text-sm mt-1">Agent-ranked ideas across themes. Click a row to open the memo.</p>
      </div>

      <div className="card-tight flex flex-wrap items-center gap-2">
        <div className="text-xs text-slate-500 mr-2">Theme:</div>
        {THEMES.map((t) => (
          <button
            key={t.key}
            onClick={() => setTheme(t.key)}
            className={`badge ${theme === t.key ? "border-accent-600 text-accent-500 bg-accent-600/15" : "border-ink-700 text-slate-300"}`}
          >
            {t.label}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2">
          <input
            className="input w-44"
            placeholder="Search ticker or name…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select className="input" value={sector} onChange={(e) => setSector(e.target.value)}>
            <option value="">All sectors</option>
            {sectors.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <button className="btn-ghost" onClick={load}>
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} /> Refresh agent scores
          </button>
        </div>
      </div>

      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-xs text-slate-500 border-b border-ink-700">
            <tr>
              <th className="text-left py-2">#</th>
              <th className="text-left">Ticker</th>
              <th className="text-left">Company</th>
              <th className="text-left">Sector</th>
              <th className="text-right">PM</th>
              <th className="text-right">Quality</th>
              <th className="text-right">Growth</th>
              <th className="text-right">Valuation</th>
              <th className="text-right">EarnMom</th>
              <th className="text-right">Risk</th>
              <th className="text-right">MacroFit</th>
              <th className="text-left">Thesis</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr key={r.ticker} className="border-b border-ink-800 table-row-hover">
                <td className="py-2 text-slate-500">{r.rank}</td>
                <td className="font-mono">
                  <Link to={`/research?ticker=${r.ticker}`} className="text-accent-500 hover:underline">
                    {r.ticker}
                  </Link>
                </td>
                <td className="text-slate-300">{r.company_name}</td>
                <td className="text-slate-400">{r.sector}</td>
                <td className="text-right font-mono">{r.pm_score.toFixed(0)}</td>
                <td className="text-right font-mono text-slate-300">{r.quality.toFixed(0)}</td>
                <td className="text-right font-mono text-slate-300">{r.growth.toFixed(0)}</td>
                <td className="text-right font-mono text-slate-300">{r.valuation.toFixed(0)}</td>
                <td className="text-right font-mono text-slate-300">{r.earnings_momentum.toFixed(0)}</td>
                <td className="text-right font-mono text-slate-300">{r.risk.toFixed(0)}</td>
                <td className="text-right font-mono text-slate-300">{r.macro_fit.toFixed(0)}</td>
                <td className="text-slate-400 text-xs max-w-[220px] truncate">{r.one_line_thesis}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
