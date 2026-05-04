import React, { useEffect, useState } from "react";
import { api } from "@/api/client";
import TickerPicker from "@/components/TickerPicker";
import type { CompanyOut, CompsResult } from "@/types";
import { fmtMultiple, fmtPct, fmtCurrency } from "@/lib/format";

const FIELDS: Array<{ key: keyof CompsResult["target"]; label: string; format: (v: unknown) => string; pct?: boolean }> = [
  { key: "market_cap", label: "Mkt Cap", format: (v) => fmtCurrency(v as number, { compact: true }) },
  { key: "revenue_growth", label: "Rev Growth", format: (v) => fmtPct(v as number) },
  { key: "gross_margin", label: "Gross %", format: (v) => fmtPct(v as number) },
  { key: "operating_margin", label: "Op %", format: (v) => fmtPct(v as number) },
  { key: "ebitda_margin", label: "EBITDA %", format: (v) => fmtPct(v as number) },
  { key: "roic", label: "ROIC", format: (v) => fmtPct(v as number) },
  { key: "pe", label: "P/E", format: (v) => fmtMultiple(v as number) },
  { key: "ev_revenue", label: "EV/Rev", format: (v) => fmtMultiple(v as number) },
  { key: "ev_ebitda", label: "EV/EBITDA", format: (v) => fmtMultiple(v as number) },
  { key: "p_fcf", label: "P/FCF", format: (v) => fmtMultiple(v as number) },
  { key: "fcf_yield", label: "FCF Yield", format: (v) => fmtPct(v as number) },
];

export default function Comps() {
  const [universe, setUniverse] = useState<CompanyOut[]>([]);
  const [ticker, setTicker] = useState("NVDA");
  const [comps, setComps] = useState<CompsResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listStocks().then(setUniverse);
  }, []);

  useEffect(() => {
    if (!ticker) return;
    setError(null);
    api.comps(ticker).then(setComps).catch((e) => setError(String(e)));
  }, [ticker]);

  return (
    <div className="space-y-4">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Comps</h1>
          <p className="text-slate-400 text-sm mt-1">Peer-relative valuation, growth, margins, and quality.</p>
        </div>
        <TickerPicker
          value={ticker}
          onChange={setTicker}
          universe={universe}
          className="w-72"
        />
      </div>

      {error && <div className="card-tight border-danger-500/40 text-danger-500 text-sm">{error}</div>}

      {comps && (
        <div className="space-y-4">
          <div className="card">
            <div className="section-title mb-2">Interpretation</div>
            <p className="text-sm text-slate-200">{comps.interpretation}</p>
          </div>

          <div className="card overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-xs text-slate-500 border-b border-ink-700">
                <tr>
                  <th className="text-left py-2">Ticker</th>
                  <th className="text-left">Company</th>
                  {FIELDS.map((f) => (
                    <th key={f.key} className="text-right">
                      {f.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-accent-600/30 bg-accent-600/5">
                  <td className="py-2 font-mono text-accent-500">{comps.target.ticker}</td>
                  <td className="text-slate-200">{comps.target.company_name}</td>
                  {FIELDS.map((f) => (
                    <td key={f.key as string} className="text-right font-mono">
                      {f.format(comps.target[f.key])}
                    </td>
                  ))}
                </tr>
                {comps.peers.map((p) => (
                  <tr key={p.ticker} className="border-b border-ink-800">
                    <td className="py-2 font-mono">{p.ticker}</td>
                    <td className="text-slate-300">{p.company_name}</td>
                    {FIELDS.map((f) => (
                      <td key={f.key as string} className="text-right font-mono text-slate-300">
                        {f.format(p[f.key])}
                      </td>
                    ))}
                  </tr>
                ))}
                <tr className="border-t border-ink-700 bg-ink-800/40">
                  <td className="py-2 font-mono text-slate-400">MEDIAN</td>
                  <td className="text-slate-400">Peer median</td>
                  {FIELDS.map((f) => (
                    <td key={f.key as string} className="text-right font-mono text-slate-400">
                      {f.format(comps.median[f.key])}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>

          {Object.keys(comps.premium_discount).length > 0 && (
            <div className="card-tight">
              <div className="section-title mb-2">Target premium / discount vs peer median</div>
              <div className="grid sm:grid-cols-3 gap-2">
                {Object.entries(comps.premium_discount).map(([k, v]) => (
                  <div key={k} className="text-xs">
                    <span className="text-slate-400">{k}: </span>
                    <span className={v >= 0 ? "text-accent-500" : "text-danger-500"}>
                      {(v * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
