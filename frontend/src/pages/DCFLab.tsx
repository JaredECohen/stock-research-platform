import React, { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "@/api/client";
import type { CompanyOut, DCFAssumptions, DCFResult, DCFSensitivity } from "@/types";
import { fmtCurrency, fmtPct } from "@/lib/format";

function NumInput(props: { label: string; value: number; onChange: (v: number) => void; step?: number; pct?: boolean; suffix?: string }) {
  const display = props.pct ? (props.value * 100).toFixed(2) : props.value.toFixed(props.step && props.step >= 1 ? 1 : 4);
  return (
    <label className="block">
      <div className="text-xs text-slate-400 mb-1">{props.label}</div>
      <div className="flex items-center gap-1">
        <input
          type="number"
          className="input w-full font-mono"
          step={props.step ?? (props.pct ? 0.1 : 0.001)}
          value={display}
          onChange={(e) => {
            const raw = parseFloat(e.target.value);
            if (Number.isNaN(raw)) return;
            props.onChange(props.pct ? raw / 100 : raw);
          }}
        />
        {props.suffix && <span className="text-xs text-slate-500">{props.suffix}</span>}
      </div>
    </label>
  );
}

function SensitivityTable({ s }: { s: DCFSensitivity }) {
  const cols = s.cols;
  const rows = s.rows;
  const cellMap = new Map<string, number>();
  s.cells.forEach((c) => cellMap.set(`${c.row_label}|${c.col_label}`, c.value));

  return (
    <div className="card-tight overflow-x-auto">
      <div className="section-title mb-2">{s.name}</div>
      <table className="w-full text-xs font-mono">
        <thead>
          <tr>
            <th className="text-left text-slate-500 p-1">{s.row_axis} \ {s.col_axis}</th>
            {cols.map((c) => (
              <th key={c} className="text-right p-1 text-slate-400">
                {c < 1 ? `${(c * 100).toFixed(2)}%` : c.toFixed(1)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const rowLabel = r < 1 ? `${(r * 100).toFixed(2)}%` : r.toFixed(1);
            return (
              <tr key={String(r)} className="border-t border-ink-800">
                <td className="text-slate-400 p-1">{rowLabel}</td>
                {cols.map((c) => {
                  const colLabel = c < 1 ? `${(c * 100).toFixed(2)}%` : c.toFixed(1);
                  const v = cellMap.get(`${rowLabel}|${colLabel}`) ?? 0;
                  return (
                    <td key={`${r}-${c}`} className="text-right p-1">
                      ${v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function DCFLab() {
  const [universe, setUniverse] = useState<CompanyOut[]>([]);
  const [ticker, setTicker] = useState("MSFT");
  const [a, setA] = useState<DCFAssumptions | null>(null);
  const [result, setResult] = useState<DCFResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.listStocks().then(setUniverse);
  }, []);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    api.dcfDefaults(ticker).then((d) => {
      setA(d);
      api.runDCF(ticker, d).then(setResult).finally(() => setLoading(false));
    });
  }, [ticker]);

  const update = (patch: Partial<DCFAssumptions>) => {
    if (!a) return;
    const next = { ...a, ...patch };
    setA(next);
  };
  const recompute = () => {
    if (!a) return;
    setLoading(true);
    api.runDCF(ticker, a).then(setResult).finally(() => setLoading(false));
  };

  const projectionData = result?.base.projections.map((p) => ({
    year: `Y${p.year}`,
    revenue: p.revenue / 1e9,
    ebit: p.ebit / 1e9,
    fcff: p.fcff / 1e9,
  })) ?? [];

  const scenarioData = result
    ? [
        { name: "Bear", price: result.bear.implied_share_price },
        { name: "Base", price: result.base.implied_share_price },
        { name: "Bull", price: result.bull.implied_share_price },
        { name: "Current", price: result.current_price },
      ]
    : [];

  return (
    <div className="space-y-4">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold">DCF Lab</h1>
          <p className="text-slate-400 text-sm mt-1">Edit assumptions and re-run base/bull/bear scenarios.</p>
        </div>
        <select className="input" value={ticker} onChange={(e) => setTicker(e.target.value)}>
          {universe.map((c) => (
            <option key={c.ticker} value={c.ticker}>
              {c.ticker} — {c.company_name}
            </option>
          ))}
        </select>
      </div>

      {a && (
        <div className="grid lg:grid-cols-[360px_1fr] gap-4">
          <div className="card">
            <div className="section-title mb-2">Base Assumptions</div>
            <div className="grid grid-cols-2 gap-3">
              <NumInput label="WACC" value={a.wacc} onChange={(v) => update({ wacc: v })} pct />
              <NumInput label="Terminal Growth" value={a.terminal_growth} onChange={(v) => update({ terminal_growth: v })} pct />
              <NumInput label="Tax Rate" value={a.tax_rate} onChange={(v) => update({ tax_rate: v })} pct />
              <NumInput label="Exit EBITDA Multiple" value={a.exit_ebitda_multiple} onChange={(v) => update({ exit_ebitda_multiple: v })} step={0.5} suffix="x" />
              <NumInput label="D&A % Revenue" value={a.da_pct_revenue} onChange={(v) => update({ da_pct_revenue: v })} pct />
              <NumInput label="Capex % Revenue" value={a.capex_pct_revenue} onChange={(v) => update({ capex_pct_revenue: v })} pct />
              <NumInput label="NWC % Revenue" value={a.nwc_pct_revenue} onChange={(v) => update({ nwc_pct_revenue: v })} pct />
            </div>
            <div className="section-title mt-4 mb-2">Year-by-year</div>
            <div className="grid grid-cols-2 gap-3">
              {a.revenue_growth.map((g, i) => (
                <NumInput
                  key={`g${i}`}
                  label={`Y${i + 1} Rev Growth`}
                  value={g}
                  onChange={(v) => {
                    const next = [...a.revenue_growth];
                    next[i] = v;
                    update({ revenue_growth: next });
                  }}
                  pct
                />
              ))}
              {a.operating_margin.map((m, i) => (
                <NumInput
                  key={`m${i}`}
                  label={`Y${i + 1} Op Margin`}
                  value={m}
                  onChange={(v) => {
                    const next = [...a.operating_margin];
                    next[i] = v;
                    update({ operating_margin: next });
                  }}
                  pct
                />
              ))}
            </div>
            <button className="btn-primary w-full mt-4" onClick={recompute} disabled={loading}>
              {loading ? "Re-computing…" : "Re-run scenarios"}
            </button>
          </div>

          <div className="space-y-4">
            {result && (
              <>
                <div className="card">
                  <div className="section-title mb-2">Scenario summary</div>
                  <p className="text-sm text-slate-300">{result.summary}</p>
                  <div className="grid grid-cols-3 gap-3 mt-3">
                    {(["bear", "base", "bull"] as const).map((k) => {
                      const s = result[k];
                      return (
                        <div key={k} className="card-tight">
                          <div className="text-xs uppercase tracking-widest text-slate-500">{s.label}</div>
                          <div className="text-xl font-mono mt-1">{fmtCurrency(s.implied_share_price)}</div>
                          <div className={`text-xs ${s.upside_pct >= 0 ? "text-accent-500" : "text-danger-500"}`}>
                            {fmtPct(s.upside_pct)} vs current
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="card">
                  <div className="section-title mb-2">Implied share price by scenario</div>
                  <div style={{ height: 220 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={scenarioData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1A2440" />
                        <XAxis dataKey="name" stroke="#94A3B8" />
                        <YAxis stroke="#94A3B8" />
                        <Tooltip contentStyle={{ background: "#0E1525", border: "1px solid #243056", color: "#E2E8F0" }} />
                        <Bar dataKey="price">
                          {scenarioData.map((d, i) => (
                            <Cell
                              key={i}
                              fill={
                                d.name === "Bull"
                                  ? "#52E0C4"
                                  : d.name === "Bear"
                                    ? "#EF6F6F"
                                    : d.name === "Base"
                                      ? "#2BC4A4"
                                      : "#94A3B8"
                              }
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="card">
                  <div className="section-title mb-2">Projection ($B)</div>
                  <div style={{ height: 240 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={projectionData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1A2440" />
                        <XAxis dataKey="year" stroke="#94A3B8" />
                        <YAxis stroke="#94A3B8" />
                        <Tooltip contentStyle={{ background: "#0E1525", border: "1px solid #243056", color: "#E2E8F0" }} />
                        <Line type="monotone" dataKey="revenue" stroke="#52E0C4" strokeWidth={2} dot={false} name="Revenue" />
                        <Line type="monotone" dataKey="ebit" stroke="#F2B045" strokeWidth={2} dot={false} name="EBIT" />
                        <Line type="monotone" dataKey="fcff" stroke="#94A3B8" strokeWidth={2} dot={false} name="FCFF" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="grid lg:grid-cols-2 gap-3">
                  {result.sensitivities.map((s) => (
                    <SensitivityTable key={s.name} s={s} />
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
