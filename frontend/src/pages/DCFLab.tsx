import React, { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "@/api/client";
import TickerPicker from "@/components/TickerPicker";
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

type Source = "saved" | "default";

type AssumptionChange = {
  field: string;
  from: unknown;
  to: unknown;
  rationale: string;
};

function ConsensusDiffBanner({
  assumptions,
  consensus,
  assumptionChanges,
}: {
  assumptions: DCFAssumptions;
  consensus: {
    consensus_revenue_growth: number[] | null;
    trailing_op_margin: number | null;
    has_consensus: boolean;
  };
  assumptionChanges?: AssumptionChange[];
}) {
  const fmtPctOne = (v: number) => `${(v * 100).toFixed(1)}%`;
  const consensusGrowth = consensus.consensus_revenue_growth;
  const modelGrowth = assumptions.revenue_growth;
  const trailingMargin = consensus.trailing_op_margin;
  const modelMargin = assumptions.operating_margin;

  // Per-year growth rows — only render if we have consensus data.
  const growthRows = consensusGrowth && modelGrowth.length === consensusGrowth.length
    ? modelGrowth.map((m, i) => {
        const c = consensusGrowth[i];
        const delta = m - c;
        const tone =
          Math.abs(delta) < 0.005
            ? "text-slate-400"
            : delta > 0
            ? "text-accent-500"
            : "text-warn-500";
        return { year: i + 1, model: m, consensus: c, delta, tone };
      })
    : null;

  // Margin row — model is held flat by default, so the diff is "model
  // vs trailing 3-yr avg" (which IS the default starting point).
  const marginRow = trailingMargin !== null
    ? {
        model: modelMargin[0] ?? 0,
        baseline: trailingMargin,
        delta: (modelMargin[0] ?? 0) - trailingMargin,
      }
    : null;

  // Map assumption-change rationale to fields so we can hover-reveal them.
  const rationaleByField: Record<string, string> = {};
  for (const c of assumptionChanges || []) {
    if (c.rationale) rationaleByField[c.field] = c.rationale;
  }

  const hasConsensus = consensus.has_consensus;

  return (
    <div className="card-tight space-y-3 !p-4 border-accent-600/25 bg-accent-600/[0.04]">
      <div className="flex items-center justify-between">
        <div className="section-title">Model vs analyst consensus</div>
        <div className="text-[10px] uppercase tracking-widest text-slate-500">
          {hasConsensus ? "consensus available" : "no consensus on file"}
        </div>
      </div>
      {!hasConsensus && (
        <div className="text-xs text-slate-400">
          The data service didn't return analyst estimates for this ticker.
          The model defaults to historical-trend revenue growth and the
          trailing 3-yr operating margin baseline shown below.
        </div>
      )}

      {growthRows && (
        <div>
          <div className="text-xs uppercase tracking-widest text-slate-500 mb-1">
            Revenue growth (5y)
          </div>
          <div className="grid grid-cols-5 gap-2">
            {growthRows.map((row) => (
              <div
                key={row.year}
                className="rounded border border-ink-700 bg-ink-800/40 p-2 text-xs"
                title={
                  rationaleByField[`revenue_growth[${row.year - 1}]`]
                    ? `Rationale: ${rationaleByField[`revenue_growth[${row.year - 1}]`]}`
                    : ""
                }
              >
                <div className="text-[10px] text-slate-500">Y{row.year}</div>
                <div className="font-mono text-slate-200">
                  {fmtPctOne(row.model)}
                </div>
                <div className="text-[10px] text-slate-500">
                  Street {fmtPctOne(row.consensus)}
                </div>
                <div className={`text-[10px] ${row.tone} font-mono`}>
                  Δ {row.delta >= 0 ? "+" : ""}
                  {(row.delta * 100).toFixed(1)}pp
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {marginRow && (
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <span className="text-slate-500 uppercase tracking-widest text-[10px]">
            Op margin (Y1)
          </span>
          <span className="font-mono text-slate-200">
            {fmtPctOne(marginRow.model)}
          </span>
          <span className="text-slate-500">
            vs trailing 3-yr avg{" "}
            <span className="font-mono text-slate-300">
              {fmtPctOne(marginRow.baseline)}
            </span>
          </span>
          <span
            className={
              Math.abs(marginRow.delta) < 0.001
                ? "text-slate-500"
                : marginRow.delta > 0
                ? "text-accent-500"
                : "text-warn-500"
            }
          >
            Δ{" "}
            {marginRow.delta >= 0 ? "+" : ""}
            {(marginRow.delta * 100).toFixed(2)}pp
          </span>
          {marginRow.delta !== 0 && (
            <span
              className="text-slate-400"
              title={
                rationaleByField["operating_margin[0]"] ||
                "Default behavior is flat at trailing 3-yr avg; deviation came from the LLM updater or your manual edit."
              }
            >
              {rationaleByField["operating_margin[0]"]
                ? "(rationale on hover)"
                : "(deviation from default)"}
            </span>
          )}
        </div>
      )}

      {(assumptionChanges?.length ?? 0) > 0 && (
        <div className="text-[11px] text-slate-400 leading-snug pt-2 border-t border-ink-700">
          <strong className="text-slate-200">
            {assumptionChanges!.length} field(s) diverge
          </strong>{" "}
          from the prior version's defaults. The LLM updater (Wave 5A) only
          moves a field when it can articulate a rationale; hover the
          changed cell above to read it.
        </div>
      )}
    </div>
  );
}

type SavedMeta = {
  version: number;
  trigger: string;
  generated_at?: string;
  assumption_changes?: Array<{
    field: string;
    from: unknown;
    to: unknown;
    rationale: string;
  }>;
};

export default function DCFLab() {
  const [universe, setUniverse] = useState<CompanyOut[]>([]);
  const [ticker, setTicker] = useState("MSFT");
  const [a, setA] = useState<DCFAssumptions | null>(null);
  const [baseline, setBaseline] = useState<DCFAssumptions | null>(null);
  const [source, setSource] = useState<Source>("saved");
  const [savedMeta, setSavedMeta] = useState<SavedMeta | null>(null);
  const [savedAvailable, setSavedAvailable] = useState<boolean>(false);
  const [result, setResult] = useState<DCFResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [consensus, setConsensus] = useState<{
    consensus_revenue_growth: number[] | null;
    trailing_op_margin: number | null;
    has_consensus: boolean;
  } | null>(null);

  useEffect(() => {
    api.listStocks().then(setUniverse);
  }, []);

  // Wave 8R — pull the analyst-consensus baseline so the lab can show
  // where current assumptions diverge.
  useEffect(() => {
    if (!ticker) return;
    let cancelled = false;
    setConsensus(null);
    api
      .dcfConsensus(ticker)
      .then((c) => {
        if (!cancelled) setConsensus(c);
      })
      .catch(() => {
        if (!cancelled) setConsensus(null);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  // Load assumptions from the chosen source whenever ticker or source changes.
  useEffect(() => {
    if (!ticker) return;
    let cancelled = false;
    setLoading(true);
    setResult(null);
    (async () => {
      let assumptions: DCFAssumptions | null = null;
      if (source === "saved") {
        try {
          const saved = await api.dcfSaved(ticker);
          if (cancelled) return;
          if (saved.has_saved && saved.assumptions) {
            assumptions = saved.assumptions;
            setSavedAvailable(true);
            setSavedMeta({
              version: saved.version!,
              trigger: saved.trigger!,
              generated_at: saved.generated_at,
              assumption_changes: saved.assumption_changes ?? [],
            });
          } else {
            setSavedAvailable(false);
            setSavedMeta(null);
          }
        } catch {
          setSavedAvailable(false);
          setSavedMeta(null);
        }
      }
      if (!assumptions) {
        // Fall through to engine defaults whenever saved isn't selected
        // OR isn't available yet for this ticker.
        assumptions = await api.dcfDefaults(ticker);
      }
      if (cancelled) return;
      setA(assumptions);
      setBaseline(assumptions);
      try {
        const r = await api.runDCF(ticker, assumptions);
        if (!cancelled) setResult(r);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ticker, source]);

  const update = (patch: Partial<DCFAssumptions>) => {
    if (!a) return;
    setA({ ...a, ...patch });
  };
  const recompute = () => {
    if (!a) return;
    setLoading(true);
    api.runDCF(ticker, a).then(setResult).finally(() => setLoading(false));
  };
  const resetToBaseline = () => {
    if (!baseline) return;
    setA(baseline);
  };
  const isDirty = a !== null && baseline !== null
    && JSON.stringify(a) !== JSON.stringify(baseline);

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
      <div className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-semibold">DCF Lab</h1>
          <p className="text-slate-400 text-sm mt-1">
            Edit assumptions and re-run base/bull/bear scenarios.{" "}
            <span className="text-slate-500">
              Edits in this lab don't persist — the saved DCF stays untouched.
            </span>
          </p>
        </div>
        <div className="flex items-end gap-3">
          <div>
            <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">
              Assumption source
            </div>
            <div className="inline-flex rounded-md border border-ink-700 overflow-hidden text-sm">
              <button
                type="button"
                onClick={() => setSource("saved")}
                className={`px-3 py-1.5 ${
                  source === "saved"
                    ? "bg-accent-600/20 text-accent-100"
                    : "bg-ink-800 text-slate-300 hover:bg-ink-700"
                }`}
                title="Load the latest DCF version persisted by the platform"
              >
                Saved DCF{savedAvailable ? "" : " (none)"}
              </button>
              <button
                type="button"
                onClick={() => setSource("default")}
                className={`px-3 py-1.5 ${
                  source === "default"
                    ? "bg-accent-600/20 text-accent-100"
                    : "bg-ink-800 text-slate-300 hover:bg-ink-700"
                }`}
                title="Engine-derived defaults (analyst-consensus growth where available)"
              >
                Defaults
              </button>
            </div>
          </div>
          <TickerPicker
            value={ticker}
            onChange={setTicker}
            universe={universe}
            className="w-72"
          />
        </div>
      </div>

      {source === "saved" && savedMeta && (
        <div className="card-tight border-accent-600/30 bg-accent-600/5 text-xs flex flex-wrap items-center gap-3">
          <span className="text-slate-400">
            Loaded saved <strong className="text-slate-100">v{savedMeta.version}</strong>{" "}
            ({savedMeta.trigger.replace("_", " ")})
            {savedMeta.generated_at
              ? ` · ${new Date(savedMeta.generated_at).toLocaleString()}`
              : ""}
          </span>
          {savedMeta.assumption_changes && savedMeta.assumption_changes.length > 0 && (
            <span className="text-slate-500">
              {savedMeta.assumption_changes.length} field(s) changed vs. prior version
            </span>
          )}
        </div>
      )}
      {source === "saved" && !savedAvailable && (
        <div className="card-tight border-warn-500/30 bg-warn-500/5 text-xs text-warn-500">
          No saved DCF for {ticker} yet — falling back to engine defaults.
          Run a memo from the Research page to seed v1.
        </div>
      )}

      {a && consensus && (
        <ConsensusDiffBanner
          assumptions={a}
          consensus={consensus}
          assumptionChanges={savedMeta?.assumption_changes}
        />
      )}

      {a && (
        <div className="grid lg:grid-cols-[360px_1fr] gap-4">
          <div className="card">
            <div className="flex items-center justify-between mb-2">
              <div className="section-title">Assumptions</div>
              {isDirty && (
                <span className="text-[10px] uppercase tracking-widest text-warn-500">
                  unsaved edits
                </span>
              )}
            </div>
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
            <div className="flex gap-2 mt-4">
              <button
                className="btn-primary flex-1"
                onClick={recompute}
                disabled={loading}
              >
                {loading ? "Re-computing…" : "Re-run scenarios"}
              </button>
              <button
                type="button"
                className="px-3 py-2 text-sm rounded-md bg-ink-800 border border-ink-700 text-slate-300 hover:bg-ink-700 disabled:opacity-50"
                onClick={resetToBaseline}
                disabled={!isDirty || loading}
                title={
                  source === "saved"
                    ? "Reset to the loaded saved DCF"
                    : "Reset to engine-derived defaults"
                }
              >
                Reset
              </button>
            </div>
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
