import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Plus, RefreshCw, Trash2 } from "lucide-react";
import { api } from "@/api/client";
import type {
  CustomScreenResult,
  CustomScreenRow,
  ScreenerMetricName,
  ScreenerOp,
  ScreenerResult,
  ScreenerRule,
} from "@/types";

type TabKey = "ai" | "factor" | "custom";

const TABS: Array<{ key: TabKey; label: string; description: string }> = [
  { key: "ai", label: "AI-First", description: "PM-conviction ranking with optional theme bias" },
  { key: "factor", label: "Factor Rank", description: "Sort by any single factor score" },
  { key: "custom", label: "Custom Screen", description: "Build rules against raw metrics (P/E, gross margin, etc.)" },
];

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

const FACTOR_COLUMNS: Array<{ key: string; label: string }> = [
  { key: "pm_score", label: "PM Conviction" },
  { key: "quality", label: "Quality" },
  { key: "growth", label: "Growth" },
  { key: "valuation", label: "Valuation" },
  { key: "earnings_momentum", label: "Earnings Momentum" },
  { key: "risk", label: "Risk" },
  { key: "macro_fit", label: "Macro Fit" },
];

const METRIC_LABELS: Record<ScreenerMetricName, string> = {
  pe_ttm: "P/E (TTM)",
  forward_pe: "Forward P/E",
  peg: "PEG",
  ev_ebitda: "EV/EBITDA",
  ev_revenue: "EV/Revenue",
  gross_margin: "Gross Margin",
  op_margin: "Operating Margin",
  fcf_margin: "FCF Margin",
  roic: "ROIC",
  roe: "ROE",
  debt_to_ebitda: "Debt/EBITDA",
  revenue_growth_yoy: "Revenue Growth YoY",
  dividend_yield: "Dividend Yield",
  market_cap: "Market Cap",
  beta: "Beta",
};

const PERCENT_METRICS: Set<ScreenerMetricName> = new Set([
  "gross_margin",
  "op_margin",
  "fcf_margin",
  "roic",
  "roe",
  "revenue_growth_yoy",
  "dividend_yield",
]);

const OPS: ScreenerOp[] = [">", "<", ">=", "<=", "=", "between"];

// localStorage keys (versioned so we can break the schema later).
const LS_TAB = "screener:tab:v1";
const LS_FACTOR = "screener:factor:v1";
const LS_CUSTOM = "screener:custom:v1";

function loadJSON<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function saveJSON(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore quota / SSR.
  }
}

function fmtMetric(name: ScreenerMetricName, value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (name === "market_cap") {
    if (Math.abs(value) >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
    if (Math.abs(value) >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
    if (Math.abs(value) >= 1e6) return `$${(value / 1e6).toFixed(0)}M`;
    return `$${value.toFixed(0)}`;
  }
  if (PERCENT_METRICS.has(name)) return `${(value * 100).toFixed(1)}%`;
  return value.toFixed(2);
}

export default function Screener() {
  const [tab, setTab] = useState<TabKey>(() => loadJSON<TabKey>(LS_TAB, "ai"));
  useEffect(() => saveJSON(LS_TAB, tab), [tab]);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Screener</h1>
        <p className="text-slate-400 text-sm mt-1">
          {TABS.find((t) => t.key === tab)?.description ?? "Agent-ranked ideas. Click a row to open the memo."}
        </p>
      </div>

      <div className="card-tight flex gap-2">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-3 py-1.5 rounded-md text-sm border ${
              tab === t.key
                ? "border-accent-600 text-accent-500 bg-accent-600/15"
                : "border-ink-700 text-slate-300 hover:bg-ink-800"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "ai" && <AIScreenerView themed />}
      {tab === "factor" && <AIScreenerView />}
      {tab === "custom" && <CustomScreenView />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// AI-First / Factor Rank — same data, different defaults.
// ---------------------------------------------------------------------------

function AIScreenerView({ themed = false }: { themed?: boolean }) {
  const factorState = loadJSON<{ sort_by: string; order: "asc" | "desc" }>(
    LS_FACTOR,
    { sort_by: "pm_score", order: "desc" },
  );
  const [theme, setTheme] = useState<string>("");
  const [sector, setSector] = useState<string>("");
  const [search, setSearch] = useState<string>("");
  const [sortBy, setSortBy] = useState<string>(themed ? "pm_score" : factorState.sort_by);
  const [order, setOrder] = useState<"asc" | "desc">(themed ? "desc" : factorState.order);
  const [data, setData] = useState<ScreenerResult | null>(null);
  const [loading, setLoading] = useState(false);

  // Persist factor-rank state (only when not on the AI-first tab)
  useEffect(() => {
    if (!themed) saveJSON(LS_FACTOR, { sort_by: sortBy, order });
  }, [themed, sortBy, order]);

  const load = () => {
    setLoading(true);
    api
      .screener({
        theme: themed ? theme || undefined : undefined,
        sector: sector || undefined,
        sort_by: sortBy,
        order,
        limit: 100,
      })
      .then(setData)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme, sector, sortBy, order]);

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
    <>
      <div className="card-tight flex flex-wrap items-center gap-2">
        {themed && (
          <>
            <div className="text-xs text-slate-500 mr-2">Theme:</div>
            {THEMES.map((t) => (
              <button
                key={t.key}
                onClick={() => setTheme(t.key)}
                className={`badge ${
                  theme === t.key
                    ? "border-accent-600 text-accent-500 bg-accent-600/15"
                    : "border-ink-700 text-slate-300"
                }`}
              >
                {t.label}
              </button>
            ))}
          </>
        )}
        {!themed && (
          <>
            <div className="text-xs text-slate-500 mr-2">Sort by:</div>
            <select className="input" value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
              {FACTOR_COLUMNS.map((c) => (
                <option key={c.key} value={c.key}>{c.label}</option>
              ))}
            </select>
            <select className="input" value={order} onChange={(e) => setOrder(e.target.value as "asc" | "desc")}>
              <option value="desc">High → Low</option>
              <option value="asc">Low → High</option>
            </select>
          </>
        )}
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
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} /> Refresh
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
    </>
  );
}

// ---------------------------------------------------------------------------
// Custom rule-based screener
// ---------------------------------------------------------------------------

interface CustomState {
  rules: ScreenerRule[];
  sort_by: ScreenerMetricName;
  order: "asc" | "desc";
}

const DEFAULT_CUSTOM_STATE: CustomState = {
  rules: [
    { metric: "gross_margin", op: ">", value: 0.5 },
    { metric: "pe_ttm", op: "<", value: 30 },
  ],
  sort_by: "market_cap",
  order: "desc",
};

function CustomScreenView() {
  const [state, setState] = useState<CustomState>(() =>
    loadJSON<CustomState>(LS_CUSTOM, DEFAULT_CUSTOM_STATE),
  );
  const [data, setData] = useState<CustomScreenResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Debounced persistence — write 250ms after last edit.
  useEffect(() => {
    const handle = window.setTimeout(() => saveJSON(LS_CUSTOM, state), 250);
    return () => window.clearTimeout(handle);
  }, [state]);

  const updateRule = (idx: number, patch: Partial<ScreenerRule>) => {
    setState((s) => ({
      ...s,
      rules: s.rules.map((r, i) => (i === idx ? { ...r, ...patch } : r)),
    }));
  };

  const addRule = () => {
    setState((s) => ({
      ...s,
      rules: [...s.rules, { metric: "pe_ttm", op: "<", value: 25 }],
    }));
  };

  const removeRule = (idx: number) => {
    setState((s) => ({ ...s, rules: s.rules.filter((_, i) => i !== idx) }));
  };

  const reset = () => {
    setState(DEFAULT_CUSTOM_STATE);
    setData(null);
    setError(null);
  };

  const run = () => {
    setLoading(true);
    setError(null);
    api
      .customScreener({
        rules: state.rules,
        sort_by: state.sort_by,
        order: state.order,
        limit: 100,
      })
      .then(setData)
      .catch((e) => setError(e?.message ?? "Custom screen failed"))
      .finally(() => setLoading(false));
  };

  // First render → run with the default rules so the user lands on results.
  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Columns shown in the result grid: every metric the user filtered on,
  // plus the sort column. Dedup + preserve order.
  const resultColumns: ScreenerMetricName[] = useMemo(() => {
    const set = new Set<ScreenerMetricName>();
    for (const r of state.rules) set.add(r.metric);
    set.add(state.sort_by);
    return Array.from(set);
  }, [state.rules, state.sort_by]);

  return (
    <>
      <div className="card-tight space-y-3">
        <div className="text-xs text-slate-500">
          Rules are AND-combined. Tickers are limited to the curated S&amp;P 100. Rows with missing
          metrics fail the rule (rather than being dropped silently).
        </div>
        <div className="space-y-2">
          {state.rules.map((rule, idx) => (
            <div key={idx} className="flex flex-wrap items-center gap-2">
              <select
                className="input"
                value={rule.metric}
                onChange={(e) => updateRule(idx, { metric: e.target.value as ScreenerMetricName })}
              >
                {(Object.keys(METRIC_LABELS) as ScreenerMetricName[]).map((m) => (
                  <option key={m} value={m}>{METRIC_LABELS[m]}</option>
                ))}
              </select>
              <select
                className="input"
                value={rule.op}
                onChange={(e) => updateRule(idx, { op: e.target.value as ScreenerOp })}
              >
                {OPS.map((op) => (
                  <option key={op} value={op}>{op}</option>
                ))}
              </select>
              <input
                type="number"
                className="input w-32"
                step="any"
                value={rule.value}
                onChange={(e) => updateRule(idx, { value: parseFloat(e.target.value) || 0 })}
              />
              {rule.op === "between" && (
                <>
                  <span className="text-slate-500 text-xs">to</span>
                  <input
                    type="number"
                    className="input w-32"
                    step="any"
                    value={rule.value2 ?? 0}
                    onChange={(e) => updateRule(idx, { value2: parseFloat(e.target.value) || 0 })}
                  />
                </>
              )}
              {PERCENT_METRICS.has(rule.metric) && (
                <span className="text-xs text-slate-500">
                  ({(rule.value * 100).toFixed(1)}% — enter as decimal)
                </span>
              )}
              <button
                onClick={() => removeRule(idx)}
                className="btn-ghost text-rose-400"
                aria-label="Remove rule"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={addRule} className="btn-ghost">
            <Plus size={14} /> Add rule
          </button>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-slate-500">Sort:</span>
            <select
              className="input"
              value={state.sort_by}
              onChange={(e) => setState((s) => ({ ...s, sort_by: e.target.value as ScreenerMetricName }))}
            >
              {(Object.keys(METRIC_LABELS) as ScreenerMetricName[]).map((m) => (
                <option key={m} value={m}>{METRIC_LABELS[m]}</option>
              ))}
            </select>
            <select
              className="input"
              value={state.order}
              onChange={(e) => setState((s) => ({ ...s, order: e.target.value as "asc" | "desc" }))}
            >
              <option value="desc">High → Low</option>
              <option value="asc">Low → High</option>
            </select>
            <button className="btn-ghost text-slate-400" onClick={reset}>Reset</button>
            <button className="btn-primary" onClick={run} disabled={loading}>
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} /> Run
            </button>
          </div>
        </div>
      </div>

      {error && <div className="card text-rose-400 text-sm">{error}</div>}

      {data && (
        <div className="card overflow-x-auto">
          <div className="text-xs text-slate-500 mb-2">
            {data.matched} match{data.matched === 1 ? "" : "es"} · {state.rules.length} rule
            {state.rules.length === 1 ? "" : "s"} · saved to this browser
          </div>
          <table className="w-full text-sm">
            <thead className="text-xs text-slate-500 border-b border-ink-700">
              <tr>
                <th className="text-left py-2">Ticker</th>
                <th className="text-left">Company</th>
                <th className="text-left">Sector</th>
                <th className="text-right">PM</th>
                {resultColumns.map((m) => (
                  <th key={m} className="text-right">{METRIC_LABELS[m]}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r: CustomScreenRow) => (
                <tr key={r.ticker} className="border-b border-ink-800 table-row-hover">
                  <td className="py-2 font-mono">
                    <Link
                      to={`/research?ticker=${r.ticker}`}
                      className="text-accent-500 hover:underline"
                    >
                      {r.ticker}
                    </Link>
                  </td>
                  <td className="text-slate-300">{r.company_name}</td>
                  <td className="text-slate-400">{r.sector}</td>
                  <td className="text-right font-mono">
                    {r.pm_score == null ? "—" : r.pm_score.toFixed(0)}
                  </td>
                  {resultColumns.map((m) => (
                    <td key={m} className="text-right font-mono text-slate-300">
                      {fmtMetric(m, r.metrics[m] as number | null | undefined)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
