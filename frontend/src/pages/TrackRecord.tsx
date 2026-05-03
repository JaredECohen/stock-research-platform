import React, { useEffect, useState } from "react";
import { api } from "@/api/client";

type TR = {
  horizon_days: number;
  total: number;
  directional_evaluations: number;
  thesis_hit_rate: number | null;
  avg_forward_return: number;
  avg_alpha: number | null;
  ticker_filter: string | null;
  sector_filter: string | null;
};

const HORIZONS = [30, 90, 180, 365];

function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export default function TrackRecord() {
  const [horizon, setHorizon] = useState(90);
  const [ticker, setTicker] = useState("");
  const [sector, setSector] = useState("");
  const [data, setData] = useState<TR | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);

  const refresh = React.useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .trackRecord({
        horizon_days: horizon,
        ticker: ticker.trim() || undefined,
        sector: sector.trim() || undefined,
      })
      .then(setData)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [horizon, ticker, sector]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const triggerEvaluator = async () => {
    setEvaluating(true);
    try {
      await fetch("/api/admin/evaluate-outcomes", { method: "POST" });
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <div className="text-2xl font-semibold tracking-tight">Track record</div>
        <div className="text-sm text-slate-400 mt-1">
          Realized forward-return scoring for every memo recommendation. SPY-relative
          alpha + thesis-held rate. Wave 4A.
        </div>
      </div>

      <div className="card-tight">
        <div className="flex flex-wrap items-end gap-3">
          <div className="space-y-1">
            <label className="text-xs uppercase tracking-widest text-slate-500">
              Horizon (days)
            </label>
            <select
              className="bg-ink-800 border border-ink-700 rounded-md px-2 py-1.5 text-sm text-slate-100"
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
            >
              {HORIZONS.map((h) => (
                <option key={h} value={h}>{h}d</option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-xs uppercase tracking-widest text-slate-500">
              Ticker (optional)
            </label>
            <input
              className="bg-ink-800 border border-ink-700 rounded-md px-2 py-1.5 text-sm text-slate-100 w-32"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="NVDA"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs uppercase tracking-widest text-slate-500">
              Sector (optional)
            </label>
            <input
              className="bg-ink-800 border border-ink-700 rounded-md px-2 py-1.5 text-sm text-slate-100 w-44"
              value={sector}
              onChange={(e) => setSector(e.target.value)}
              placeholder="Technology"
            />
          </div>
          <button
            type="button"
            onClick={refresh}
            disabled={loading}
            className="px-3 py-1.5 text-sm rounded-md bg-accent-600/30 border border-accent-600/40 text-accent-100 hover:bg-accent-600/45 disabled:opacity-50"
          >
            {loading ? "Loading…" : "Refresh"}
          </button>
          <button
            type="button"
            onClick={triggerEvaluator}
            disabled={evaluating}
            className="px-3 py-1.5 text-sm rounded-md bg-ink-800 border border-ink-700 text-slate-200 hover:bg-ink-700 disabled:opacity-50"
            title="Run the daily outcome scorer now (production runs it via APScheduler)."
          >
            {evaluating ? "Evaluating…" : "Score now"}
          </button>
        </div>
      </div>

      {error && (
        <div className="card-tight border-danger-500/40 bg-danger-500/5 text-danger-500 text-sm">
          {error}
        </div>
      )}

      {data && (
        <div className="grid md:grid-cols-3 gap-4">
          <Stat
            label="Total evaluations"
            value={String(data.total)}
            sub={`(${data.horizon_days}d horizon)`}
          />
          <Stat
            label="Thesis hit rate"
            value={fmtPct(data.thesis_hit_rate, 0)}
            sub={`${data.directional_evaluations} directional calls`}
          />
          <Stat
            label="Avg alpha vs SPY"
            value={data.avg_alpha === null ? "—" : fmtPct(data.avg_alpha)}
            sub={`avg return ${fmtPct(data.avg_forward_return)}`}
          />
        </div>
      )}

      {data && data.total === 0 && (
        <div className="card-tight border-warn-500/40 bg-warn-500/5 text-warn-500 text-sm">
          No outcomes for this filter yet. Click <strong>Score now</strong> to
          evaluate any memos whose forward window has come of age.
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="card-tight">
      <div className="text-xs uppercase tracking-widest text-slate-500">
        {label}
      </div>
      <div className="text-2xl font-semibold mt-1">{value}</div>
      {sub && <div className="text-xs text-slate-400 mt-0.5">{sub}</div>}
    </div>
  );
}
