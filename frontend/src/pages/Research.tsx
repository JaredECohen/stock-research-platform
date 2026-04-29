import React, { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "@/api/client";
import MemoCard from "@/components/MemoCard";
import { AgentTrace } from "@/components/AgentTrace";
import type { CompanyOut, StockMemoOut, AgentTrace as AgentTraceT } from "@/types";

export default function Research() {
  const [params, setParams] = useSearchParams();
  const ticker = params.get("ticker") || "";
  const [universe, setUniverse] = useState<CompanyOut[]>([]);
  const [memo, setMemo] = useState<StockMemoOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trace, setTrace] = useState<AgentTraceT[]>([]);

  useEffect(() => {
    api.listStocks().then(setUniverse).catch(() => {});
  }, []);

  useEffect(() => {
    if (!ticker) {
      setMemo(null);
      return;
    }
    setLoading(true);
    setError(null);
    setTrace([
      { agent: "PM Orchestrator", status: "running", detail: "Routing single-stock memo workflow." },
      { agent: "Sector Analyst", status: "queued", detail: "" },
      { agent: "Earnings Analyst", status: "queued", detail: "" },
      { agent: "Filing Analyst", status: "queued", detail: "" },
      { agent: "Valuation Analyst", status: "queued", detail: "" },
      { agent: "Comps Analyst", status: "queued", detail: "" },
      { agent: "Macro Analyst", status: "queued", detail: "" },
      { agent: "Risk Committee", status: "queued", detail: "" },
    ]);
    api
      .getStockMemo(ticker)
      .then((m) => {
        setMemo(m);
        setTrace((cur) => cur.map((t) => ({ ...t, status: "done" as const, detail: t.detail || "complete" })));
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [ticker]);

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Stock Research</h1>
      <div className="card-tight">
        <div className="flex flex-col md:flex-row gap-3 md:items-center">
          <select
            className="input flex-1"
            value={ticker}
            onChange={(e) => {
              const t = e.target.value.toUpperCase();
              if (t) setParams({ ticker: t });
              else setParams({});
            }}
          >
            <option value="">Select a ticker…</option>
            {universe.map((c) => (
              <option key={c.ticker} value={c.ticker}>
                {c.ticker} — {c.company_name}
              </option>
            ))}
          </select>
          <button
            type="button"
            disabled={!ticker || loading}
            className="btn-primary"
            onClick={() => ticker && api.analyzeStock(ticker).then(setMemo)}
          >
            {loading ? "Generating…" : "Refresh memo"}
          </button>
        </div>
      </div>

      {error && <div className="card-tight border-danger-500/40 text-danger-500 text-sm">{error}</div>}

      {ticker && !memo && !error && (
        <div className="grid lg:grid-cols-[1fr_280px] gap-4">
          <div className="card">
            <div className="text-sm text-slate-400">Generating memo for {ticker}…</div>
          </div>
          <AgentTrace trace={trace} />
        </div>
      )}

      {memo && (
        <div className="grid lg:grid-cols-[1fr_280px] gap-4">
          <MemoCard memo={memo} />
          <AgentTrace trace={trace} />
        </div>
      )}
    </div>
  );
}
