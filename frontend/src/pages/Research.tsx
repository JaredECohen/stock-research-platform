import React, { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "@/api/client";
import MemoCard from "@/components/MemoCard";
import { AgentTrace } from "@/components/AgentTrace";
import NewsAlertPanel from "@/components/NewsAlertPanel";
import AnalyzeStockGate, { TierBadge } from "@/components/AnalyzeStockGate";
import MemoVersionTimeline from "@/components/MemoVersionTimeline";
import DCFVersionHistory from "@/components/DCFVersionHistory";
import MemoryTrail from "@/components/MemoryTrail";
import TickerPicker from "@/components/TickerPicker";
import type { CompanyOut, StockMemoOut, AgentTrace as AgentTraceT } from "@/types";

type FetchError = Error & { status?: number; detail?: string };

export default function Research() {
  const [params, setParams] = useSearchParams();
  const ticker = params.get("ticker") || "";
  const [universe, setUniverse] = useState<CompanyOut[]>([]);
  const [universeError, setUniverseError] = useState<string | null>(null);
  const [universeLoading, setUniverseLoading] = useState(false);
  const [memo, setMemo] = useState<StockMemoOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [needsGate, setNeedsGate] = useState(false);   // 409 from backend
  const [error, setError] = useState<string | null>(null);
  const [trace, setTrace] = useState<AgentTraceT[]>([]);

  const loadUniverse = React.useCallback(() => {
    setUniverseLoading(true);
    setUniverseError(null);
    api
      .listStocks()
      .then((rows) => {
        setUniverse(rows);
        setUniverseLoading(false);
      })
      .catch((e: Error) => {
        setUniverseError(e.message || "Failed to load tickers");
        setUniverseLoading(false);
      });
  }, []);

  useEffect(() => {
    loadUniverse();
  }, []);

  const company = universe.find((c) => c.ticker === ticker);

  // Fetch memo. When the backend returns 409 (data_only tier without
  // ondemand=true), surface the analyze gate instead of raising.
  function loadMemo(opts?: { ondemand?: boolean }) {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    setMemo(null);
    setNeedsGate(false);
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
      .getStockMemo(ticker, opts)
      .then((m) => {
        setMemo(m);
        setTrace((cur) => cur.map((t) => ({ ...t, status: "done" as const, detail: t.detail || "complete" })));
      })
      .catch((e: FetchError) => {
        if (e.status === 409) {
          setNeedsGate(true);
        } else {
          setError(e.detail || String(e));
        }
      })
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    if (!ticker) {
      setMemo(null);
      setNeedsGate(false);
      return;
    }
    loadMemo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker]);

  // After a successful on-demand analysis the backend promotes the ticker
  // to `analyzed_on_demand`; refresh the universe list so the badge updates.
  function onAnalyzed(m: StockMemoOut) {
    setMemo(m);
    loadUniverse();
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Stock Research</h1>
      <div className="card-tight">
        <div className="flex flex-col md:flex-row gap-3 md:items-center">
          <TickerPicker
            value={ticker}
            onChange={(t) => (t ? setParams({ ticker: t }) : setParams({}))}
            universe={universe}
            loading={universeLoading}
            className="flex-1"
          />
          {(universeError || universe.length === 0) && !universeLoading && (
            <button
              type="button"
              onClick={loadUniverse}
              className="px-3 py-1.5 text-sm rounded-md bg-ink-800 border border-ink-700 text-slate-200 hover:bg-ink-700"
              title={universeError || ""}
            >
              Reload tickers
            </button>
          )}
          <button
            type="button"
            disabled={!ticker || loading}
            className="btn-primary"
            onClick={() =>
              ticker &&
              api
                .analyzeStock(ticker)
                .then(onAnalyzed)
                .catch((e: FetchError) =>
                  setError(e.detail || String(e)),
                )
            }
          >
            {loading ? "Generating…" : "Refresh memo"}
          </button>
        </div>
        {company && (
          <div className="flex items-center gap-2 mt-2 text-xs text-slate-400">
            <TierBadge tier={company.universe_tier} />
            <span>{company.sector}</span>
            {company.industry && <span>· {company.industry}</span>}
          </div>
        )}
      </div>

      {error && <div className="card-tight border-danger-500/40 text-danger-500 text-sm">{error}</div>}

      {needsGate && company && (
        <AnalyzeStockGate
          company={company}
          loading={loading}
          onAnalyze={() => loadMemo({ ondemand: true })}
        />
      )}

      {ticker && !memo && !error && !needsGate && loading && (
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
          <div className="space-y-4">
            <AgentTrace trace={trace} />
            <NewsAlertPanel alerts={memo.sector_agent_view.data?.pending_news_alerts} />
            <div className="card-tight">
              <MemoVersionTimeline ticker={memo.ticker} />
            </div>
            <div className="card-tight">
              <DCFVersionHistory ticker={memo.ticker} />
            </div>
            <div className="card-tight">
              <MemoryTrail ticker={memo.ticker} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
