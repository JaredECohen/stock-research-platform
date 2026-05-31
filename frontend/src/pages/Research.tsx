import React, { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "@/api/client";
import MemoCard from "@/components/MemoCard";
import { AgentTrace } from "@/components/AgentTrace";
import FullInvestmentMemo from "@/components/FullInvestmentMemo";
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
  const [fullMemoOpen, setFullMemoOpen] = useState(false);
  // Async memo regen state. `regenStartedAt` is the server-side
  // started_at returned by POST /analyze; polling stops when the
  // status endpoint reports a `latest_memo_at` strictly newer than
  // it (i.e. the background regen finished and persisted the new
  // version). `regenElapsedSec` is purely UI feedback so the user
  // sees progress instead of a static "Generating…" label.
  const [regenStartedAt, setRegenStartedAt] = useState<string | null>(null);
  const [regenElapsedSec, setRegenElapsedSec] = useState(0);

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

  // Kick off an async memo regen. Returns immediately; the polling
  // effect below watches for completion and re-fetches the memo when
  // the persisted `latest_memo_at` advances past the regen's
  // `started_at`. Survives page reloads — re-clicking just coalesces
  // against the in-flight job server-side.
  function startRegen() {
    if (!ticker) return;
    setError(null);
    api
      .analyzeStock(ticker)
      .then((res) => {
        setRegenStartedAt(res.started_at);
        setRegenElapsedSec(0);
      })
      .catch((e: FetchError) => setError(e.detail || String(e)));
  }

  // Poll `/analyze/status` while a regen is in flight. Stop when the
  // server's latest memo is newer than our `started_at` — at which
  // point fetch the new memo + clear local state.
  useEffect(() => {
    if (!ticker || !regenStartedAt) return;
    const tickStart = Date.now();
    const tick = setInterval(() => {
      setRegenElapsedSec(Math.round((Date.now() - tickStart) / 1000));
      api
        .analyzeStatus(ticker)
        .then((s) => {
          const newer =
            s.latest_memo_at !== null &&
            new Date(s.latest_memo_at).getTime() >
              new Date(regenStartedAt).getTime();
          if (newer) {
            clearInterval(tick);
            setRegenStartedAt(null);
            setRegenElapsedSec(0);
            // Refetch the fresh memo from the server.
            api
              .getStockMemo(ticker)
              .then(onAnalyzed)
              .catch((e: FetchError) =>
                setError(e.detail || String(e)),
              );
          }
        })
        .catch(() => {
          // Network blip — keep polling. Persistent failures will be
          // visible via the lack of progress in the elapsed counter.
        });
    }, 10_000);
    return () => clearInterval(tick);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker, regenStartedAt]);

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
            disabled={!ticker || loading || !!regenStartedAt}
            className="btn-primary"
            onClick={startRegen}
            title={
              regenStartedAt
                ? "Memo regeneration is already running in the background"
                : undefined
            }
          >
            {regenStartedAt
              ? `Regenerating… ${Math.floor(regenElapsedSec / 60)}:${String(regenElapsedSec % 60).padStart(2, "0")}`
              : loading
                ? "Generating…"
                : "Refresh memo"}
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

      {universeError && !universeLoading && (
        <div className="card-tight border-warn-500/40 bg-warn-500/5 text-sm">
          <div className="text-warn-500 font-medium">Ticker universe failed to load</div>
          <div className="text-xs text-slate-300 mt-1 leading-relaxed">
            {universeError}
            <br />
            The picker is empty until this resolves. Try the{" "}
            <button
              type="button"
              onClick={loadUniverse}
              className="underline text-accent-400 hover:text-accent-300"
            >
              Reload tickers
            </button>{" "}
            button above. If the failure persists, the backend may be deploying
            a schema migration — wait 1-2 minutes and reload.
          </div>
        </div>
      )}

      {regenStartedAt && (
        <div className="card-tight border-accent-600/30 bg-accent-600/[0.04] text-sm text-slate-200">
          <div className="font-medium text-accent-400">
            Regenerating memo in background ({Math.floor(regenElapsedSec / 60)}:{String(regenElapsedSec % 60).padStart(2, "0")} elapsed)
          </div>
          <div className="text-xs text-slate-400 mt-1">
            A full memo regen takes 5-9 minutes. The page will refresh automatically when it's ready — you can navigate away and come back. The current memo below is the previous version.
          </div>
        </div>
      )}

      {needsGate && company && (
        <AnalyzeStockGate
          company={company}
          loading={loading}
          onAnalyze={() => loadMemo({ ondemand: true })}
        />
      )}

      {ticker && !memo && !error && !needsGate && loading && (
        <div className="grid lg:grid-cols-[1fr_280px] gap-4">
          <div className="card flex items-center gap-3 py-8">
            <span
              aria-hidden
              className="inline-block h-5 w-5 rounded-full border-2 border-accent-500 border-t-transparent animate-spin"
            />
            <div className="text-sm text-slate-200">
              Loading memo for <span className="font-semibold">{ticker}</span>…
              <div className="text-xs text-slate-500 mt-0.5">
                Fetching the latest cached analysis. If this is a fresh ticker the first load can take ~30 seconds.
              </div>
            </div>
          </div>
          <AgentTrace trace={trace} />
        </div>
      )}

      {memo && (
        <div className="grid lg:grid-cols-[1fr_280px] gap-4">
          <div className="space-y-3">
            <div className="flex justify-end">
              <button
                type="button"
                className="px-3 py-1.5 text-sm rounded-md bg-accent-600 hover:bg-accent-500 text-white font-medium shadow"
                onClick={() => setFullMemoOpen(true)}
                title="Open the full structured investment memo (with PDF download)"
              >
                View Full Investment Memo
              </button>
            </div>
            <MemoCard memo={memo} />
          </div>
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

      {memo && (
        <FullInvestmentMemo
          memo={memo}
          open={fullMemoOpen}
          onClose={() => setFullMemoOpen(false)}
        />
      )}
    </div>
  );
}
