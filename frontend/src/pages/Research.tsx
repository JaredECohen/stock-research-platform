import React, { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "@/api/client";
import MemoCard from "@/components/MemoCard";
import { AgentTrace } from "@/components/AgentTrace";
import NewsAlertPanel from "@/components/NewsAlertPanel";
import AnalyzeStockGate, { TierBadge } from "@/components/AnalyzeStockGate";
import MemoVersionTimeline from "@/components/MemoVersionTimeline";
import DCFVersionHistory from "@/components/DCFVersionHistory";
import MemoryTrail from "@/components/MemoryTrail";
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

  // Wave 9b — typeahead-friendly ticker picker.
  // `draft` is the live input value; we commit to the URL only when the
  // user lands on a known ticker (via dropdown selection or Enter on a
  // ticker-shaped string). Filters the visible options against ticker +
  // company name so typing "app" shows AAPL.
  const [draft, setDraft] = useState<string>(ticker);
  useEffect(() => setDraft(ticker), [ticker]);
  const tickerSet = useMemo(() => new Set(universe.map((c) => c.ticker)), [universe]);
  const filteredUniverse = useMemo(() => {
    const q = draft.trim().toLowerCase();
    if (!q) return universe;
    return universe.filter(
      (c) => c.ticker.toLowerCase().includes(q) || c.company_name.toLowerCase().includes(q),
    );
  }, [universe, draft]);
  const [showList, setShowList] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setShowList(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  function commitTicker(t: string) {
    const upper = t.trim().toUpperCase();
    if (!upper) {
      setParams({});
      setDraft("");
      setShowList(false);
      return;
    }
    setParams({ ticker: upper });
    setDraft(upper);
    setShowList(false);
  }

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
          <div ref={containerRef} className="relative flex-1">
            <input
              type="text"
              className="input w-full"
              value={draft}
              onChange={(e) => {
                setDraft(e.target.value.toUpperCase());
                setShowList(true);
              }}
              onFocus={() => setShowList(true)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  // Prefer the first filtered match; otherwise commit the
                  // raw text (lazy-research path covers unknown tickers).
                  const target = filteredUniverse[0]?.ticker ?? draft;
                  commitTicker(target);
                } else if (e.key === "Escape") {
                  setShowList(false);
                  setDraft(ticker);
                }
              }}
              placeholder={
                universeLoading
                  ? "Loading tickers…"
                  : universe.length === 0
                  ? "No tickers available — click Reload"
                  : "Type a ticker or company name…"
              }
              disabled={universeLoading}
              spellCheck={false}
              autoComplete="off"
            />
            {showList && filteredUniverse.length > 0 && (
              <ul
                className="absolute z-10 left-0 right-0 mt-1 max-h-72 overflow-y-auto rounded-md border border-ink-700 bg-ink-900 shadow-lg"
                role="listbox"
              >
                {filteredUniverse.slice(0, 60).map((c) => (
                  <li
                    key={c.ticker}
                    role="option"
                    aria-selected={c.ticker === ticker}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      commitTicker(c.ticker);
                    }}
                    className={`px-3 py-1.5 cursor-pointer text-sm flex items-center gap-2 ${
                      c.ticker === ticker
                        ? "bg-accent-600/15 text-accent-400"
                        : "text-slate-200 hover:bg-ink-800"
                    }`}
                  >
                    <span className="font-mono w-14">{c.ticker}</span>
                    <span className="text-slate-400 truncate">{c.company_name}</span>
                    {c.sector && (
                      <span className="ml-auto text-xs text-slate-500">{c.sector}</span>
                    )}
                  </li>
                ))}
                {filteredUniverse.length > 60 && (
                  <li className="px-3 py-1 text-xs text-slate-500 border-t border-ink-800">
                    {filteredUniverse.length - 60} more — keep typing to narrow
                  </li>
                )}
              </ul>
            )}
          </div>
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
