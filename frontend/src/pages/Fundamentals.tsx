import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, isApiError } from "@/api/client";
import { useConfig } from "@/auth/ConfigProvider";
import { useAccount } from "@/auth/useAccount";
import RateLimitNotice from "@/components/RateLimitNotice";
import CommentaryPanel from "@/components/fundamentals/CommentaryPanel";
import CompanySelector from "@/components/fundamentals/CompanySelector";
import EntitlementNotice from "@/components/fundamentals/EntitlementNotice";
import MetricPicker from "@/components/fundamentals/MetricPicker";
import SeriesStateNotice from "@/components/fundamentals/SeriesStateNotice";
import TimeRangeControl from "@/components/fundamentals/TimeRangeControl";
import ViewModeControl from "@/components/fundamentals/ViewModeControl";
import { FundamentalsChart } from "@/components/fundamentals/chart";
import { MAX_METRICS, MAX_TICKERS, useFundamentalsState } from "@/hooks/useFundamentalsState";
import type { LayoutResult } from "@/lib/fundamentals/layout";
import type { CatalogResponse, CompanyOut, RateLimitRefusal, SeriesResponseWire, StructuredErrorDetail } from "@/types";

/**
 * FEAT-001 Fundamentals Explorer. The URL is the state (`useFundamentalsState`);
 * this page loads the catalog and the ticker universe once, fetches the
 * series whenever the selection changes, and hands the chart engine exactly
 * what the server sent. Every non-happy state is explicit and keeps the
 * reader's inputs: 402 names the plan's limits, 429 keeps the chart and
 * offers a retry, a company with no stored history gets the research CTA
 * (the only path that loads history — nothing is backfilled in a page
 * request), stale and estimated data are labelled rather than smoothed.
 */

/** Tailwind `md` is 768px; dual axis is unreadable below it. */
const NARROW_QUERY = "(max-width: 767px)";

function useNarrowViewport(): boolean {
  const [narrow, setNarrow] = useState<boolean>(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
    return window.matchMedia(NARROW_QUERY).matches;
  });
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mql = window.matchMedia(NARROW_QUERY);
    const onChange = (e: MediaQueryListEvent) => setNarrow(e.matches);
    if (typeof mql.addEventListener === "function") {
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    }
    return undefined;
  }, []);
  return narrow;
}

export default function Fundamentals() {
  const { config } = useConfig();
  const { account, enabled: accountEnabled } = useAccount();
  const narrow = useNarrowViewport();

  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [universe, setUniverse] = useState<CompanyOut[]>([]);
  const [universeLoading, setUniverseLoading] = useState(true);

  const knownMetrics = useMemo(() => (catalog ? new Set(catalog.metrics.map((m) => m.id)) : null), [catalog]);
  const metricLabels = useMemo(() => {
    const out: Record<string, string> = {};
    for (const m of catalog?.metrics ?? []) out[m.id] = m.label;
    return out;
  }, [catalog]);

  const { state, setTickers, setMetrics, setYears, setView } = useFundamentalsState(knownMetrics);

  const [data, setData] = useState<SeriesResponseWire | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refusal, setRefusal] = useState<StructuredErrorDetail | null>(null);
  const [rateLimit, setRateLimit] = useState<RateLimitRefusal | null>(null);
  const [layout, setLayout] = useState<LayoutResult | null>(null);
  const requestSeq = useRef(0);

  const loadCatalog = useCallback(() => {
    setCatalogError(null);
    api
      .fundamentalsCatalog()
      .then(setCatalog)
      .catch((e: Error) => setCatalogError(isApiError(e) ? e.detail || e.message : e.message || "Failed to load the metric catalog"));
  }, []);

  useEffect(() => {
    loadCatalog();
    api
      .listStocks()
      .then(setUniverse)
      .catch(() => setUniverse([]))
      .finally(() => setUniverseLoading(false));
  }, [loadCatalog]);

  // One string so the effect below re-runs only when the *request* changes,
  // not on every render that rebuilds the arrays.
  const requestKey = JSON.stringify({ t: state.tickers, m: state.metrics, y: state.years });
  const canRequest = !!knownMetrics && state.tickers.length > 0 && state.metrics.length > 0;

  const loadSeries = useCallback(() => {
    const seq = ++requestSeq.current;
    const { t: tickers, m: metrics, y: years } = JSON.parse(requestKey) as { t: string[]; m: string[]; y: number | null };
    setLoading(true);
    setError(null);
    setRefusal(null);
    setRateLimit(null);
    api
      .fundamentalsSeries({ tickers, metrics, years: years ?? undefined })
      .then((r) => {
        if (seq !== requestSeq.current) return;
        setData(r);
      })
      .catch((e: unknown) => {
        if (seq !== requestSeq.current) return;
        if (isApiError(e) && e.code === "plan_required" && e.structured) {
          // The shape is wrong, so nothing was drawn; the previous chart
          // would describe a different selection than the URL.
          setData(null);
          setRefusal(e.structured);
        } else if (isApiError(e) && e.rateLimit) {
          // Inputs and the last chart stay; the notice offers the retry.
          setRateLimit(e.rateLimit);
        } else {
          setData(null);
          setError(isApiError(e) ? e.detail || e.message : String(e));
        }
      })
      .finally(() => {
        if (seq === requestSeq.current) setLoading(false);
      });
  }, [requestKey]);

  useEffect(() => {
    if (!canRequest) {
      requestSeq.current += 1;
      setData(null);
      setLoading(false);
      setError(null);
      setRefusal(null);
      setRateLimit(null);
      return;
    }
    loadSeries();
  }, [canRequest, loadSeries]);

  const onLayout = useCallback((l: LayoutResult) => setLayout(l), []);
  const onSuggestion = useCallback((mode: "indexed" | "small-multiples") => setView(mode), [setView]);

  const applied = data?.limits.applied;
  const planMaxCompanies = applied?.max_companies ?? MAX_TICKERS;
  const planMaxMetrics = applied?.max_metrics ?? MAX_METRICS;
  const capped = data?.limits.capped_by_plan && state.years !== null ? { applied: data.limits.applied, requestedYears: state.years } : null;
  const commentaryEntitlement = account?.entitlements.chart_commentary ?? null;

  const samples = config.sample_tickers.length >= 2 ? config.sample_tickers.slice(0, 2) : ["AAPL", "MSFT"];
  const examples = [
    { label: `Revenue and gross margin — ${samples[0]} vs ${samples[1]}`, to: `/app/fundamentals?t=${samples.join(",")}&m=revenue,gross_margin` },
    { label: `Free cash flow before and after SBC — ${samples[0]}`, to: `/app/fundamentals?t=${samples[0]}&m=free_cash_flow,fcf_after_sbc` },
    { label: `Valuation at each fiscal year end — ${samples[0]} vs ${samples[1]}`, to: `/app/fundamentals?t=${samples.join(",")}&m=pe_ttm,fcf_yield&y=10` },
  ];

  const hasSeries = !!data && data.series.length > 0;
  const showChart = hasSeries || (!!data && data.unavailable.length === 0);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold">Fundamentals</h1>
        <p className="text-slate-400 text-sm mt-1 max-w-prose">
          Annual reported statements and the ratios derived from them, drawn exactly as stored — a missing value is shown as n/a with its
          reason, never as zero. Research and education only; not investment advice.
        </p>
      </div>

      {catalogError && (
        <div className="card-tight border-danger-500/40 text-sm" role="alert">
          <span className="text-danger-500">The metric catalog failed to load: {catalogError}</span>{" "}
          <button type="button" onClick={loadCatalog} className="underline text-accent-500">
            Retry
          </button>
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-[280px_minmax(0,1fr)]">
        <aside className="space-y-5" aria-label="Chart controls">
          <CompanySelector
            tickers={state.tickers}
            onChange={setTickers}
            universe={universe}
            universeLoading={universeLoading}
            max={planMaxCompanies}
            maxSource={applied ? "plan" : "absolute"}
          />
          <MetricPicker catalog={catalog?.metrics ?? null} selected={state.metrics} onChange={setMetrics} max={planMaxMetrics} maxSource={applied ? "plan" : "absolute"} />
          <TimeRangeControl years={state.years} onChange={setYears} appliedYears={applied ? applied.max_years : undefined} capped={!!data?.limits.capped_by_plan} />
          <ViewModeControl view={state.view} onChange={setView} availability={hasSeries ? layout?.availability : null} />
        </aside>

        <div className="space-y-4 min-w-0">
          {refusal && <EntitlementNotice refusal={refusal} onDismiss={() => setRefusal(null)} />}
          {capped && !refusal && <EntitlementNotice capped={capped} />}
          {rateLimit && (
            <RateLimitNotice
              refusal={rateLimit}
              onRetry={loadSeries}
              onDismiss={() => setRateLimit(null)}
              preservedNote="Your companies, metrics and range are unchanged — retry when the timer ends."
            />
          )}
          {error && (
            <div className="card-tight border-danger-500/40 text-sm" role="alert">
              <span className="text-danger-500">{error}</span>{" "}
              <button type="button" onClick={loadSeries} className="underline text-accent-500">
                Retry
              </button>
            </div>
          )}

          {state.tickers.length === 0 && (
            <div className="card" data-testid="fundamentals-empty">
              <div className="section-title mb-2">Start with a company</div>
              <p className="text-sm text-slate-300">
                Add up to {MAX_TICKERS} companies and up to {MAX_METRICS} metrics; the chart and its URL update together, so any view can be
                shared. Or open one of these:
              </p>
              <ul className="mt-3 space-y-1.5 text-sm">
                {examples.map((ex) => (
                  <li key={ex.to}>
                    <Link to={ex.to} className="text-accent-500 underline underline-offset-2">
                      {ex.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {state.tickers.length > 0 && state.metrics.length === 0 && !catalogError && (
            <p className="text-sm text-slate-400" data-testid="fundamentals-no-metrics">
              Pick at least one metric to draw {state.tickers.join(", ")}.
            </p>
          )}

          {loading && (
            <div className="card-tight text-sm text-slate-400" aria-busy="true" role="status" data-testid="fundamentals-loading">
              Loading series…
            </div>
          )}

          {data && (
            <SeriesStateNotice unavailable={data.unavailable} series={data.series} warnings={data.warnings} metricLabels={metricLabels} />
          )}

          {data && showChart && (
            <div className="card" aria-busy={loading || undefined}>
              <FundamentalsChart
                series={data.series}
                view={state.view}
                metricLabels={metricLabels}
                metricOrder={state.metrics}
                tickerOrder={state.tickers}
                narrow={narrow}
                onLayout={onLayout}
                onSuggestion={onSuggestion}
              />
            </div>
          )}

          {data && (
            <CommentaryPanel
              tickers={state.tickers}
              metrics={state.metrics}
              years={data.limits.applied.max_years}
              fingerprint={hasSeries && data.fingerprint ? data.fingerprint : null}
              entitlement={commentaryEntitlement}
              accountEnabled={accountEnabled}
            />
          )}
        </div>
      </div>
    </div>
  );
}
