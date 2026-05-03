// Thin fetch client for the MarketMosaic API.
// All endpoints proxied through Vite to the FastAPI backend.

import type {
  ChatResponse,
  CompanyOut,
  CompsResult,
  DCFAssumptions,
  DCFResult,
  MacroScenarioResult,
  ModelPortfolio,
  PortfolioRequest,
  ProvidersStatusResponse,
  ScreenerResult,
  StockMemoOut,
} from "@/types";

const BASE = (import.meta.env.VITE_BACKEND_URL as string | undefined) || "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    // Tag 409 (data-only ticker requires ondemand) so callers can detect it
    // explicitly and render the gate instead of a generic error.
    const err = new Error(`API ${res.status}: ${text || res.statusText}`) as Error & {
      status?: number;
      detail?: string;
    };
    err.status = res.status;
    try {
      const parsed = JSON.parse(text || "{}") as { detail?: string };
      err.detail = parsed.detail;
    } catch {}
    throw err;
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string; mode: string; llm_configured: boolean }>("/health"),
  providersStatus: () => request<ProvidersStatusResponse>("/api/providers/status"),

  listStocks: () => request<CompanyOut[]>("/api/stocks"),
  getStock: (ticker: string) =>
    request<{
      profile: CompanyOut & {
        drivers?: string[];
        risks?: string[];
        segments?: string[];
        cik?: string;
      };
      ratios: Record<string, number | null>;
      income: Array<Record<string, number | null>>;
      balance: Array<Record<string, number | null>>;
      cash: Array<Record<string, number | null>>;
      earnings: Record<string, unknown>;
      market_stats: Record<string, number>;
    }>(`/api/stocks/${ticker}`),
  getStockMemo: (ticker: string, opts?: { scenario?: string; ondemand?: boolean }) => {
    const qs: string[] = [];
    if (opts?.scenario) qs.push(`scenario=${encodeURIComponent(opts.scenario)}`);
    if (opts?.ondemand) qs.push(`ondemand=true`);
    const suffix = qs.length ? `?${qs.join("&")}` : "";
    return request<StockMemoOut>(`/api/stocks/${ticker}/memo${suffix}`);
  },
  analyzeStock: (ticker: string) =>
    request<StockMemoOut>(`/api/stocks/${ticker}/analyze`, { method: "POST", body: "{}" }),
  getStockPrices: (ticker: string, days = 252) =>
    request<Array<{ date: string; close: number; volume?: number }>>(
      `/api/stocks/${ticker}/prices?days=${days}`,
    ),

  screener: (params?: { theme?: string; sector?: string; limit?: number }) => {
    const q = new URLSearchParams();
    if (params?.theme) q.set("theme", params.theme);
    if (params?.sector) q.set("sector", params.sector);
    if (params?.limit) q.set("limit", String(params.limit));
    return request<ScreenerResult>(`/api/screener?${q.toString()}`);
  },

  chat: (message: string) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, history: [] }),
    }),

  dcfDefaults: (ticker: string) =>
    request<DCFAssumptions>(`/api/dcf/${ticker}/default-assumptions`),
  runDCF: (ticker: string, assumptions: DCFAssumptions) =>
    request<DCFResult>(`/api/dcf/${ticker}`, { method: "POST", body: JSON.stringify(assumptions) }),

  comps: (ticker: string) => request<CompsResult>(`/api/comps/${ticker}`),

  buildPortfolio: (req: PortfolioRequest) =>
    request<ModelPortfolio>(`/api/portfolio/build`, {
      method: "POST",
      body: JSON.stringify(req),
    }),

  macroAnalyze: (scenario: string) =>
    request<MacroScenarioResult>(`/api/macro/analyze`, {
      method: "POST",
      body: JSON.stringify({ scenario }),
    }),
  macroSeries: (seriesId?: string) => {
    const q = seriesId ? `?series_id=${encodeURIComponent(seriesId)}` : "";
    return request<unknown>(`/api/macro/series${q}`);
  },

  // Wave 8D — historical / governance views.
  memoHistory: (ticker: string, limit = 25) =>
    request<
      Array<{
        version: number;
        trigger: string;
        parent_version: number | null;
        generated_at: string;
        revision_log: Array<{
          version: number;
          trigger: string;
          at?: string;
          parent_version?: number | null;
          fields_patched?: string[];
          rationales?: Record<string, string>;
          delta_summary?: string;
          critic_skipped?: boolean;
          alert?: { title?: string; severity?: string; source?: string };
        }>;
        rating_label: string | null;
        confidence_score: number | null;
      }>
    >(`/api/stocks/${ticker}/memos?limit=${limit}`),

  trackRecord: (params?: {
    horizon_days?: number;
    ticker?: string;
    sector?: string;
  }) => {
    const q = new URLSearchParams();
    if (params?.horizon_days) q.set("horizon_days", String(params.horizon_days));
    if (params?.ticker) q.set("ticker", params.ticker);
    if (params?.sector) q.set("sector", params.sector);
    return request<{
      horizon_days: number;
      total: number;
      directional_evaluations: number;
      thesis_hit_rate: number | null;
      avg_forward_return: number;
      avg_alpha: number | null;
      ticker_filter: string | null;
      sector_filter: string | null;
    }>(`/api/admin/track-record?${q.toString()}`);
  },

  dcfVersionHistory: (ticker: string, limit = 25) =>
    request<{
      ticker: string;
      versions: Array<{
        version: number;
        parent_version: number | null;
        trigger: string;
        generated_at: string;
        assumption_changes: Array<{
          field: string;
          from: unknown;
          to: unknown;
          rationale: string;
        }>;
        has_result: boolean;
      }>;
    }>(`/api/admin/dcf-versions/${ticker}?limit=${limit}`),

  stockMemory: (ticker: string, limit = 10) =>
    request<{
      ticker: string;
      path: string;
      entry_count: number;
      historical_context: string;
      entries: Array<{
        date: string;
        trigger: string;
        body: string;
        structured_facts: {
          sources?: Array<{
            source_kind: string;
            source_id: string;
            facts: Record<string, string[]>;
          }>;
          extractor_version?: number;
        } | null;
      }>;
    }>(`/api/stocks/${ticker}/memory?limit=${limit}`),

  lopsidednessAudit: (n = 10) =>
    request<{
      inspected: number;
      avg_bull_key_points: number;
      avg_bear_key_points: number;
      key_point_skew: number;
      sector_lean_counts: { bull: number; bear: number; balanced: number };
      lean_skew: number;
      avg_falsifiable_tests_per_memo: number;
      rows: Array<{
        ticker: string;
        version: number;
        rating: string | null;
        sector_lean: string;
        bull_kp: number;
        bear_kp: number;
        falsifiable_tests: number;
      }>;
    }>(`/api/admin/lopsidedness-audit?n=${n}`),
};
