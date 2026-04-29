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
    throw new Error(`API ${res.status}: ${text || res.statusText}`);
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
  getStockMemo: (ticker: string, scenario?: string) =>
    request<StockMemoOut>(
      `/api/stocks/${ticker}/memo${scenario ? `?scenario=${encodeURIComponent(scenario)}` : ""}`,
    ),
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
};
