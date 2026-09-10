// Thin fetch client for the MarketMosaic API.
// All endpoints proxied through Vite to the FastAPI backend.
//
// FEAT-002: when an auth provider is mounted it hands us `getToken`, and
// every request carries `Authorization: Bearer …` (header only — never a
// query param, which the backend persists to ui_logs). Structured refusals
// (402 entitlement, 429 rate limit, 409 no_memo, …) surface as `ApiError`
// with typed fields so pages can render a specific prompt, not "an error
// occurred". The backend authorises every call regardless of what the UI
// believes; nothing here is a security boundary.

import type {
  Account,
  AnalyzeJob,
  BillingInterval,
  BootstrapResponse,
  CatalogResponse,
  ChatResponse,
  CommentaryRequestWire,
  CommentaryResponse,
  CompanyOut,
  CompsResult,
  DCFAssumptions,
  DCFResult,
  EntitlementRefusal,
  MacroScenarioResult,
  ModelPortfolio,
  PortfolioRequest,
  ProvidersStatusResponse,
  RateLimitRefusal,
  ScorecardDetailWire,
  ScorecardEvaluationWire,
  ScorecardHistory,
  ScorecardSpecWire,
  ScorecardUniverseWire,
  ScreenerResult,
  SeriesRequest,
  SeriesResponseWire,
  StockMemoOut,
  StructuredErrorDetail,
  UsageResponse,
} from "@/types";
import { SCORECARD_EXPORT_CONTRACT, SCORECARD_FAMILIES } from "@/types/scorecard";
import { getSessionId, logEvent } from "@/lib/logger";
import { getAnonId } from "@/lib/analytics";

const BASE = (import.meta.env.VITE_BACKEND_URL as string | undefined) || "";

/** Fired on any 401 so RequireAuth can route to sign-in with a returnTo. */
export const AUTH_REQUIRED_EVENT = "mm:auth-required";
/** Fired on any 402 so `useAccount` refreshes the meters. */
export const ACCOUNT_REFRESH_EVENT = "mm:account-refresh";

export type TokenProvider = () => Promise<string | null>;

let tokenProvider: TokenProvider | null = null;

/** Installed by the auth provider; null clears it (sign-out, wall off). */
export function setTokenProvider(fn: TokenProvider | null): void {
  tokenProvider = fn;
}

export function hasTokenProvider(): boolean {
  return tokenProvider !== null;
}

/**
 * Error for any non-2xx response. `detail` stays a string for the many
 * call sites that render `e.detail || String(e)`; the structured object
 * (when the backend sent one) is on `structured`, with the two shapes the
 * UI cares about pre-narrowed on `entitlement` / `rateLimit`.
 */
export class ApiError extends Error {
  status: number;
  /** Human-readable detail: the backend `message` or the raw string detail. */
  detail?: string;
  code?: string;
  structured?: StructuredErrorDetail;
  entitlement?: EntitlementRefusal;
  rateLimit?: RateLimitRefusal;
  /** 409 `no_memo`: where to POST to queue a research run. */
  analyzePath?: string;

  constructor(status: number, text: string, statusText?: string) {
    super(`API ${status}: ${text || statusText || ""}`);
    this.name = "ApiError";
    this.status = status;
    let parsed: { detail?: unknown } = {};
    try {
      parsed = JSON.parse(text || "{}") as { detail?: unknown };
    } catch {}
    const d = parsed.detail;
    if (typeof d === "string") {
      this.detail = d;
    } else if (d && typeof d === "object") {
      const s = d as StructuredErrorDetail;
      this.structured = s;
      this.code = typeof s.code === "string" ? s.code : undefined;
      this.detail = s.message || this.code || text;
      if (this.code === "plan_required" || this.code === "quota_exceeded") {
        this.entitlement = {
          code: this.code,
          feature: s.feature ?? null,
          plan: s.plan ?? null,
          used: s.used ?? null,
          limit: s.limit ?? null,
          resets_at: s.resets_at ?? null,
          upgrade_url: s.upgrade_url || "/pricing",
          message: s.message || "",
        };
      }
      if (this.code === "rate_limited" || this.code === "concurrency_limited") {
        this.rateLimit = {
          code: this.code,
          scope: s.scope || "unknown",
          retry_after: Math.max(0, Number(s.retry_after ?? 0) || 0),
          window_seconds: s.window_seconds ?? null,
          message: s.message || "",
        };
      }
      if (this.code === "no_memo") {
        const ap = s.extra?.analyze_path ?? (s as unknown as { analyze_path?: unknown }).analyze_path;
        this.analyzePath = typeof ap === "string" ? ap : undefined;
      }
    } else if (text) {
      this.detail = text;
    }
  }
}

export function isApiError(e: unknown): e is ApiError {
  return e instanceof ApiError || (typeof e === "object" && e !== null && "status" in e && "message" in e);
}

interface Raw<T> {
  status: number;
  body: T;
  headers: Headers;
}

async function requestRaw<T>(path: string, init?: RequestInit): Promise<Raw<T>> {
  const method = (init?.method || "GET").toUpperCase();
  // Don't trace the trace endpoint — would self-recurse on every flush.
  const trace = !path.startsWith("/api/admin/ui-log");
  const started = performance.now();
  let res: Response | null = null;
  let errorMsg: string | undefined;

  const headers = new Headers({
    "Content-Type": "application/json",
    // Match the session header the backend middleware reads.
    "X-Session-Id": getSessionId(),
    "X-Anon-Id": getAnonId(),
  });
  if (init?.headers) new Headers(init.headers).forEach((v, k) => headers.set(k, v));
  if (tokenProvider) {
    let token: string | null = null;
    try {
      token = await tokenProvider();
    } catch {
      token = null;
    }
    if (token) headers.set("Authorization", `Bearer ${token}`);
  }

  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers });
  } catch (e) {
    errorMsg = (e as Error).message;
    throw e;
  } finally {
    if (trace) {
      const duration_ms = Math.round(performance.now() - started);
      logEvent({
        kind: "api_call",
        path,
        method,
        status_code: res?.status,
        duration_ms,
        payload: errorMsg ? { network_error: errorMsg } : {},
      });
    }
  }
  if (!res.ok) {
    const text = await res.text();
    const err = new ApiError(res.status, text, res.statusText);
    if (typeof window !== "undefined") {
      if (res.status === 401) {
        window.dispatchEvent(new CustomEvent(AUTH_REQUIRED_EVENT, { detail: { path } }));
      } else if (res.status === 402) {
        window.dispatchEvent(new CustomEvent(ACCOUNT_REFRESH_EVENT));
      }
    }
    throw err;
  }
  const body = res.status === 204 ? (undefined as unknown as T) : ((await res.json()) as T);
  return { status: res.status, body, headers: res.headers };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return (await requestRaw<T>(path, init)).body;
}

/** GET /memo can answer with the memo (200) or, under the login wall with
 *  `ondemand=true`, with a queued worker job (202). */
export type MemoFetchResult =
  | { kind: "memo"; memo: StockMemoOut; stale: boolean; staleReason: string | null }
  | { kind: "queued"; job: AnalyzeJob };

/**
 * FEAT-001: the backend serialises `limits.applied` as
 * `{companies, metrics, years}` (schemas/fundamentals.py `AppliedLimits`)
 * while a 402's `extra.limits` and the chart engine's TypeScript mirror use
 * `{max_companies, max_metrics, max_years}` (auth/features.py `Shape`).
 * Fold both spellings into the mirror here so every page and test sees one
 * shape. Nothing is invented: an absent count stays 0 and absent years stay
 * null (the full history).
 */
export function normaliseSeriesResponse(raw: unknown): SeriesResponseWire {
  const r = (raw ?? {}) as Record<string, unknown>;
  const limits = (r.limits ?? {}) as Record<string, unknown>;
  const a = (limits.applied ?? {}) as Record<string, unknown>;
  const num = (...vals: unknown[]): number => {
    for (const v of vals) if (typeof v === "number" && Number.isFinite(v)) return v;
    return 0;
  };
  const years = [a.max_years, a.years].find((v) => typeof v === "number") as number | undefined;
  return {
    ...(r as unknown as SeriesResponseWire),
    series: Array.isArray(r.series) ? (r.series as SeriesResponseWire["series"]) : [],
    unavailable: Array.isArray(r.unavailable) ? (r.unavailable as SeriesResponseWire["unavailable"]) : [],
    periods: Array.isArray(r.periods) ? (r.periods as string[]) : [],
    warnings: Array.isArray(r.warnings) ? (r.warnings as string[]) : [],
    fingerprint: typeof r.fingerprint === "string" ? r.fingerprint : "",
    limits: {
      applied: {
        max_companies: num(a.max_companies, a.companies),
        max_metrics: num(a.max_metrics, a.metrics),
        max_years: years ?? null,
      },
      capped_by_plan: limits.capped_by_plan === true,
    },
  };
}

// ---------------------------------------------------------------------------
// Phase 6 — Fundamental Factor Scorecard (browser-called reads under
// /api/scorecard; the admin refresh/evaluate/backfill endpoints are
// deliberately not here). The page never computes a score: every call is
// a read of what the worker persisted.
// ---------------------------------------------------------------------------

/** `sort_by` values `GET /api/scorecard` accepts (`scorecard_service.
 *  UNIVERSE_SORT_COLUMNS`). Anything else is a 422 there, so the client
 *  never sends a key outside this list — an unknown key is dropped and
 *  the backend's default (`overall_score`) applies. */
export const SCORECARD_UNIVERSE_SORT_KEYS = [
  "overall_score",
  "overall_z",
  "universe_percentile",
  "sector_percentile",
  "coverage",
  "ticker",
  ...SCORECARD_FAMILIES,
] as const;
export type ScorecardUniverseSortKey = (typeof SCORECARD_UNIVERSE_SORT_KEYS)[number];

export function isScorecardSortKey(key: string | null | undefined): key is ScorecardUniverseSortKey {
  return !!key && (SCORECARD_UNIVERSE_SORT_KEYS as readonly string[]).includes(key);
}

export interface ScorecardUniverseParams {
  as_of?: string;
  version?: string;
  sector?: string;
  sort_by?: string;
  order?: "asc" | "desc";
  /** 1–600 (the route's ceiling; the curated universe is 100–600 names). */
  limit?: number;
  min_coverage?: number;
}

export interface ScorecardExportParams {
  format?: "csv" | "json";
  version?: string;
  as_of?: string;
  /** JSON only: append feature_raw / feature_z objects. */
  include_features?: boolean;
}

/**
 * The export is a plain `<a href>` download under the FROZEN v1 column
 * contract. The URL carries no credential of any kind: the bearer (when
 * the wall is on) rides in a header the client attaches, and the
 * optional `SCORECARD_EXPORT_TOKEN` is for downstream systems to present
 * themselves — a token in a query string would land in ui_logs.
 */
export function scorecardExportUrl(params: ScorecardExportParams = {}): string {
  const q = new URLSearchParams();
  q.set("format", params.format ?? "csv");
  q.set("contract", SCORECARD_EXPORT_CONTRACT);
  if (params.version) q.set("version", params.version);
  if (params.as_of) q.set("as_of", params.as_of);
  if (params.include_features) q.set("include_features", "true");
  return `${BASE}/api/scorecard/export?${q.toString()}`;
}

/** The detail row embeds its month-end history (`ScorecardDetailOut.
 *  history`); there is no separate history route. Lift it into the shape
 *  the history chart draws, oldest first as the backend orders it. */
export function historyFromDetail(detail: ScorecardDetailWire): ScorecardHistory {
  return {
    ticker: detail.ticker,
    version_key: detail.version_key,
    points: Array.isArray(detail.history) ? detail.history : [],
  };
}

function stringList(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((s): s is string => typeof s === "string") : [];
}

/**
 * `ScorecardEvaluationOut` keys `evaluations` by kind and carries the
 * caveats once, at the top level; the evaluation component renders an
 * array with the caveats on each result. Fold the wire shape into that
 * without inventing anything: a result that carries its own `caveats`
 * keeps them, one that does not gets the response's list verbatim, and
 * the quintile bucket table is renamed from the backend's
 * `quantile_table` / `n_months` to the mirror's `quintile_table` / `n`.
 * Every other key (`reasons`, `interpretation`, `stats_note`, …) passes
 * through untouched so nothing the worker said is lost.
 */
export function normaliseEvaluationResponse(raw: unknown): ScorecardEvaluationWire {
  const r = (raw ?? {}) as Record<string, unknown>;
  const caveats = stringList(r.caveats);
  const wire = r.evaluations;
  const items: Array<Record<string, unknown>> = Array.isArray(wire)
    ? (wire as Array<Record<string, unknown>>)
    : wire && typeof wire === "object"
      ? Object.values(wire as Record<string, Record<string, unknown>>)
      : [];
  const evaluations = items
    .filter((item) => item && typeof item.kind === "string")
    .map((item) => {
      const result: Record<string, unknown> = { ...((item.result as Record<string, unknown> | undefined) ?? {}) };
      if (!Array.isArray(result.caveats)) result.caveats = caveats;
      if (item.kind === "quintile_ls") {
        if (!Array.isArray(result.months)) result.months = [];
        if (!Array.isArray(result.skipped_months)) result.skipped_months = [];
        if (!Array.isArray(result.quintile_table)) {
          const buckets = Array.isArray(result.quantile_table) ? (result.quantile_table as Array<Record<string, unknown>>) : [];
          result.quintile_table = buckets.map((b) => ({
            q: b.q,
            mean_ret: typeof b.mean_ret === "number" ? b.mean_ret : null,
            n: typeof b.n === "number" ? b.n : typeof b.n_months === "number" ? b.n_months : 0,
          }));
        }
        if (typeof result.n_months !== "number") result.n_months = (result.months as unknown[]).length;
      }
      return {
        kind: item.kind,
        created_at: typeof item.created_at === "string" ? item.created_at : "",
        sample_start: typeof item.sample_start === "string" ? item.sample_start : null,
        sample_end: typeof item.sample_end === "string" ? item.sample_end : null,
        n_obs: typeof item.n_obs === "number" ? item.n_obs : 0,
        params: (item.params && typeof item.params === "object" ? item.params : {}) as Record<string, unknown>,
        result,
      } as unknown as ScorecardEvaluationWire["evaluations"][number];
    });
  return {
    version_key: typeof r.version_key === "string" ? r.version_key : "",
    evaluations,
    caveats,
    note: typeof r.note === "string" ? r.note : "",
  };
}

export const api = {
  health: () => request<{ status: string; mode: string; llm_configured: boolean }>("/health"),

  // --- FEAT-001 fundamentals explorer ---------------------------------
  // Browser-called, outside /api/admin. The backend shapes each request to
  // the plan (402 `plan_required` names the exact limits) and meters
  // commentary. `years` omitted = the plan's default range, which the
  // backend does NOT report as a cap.
  fundamentalsCatalog: () => request<CatalogResponse>("/api/fundamentals/catalog"),
  fundamentalsSeries: async (req: SeriesRequest): Promise<SeriesResponseWire> => {
    const body: Record<string, unknown> = { tickers: req.tickers, metrics: req.metrics };
    if (typeof req.years === "number") body.years = req.years;
    if (req.normalize && req.normalize !== "none") body.normalize = req.normalize;
    const raw = await request<unknown>("/api/fundamentals/series", { method: "POST", body: JSON.stringify(body) });
    return normaliseSeriesResponse(raw);
  },
  /** Charged before the model call and released when it returns nothing;
   *  a degraded body (no LLM, anonymous visitor) costs nothing. */
  fundamentalsCommentary: (req: CommentaryRequestWire) =>
    request<CommentaryResponse>("/api/fundamentals/commentary", { method: "POST", body: JSON.stringify(req) }),
  providersStatus: () => request<ProvidersStatusResponse>("/api/providers/status"),

  // --- FEAT-002 account / billing -------------------------------------
  // `/api/public/config` is deliberately NOT here: everything on this
  // object carries the bearer, X-Anon-Id and X-Session-Id, and the public
  // config is a `Cache-Control: public` endpoint that must never see them.
  // Use `fetchPublicConfig` in auth/ConfigProvider (plan §6.3).
  me: () => request<Account>("/api/me"),
  /** Idempotent; starts the trial once server-side. */
  bootstrap: () => request<BootstrapResponse>("/api/me/bootstrap", { method: "POST", body: "{}" }),
  usage: (period?: string) =>
    request<UsageResponse>(`/api/me/usage${period ? `?period=${encodeURIComponent(period)}` : ""}`),
  /** Returns the Stripe-hosted Checkout URL. Landing on /app/billing/success
   *  afterwards proves nothing — only `me()` reflecting a subscription does. */
  checkout: (interval: BillingInterval) =>
    request<{ url: string }>("/api/billing/checkout", { method: "POST", body: JSON.stringify({ interval }) }),
  portal: () => request<{ url: string }>("/api/billing/portal", { method: "POST", body: "{}" }),
  reconcile: () => request<Account>("/api/billing/reconcile", { method: "POST", body: "{}" }),
  /** TrackRecord "Score now" — goes through the client so the bearer rides along. */
  evaluateOutcomes: () => request<unknown>("/api/admin/evaluate-outcomes", { method: "POST", body: "{}" }),

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
  /** Memo or queued job. Throws `ApiError` with `code === "no_memo"` (409)
   *  when nothing is stored and generation must go through a research run. */
  getStockMemo: async (ticker: string, opts?: { scenario?: string; ondemand?: boolean }): Promise<MemoFetchResult> => {
    const qs: string[] = [];
    if (opts?.scenario) qs.push(`scenario=${encodeURIComponent(opts.scenario)}`);
    if (opts?.ondemand) qs.push(`ondemand=true`);
    const suffix = qs.length ? `?${qs.join("&")}` : "";
    const raw = await requestRaw<StockMemoOut | AnalyzeJob>(`/api/stocks/${ticker}/memo${suffix}`);
    if (raw.status === 202) {
      return { kind: "queued", job: raw.body as AnalyzeJob };
    }
    return {
      kind: "memo",
      memo: raw.body as StockMemoOut,
      stale: raw.headers.get("X-Memo-Stale") === "true",
      staleReason: raw.headers.get("X-Memo-Stale-Reason"),
    };
  },
  /** Kick off an async memo regeneration. Returns 202 immediately with
   *  the job's `started_at`. Poll `analyzeStatus` to detect completion;
   *  when `latest_memo_at > started_at`, the new memo is ready and can
   *  be fetched via `getStockMemo`. Under the login wall this is the
   *  metered `research_run`; a 402 carries the entitlement refusal. */
  analyzeStock: (ticker: string) =>
    request<AnalyzeJob>(`/api/stocks/${ticker}/analyze`, { method: "POST", body: "{}" }),
  /** Poll for async memo regen completion. Returns the in-flight flag
   *  + the latest persisted memo's timestamp. `last_failure` carries
   *  the regen job's error telemetry when the most recent run failed
   *  (cleared on next success); `last_progress` is its step trace. */
  analyzeStatus: (ticker: string) =>
    request<{
      ticker: string;
      in_progress: boolean;
      started_at: string | null;
      latest_memo_at: string | null;
      latest_version: number | null;
      last_failure: {
        ticker: string;
        error_type: string;
        error_message: string;
        traceback_tail: string;
        started_at: string | null;
        failed_at: string | null;
        duration_seconds: number | null;
      } | null;
      last_progress: Array<{ step: string; at: string }>;
      job_id: number | null;
      job_status: "queued" | "running" | "succeeded" | "failed" | null;
    }>(`/api/stocks/${ticker}/analyze/status`),
  /** Synchronous escape hatch for dev — returns the memo inline. Will
   *  504 on Render (HTTP timeout ~100s) and is refused (403) under the
   *  login wall. Use `analyzeStock` + polling in production. */
  analyzeStockSync: (ticker: string) =>
    request<StockMemoOut>(`/api/stocks/${ticker}/analyze?sync=true`, {
      method: "POST",
      body: "{}",
    }),
  getStockPrices: (ticker: string, days = 252) =>
    request<Array<{ date: string; close: number; volume?: number }>>(
      `/api/stocks/${ticker}/prices?days=${days}`,
    ),

  screener: (params?: {
    theme?: string;
    sector?: string;
    sort_by?: string;
    order?: "asc" | "desc";
    limit?: number;
  }) => {
    const q = new URLSearchParams();
    if (params?.theme) q.set("theme", params.theme);
    if (params?.sector) q.set("sector", params.sector);
    if (params?.sort_by) q.set("sort_by", params.sort_by);
    if (params?.order) q.set("order", params.order);
    if (params?.limit) q.set("limit", String(params.limit));
    return request<ScreenerResult>(`/api/screener?${q.toString()}`);
  },

  customScreener: (req: import("@/types").CustomScreenRequest) =>
    request<import("@/types").CustomScreenResult>("/api/screener/custom", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  // --- Phase 6 scorecard reads ----------------------------------------
  /** The cross-section from the latest succeeded run. `sort_by` outside
   *  `SCORECARD_UNIVERSE_SORT_KEYS` is dropped rather than sent (422). */
  scorecardUniverse: (params: ScorecardUniverseParams = {}) => {
    const q = new URLSearchParams();
    if (params.as_of) q.set("as_of", params.as_of);
    if (params.version) q.set("version", params.version);
    if (params.sector) q.set("sector", params.sector);
    if (isScorecardSortKey(params.sort_by)) q.set("sort_by", params.sort_by);
    if (params.order) q.set("order", params.order);
    if (typeof params.limit === "number") q.set("limit", String(Math.max(1, Math.min(600, Math.round(params.limit)))));
    if (typeof params.min_coverage === "number") q.set("min_coverage", String(params.min_coverage));
    const qs = q.toString();
    return request<ScorecardUniverseWire>(`/api/scorecard${qs ? `?${qs}` : ""}`);
  },
  /** Latest score for one name with every feature's observed value and
   *  model read, plus `months` of month-end history. 404 when no
   *  succeeded run scored the ticker. */
  scorecard: (ticker: string, opts: { as_of?: string; version?: string; months?: number } = {}) => {
    const q = new URLSearchParams();
    if (opts.as_of) q.set("as_of", opts.as_of);
    if (opts.version) q.set("version", opts.version);
    if (typeof opts.months === "number") q.set("months", String(opts.months));
    const qs = q.toString();
    return request<ScorecardDetailWire>(`/api/scorecard/${encodeURIComponent(ticker.toUpperCase())}${qs ? `?${qs}` : ""}`);
  },
  /** Month-end history for the chart, lifted from the detail row. */
  scorecardHistory: async (ticker: string, months = 36): Promise<ScorecardHistory> =>
    historyFromDetail(await request<ScorecardDetailWire>(`/api/scorecard/${encodeURIComponent(ticker.toUpperCase())}?months=${months}`)),
  /** The methodology: families, features, normalisation, the score-scale
   *  caption and the version label the page shows. */
  scorecardSpec: (version?: string) =>
    request<ScorecardSpecWire>(`/api/scorecard/spec${version ? `?version=${encodeURIComponent(version)}` : ""}`),
  /** Latest evaluation per kind with the caveats folded onto each result. */
  scorecardEvaluation: async (opts: { version?: string; kind?: string } = {}): Promise<ScorecardEvaluationWire> => {
    const q = new URLSearchParams();
    if (opts.version) q.set("version", opts.version);
    if (opts.kind) q.set("kind", opts.kind);
    const qs = q.toString();
    return normaliseEvaluationResponse(await request<unknown>(`/api/scorecard/evaluation${qs ? `?${qs}` : ""}`));
  },
  /** Plain href for the frozen v1 export (see `scorecardExportUrl`). */
  scorecardExportUrl,

  chat: (message: string) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, history: [] }),
    }),

  dcfDefaults: (ticker: string) =>
    request<DCFAssumptions>(`/api/dcf/${ticker}/default-assumptions`),
  dcfConsensus: (ticker: string) =>
    request<{
      ticker: string;
      consensus_revenue_growth: number[] | null;
      trailing_op_margin: number | null;
      has_consensus: boolean;
    }>(`/api/dcf/${ticker}/consensus`),
  dcfSaved: (ticker: string) =>
    request<{
      has_saved: boolean;
      ticker: string;
      version?: number;
      trigger?: string;
      parent_version?: number | null;
      generated_at?: string;
      assumption_changes?: Array<{
        field: string;
        from: unknown;
        to: unknown;
        rationale: string;
      }>;
      assumptions?: DCFAssumptions;
    }>(`/api/dcf/${ticker}/saved`),
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
