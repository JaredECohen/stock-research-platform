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
  IndustryChanges,
  IndustryCompanies,
  IndustryHistory,
  IndustryReport,
  IndustryTaxonomy,
  MacroScenarioResult,
  ModelPortfolio,
  PortfolioRequest,
  ProvidersStatusResponse,
  RateLimitRefusal,
  ScorecardDetail,
  ScorecardEvaluationOut,
  ScorecardEvaluationResponse,
  ScorecardHistory,
  ScorecardSpec,
  ScorecardUniverse,
  ScorecardUniverseRow,
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

/** Every API call's headers: JSON unless the caller says otherwise and
 *  the session / anon ids the backend middleware reads. Synchronous on
 *  purpose — the bearer is added by `fetchApi` and awaited only when a
 *  provider is installed, so with the wall off a request still reaches
 *  `fetch` in the same tick it was made (pollers assert on that). */
function baseHeaders(init?: RequestInit): Headers {
  const headers = new Headers({
    "Content-Type": "application/json",
    // Match the session header the backend middleware reads.
    "X-Session-Id": getSessionId(),
    "X-Anon-Id": getAnonId(),
  });
  if (init?.headers) new Headers(init.headers).forEach((v, k) => headers.set(k, v));
  return headers;
}

/** `RequestInit` plus the one behaviour a caller may opt out of. By
 *  default a 401 raises `AUTH_REQUIRED_EVENT`, which `RequireAuth` answers
 *  by signing the session out and routing to sign-in — right for a token
 *  the backend rejected, wrong for a route whose 401 is not a verdict on
 *  the session at all. The scorecard export on a deployment that sets
 *  `SCORECARD_EXPORT_TOKEN` is that route: it demands its own token and
 *  ignores the customer bearer (`routes_scorecard._check_export_token`),
 *  so its 401 must reach the page as an `ApiError` and nothing more. */
export interface FetchApiInit extends RequestInit {
  unauthorized?: "event" | "throw";
}

/** `fetch` with the shared headers, the bearer (header only, never a
 *  query param), the `api_call` trace and the refusal mapping (non-2xx →
 *  `ApiError`; 401 raises the auth-required event unless the caller
 *  opted out, 402 the account refresh). Returns the raw `Response` so a
 *  caller can read a body that is not JSON — the scorecard export streams
 *  CSV. */
async function fetchApi(path: string, init?: FetchApiInit): Promise<Response> {
  const { unauthorized = "event", ...requestInit } = init ?? {};
  const method = (requestInit.method || "GET").toUpperCase();
  // Don't trace the trace endpoint — would self-recurse on every flush.
  const trace = !path.startsWith("/api/admin/ui-log");
  const started = performance.now();
  let res: Response | null = null;
  let errorMsg: string | undefined;
  const headers = baseHeaders(requestInit);
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
    res = await fetch(`${BASE}${path}`, { ...requestInit, headers });
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
        if (unauthorized === "event") window.dispatchEvent(new CustomEvent(AUTH_REQUIRED_EVENT, { detail: { path } }));
      } else if (res.status === 402) {
        window.dispatchEvent(new CustomEvent(ACCOUNT_REFRESH_EVENT));
      }
    }
    throw err;
  }
  return res;
}

async function requestRaw<T>(path: string, init?: RequestInit): Promise<Raw<T>> {
  const res = await fetchApi(path, init);
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

/** `/api/scorecard/export?…&contract=v1`, BASE-relative. The query carries
 *  no credential of any kind: the bearer (when the wall is on) rides in a
 *  header the client attaches, and the optional `SCORECARD_EXPORT_TOKEN`
 *  is for downstream systems to present themselves — a token in a query
 *  string would land in ui_logs. */
export function scorecardExportPath(params: ScorecardExportParams = {}): string {
  const q = new URLSearchParams();
  q.set("format", params.format ?? "csv");
  q.set("contract", SCORECARD_EXPORT_CONTRACT);
  if (params.version) q.set("version", params.version);
  if (params.as_of) q.set("as_of", params.as_of);
  if (params.include_features) q.set("include_features", "true");
  return `/api/scorecard/export?${q.toString()}`;
}

/**
 * The plain `<a href>` for the export under the FROZEN v1 column
 * contract. Only usable when the login wall is off: a navigation cannot
 * carry the bearer, so under `AUTH_ENABLED` the page goes through
 * `scorecardExportDownload` instead.
 */
export function scorecardExportUrl(params: ScorecardExportParams = {}): string {
  return `${BASE}${scorecardExportPath(params)}`;
}

/** What `scorecardExportDownload` hands the page: the streamed bytes and
 *  the contract headers the route stamps on the response. */
export interface ScorecardExportFile {
  blob: Blob;
  filename: string;
  contract: string;
  version: string;
  as_of: string;
  run_id: string;
}

/** The `filename` of a `Content-Disposition: attachment` header, or the
 *  fallback. Only a bare basename is accepted — a header must never pick
 *  a path on the viewer's machine. */
export function filenameFromDisposition(header: string | null | undefined, fallback: string): string {
  const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(header || "");
  let name = "";
  if (m) {
    try {
      name = decodeURIComponent(m[1]).trim();
    } catch {
      name = m[1].trim();
    }
  }
  return name && !/[\\/]/.test(name) ? name : fallback;
}

/**
 * Fetch the export with the client's headers — the bearer under the wall,
 * the same structured refusals as every other read (402 `plan_required`
 * renders the upgrade prompt) — so the page can hand the viewer the bytes
 * as an object-URL download. The URL is the same credential-free one
 * `scorecardExportUrl` builds.
 *
 * A 401 here is NOT a session verdict: a deployment that sets
 * `SCORECARD_EXPORT_TOKEN` refuses every browser export with one, bearer
 * or not, because the route then wants that token alone. Raising the
 * auth-required event on it would have `RequireAuth` sign a valid session
 * out and bounce the viewer to sign-in, so the refusal is thrown to the
 * page instead, which renders it verbatim. A bearer that really has
 * expired is caught by the next ordinary read.
 */
export async function scorecardExportDownload(params: ScorecardExportParams = {}): Promise<ScorecardExportFile> {
  const format = params.format ?? "csv";
  const res = await fetchApi(scorecardExportPath(params), {
    headers: { Accept: format === "json" ? "application/json" : "text/csv" },
    unauthorized: "throw",
  });
  const blob = await res.blob();
  const version = res.headers.get("X-Scorecard-Version") || params.version || "";
  const asOf = res.headers.get("X-Scorecard-As-Of") || params.as_of || "";
  const fallback = `scorecard_${version || "active"}_${asOf || "latest"}.${format}`;
  return {
    blob,
    filename: filenameFromDisposition(res.headers.get("Content-Disposition"), fallback),
    contract: res.headers.get("X-Scorecard-Contract") || SCORECARD_EXPORT_CONTRACT,
    version,
    as_of: asOf,
    run_id: res.headers.get("X-Scorecard-Run-Id") || "",
  };
}

/**
 * `scorecard_service.universe_table` numbers every row it returns with
 * `enumerate(ordered, start=1)` — the names it could not score are
 * appended after the scored ones and numbered with them, so a name with
 * no overall arrives with a positional integer (166 of 170) where the
 * table wants "nothing to rank". A position beside an unscored overall
 * is not evidence of standing, and missing evidence renders as n/a, so
 * the client blanks it here: the table's null path prints
 * `n/a (unscored)` and sorts the row last in either direction. Scored
 * rows keep the wire value — the page requests the route's default
 * `sort_by=overall_score` descending, under which the position is the
 * rank. Nothing else in the row is touched.
 */
export function normaliseUniverseResponse(raw: ScorecardUniverse): ScorecardUniverse {
  const rows = Array.isArray(raw?.rows) ? raw.rows : [];
  const scored = (r: ScorecardUniverseRow): boolean => typeof r.overall_score === "number" && Number.isFinite(r.overall_score);
  return { ...raw, rows: rows.map((r) => (scored(r) ? r : { ...r, rank: null })) };
}

/** The detail row embeds its month-end history (`ScorecardDetailOut.
 *  history`); there is no separate history route. Lift it into the shape
 *  the history chart draws, oldest first as the backend orders it. */
export function historyFromDetail(detail: ScorecardDetail): ScorecardHistory {
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
 * caveats once at the top level as well as on every result (the worker
 * attaches `EVALUATION_CAVEATS` to each kind); the evaluation component
 * renders an array with the caveats read from each result. Fold the wire
 * shape into that without inventing anything: a result keeps its own
 * `caveats`, a result without any (defensive — the backend always writes
 * them) gets the response's list verbatim, and the quintile bucket table
 * is renamed from the backend's `quantile_table` / `n_months` to the
 * mirror's `quintile_table` / `n`. Every other key (`reasons`,
 * `interpretation`, `stats_note`, …) passes through untouched so nothing
 * the worker said is lost.
 */
export function normaliseEvaluationResponse(raw: ScorecardEvaluationOut | Record<string, unknown> | null | undefined): ScorecardEvaluationResponse {
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
      } as unknown as ScorecardEvaluationResponse["evaluations"][number];
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
   *  `SCORECARD_UNIVERSE_SORT_KEYS` is dropped rather than sent (422); an
   *  unscored name's positional `rank` is blanked (`normaliseUniverseResponse`). */
  scorecardUniverse: async (params: ScorecardUniverseParams = {}): Promise<ScorecardUniverse> => {
    const q = new URLSearchParams();
    if (params.as_of) q.set("as_of", params.as_of);
    if (params.version) q.set("version", params.version);
    if (params.sector) q.set("sector", params.sector);
    if (isScorecardSortKey(params.sort_by)) q.set("sort_by", params.sort_by);
    if (params.order) q.set("order", params.order);
    if (typeof params.limit === "number") q.set("limit", String(Math.max(1, Math.min(600, Math.round(params.limit)))));
    if (typeof params.min_coverage === "number") q.set("min_coverage", String(params.min_coverage));
    const qs = q.toString();
    return normaliseUniverseResponse(await request<ScorecardUniverse>(`/api/scorecard${qs ? `?${qs}` : ""}`));
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
    return request<ScorecardDetail>(`/api/scorecard/${encodeURIComponent(ticker.toUpperCase())}${qs ? `?${qs}` : ""}`);
  },
  /** Month-end history for the chart, lifted from the detail row. */
  scorecardHistory: async (ticker: string, months = 36): Promise<ScorecardHistory> =>
    historyFromDetail(await request<ScorecardDetail>(`/api/scorecard/${encodeURIComponent(ticker.toUpperCase())}?months=${months}`)),
  /** The methodology: families, features, normalisation, the score-scale
   *  caption and the version label the page shows. */
  scorecardSpec: (version?: string) =>
    request<ScorecardSpec>(`/api/scorecard/spec${version ? `?version=${encodeURIComponent(version)}` : ""}`),
  /** Latest evaluation per kind with the caveats folded onto each result. */
  scorecardEvaluation: async (opts: { version?: string; kind?: string } = {}): Promise<ScorecardEvaluationResponse> => {
    const q = new URLSearchParams();
    if (opts.version) q.set("version", opts.version);
    if (opts.kind) q.set("kind", opts.kind);
    const qs = q.toString();
    return normaliseEvaluationResponse(await request<ScorecardEvaluationOut>(`/api/scorecard/evaluation${qs ? `?${qs}` : ""}`));
  },
  /** Plain href for the frozen v1 export (wall off; see `scorecardExportUrl`). */
  scorecardExportUrl,
  /** The export fetched with the bearer and handed back as a file (wall on). */
  scorecardExport: scorecardExportDownload,

  // --- FEAT-003 Industry Analysis reads -------------------------------
  // Browser-called, outside /api/admin, and every one of them is a row
  // fetch: the weekly worker generates, the page only reads. There is
  // deliberately no regenerate/publish method here — a page view must
  // never queue work (the repo's "no expensive work in a page request"
  // rule, and the reason these routes can be public at all).
  /** Sectors → industry groups with per-group counts and a pointer at the
   *  latest edition. Answers in every configuration, including before the
   *  taxonomy is imported (503 with a remedy) and when the reports behind
   *  it are Pro — its `access.surfaces` block is how the UI learns that. */
  industryTaxonomy: () => request<IndustryTaxonomy>("/api/industries/taxonomy"),
  /** One edition: `latest` (the published one) or an edition number. */
  industryReport: (code: string, version: string | number = "latest") =>
    request<IndustryReport>(
      `/api/industries/${encodeURIComponent(code)}/report?version=${encodeURIComponent(String(version))}`,
    ),
  /** The group's classified membership with the price coverage of each
   *  name. `limit` caps the page; the response counts what it dropped. */
  industryCompanies: (code: string, limit?: number) =>
    request<IndustryCompanies>(
      `/api/industries/${encodeURIComponent(code)}/companies${typeof limit === "number" ? `?limit=${Math.max(1, Math.min(500, Math.round(limit)))}` : ""}`,
    ),
  /** Prior editions, newest first, metadata only. Pro under the wall. */
  industryHistory: (code: string, limit?: number) =>
    request<IndustryHistory>(
      `/api/industries/${encodeURIComponent(code)}/history${typeof limit === "number" ? `?limit=${Math.max(1, Math.min(104, Math.round(limit)))}` : ""}`,
    ),
  /** What moved between two editions. `from` defaults to the edition the
   *  target actually replaced (its parent), not `version - 1`. Pro. */
  industryChanges: (code: string, opts: { from?: number; to?: string | number } = {}) => {
    const q = new URLSearchParams();
    if (typeof opts.from === "number") q.set("from", String(opts.from));
    if (opts.to !== undefined) q.set("to", String(opts.to));
    const qs = q.toString();
    return request<IndustryChanges>(`/api/industries/${encodeURIComponent(code)}/changes${qs ? `?${qs}` : ""}`);
  },

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
