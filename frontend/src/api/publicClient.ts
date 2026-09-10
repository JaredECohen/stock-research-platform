// FEAT-002 (S6) — the token-free client for `/api/public/*`.
//
// These responses are `Cache-Control: public`, which is exactly why they
// must never ride `api/client.ts`: a bearer, the anon id or the session id
// on a publicly cacheable GET is the kind of thing a shared cache would
// happily replay to the next visitor. So this module builds its own plain
// `fetch` calls with no headers beyond what the browser adds, and the
// authenticated client never learns these endpoints exist
// (`ConfigProvider.test.tsx` pins `"publicConfig" in api === false`).
//
// Every function resolves to `null` (or a typed failure) instead of
// throwing: the marketing pages degrade — "samples unavailable" — rather
// than blank out, and nothing here can cost money on the backend side.

import type { FundamentalsSeries, LedgerCell, LedgerColumn, LedgerStatus, SamplePayload, SampleSummary } from "@/types/public";

const BASE = (import.meta.env.VITE_BACKEND_URL as string | undefined) || "";

/** Config fetch: 2s so a slow backend cannot become a blank screen. */
export const CONFIG_TIMEOUT_MS = 2000;
/** Samples carry a whole memo; give them a little longer. */
export const SAMPLE_TIMEOUT_MS = 8000;

/** Tickers are `[A-Z.-]`, at most 10 characters; anything else is never
 *  sent, so a crafted route param cannot become a request path. */
const TICKER_RE = /^[A-Z][A-Z0-9.-]{0,9}$/;

export function normalizeTicker(raw: string | null | undefined): string | null {
  const t = (raw || "").trim().toUpperCase();
  return TICKER_RE.test(t) ? t : null;
}

async function publicGet(path: string, timeoutMs: number): Promise<Response | null> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    // No `headers` at all — see the module comment.
    return await fetch(`${BASE}${path}`, { method: "GET", signal: controller.signal });
  } catch {
    return null;
  } finally {
    window.clearTimeout(timer);
  }
}

/** Raw JSON from `GET /api/public/config`, or null on any failure
 *  (non-2xx, network, timeout, malformed body). `auth/ConfigProvider`
 *  owns the coercion into `PublicConfig` and the safe defaults. */
export async function fetchPublicConfigJson(timeoutMs = CONFIG_TIMEOUT_MS): Promise<unknown | null> {
  const res = await publicGet("/api/public/config", timeoutMs);
  if (!res || !res.ok) return null;
  try {
    return await res.json();
  } catch {
    return null;
  }
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === "object" && !Array.isArray(v);
}

function coerceSummary(v: unknown): SampleSummary | null {
  if (!isRecord(v) || typeof v.ticker !== "string") return null;
  return {
    ticker: v.ticker,
    company_name: typeof v.company_name === "string" ? v.company_name : null,
    sector: typeof v.sector === "string" ? v.sector : null,
    built_at: typeof v.built_at === "string" ? v.built_at : null,
    kinds: Array.isArray(v.kinds) ? v.kinds.filter((k): k is string => typeof k === "string") : [],
  };
}

/** `GET /api/public/samples` — every allowlisted ticker, built or not.
 *  Null when the endpoint is unreachable; an empty list when the
 *  allowlist is empty. */
export async function listSamples(): Promise<SampleSummary[] | null> {
  const res = await publicGet("/api/public/samples", SAMPLE_TIMEOUT_MS);
  if (!res || !res.ok) return null;
  try {
    const body: unknown = await res.json();
    if (!Array.isArray(body)) return null;
    return body.map(coerceSummary).filter((s): s is SampleSummary => s !== null);
  } catch {
    return null;
  }
}

export type SampleResult =
  | { status: "ok"; sample: SamplePayload }
  | { status: "not_found"; sampleTickers: string[] }
  | { status: "unavailable" };

const EMPTY_LEDGER_REASON = "no stored memo";

/** Fill in anything the backend might omit so components can rely on
 *  the shape without a null check per field. */
function coerceSample(v: unknown, ticker: string): SamplePayload | null {
  if (!isRecord(v)) return null;
  const strings = (x: unknown): string[] => (Array.isArray(x) ? x.filter((s): s is string => typeof s === "string") : []);
  const column = (x: unknown): LedgerColumn => {
    if (isRecord(x) && (x.status === "available" || x.status === "not_captured" || x.status === "n/a")) {
      return {
        status: x.status as LedgerStatus,
        items: Array.isArray(x.items) ? (x.items as LedgerCell[]) : [],
        reason: typeof x.reason === "string" ? x.reason : null,
      };
    }
    return { status: "n/a" as const, items: [], reason: EMPTY_LEDGER_REASON };
  };
  const ledger = isRecord(v.expectations_ledger) ? v.expectations_ledger : {};
  const pricesRaw = Array.isArray(v.prices) ? v.prices : null;
  return {
    ticker: typeof v.ticker === "string" ? v.ticker : ticker,
    company_name: typeof v.company_name === "string" ? v.company_name : null,
    sector: typeof v.sector === "string" ? v.sector : null,
    built_at: typeof v.built_at === "string" ? v.built_at : null,
    memo: isRecord(v.memo) ? (v.memo as unknown as SamplePayload["memo"]) : null,
    dcf: isRecord(v.dcf) ? (v.dcf as unknown as SamplePayload["dcf"]) : null,
    comps: isRecord(v.comps) ? (v.comps as unknown as SamplePayload["comps"]) : null,
    fundamentals:
      isRecord(v.fundamentals) && Array.isArray(v.fundamentals.series)
        ? { series: v.fundamentals.series as FundamentalsSeries[] }
        : null,
    prices: pricesRaw
      ? pricesRaw
          .filter((p): p is { date: string; close: number } => isRecord(p) && typeof p.date === "string" && typeof p.close === "number")
          .map((p) => ({ date: p.date, close: p.close }))
      : null,
    screener_row: isRecord(v.screener_row) ? (v.screener_row as unknown as SamplePayload["screener_row"]) : null,
    commentary:
      isRecord(v.commentary) && typeof v.commentary.text === "string"
        ? {
            text: v.commentary.text,
            generated_at: typeof v.commentary.generated_at === "string" ? v.commentary.generated_at : "",
            model: typeof v.commentary.model === "string" ? v.commentary.model : "",
          }
        : null,
    expectations_ledger: {
      columns: ["reported_consensus", "management_guidance", "price_implied", "our_forecast"],
      note: typeof ledger.note === "string" ? ledger.note : "",
      reported_consensus: column(ledger.reported_consensus),
      management_guidance: column(ledger.management_guidance),
      price_implied: column(ledger.price_implied),
      our_forecast: column(ledger.our_forecast),
    },
    kinds_built: strings(v.kinds_built),
    degraded: strings(v.degraded),
    disclosures: strings(v.disclosures),
  };
}

/** `GET /api/public/samples/{ticker}`. A 404 is a typed result (the
 *  backend lists which tickers are public) so the page can point at the
 *  real samples instead of showing a generic error. */
export async function getSample(rawTicker: string): Promise<SampleResult> {
  const ticker = normalizeTicker(rawTicker);
  if (!ticker) return { status: "not_found", sampleTickers: [] };
  const res = await publicGet(`/api/public/samples/${encodeURIComponent(ticker)}`, SAMPLE_TIMEOUT_MS);
  if (!res) return { status: "unavailable" };
  if (res.status === 404) {
    try {
      const body: unknown = await res.json();
      const detail = isRecord(body) && isRecord(body.detail) ? body.detail : {};
      const tickers = Array.isArray(detail.sample_tickers) ? detail.sample_tickers.filter((t): t is string => typeof t === "string") : [];
      return { status: "not_found", sampleTickers: tickers };
    } catch {
      return { status: "not_found", sampleTickers: [] };
    }
  }
  if (!res.ok) return { status: "unavailable" };
  try {
    const sample = coerceSample(await res.json(), ticker);
    return sample ? { status: "ok", sample } : { status: "unavailable" };
  } catch {
    return { status: "unavailable" };
  }
}
