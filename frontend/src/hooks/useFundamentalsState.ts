// FEAT-001 — the Fundamentals Explorer's state lives in the URL
// (`?t=AAPL,MSFT&m=revenue,gross_margin&y=10&v=dual-axis`) so a chart is
// shareable, refreshable, and logged by RouteTracker for free. This hook is
// the only reader and writer of those params: it parses leniently (case,
// duplicates, unknown metric ids once the catalog is known), clamps to the
// absolute request ceilings, and rewrites the URL to its canonical form
// exactly once when what it parsed differs from what was written.
//
// Plan limits are deliberately NOT applied here. The backend is the gate
// and reports the shape it applied; hiding part of a selection client-side
// would let the page silently disagree with a 402 that names the request.
import { useCallback, useEffect, useMemo } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { isViewMode, type ViewMode } from "@/types/fundamentals";

/** Absolute request ceilings (schemas/fundamentals.py MAX_TICKERS/MAX_METRICS). */
export const MAX_TICKERS = 5;
export const MAX_METRICS = 4;

/** `y=` values. `null` is "max": `years` is omitted from the request and
 *  the plan's full allowance applies — which the backend does not count as
 *  a cap. Only an explicit 5/10 above the plan ceiling is capped. */
export type YearsChoice = 5 | 10 | null;
export const YEAR_CHOICES: readonly YearsChoice[] = [5, 10, null];

export interface FundamentalsState {
  tickers: string[];
  metrics: string[];
  years: YearsChoice;
  view: ViewMode;
}

const OWN_KEYS = new Set(["t", "m", "y", "v"]);
const TICKER_RE = /^[A-Z0-9.\-]{1,12}$/;
const METRIC_RE = /^[a-z0-9_]{1,64}$/;

function splitList(raw: string | null): string[] {
  return (raw ?? "")
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

export function parseTickers(raw: string | null): string[] {
  const out: string[] = [];
  for (const item of splitList(raw)) {
    const t = item.toUpperCase();
    if (!TICKER_RE.test(t) || out.includes(t)) continue;
    out.push(t);
    if (out.length >= MAX_TICKERS) break;
  }
  return out;
}

/** Lower-cased, de-duplicated, clamped; filtered to `known` ids when the
 *  catalog is loaded (null = not yet known, keep everything plausible so
 *  the selection does not flicker before the catalog answers). */
export function parseMetrics(raw: string | null, known: ReadonlySet<string> | null): string[] {
  const out: string[] = [];
  for (const item of splitList(raw)) {
    const m = item.toLowerCase();
    if (!METRIC_RE.test(m) || out.includes(m)) continue;
    if (known && !known.has(m)) continue;
    out.push(m);
    if (out.length >= MAX_METRICS) break;
  }
  return out;
}

export function parseYears(raw: string | null): YearsChoice {
  return raw === "5" ? 5 : raw === "10" ? 10 : null;
}

export function parseView(raw: string | null): ViewMode {
  return isViewMode(raw) ? raw : "auto";
}

export function parseState(params: URLSearchParams, known: ReadonlySet<string> | null): FundamentalsState {
  return {
    tickers: parseTickers(params.get("t")),
    metrics: parseMetrics(params.get("m"), known),
    years: parseYears(params.get("y")),
    view: parseView(params.get("v")),
  };
}

/** `?t=AAPL,MSFT&m=revenue&y=10&v=table` — defaults are omitted so the
 *  shortest URL is the plain page. Commas stay literal (they are legal in a
 *  query string and the URL is meant to be read). Params this hook does not
 *  own are carried over unchanged. */
export function encodeState(state: FundamentalsState, base?: URLSearchParams): string {
  const pairs: Array<[string, string]> = [];
  if (state.tickers.length) pairs.push(["t", state.tickers.join(",")]);
  if (state.metrics.length) pairs.push(["m", state.metrics.join(",")]);
  if (state.years !== null) pairs.push(["y", String(state.years)]);
  if (state.view !== "auto") pairs.push(["v", state.view]);
  base?.forEach((v, k) => {
    if (!OWN_KEYS.has(k)) pairs.push([k, v]);
  });
  const q = pairs.map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v).replace(/%2C/gi, ",")}`).join("&");
  return q ? `?${q}` : "";
}

export interface UseFundamentalsState {
  state: FundamentalsState;
  /** Company and metric changes push history so Back undoes one step. */
  setTickers: (tickers: string[]) => void;
  setMetrics: (metrics: string[]) => void;
  /** Range and view are presentation: they replace the entry. */
  setYears: (years: YearsChoice) => void;
  setView: (view: ViewMode) => void;
}

export function useFundamentalsState(knownMetrics: ReadonlySet<string> | null): UseFundamentalsState {
  const [params] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();

  const state = useMemo(() => parseState(params, knownMetrics), [params, knownMetrics]);
  const canonical = useMemo(() => encodeState(state, params), [state, params]);

  // Normalise the URL once the catalog can vouch for metric ids. Comparing
  // the raw search string against the canonical encoding (not re-parsing)
  // is what makes this converge: after one replace they are byte-equal.
  useEffect(() => {
    if (!knownMetrics) return;
    if (location.search === canonical) return;
    navigate({ pathname: location.pathname, search: canonical, hash: location.hash }, { replace: true });
  }, [knownMetrics, canonical, location.pathname, location.search, location.hash, navigate]);

  const write = useCallback(
    (next: FundamentalsState, replace: boolean) => {
      navigate({ pathname: location.pathname, search: encodeState(next, params), hash: location.hash }, { replace });
    },
    [navigate, location.pathname, location.hash, params],
  );

  const setTickers = useCallback((tickers: string[]) => write({ ...state, tickers: parseTickers(tickers.join(",")) }, false), [state, write]);
  const setMetrics = useCallback(
    (metrics: string[]) => write({ ...state, metrics: parseMetrics(metrics.join(","), knownMetrics) }, false),
    [state, write, knownMetrics],
  );
  const setYears = useCallback((years: YearsChoice) => write({ ...state, years }, true), [state, write]);
  const setView = useCallback((view: ViewMode) => write({ ...state, view }, true), [state, write]);

  return { state, setTickers, setMetrics, setYears, setView };
}
