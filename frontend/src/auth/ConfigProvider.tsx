import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { FeatureAllowance, FeatureMatrix, FeatureMatrixEntry, PublicConfig } from "@/types";
import { CONFIG_TIMEOUT_MS, fetchPublicConfigJson } from "@/api/publicClient";

/**
 * Loads `GET /api/public/config` once at boot. The flags here drive UX
 * only (which provider to mount, which nav items show a lock); the backend
 * authorises every request regardless, so an unreachable config endpoint
 * falls back to "auth off" defaults rather than blocking the page. A 2s
 * timeout keeps a slow backend from turning into a blank screen.
 *
 * The fetch itself lives in `api/publicClient.ts` with the other token-free
 * calls: the endpoint is public and `Cache-Control: public`, so it must
 * never carry a bearer, the anon id or the session id (plan §6.3). This
 * module only coerces the body into `PublicConfig`.
 *
 * `trial_days` and `features` have NO fallback numbers on purpose: copy
 * that mentions an allowance renders without the number rather than
 * promising something the backend may not enforce.
 */
export const DEFAULT_CONFIG: PublicConfig = {
  auth_enabled: false,
  billing_enabled: false,
  usage_limits_enabled: false,
  clerk_publishable_key: null,
  clerk_frontend_api: null,
  sample_tickers: [],
  prices: { monthly_cents: 2999, annual_cents: 29900, currency: "usd" },
  legal_reviewed: false,
  app_env: "development",
  trial_days: null,
  features: {},
};

function coerceAllowance(v: unknown): FeatureAllowance | undefined {
  if (v === null || typeof v === "boolean") return v;
  if (typeof v === "number" && Number.isFinite(v)) return Math.trunc(v);
  if (v === "follows_memo") return v;
  return undefined;
}

/** Keep only well-formed entries; a malformed row is dropped rather than
 *  rendered as a nonsense allowance. */
export function coerceFeatures(raw: unknown): FeatureMatrix {
  if (!raw || typeof raw !== "object") return {};
  const out: FeatureMatrix = {};
  for (const [name, entry] of Object.entries(raw as Record<string, unknown>)) {
    if (!entry || typeof entry !== "object") continue;
    const e = entry as Partial<Record<keyof FeatureMatrixEntry, unknown>>;
    const free = coerceAllowance(e.free);
    const pro = coerceAllowance(e.pro);
    if (free === undefined || pro === undefined) continue;
    out[name] = {
      description: typeof e.description === "string" ? e.description : "",
      free,
      pro,
      metered: e.metered === true,
      period: typeof e.period === "string" ? e.period : "month",
      distinct_resources: e.distinct_resources === true,
    };
  }
  return out;
}

export interface ConfigContextValue {
  config: PublicConfig;
  /** False until the fetch settled (success, failure or timeout). */
  loaded: boolean;
  /** True when we are running on defaults because the fetch failed. */
  fallback: boolean;
}

export const ConfigContext = createContext<ConfigContextValue>({
  config: DEFAULT_CONFIG,
  loaded: true,
  fallback: false,
});

export { CONFIG_TIMEOUT_MS };

function coerce(raw: unknown): PublicConfig {
  const r = (raw && typeof raw === "object" ? raw : {}) as Partial<PublicConfig>;
  return {
    ...DEFAULT_CONFIG,
    ...r,
    auth_enabled: r.auth_enabled === true,
    billing_enabled: r.billing_enabled === true,
    usage_limits_enabled: r.usage_limits_enabled === true,
    clerk_publishable_key: typeof r.clerk_publishable_key === "string" && r.clerk_publishable_key ? r.clerk_publishable_key : null,
    clerk_frontend_api: typeof r.clerk_frontend_api === "string" && r.clerk_frontend_api ? r.clerk_frontend_api : null,
    sample_tickers: Array.isArray(r.sample_tickers) ? r.sample_tickers.filter((t) => typeof t === "string") : [],
    prices: { ...DEFAULT_CONFIG.prices, ...(r.prices || {}) },
    trial_days: typeof r.trial_days === "number" && Number.isInteger(r.trial_days) && r.trial_days > 0 ? r.trial_days : null,
    features: coerceFeatures(r.features),
  };
}

export async function fetchPublicConfig(timeoutMs = CONFIG_TIMEOUT_MS): Promise<PublicConfig | null> {
  const raw = await fetchPublicConfigJson(timeoutMs);
  return raw === null ? null : coerce(raw);
}

interface Props {
  children: React.ReactNode;
  /** Tests and storybook-style callers: skip the fetch entirely. */
  initial?: Partial<PublicConfig>;
}

export function ConfigProvider({ children, initial }: Props) {
  const [state, setState] = useState<ConfigContextValue>(() =>
    initial
      ? { config: { ...DEFAULT_CONFIG, ...initial }, loaded: true, fallback: false }
      : { config: DEFAULT_CONFIG, loaded: false, fallback: false },
  );

  useEffect(() => {
    if (initial) return;
    let cancelled = false;
    void fetchPublicConfig().then((cfg) => {
      if (cancelled) return;
      setState(cfg ? { config: cfg, loaded: true, fallback: false } : { config: DEFAULT_CONFIG, loaded: true, fallback: true });
    });
    return () => {
      cancelled = true;
    };
  }, [initial]);

  const value = useMemo(() => state, [state]);
  return <ConfigContext.Provider value={value}>{children}</ConfigContext.Provider>;
}

export function useConfig(): ConfigContextValue {
  return useContext(ConfigContext);
}
