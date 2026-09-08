import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { PublicConfig } from "@/types";

/**
 * Loads `GET /api/public/config` once at boot. The flags here drive UX
 * only (which provider to mount, which nav items show a lock); the backend
 * authorises every request regardless, so an unreachable config endpoint
 * falls back to "auth off" defaults rather than blocking the page. A 2s
 * timeout keeps a slow backend from turning into a blank screen.
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
  trial_days: 7,
  features: {},
};

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

export const CONFIG_TIMEOUT_MS = 2000;
const BASE = (import.meta.env.VITE_BACKEND_URL as string | undefined) || "";

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
  };
}

export async function fetchPublicConfig(timeoutMs = CONFIG_TIMEOUT_MS): Promise<PublicConfig | null> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${BASE}/api/public/config`, { signal: controller.signal });
    if (!res.ok) return null;
    return coerce(await res.json());
  } catch {
    return null;
  } finally {
    window.clearTimeout(timer);
  }
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
