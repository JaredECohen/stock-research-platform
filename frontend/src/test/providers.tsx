// Test scaffolding for FEAT-002 components: a URL-keyed fetch stub with
// safe defaults for every endpoint the shell touches, account fixtures,
// and a render helper that mounts Config + Auth contexts around a
// MemoryRouter.
import React from "react";
import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { vi } from "vitest";
import { AuthContext, DISABLED_AUTH, type AuthContextValue } from "@/auth/AuthContext";
import { ConfigContext, DEFAULT_CONFIG } from "@/auth/ConfigProvider";
import type { Account, Entitlement, FeatureAllowance, FeatureMatrix, FeatureMatrixEntry, PublicConfig } from "@/types";

// Node ≥22 ships an experimental `localStorage` global that is `undefined`
// unless `--localstorage-file` is passed, and it shadows jsdom's. Install an
// in-memory Storage when that happens so code under test (anon id, drafts)
// behaves as it does in a browser. Configurable, so this is a plain define.
class MemoryStorage implements Storage {
  private m = new Map<string, string>();
  get length() {
    return this.m.size;
  }
  clear() {
    this.m.clear();
  }
  getItem(k: string) {
    return this.m.has(k) ? this.m.get(k)! : null;
  }
  key(i: number) {
    return Array.from(this.m.keys())[i] ?? null;
  }
  removeItem(k: string) {
    this.m.delete(k);
  }
  setItem(k: string, v: string) {
    this.m.set(k, String(v));
  }
}
if (typeof globalThis.localStorage === "undefined" || globalThis.localStorage === null) {
  Object.defineProperty(globalThis, "localStorage", { value: new MemoryStorage(), configurable: true, writable: true });
}

export type Responder = (url: string, init?: RequestInit) => Promise<Partial<Response>> | Partial<Response>;
/** A substring of the URL, or a RegExp for exact-ish matches. */
export type Matcher = string | RegExp;

export function okJson(payload: unknown, status = 200): Partial<Response> {
  return {
    ok: status < 400,
    status,
    headers: new Headers(),
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  };
}

export function errJson(status: number, detail: unknown, headers: Record<string, string> = {}): Partial<Response> {
  const body = JSON.stringify({ detail });
  return {
    ok: false,
    status,
    statusText: `status ${status}`,
    headers: new Headers(headers),
    json: async () => JSON.parse(body),
    text: async () => body,
  };
}

const HEALTHY_STATUS = {
  mode: "demo",
  providers: {},
  missing_api_keys: [],
  llm_configured: true,
  llm: { configured: true, degraded: false, degradation_reasons: [], breakers: {} },
  feature_flags: {},
};

/** Fetch stub keyed by URL matcher; first match wins. Unknown API URLs
 *  stay pending forever — the page under test keeps its loading state
 *  instead of crashing on an empty object or raising an unhandled
 *  rejection from a page this slice does not own. Telemetry endpoints
 *  (ui-log, analytics) get an empty 200 so flushes never fail a test. */
export function stubFetch(routes: Array<[Matcher, Responder]> = []) {
  const defaults: Array<[Matcher, Responder]> = [
    ["/api/providers/status", () => okJson(HEALTHY_STATUS)],
    [/\/api\/screener(\?|$)/, () => okJson({ theme: null, rows: [], generated_at: "", total: 0 })],
    [/\/api\/stocks$/, () => okJson([])],
  ];
  const all = [...routes, ...defaults];
  const mock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    for (const [needle, fn] of all) {
      const hit = typeof needle === "string" ? url.includes(needle) : needle.test(url);
      if (hit) return Promise.resolve(fn(url, init)) as Promise<Response>;
    }
    if (url.includes("/api/") && !url.includes("/api/admin/ui-log") && !url.includes("/api/public/events")) {
      return new Promise<Response>(() => {});
    }
    return Promise.resolve(okJson({}) as Response);
  });
  vi.stubGlobal("fetch", mock);
  return mock;
}

export function calls(mock: ReturnType<typeof stubFetch>, needle: string) {
  return mock.mock.calls.filter(([u]) => String(u).includes(needle));
}

export function requestHeaders(init?: RequestInit): Headers {
  return new Headers(init?.headers || {});
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** The default matrix from backend `auth/features.py` as
 *  `/api/public/config.features` serialises it. Tests that assert copy
 *  numbers pass this (or an override of it) in `config.features`. */
export function featureMatrix(over: Partial<Record<string, Partial<FeatureMatrixEntry>>> = {}): FeatureMatrix {
  const row = (description: string, free: FeatureAllowance, pro: FeatureAllowance, extra: Partial<FeatureMatrixEntry> = {}): FeatureMatrixEntry => ({
    description, free, pro, metered: false, period: "month", distinct_resources: false, ...extra,
  });
  const base: FeatureMatrix = {
    memo_view: row("Open a stored investment memo", 3, null, { metered: true, distinct_resources: true }),
    research_run: row("Run the full agent committee on a ticker", 1, 20, { metered: true }),
    pm_chat: row("Ask-the-PM chat turn (also macro analysis and the NL screener)", 10, 300, { metered: true }),
    chart_commentary: row("AI commentary on a chart (reserved for FEAT-001)", 5, 100, { metered: true }),
    fundamentals_explorer: row("Fundamentals explorer (reserved for FEAT-001)", true, true),
    dcf: row("DCF model on a ticker", "follows_memo", true),
    comps: row("Comparable-company table", "follows_memo", true),
    portfolio: row("Model portfolio builder", false, true),
    macro: row("Macro series and scenario analysis", false, true),
    track_record: row("Track record and outcome evaluation", false, true),
    memo_history: row("Memo version history, agent memory and DCF versions", false, true),
    data_catalog: row("Data catalog and sector overlays", false, true),
    scorecard: row("Fundamental factor scorecard", false, true),
  };
  for (const [name, patch] of Object.entries(over)) {
    base[name] = { ...base[name], ...(patch || {}) };
  }
  return base;
}

export function ent(feature: string, over: Partial<Entitlement> = {}): Entitlement {
  return { feature, allowed: true, limit: null, used: 0, remaining: null, resets_at: "2026-10-01T00:00:00", ...over };
}

export function makeAccount(over: Partial<Account> = {}): Account {
  return {
    user: {
      id: 1,
      external_id: "user_stub",
      email_verified: true,
      created_at: "2026-09-01T00:00:00",
      account_state: "active",
      trial_started_at: "2026-09-08T14:03:00",
      trial_ends_at: "2026-09-15T14:03:00",
    },
    plan: {
      plan: "pro",
      source: "trial",
      trial_ends_at: "2026-09-15T14:03:00",
      period_end: null,
      cancel_at_period_end: false,
      grace_until: null,
      warning: null,
    },
    entitlements: {
      memo_view: ent("memo_view", { metered: true }),
      research_run: ent("research_run", { limit: 20, used: 2, remaining: 18, metered: true }),
      pm_chat: ent("pm_chat", { limit: 300, used: 12, remaining: 288, metered: true }),
      dcf: ent("dcf"),
      comps: ent("comps"),
      portfolio: ent("portfolio"),
      macro: ent("macro"),
      track_record: ent("track_record"),
      memo_history: ent("memo_history"),
    },
    billing: { has_subscription: false, stripe_status: null, interval: null, portal_available: false, billing_enabled: true },
    period_key: "2026-09",
    usage_limits_enabled: true,
    ...over,
  };
}

export function freeAccount(over: Partial<Account> = {}): Account {
  return makeAccount({
    plan: { plan: "free", source: "default", trial_ends_at: null, period_end: null, cancel_at_period_end: false, grace_until: null, warning: null },
    entitlements: {
      memo_view: ent("memo_view", { limit: 3, used: 3, remaining: 0, metered: true }),
      research_run: ent("research_run", { limit: 1, used: 0, remaining: 1, metered: true }),
      pm_chat: ent("pm_chat", { limit: 10, used: 4, remaining: 6, metered: true }),
      dcf: ent("dcf", { follows_memo: true }),
      comps: ent("comps", { follows_memo: true }),
      portfolio: ent("portfolio", { allowed: false }),
      macro: ent("macro", { allowed: false }),
      track_record: ent("track_record", { allowed: false }),
      memo_history: ent("memo_history", { allowed: false }),
    },
    ...over,
  });
}

export const SIGNED_IN: AuthContextValue = {
  status: "signed_in",
  mode: "stub",
  user: { id: "user_stub", email: "stub@example.com", emailVerified: true },
  getToken: async () => "stub-token",
  signOut: async () => {},
  openSignIn: () => {},
};

export const SIGNED_OUT: AuthContextValue = { ...DISABLED_AUTH, status: "signed_out", mode: "stub" };

// ---------------------------------------------------------------------------
// Render helper
// ---------------------------------------------------------------------------

/** Prints the live location so router tests can assert on it. */
export function LocationSpy() {
  const loc = useLocation();
  return <div data-testid="location">{loc.pathname + loc.search + loc.hash}</div>;
}

interface Options {
  route?: string;
  config?: Partial<PublicConfig>;
  auth?: AuthContextValue;
  /** Wrap `ui` in a Routes tree at `path` (default: render `ui` directly). */
  path?: string;
}

export function renderWithProviders(ui: React.ReactElement, opts: Options = {}) {
  const config = { ...DEFAULT_CONFIG, ...(opts.config || {}) };
  const auth = opts.auth || DISABLED_AUTH;
  const body = opts.path ? (
    <Routes>
      <Route path={opts.path} element={ui} />
    </Routes>
  ) : (
    ui
  );
  return render(
    <MemoryRouter initialEntries={[opts.route || "/"]}>
      <ConfigContext.Provider value={{ config, loaded: true, fallback: false }}>
        <AuthContext.Provider value={auth}>
          <LocationSpy />
          {body}
        </AuthContext.Provider>
      </ConfigContext.Provider>
    </MemoryRouter>,
  );
}
