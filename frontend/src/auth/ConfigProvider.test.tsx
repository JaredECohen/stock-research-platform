import { afterEach, describe, expect, it, vi } from "vitest";
import { api, setTokenProvider } from "@/api/client";
import { DEFAULT_CONFIG, coerceFeatures, fetchPublicConfig } from "@/auth/ConfigProvider";
import { featureMatrix, okJson, requestHeaders, stubFetch } from "@/test/providers";

const SERVER_CONFIG = {
  auth_enabled: true,
  billing_enabled: true,
  usage_limits_enabled: true,
  clerk_publishable_key: "pk_test_x",
  clerk_frontend_api: "https://clerk.example",
  sample_tickers: ["NVDA", "COST", "JPM"],
  prices: { monthly_cents: 2999, annual_cents: 29900, currency: "usd" },
  legal_reviewed: false,
  app_env: "test",
  trial_days: 14,
  features: featureMatrix({ pm_chat: { pro: 500 } }),
};

describe("fetchPublicConfig", () => {
  afterEach(() => {
    setTokenProvider(null);
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("is token-free: no bearer, anon id or session id even with a token provider installed", async () => {
    // Plan §6.3 — public, `Cache-Control: public` endpoints never ride the
    // authenticated client. A bearer on a publicly cacheable response is
    // the kind of thing a shared cache would happily replay.
    setTokenProvider(async () => "stub-token");
    const mock = stubFetch([["/api/public/config", () => okJson(SERVER_CONFIG)]]);
    const cfg = await fetchPublicConfig();
    expect(cfg?.auth_enabled).toBe(true);
    const [, init] = mock.mock.calls.find(([u]) => String(u).includes("/api/public/config"))!;
    const headers = requestHeaders(init);
    expect(headers.get("Authorization")).toBeNull();
    expect(headers.get("X-Anon-Id")).toBeNull();
    expect(headers.get("X-Session-Id")).toBeNull();
    // And the authenticated client does not expose it at all.
    expect("publicConfig" in api).toBe(false);
  });

  it("carries trial_days and the feature matrix through, overrides included", async () => {
    stubFetch([["/api/public/config", () => okJson(SERVER_CONFIG)]]);
    const cfg = await fetchPublicConfig();
    expect(cfg?.trial_days).toBe(14);
    expect(cfg?.features.pm_chat.pro).toBe(500);
    expect(cfg?.features.memo_view).toEqual(expect.objectContaining({ free: 3, pro: null, distinct_resources: true }));
    expect(cfg?.features.dcf.free).toBe("follows_memo");
  });

  it("has no fallback numbers: defaults carry null trial_days and an empty matrix", async () => {
    expect(DEFAULT_CONFIG.trial_days).toBeNull();
    expect(DEFAULT_CONFIG.features).toEqual({});
    stubFetch([["/api/public/config", () => okJson({ auth_enabled: true, trial_days: "7", features: "nope" })]]);
    const cfg = await fetchPublicConfig();
    expect(cfg?.trial_days).toBeNull();
    expect(cfg?.features).toEqual({});
  });

  it("drops malformed matrix rows and keeps well-formed ones", () => {
    const out = coerceFeatures({
      research_run: { description: "x", free: 1, pro: 20, metered: true, period: "month", distinct_resources: false },
      bogus: { free: "sometimes", pro: 20 },
      also_bogus: "string",
      pm_chat: { free: 10, pro: 300.7 },
    });
    expect(Object.keys(out).sort()).toEqual(["pm_chat", "research_run"]);
    expect(out.pm_chat).toEqual({ description: "", free: 10, pro: 300, metered: false, period: "month", distinct_resources: false });
  });

  it("returns null on a non-2xx and on timeout so the app falls back to auth-off defaults", async () => {
    stubFetch([["/api/public/config", () => okJson({}, 503)]]);
    expect(await fetchPublicConfig()).toBeNull();
    vi.unstubAllGlobals();

    vi.useFakeTimers();
    const mock = vi.fn((_input: RequestInfo | URL, init?: RequestInit) =>
      new Promise<Response>((_, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }),
    );
    vi.stubGlobal("fetch", mock);
    const pending = fetchPublicConfig(50);
    await vi.advanceTimersByTimeAsync(60);
    expect(await pending).toBeNull();
  });
});
