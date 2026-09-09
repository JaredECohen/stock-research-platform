import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ACCOUNT_REFRESH_EVENT, AUTH_REQUIRED_EVENT, ApiError, api, setTokenProvider } from "@/api/client";
import { resetAnalyticsForTests } from "@/lib/analytics";
import { calls, errJson, okJson, requestHeaders, stubFetch } from "@/test/providers";

describe("api client auth headers", () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    resetAnalyticsForTests();
  });
  afterEach(() => {
    setTokenProvider(null);
    vi.unstubAllGlobals();
  });

  it("sends no Authorization header when no provider is installed", async () => {
    const mock = stubFetch([["/api/me", () => okJson({ ok: true })]]);
    await api.me();
    const [, init] = calls(mock, "/api/me")[0];
    const h = requestHeaders(init);
    expect(h.get("authorization")).toBeNull();
    expect(h.get("x-session-id")).toBeTruthy();
    expect(h.get("x-anon-id")).toBeTruthy();
  });

  it("adds a bearer header from the token provider, in the header only", async () => {
    const mock = stubFetch([["/api/me", () => okJson({ ok: true })]]);
    setTokenProvider(async () => "tok-123");
    await api.me();
    const [url, init] = calls(mock, "/api/me")[0];
    expect(requestHeaders(init).get("authorization")).toBe("Bearer tok-123");
    expect(String(url)).not.toContain("tok-123");
  });

  it("omits the header when the provider yields null or throws", async () => {
    const mock = stubFetch([["/api/me", () => okJson({ ok: true })]]);
    setTokenProvider(async () => null);
    await api.me();
    setTokenProvider(async () => {
      throw new Error("no session");
    });
    await api.me();
    for (const [, init] of calls(mock, "/api/me")) {
      expect(requestHeaders(init).get("authorization")).toBeNull();
    }
  });

  it("keeps the anon id stable across requests", async () => {
    const mock = stubFetch([["/api/me", () => okJson({ ok: true })]]);
    await api.me();
    await api.me();
    const ids = calls(mock, "/api/me").map(([, init]) => requestHeaders(init).get("x-anon-id"));
    expect(ids[0]).toBe(ids[1]);
    expect(localStorage.getItem("mm_anon_id")).toBe(ids[0]);
  });
});

describe("api client structured errors", () => {
  afterEach(() => {
    setTokenProvider(null);
    vi.unstubAllGlobals();
  });

  it("parses a 402 quota_exceeded body into ApiError.entitlement and asks for an account refresh", async () => {
    stubFetch([
      [
        "/api/stocks/NVDA/analyze",
        () =>
          errJson(402, {
            code: "quota_exceeded",
            feature: "research_run",
            plan: "free",
            used: 1,
            limit: 1,
            resets_at: "2026-10-01T00:00:00",
            upgrade_url: "/pricing",
            message: "Free Explorer includes 1 full research run per month.",
          }),
      ],
    ]);
    const refresh = vi.fn();
    window.addEventListener(ACCOUNT_REFRESH_EVENT, refresh);
    const err = await api.analyzeStock("NVDA").catch((e: unknown) => e);
    window.removeEventListener(ACCOUNT_REFRESH_EVENT, refresh);
    expect(err).toBeInstanceOf(ApiError);
    const e = err as ApiError;
    expect(e.status).toBe(402);
    expect(e.code).toBe("quota_exceeded");
    expect(e.entitlement).toEqual({
      code: "quota_exceeded",
      feature: "research_run",
      plan: "free",
      used: 1,
      limit: 1,
      resets_at: "2026-10-01T00:00:00",
      upgrade_url: "/pricing",
      message: "Free Explorer includes 1 full research run per month.",
    });
    expect(e.rateLimit).toBeUndefined();
    // Legacy call sites render `e.detail || String(e)`: still a string.
    expect(e.detail).toBe("Free Explorer includes 1 full research run per month.");
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it("parses a 402 plan_required body", async () => {
    stubFetch([["/api/portfolio/build", () => errJson(402, { code: "plan_required", feature: "portfolio", plan: "free", upgrade_url: "/pricing" })]]);
    const e = (await api.buildPortfolio({} as never).catch((x: unknown) => x)) as ApiError;
    expect(e.entitlement?.code).toBe("plan_required");
    expect(e.entitlement?.feature).toBe("portfolio");
    expect(e.entitlement?.used).toBeNull();
  });

  it("parses a 429 rate_limited body into ApiError.rateLimit", async () => {
    stubFetch([
      [
        "/api/chat",
        () =>
          errJson(429, { code: "rate_limited", scope: "user:llm_light", retry_after: 17, window_seconds: 60, message: "slow down" }, { "Retry-After": "17" }),
      ],
    ]);
    const e = (await api.chat("hi").catch((x: unknown) => x)) as ApiError;
    expect(e.status).toBe(429);
    expect(e.rateLimit).toEqual({ code: "rate_limited", scope: "user:llm_light", retry_after: 17, window_seconds: 60, message: "slow down" });
    expect(e.entitlement).toBeUndefined();
  });

  it("parses a slowapi-shaped 429 (scope ip) the same way", async () => {
    stubFetch([["/api/chat", () => errJson(429, { code: "rate_limited", scope: "ip", retry_after: 5, window_seconds: 60, limit: 60, message: "Too many" })]]);
    const e = (await api.chat("hi").catch((x: unknown) => x)) as ApiError;
    expect(e.rateLimit?.scope).toBe("ip");
    expect(e.rateLimit?.retry_after).toBe(5);
  });

  it("parses a 409 no_memo body and exposes analyze_path", async () => {
    stubFetch([["/api/stocks/NVDA/memo", () => errJson(409, { code: "no_memo", message: "no memo stored", extra: { analyze_path: "/api/stocks/NVDA/analyze" } })]]);
    const e = (await api.getStockMemo("NVDA").catch((x: unknown) => x)) as ApiError;
    expect(e.status).toBe(409);
    expect(e.code).toBe("no_memo");
    expect(e.analyzePath).toBe("/api/stocks/NVDA/analyze");
  });

  it("returns a queued job for a 202 memo response and a memo for 200", async () => {
    stubFetch([
      [
        "/api/stocks/NVDA/memo?ondemand=true",
        () => ({ ...okJson({ ticker: "NVDA", status: "started", started_at: "2026-09-08T00:00:00", job_id: 7 }, 202) }),
      ],
      [
        "/api/stocks/NVDA/memo",
        () => ({
          ...okJson({ ticker: "NVDA" }),
          headers: new Headers({ "X-Memo-Stale": "true", "X-Memo-Stale-Reason": "earnings" }),
        }),
      ],
    ]);
    const queued = await api.getStockMemo("NVDA", { ondemand: true });
    expect(queued.kind).toBe("queued");
    if (queued.kind === "queued") expect(queued.job.job_id).toBe(7);
    const memo = await api.getStockMemo("NVDA");
    expect(memo.kind).toBe("memo");
    if (memo.kind === "memo") {
      expect(memo.stale).toBe(true);
      expect(memo.staleReason).toBe("earnings");
    }
  });

  it("keeps a plain-string detail as before", async () => {
    stubFetch([["/api/stocks/XYZ/memo", () => errJson(409, "ticker is data_only; pass ondemand=true")]]);
    const e = (await api.getStockMemo("XYZ").catch((x: unknown) => x)) as ApiError;
    expect(e.status).toBe(409);
    expect(e.code).toBeUndefined();
    expect(e.detail).toBe("ticker is data_only; pass ondemand=true");
  });

  it("dispatches mm:auth-required on a 401", async () => {
    stubFetch([["/api/me", () => errJson(401, { code: "auth_required", message: "sign in" })]]);
    const handler = vi.fn();
    window.addEventListener(AUTH_REQUIRED_EVENT, handler);
    const e = (await api.me().catch((x: unknown) => x)) as ApiError;
    window.removeEventListener(AUTH_REQUIRED_EVENT, handler);
    expect(e.status).toBe(401);
    expect(e.code).toBe("auth_required");
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("does not dispatch the auth event on other statuses", async () => {
    stubFetch([["/api/me", () => errJson(503, { code: "auth_unavailable" })]]);
    const handler = vi.fn();
    window.addEventListener(AUTH_REQUIRED_EVENT, handler);
    await api.me().catch(() => {});
    window.removeEventListener(AUTH_REQUIRED_EVENT, handler);
    expect(handler).not.toHaveBeenCalled();
  });

  it("posts checkout with the interval and evaluate-outcomes with a bearer", async () => {
    const mock = stubFetch([
      ["/api/billing/checkout", () => okJson({ url: "https://checkout.stripe.com/x" })],
      ["/api/admin/evaluate-outcomes", () => okJson({ evaluated: 0 })],
    ]);
    setTokenProvider(async () => "tok");
    const { url } = await api.checkout("year");
    expect(url).toContain("stripe");
    const [, checkoutInit] = calls(mock, "/api/billing/checkout")[0];
    expect(JSON.parse(String(checkoutInit?.body))).toEqual({ interval: "year" });
    await api.evaluateOutcomes();
    const [, evalInit] = calls(mock, "/api/admin/evaluate-outcomes")[0];
    expect(evalInit?.method).toBe("POST");
    expect(requestHeaders(evalInit).get("authorization")).toBe("Bearer tok");
  });
});
