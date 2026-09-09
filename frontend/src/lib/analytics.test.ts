import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushAnalytics, getAnonId, resetAnalyticsForTests, track } from "@/lib/analytics";
import { calls, okJson, stubFetch } from "@/test/providers";

describe("analytics", () => {
  beforeEach(() => {
    localStorage.clear();
    resetAnalyticsForTests();
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    resetAnalyticsForTests();
  });

  it("keeps one anon id per browser in localStorage", () => {
    const a = getAnonId();
    expect(localStorage.getItem("mm_anon_id")).toBe(a);
    resetAnalyticsForTests();
    localStorage.setItem("mm_anon_id", a);
    expect(getAnonId()).toBe(a);
  });

  it("drops names outside the allowlist and never sends them", async () => {
    const mock = stubFetch([["/api/public/events", () => okJson({ accepted: 0 })]]);
    expect(track("not_a_real_event")).toBe(false);
    expect(track("password_typed", { path: "/x" })).toBe(false);
    await vi.advanceTimersByTimeAsync(5000);
    expect(calls(mock, "/api/public/events")).toHaveLength(0);
  });

  it("batches allowlisted events into one POST with the anon id and allowlisted props only", async () => {
    const mock = stubFetch([["/api/public/events", () => okJson({ accepted: 2 })]]);
    expect(track("pricing_view", { page: "/pricing", secret: "nope", ticker: "NVDA" })).toBe(true);
    expect(track("checkout_started", { interval: "month" })).toBe(true);
    expect(calls(mock, "/api/public/events")).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(2100);
    const posts = calls(mock, "/api/public/events");
    expect(posts).toHaveLength(1);
    const [, init] = posts[0];
    expect(new Headers(init?.headers).get("x-anon-id")).toBe(getAnonId());
    const body = JSON.parse(String(init?.body)) as { events: Array<{ name: string; props: Record<string, unknown> }> };
    expect(body.events.map((e) => e.name)).toEqual(["pricing_view", "checkout_started"]);
    expect(body.events[0].props).toEqual({ page: "/pricing", ticker: "NVDA" });
  });

  it("caps a batch at 50 events per request", async () => {
    const mock = stubFetch([["/api/public/events", () => okJson({ accepted: 50 })]]);
    for (let i = 0; i < 60; i++) track("sample_interact", { kind: "x" });
    await flushAnalytics();
    await vi.advanceTimersByTimeAsync(3000);
    const posts = calls(mock, "/api/public/events");
    const sizes = posts.map(([, init]) => (JSON.parse(String(init?.body)) as { events: unknown[] }).events.length);
    expect(sizes.reduce((a, b) => a + b, 0)).toBe(60);
    expect(Math.max(...sizes)).toBeLessThanOrEqual(50);
  });

  it("swallows network failures", async () => {
    stubFetch([["/api/public/events", () => Promise.reject(new Error("down"))]]);
    track("landing_view");
    await expect(flushAnalytics()).resolves.toBeUndefined();
  });
});
