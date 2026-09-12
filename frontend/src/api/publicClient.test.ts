import { afterEach, describe, expect, it, vi } from "vitest";
import { setTokenProvider } from "@/api/client";
import { fetchPublicConfigJson, getSample, listSamples, normalizeTicker } from "@/api/publicClient";
import { errJson, okJson, requestHeaders, stubFetch } from "@/test/providers";
import { SAMPLE_LIST, SAMPLE_TICKERS, makeSample, sampleRoutes } from "@/test/fixtures/sample";

describe("publicClient", () => {
  afterEach(() => {
    setTokenProvider(null);
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("never attaches a bearer, anon id or session id to a public GET", async () => {
    // A token provider is installed (a signed-in session) and the anon id
    // exists in storage — none of it may reach a publicly cacheable URL.
    setTokenProvider(async () => "stub-token");
    localStorage.setItem("mm_anon_id", "anon-123");
    const mock = stubFetch([...sampleRoutes(), ["/api/public/config", () => okJson({ auth_enabled: true })]]);
    await fetchPublicConfigJson();
    await listSamples();
    await getSample("COST");
    const publicCalls = mock.mock.calls.filter(([u]) => String(u).includes("/api/public/"));
    expect(publicCalls).toHaveLength(3);
    for (const [, init] of publicCalls) {
      const h = requestHeaders(init);
      expect(h.get("Authorization")).toBeNull();
      expect(h.get("X-Anon-Id")).toBeNull();
      expect(h.get("X-Session-Id")).toBeNull();
      expect(init?.method ?? "GET").toBe("GET");
    }
  });

  it("lists samples with coerced summaries and drops malformed rows", async () => {
    stubFetch([[/\/api\/public\/samples(\?|$)/, () => okJson([...SAMPLE_LIST, { nope: true }, "x", { ticker: "ZZ", kinds: "bad" }])]]);
    const out = await listSamples();
    expect(out?.map((s) => s.ticker)).toEqual([...SAMPLE_TICKERS, "ZZ"]);
    expect(out?.[2]).toEqual({ ticker: "JPM", company_name: "JPMorgan Chase", sector: "Financials", built_at: null, kinds: [] });
    expect(out?.[3].kinds).toEqual([]);
  });

  it("returns null for the list on a non-2xx, a network failure and a non-array body", async () => {
    stubFetch([[/\/api\/public\/samples(\?|$)/, () => okJson({ detail: "nope" }, 503)]]);
    expect(await listSamples()).toBeNull();
    vi.unstubAllGlobals();
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    expect(await listSamples()).toBeNull();
    vi.unstubAllGlobals();
    stubFetch([[/\/api\/public\/samples(\?|$)/, () => okJson({ rows: [] })]]);
    expect(await listSamples()).toBeNull();
  });

  it("returns a typed 404 carrying the tickers that are public", async () => {
    stubFetch(sampleRoutes());
    const res = await getSample("aapl");
    expect(res).toEqual({ status: "not_found", sampleTickers: SAMPLE_TICKERS });
  });

  it("never sends a request for a ticker that fails validation", async () => {
    const mock = stubFetch(sampleRoutes());
    for (const bad of ["", "..", "../admin", "a b", "NVDA?x=1", "TOOLONGTICKER1"]) {
      expect(await getSample(bad)).toEqual({ status: "not_found", sampleTickers: [] });
    }
    expect(mock).not.toHaveBeenCalled();
    expect(normalizeTicker(" cost ")).toBe("COST");
    expect(normalizeTicker("BRK.B")).toBe("BRK.B");
    expect(normalizeTicker("1abc")).toBeNull();
  });

  it("coerces a sample: prices as a bare list, ledger columns always present", async () => {
    const raw = { ...makeSample(), prices: [{ date: "2026-01-02", close: 1.5 }, { date: "x" }, "junk"], expectations_ledger: { note: "n" } };
    stubFetch([[/\/api\/public\/samples\/COST/, () => okJson(raw)]]);
    const res = await getSample("COST");
    expect(res.status).toBe("ok");
    if (res.status !== "ok") return;
    expect(res.sample.prices).toEqual([{ date: "2026-01-02", close: 1.5 }]);
    expect(res.sample.expectations_ledger.price_implied).toEqual({ status: "n/a", items: [], reason: "no stored memo" });
    expect(res.sample.expectations_ledger.columns).toHaveLength(4);
    expect(res.sample.memo?.ticker).toBe("COST");
  });

  it("is 'unavailable' on a 5xx, a network failure, a bad body and a timeout", async () => {
    stubFetch([[/\/api\/public\/samples\/COST/, () => errJson(503, { code: "db" })]]);
    expect(await getSample("COST")).toEqual({ status: "unavailable" });
    vi.unstubAllGlobals();
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    expect(await getSample("COST")).toEqual({ status: "unavailable" });
    vi.unstubAllGlobals();
    stubFetch([[/\/api\/public\/samples\/COST/, () => ({ ok: true, status: 200, headers: new Headers(), json: async () => "not an object" })]]);
    expect(await getSample("COST")).toEqual({ status: "unavailable" });
    vi.unstubAllGlobals();

    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn((_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
      ),
    );
    const pending = getSample("COST");
    await vi.advanceTimersByTimeAsync(9000);
    expect(await pending).toEqual({ status: "unavailable" });
  });
});
