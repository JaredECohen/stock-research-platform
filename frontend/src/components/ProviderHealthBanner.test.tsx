import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import ProviderHealthBanner, { assessHealth } from "@/components/ProviderHealthBanner";
import type { ProvidersStatusResponse } from "@/types";

// The banner goes through api.providersStatus → global fetch, so we mock
// fetch by URL: the status call gets the payload under test, anything else
// (the logger's /api/admin/ui-log flush) gets an empty 200. Rejecting the
// status call exercises the "never render an error banner" path.
type Responder = () => Promise<Partial<Response>>;

function stubFetch(status: Responder) {
  const mock = vi.fn((input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (url.includes("/api/providers/status")) return status();
    return Promise.resolve({ ok: true, status: 200, json: async () => ({}), text: async () => "" });
  });
  vi.stubGlobal("fetch", mock);
  return mock;
}

function okWith(payload: unknown): Responder {
  return () =>
    Promise.resolve({
      ok: true,
      status: 200,
      json: async () => payload,
      text: async () => JSON.stringify(payload),
    });
}

const HEALTHY: ProvidersStatusResponse = {
  mode: "live",
  providers: {},
  missing_api_keys: [],
  llm_configured: true,
  llm: {
    configured: true,
    provider_choice: "auto",
    active_provider: "openai",
    openai_configured: true,
    anthropic_configured: true,
    breakers: {
      openai: { failure_count: 0, is_open: false, seconds_since_last_failure: null, cooldown_seconds: 300 },
      anthropic: { failure_count: 0, is_open: false, seconds_since_last_failure: null, cooldown_seconds: 300 },
    },
    failover: { enabled: true, count: 0, last_from: null, last_to: null, last_at: null, last_reason: null },
    degraded: false,
    degradation_reasons: [],
  },
  feature_flags: {},
};

function withLLM(overrides: Partial<NonNullable<ProvidersStatusResponse["llm"]>>): ProvidersStatusResponse {
  return { ...HEALTHY, llm: { ...HEALTHY.llm!, ...overrides } };
}

const DEGRADED = withLLM({
  degraded: true,
  degradation_reasons: ["OpenAI quota exhausted (insufficient_quota)"],
});

const BREAKER_OPEN = withLLM({
  breakers: {
    openai: { failure_count: 5, is_open: true, seconds_since_last_failure: 40, cooldown_seconds: 300 },
    anthropic: { failure_count: 0, is_open: false, seconds_since_last_failure: null, cooldown_seconds: 300 },
  },
});

const FAILOVER_ONLY = withLLM({
  failover: {
    enabled: true,
    count: 1,
    last_from: "openai",
    last_to: "anthropic",
    last_at: new Date().toISOString(),
    last_reason: "rate_limit",
  },
});

// Pre-breaker backend: no breakers/failover/degraded fields at all.
const OLD_BACKEND: ProvidersStatusResponse = {
  mode: "demo",
  providers: {},
  missing_api_keys: [],
  llm_configured: true,
  llm: {
    configured: true,
    provider_choice: "auto",
    active_provider: "openai",
    openai_configured: true,
    anthropic_configured: false,
    openai_strong_model: "gpt-x",
    openai_cheap_model: "gpt-x-mini",
    anthropic_strong_model: "claude-x",
    anthropic_cheap_model: "claude-x-mini",
  },
  feature_flags: {},
};

// Let the fetch promise and the resulting setState settle.
async function settle() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

function renderBanner() {
  return render(
    <MemoryRouter>
      <ProviderHealthBanner />
    </MemoryRouter>,
  );
}

describe("ProviderHealthBanner", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("renders nothing on a healthy payload", async () => {
    const fetchMock = stubFetch(okWith(HEALTHY));
    renderBanner();
    await settle();
    expect(fetchMock).toHaveBeenCalled();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("renders the degraded banner with reasons and role=status", async () => {
    stubFetch(okWith(DEGRADED));
    renderBanner();
    const banner = await screen.findByRole("status");
    expect(banner).toHaveAttribute("aria-live", "polite");
    expect(banner).toHaveAttribute("data-variant", "degraded");
    expect(screen.getByText("AI analysis is degraded")).toBeInTheDocument();
    expect(screen.getByText("OpenAI quota exhausted (insufficient_quota)")).toBeInTheDocument();
    expect(
      screen.getByText("Memos and chat may fall back to deterministic sections until the provider recovers."),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /provider health details/i })).toHaveAttribute("href", "/settings");
  });

  it("derives a breaker sentence when a breaker is open without the degraded flag", async () => {
    stubFetch(okWith(BREAKER_OPEN));
    renderBanner();
    await screen.findByRole("status");
    expect(
      screen.getByText("OpenAI circuit breaker is open after 5 failures — retrying in ~260s"),
    ).toBeInTheDocument();
    // The closed Anthropic breaker must not produce a line.
    expect(screen.queryByText(/Anthropic circuit breaker/)).not.toBeInTheDocument();
  });

  it("shows the banner in live mode when no LLM is configured", async () => {
    stubFetch(okWith({ ...HEALTHY, llm_configured: false, llm: undefined }));
    renderBanner();
    await screen.findByRole("status");
    expect(screen.getByText("No LLM API key is configured in live mode.")).toBeInTheDocument();
  });

  it("stays hidden in demo mode with no LLM (expected, not degraded)", async () => {
    stubFetch(okWith({ ...HEALTHY, mode: "demo", llm_configured: false, llm: undefined }));
    renderBanner();
    await settle();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("dismiss hides it, keeps the same payload hidden, and a new reason re-shows it", async () => {
    stubFetch(okWith(DEGRADED));
    const first = renderBanner();
    await screen.findByRole("status");
    fireEvent.click(screen.getByRole("button", { name: /dismiss provider health notice/i }));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    first.unmount();

    // Same reasons on a fresh mount (new page, same tab) → still dismissed.
    renderBanner();
    await settle();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    vi.unstubAllGlobals();

    // A different reason → banner returns.
    stubFetch(okWith(withLLM({ degraded: true, degradation_reasons: ["Anthropic credits exhausted"] })));
    renderBanner();
    await screen.findByRole("status");
    expect(screen.getByText("Anthropic credits exhausted")).toBeInTheDocument();
  });

  it("renders the quiet info variant for a recent failover with degraded=false", async () => {
    stubFetch(okWith(FAILOVER_ONLY));
    renderBanner();
    const banner = await screen.findByRole("status");
    expect(banner).toHaveAttribute("data-variant", "info");
    expect(screen.getByText("Using Anthropic after OpenAI failed (rate_limit).")).toBeInTheDocument();
    expect(screen.queryByText("AI analysis is degraded")).not.toBeInTheDocument();
    expect(screen.queryByText(/deterministic sections/)).not.toBeInTheDocument();
  });

  it("ignores a stale failover", () => {
    const stale = withLLM({
      failover: { ...FAILOVER_ONLY.llm!.failover!, last_at: "2020-01-01T00:00:00Z" },
    });
    expect(assessHealth(stale)).toBeNull();
  });

  it("renders nothing and does not crash on an old backend payload", async () => {
    stubFetch(okWith(OLD_BACKEND));
    renderBanner();
    await settle();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("renders nothing when the status fetch rejects", async () => {
    stubFetch(() => Promise.reject(new Error("network down")));
    renderBanner();
    await settle();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("renders nothing when the status endpoint returns an error status", async () => {
    stubFetch(() => Promise.resolve({ ok: false, status: 503, statusText: "unavailable", text: async () => "down" }));
    renderBanner();
    await settle();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("polls every 60s and clears the interval on unmount", async () => {
    vi.useFakeTimers();
    const fetchMock = stubFetch(okWith(HEALTHY));
    const view = renderBanner();
    const statusCalls = () =>
      fetchMock.mock.calls.filter(([u]) => String(u).includes("/api/providers/status")).length;
    expect(statusCalls()).toBe(1);
    await act(async () => {
      vi.advanceTimersByTime(60_000);
    });
    expect(statusCalls()).toBe(2);
    view.unmount();
    await act(async () => {
      vi.advanceTimersByTime(120_000);
    });
    expect(statusCalls()).toBe(2);
  });
});
