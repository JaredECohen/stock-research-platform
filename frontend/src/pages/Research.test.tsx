import { afterEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen } from "@testing-library/react";
import Research from "@/pages/Research";
import { SIGNED_IN, calls, errJson, okJson, renderWithProviders, stubFetch } from "@/test/providers";

const NVDA = { ticker: "NVDA", company_name: "NVIDIA", exchange: "NASDAQ", sector: "Technology", industry: "Semis", universe_tier: "data_only" };
const JOB = { ticker: "NVDA", status: "started", started_at: "2026-09-08T10:00:00", job_id: 3, current_version: null, current_generated_at: null, note: "" };

async function settle() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

describe("Research under the login wall", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("turns a 409 no_memo into a research run (POST /analyze), never an inline ondemand fetch", async () => {
    const mock = stubFetch([
      ["/api/stocks/NVDA/memo", () => errJson(409, { code: "no_memo", message: "no memo stored", extra: { analyze_path: "/api/stocks/NVDA/analyze" } })],
      ["/api/stocks/NVDA/analyze", () => okJson(JOB, 202)],
      [/\/api\/stocks$/, () => okJson([NVDA])],
    ]);
    renderWithProviders(<Research />, { route: "/app/research?ticker=NVDA", config: { auth_enabled: true }, auth: SIGNED_IN });
    await settle();
    await settle();
    fireEvent.click(await screen.findByRole("button", { name: "Analyze this stock" }));
    await settle();
    expect(calls(mock, "/api/stocks/NVDA/analyze")).toHaveLength(1);
    expect(calls(mock, "ondemand=true")).toHaveLength(0);
    expect(screen.getByText(/Regenerating memo in background/)).toBeInTheDocument();
  });

  it("renders the upgrade prompt when the research run is refused with 402 and keeps the ticker", async () => {
    stubFetch([
      ["/api/stocks/NVDA/memo", () => errJson(409, { code: "no_memo", message: "no memo stored" })],
      [
        "/api/stocks/NVDA/analyze",
        () => errJson(402, { code: "quota_exceeded", feature: "research_run", plan: "free", used: 1, limit: 1, resets_at: "2026-10-01T00:00:00", upgrade_url: "/pricing" }),
      ],
      [/\/api\/stocks$/, () => okJson([NVDA])],
    ]);
    renderWithProviders(<Research />, { route: "/app/research?ticker=NVDA", config: { auth_enabled: true }, auth: SIGNED_IN });
    await settle();
    await settle();
    fireEvent.click(await screen.findByRole("button", { name: "Analyze this stock" }));
    await settle();
    expect(screen.getByTestId("upgrade-prompt")).toHaveTextContent("You've used 1 of 1 research runs this month");
    expect(screen.getByTestId("location")).toHaveTextContent("/app/research?ticker=NVDA");
  });

  it("shows the rate-limit notice with a retry when a memo view is throttled", async () => {
    stubFetch([
      ["/api/stocks/NVDA/memo", () => errJson(429, { code: "rate_limited", scope: "user:data", retry_after: 0, window_seconds: 60 })],
      [/\/api\/stocks$/, () => okJson([NVDA])],
    ]);
    renderWithProviders(<Research />, { route: "/app/research?ticker=NVDA", config: { auth_enabled: true }, auth: SIGNED_IN });
    await settle();
    await settle();
    expect(screen.getByTestId("rate-limit-notice")).toHaveTextContent("NVDA stays selected");
    expect(screen.getByRole("button", { name: "Retry" })).toBeEnabled();
  });

  it("shows the unreadable-stored-memo message without offering Analyze", async () => {
    // FIX-004: a stored snapshot that no longer validates is a 422, not a 409,
    // so the page must say so rather than route into the (charged) Analyze gate.
    const mock = stubFetch([
      [
        "/api/stocks/NVDA/memo",
        () =>
          errJson(422, {
            code: "memo_unreadable",
            message:
              "The stored memo for NVDA (version 7) cannot be displayed: it was saved in a format this version of MarketMosaic cannot read. The stored record has not been changed.",
            feature: "memo_view",
            extra: { ticker: "NVDA", version: 7, fields: ["bull_case"] },
          }),
      ],
      [/\/api\/stocks$/, () => okJson([NVDA])],
    ]);
    renderWithProviders(<Research />, { route: "/app/research?ticker=NVDA", config: { auth_enabled: true }, auth: SIGNED_IN });
    await settle();
    await settle();
    expect(screen.getByText(/cannot be displayed/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Analyze this stock" })).toBeNull();
    expect(calls(mock, "/analyze")).toHaveLength(0);
  });
});
