import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen } from "@testing-library/react";
import Chat from "@/pages/Chat";
import { SIGNED_IN, calls, errJson, okJson, renderWithProviders, stubFetch } from "@/test/providers";

async function settle() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

function typeAndSend(text: string) {
  fireEvent.change(screen.getByPlaceholderText("Ask the PM…"), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: /Send/ }));
}

describe("Chat refusals", () => {
  beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }));
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("keeps the typed message and offers a timed retry on 429", async () => {
    let n = 0;
    const mock = stubFetch([
      [
        "/api/chat",
        () =>
          n++ === 0
            ? errJson(429, { code: "rate_limited", scope: "user:llm_light", retry_after: 2, window_seconds: 60 })
            : okJson({ intent: "general_research_chat", answer: "Here you go.", agent_trace: [], sources: [], disclaimer: "" }),
      ],
    ]);
    renderWithProviders(<Chat />, { route: "/app/chat", config: { auth_enabled: true }, auth: SIGNED_IN });
    typeAndSend("What sectors benefit if inflation stays sticky?");
    await settle();
    expect(screen.getByTestId("rate-limit-notice")).toHaveTextContent("Your message is kept in the box below.");
    // The input was restored, not discarded.
    expect(screen.getByPlaceholderText("Ask the PM…")).toHaveValue("What sectors benefit if inflation stays sticky?");
    expect(screen.getByRole("button", { name: "Retry in 2s" })).toBeDisabled();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2100);
    });
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await settle();
    expect(calls(mock, "/api/chat")).toHaveLength(2);
    expect(screen.getByText("Here you go.")).toBeInTheDocument();
    expect(screen.queryByTestId("rate-limit-notice")).not.toBeInTheDocument();
  });

  it("shows the upgrade prompt with the allowance numbers on 402 and keeps the message", async () => {
    stubFetch([
      [
        "/api/chat",
        () =>
          errJson(402, {
            code: "quota_exceeded",
            feature: "pm_chat",
            plan: "free",
            used: 10,
            limit: 10,
            resets_at: "2026-10-01T00:00:00",
            upgrade_url: "/pricing",
            message: "Free Explorer includes 10 Ask-the-PM turns per month.",
          }),
      ],
    ]);
    renderWithProviders(<Chat />, { route: "/app/chat", config: { auth_enabled: true }, auth: SIGNED_IN });
    typeAndSend("Analyze NVDA");
    await settle();
    expect(screen.getByTestId("upgrade-prompt")).toHaveTextContent("You've used 10 of 10 Ask-the-PM turns this month");
    expect(screen.getByPlaceholderText("Ask the PM…")).toHaveValue("Analyze NVDA");
    expect(screen.queryByText(/API 402/)).not.toBeInTheDocument();
  });

  it("links each needs_analysis ticker to the research page under /app", async () => {
    stubFetch([
      [
        "/api/chat",
        () =>
          okJson({
            intent: "single_stock_analysis",
            answer: "No memo is stored for MOD yet.",
            agent_trace: [],
            sources: [],
            disclaimer: "",
            needs_analysis: ["MOD"],
          }),
      ],
    ]);
    renderWithProviders(<Chat />, { route: "/app/chat", config: { auth_enabled: true }, auth: SIGNED_IN });
    typeAndSend("Analyze MOD");
    await settle();
    expect(screen.getByRole("link", { name: "MOD" })).toHaveAttribute("href", "/app/research?ticker=MOD");
  });
});
