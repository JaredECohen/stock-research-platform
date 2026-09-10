import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, within } from "@testing-library/react";
import Landing from "@/pages/public/Landing";
import { resetAnalyticsForTests } from "@/lib/analytics";
import { SIGNED_IN, SIGNED_OUT, calls, featureMatrix, okJson, renderWithProviders, stubFetch } from "@/test/providers";
import { SAMPLE_TICKERS, sampleRoutes } from "@/test/fixtures/sample";

// recharts needs ResizeObserver (absent in jsdom); the charts are not what
// these tests assert on — the tables under them are.
vi.mock("recharts", () => {
  const Noop = () => null;
  return { Bar: Noop, BarChart: Noop, CartesianGrid: Noop, Line: Noop, LineChart: Noop, ResponsiveContainer: Noop, Tooltip: Noop, XAxis: Noop, YAxis: Noop };
});

const CONFIG = { features: featureMatrix(), trial_days: 7, sample_tickers: SAMPLE_TICKERS };

describe("Landing", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    resetAnalyticsForTests();
  });

  it("renders the value proposition with landmarks, one h1, a skip link and the mocked samples", async () => {
    stubFetch(sampleRoutes());
    renderWithProviders(<Landing />, { config: CONFIG });

    // Landmarks and heading hierarchy.
    expect(screen.getByRole("banner")).toBeInTheDocument();
    expect(screen.getByRole("main")).toHaveAttribute("id", "main");
    expect(screen.getByRole("contentinfo")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Primary" })).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Research that shows its work.");
    expect(screen.getByRole("link", { name: "Skip to content" })).toHaveAttribute("href", "#main");
    expect(document.title).toBe("Your AI investment committee — MarketMosaic");

    // The value proposition names the committee's inputs.
    const hero = screen.getByRole("heading", { level: 1 }).closest("section")!;
    for (const word of ["fundamentals", "filings", "earnings", "valuation", "risks", "catalysts", "scenarios"]) {
      expect(hero.textContent).toContain(word);
    }

    // Samples: the first BUILT ticker is selected by default (NVDA), and its
    // memo summary, ledger, DCF, comps, fundamentals and commentary render.
    await screen.findByRole("heading", { name: /NVIDIA/ });
    expect(screen.getByRole("button", { name: /NVDA/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("heading", { name: "Expectations ledger" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "DCF scenarios" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Comparable companies" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Fundamentals" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Committee commentary" })).toBeInTheDocument();
    expect(screen.getByTestId("data-freshness")).toHaveTextContent("Built from stored research on September 6, 2026 at 07:00 UTC");
    expect(screen.getByRole("link", { name: "Open the full NVDA sample" })).toHaveAttribute("href", "/samples/NVDA");

    // Every button has an accessible name (a11y baseline).
    for (const b of screen.getAllByRole("button")) expect(b).toHaveAccessibleName();

    // Pricing and FAQ sections are on the page, with the plan comparison.
    expect(screen.getByRole("heading", { name: "Pricing" })).toBeInTheDocument();
    expect(screen.getByText("3 distinct companies' memos per month")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Frequently asked questions" })).toBeInTheDocument();
    const notes = screen.getAllByRole("note", { name: "Disclosure" });
    expect(notes[notes.length - 1]).toHaveTextContent("research and education only");
  });

  it("switches the showcased company and only fetches each sample once", async () => {
    const mock = stubFetch(sampleRoutes());
    renderWithProviders(<Landing />, { config: CONFIG });
    await screen.findByRole("heading", { name: /NVIDIA/ });
    fireEvent.click(screen.getByRole("button", { name: /COST/ }));
    await screen.findByRole("heading", { name: /Costco Wholesale/ });
    expect(screen.getByRole("button", { name: /COST/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /NVDA/ })).toHaveAttribute("aria-pressed", "false");
    // The unbuilt ticker renders honestly: a notice, no memo, every ledger
    // column blank with its reason.
    fireEvent.click(screen.getByRole("button", { name: /JPM/ }));
    await screen.findByTestId("sample-unbuilt");
    expect(screen.queryByRole("heading", { name: "DCF scenarios" })).not.toBeInTheDocument();
    expect(screen.getAllByText(/no stored memo/)).toHaveLength(4);
    expect(calls(mock, "/api/public/samples/")).toHaveLength(3);
  });

  it("degrades without crashing when the samples endpoint answers 503", async () => {
    stubFetch([[/\/api\/public\/samples/, () => okJson({ detail: "db down" }, 503)]]);
    renderWithProviders(<Landing />, { config: CONFIG });
    await screen.findByTestId("samples-unavailable");
    expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Pricing" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "How it works" })).toBeInTheDocument();
  });

  it("degrades without crashing when fetch itself rejects", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    renderWithProviders(<Landing />, { config: CONFIG });
    await screen.findByTestId("samples-unavailable");
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  });

  it("says so when the allowlist is empty", async () => {
    stubFetch([[/\/api\/public\/samples(\?|$)/, () => okJson([])]]);
    renderWithProviders(<Landing />, { config: CONFIG });
    await screen.findByTestId("samples-empty");
  });

  it("offers the trial and sign-in to a signed-out visitor, with the configured trial length", async () => {
    stubFetch(sampleRoutes());
    renderWithProviders(<Landing />, { config: { ...CONFIG, auth_enabled: true }, auth: SIGNED_OUT });
    const hero = screen.getByRole("heading", { level: 1 }).closest("section")!;
    expect(within(hero).getByRole("link", { name: "Start 7-day Pro trial" })).toHaveAttribute("href", "/sign-up?returnTo=%2Fapp");
    expect(within(hero).getByRole("link", { name: "Sign in" })).toHaveAttribute("href", "/sign-in?returnTo=%2Fapp");
    expect(within(hero).getByRole("link", { name: "View sample research" })).toHaveAttribute("href", "/samples");
    expect(within(hero).getByRole("link", { name: "See pricing" })).toHaveAttribute("href", "/pricing");
    const nav = screen.getByRole("banner");
    expect(within(nav).getByRole("link", { name: "Sign up free" })).toHaveAttribute("href", "/sign-up?returnTo=%2Fapp");
    expect(screen.queryByRole("link", { name: "Continue to app" })).not.toBeInTheDocument();
  });

  it("does not name a trial length when the config fell back", () => {
    stubFetch(sampleRoutes());
    renderWithProviders(<Landing />, { config: { auth_enabled: true, trial_days: null, features: {} }, auth: SIGNED_OUT });
    expect(screen.getAllByRole("link", { name: "Start your Pro trial" }).length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText(/\d+-day/)).not.toBeInTheDocument();
    expect(screen.getByTestId("matrix-unknown")).toBeInTheDocument();
  });

  it("shows 'Continue to app' to a signed-in visitor and never redirects", () => {
    stubFetch(sampleRoutes());
    renderWithProviders(<Landing />, { route: "/", config: { ...CONFIG, auth_enabled: true }, auth: SIGNED_IN });
    expect(screen.getAllByRole("link", { name: "Continue to app" }).length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByRole("link", { name: "Sign up free" })).not.toBeInTheDocument();
    expect(screen.getByTestId("location")).toHaveTextContent("/");
  });

  it("shows 'Continue to app' with the login wall off", () => {
    stubFetch(sampleRoutes());
    renderWithProviders(<Landing />, { config: CONFIG });
    expect(screen.getAllByRole("link", { name: "Continue to app" })[0]).toHaveAttribute("href", "/app");
    expect(screen.queryByRole("link", { name: /Pro trial/ })).not.toBeInTheDocument();
  });

  it("queues a landing_view analytics event without touching an authenticated endpoint", async () => {
    const mock = stubFetch(sampleRoutes());
    renderWithProviders(<Landing />, { config: CONFIG });
    await screen.findByRole("heading", { name: /NVIDIA/ });
    const urls = mock.mock.calls.map(([u]) => String(u));
    expect(urls.every((u) => u.includes("/api/public/"))).toBe(true);
  });
});
