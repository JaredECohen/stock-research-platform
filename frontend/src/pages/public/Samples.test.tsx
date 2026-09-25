import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import Samples from "@/pages/public/Samples";
import SampleDetail from "@/pages/public/SampleDetail";
import { resetAnalyticsForTests } from "@/lib/analytics";
import { SIGNED_OUT, calls, errJson, okJson, renderWithProviders, stubFetch } from "@/test/providers";
import { NVDA_SAMPLE, SAMPLE_TICKERS, makeSample, sampleRoutes, unbuiltSample } from "@/test/fixtures/sample";
import { QUALITY_EXPECT, qualityMemo } from "@/test/fixtures/memoQuality";

vi.mock("recharts", () => {
  const Noop = () => null;
  return { Bar: Noop, BarChart: Noop, CartesianGrid: Noop, Line: Noop, LineChart: Noop, ResponsiveContainer: Noop, Tooltip: Noop, XAxis: Noop, YAxis: Noop };
});

describe("/samples", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    resetAnalyticsForTests();
  });

  it("lists every allowlisted ticker, built or not", async () => {
    stubFetch(sampleRoutes());
    renderWithProviders(<Samples />, { route: "/samples" });
    await screen.findByRole("link", { name: "Open the NVDA sample" });
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByRole("link", { name: "Open the JPM sample" })).toHaveAttribute("href", "/samples/JPM");
    expect(screen.getByText("Not built yet")).toBeInTheDocument();
    expect(screen.getAllByText(/Built Sep 6, 2026/)).toHaveLength(2);
    expect(screen.getByRole("list", { name: "Sections available for NVDA" })).toBeInTheDocument();
  });

  it("degrades when the list is unavailable", async () => {
    stubFetch([[/\/api\/public\/samples/, () => okJson({}, 500)]]);
    renderWithProviders(<Samples />, { route: "/samples" });
    await screen.findByTestId("samples-unavailable");
  });
});

describe("/samples/:ticker", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    resetAnalyticsForTests();
  });

  const mount = (route: string) => renderWithProviders(<SampleDetail />, { route, path: "/samples/:ticker", auth: SIGNED_OUT, config: { auth_enabled: true } });

  it("renders the full sample with one h1, the ledger, every section and the read-only memo", async () => {
    stubFetch(sampleRoutes());
    mount("/samples/cost");
    await screen.findByRole("heading", { level: 1, name: /Costco Wholesale/ });
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(document.title).toBe("Costco Wholesale (COST) sample — MarketMosaic");
    for (const name of ["Expectations ledger", "Committee commentary", "DCF scenarios", "Comparable companies", "Fundamentals", "Price history", "Screener rank", "The full committee memo"]) {
      expect(screen.getByRole("heading", { name })).toBeInTheDocument();
    }
    // Observed vs interpretation cells are distinguishable.
    expect(document.querySelectorAll('[data-basis="observed"]').length).toBeGreaterThan(0);
    expect(document.querySelectorAll('[data-basis="interpretation"]').length).toBeGreaterThan(0);
    // Disclosures from the payload, including the build time.
    expect(screen.getByRole("note", { name: "Disclosure" })).toHaveTextContent("Built from stored research on 2026-09-06T07:00:00Z");
    for (const a of screen.getAllByRole("link", { name: "Sign up free" })) expect(a).toHaveAttribute("href", "/sign-up?returnTo=%2Fapp");
  });

  it("shows a friendly not-found for an unlisted ticker, with links to the public ones", async () => {
    stubFetch(sampleRoutes());
    mount("/samples/AAPL");
    await screen.findByTestId("sample-not-found");
    const list = screen.getByRole("list", { name: "Public samples" });
    for (const t of SAMPLE_TICKERS) expect(within(list).getByRole("link", { name: t })).toHaveAttribute("href", `/samples/${t}`);
  });

  it("never requests an invalid ticker", async () => {
    const mock = stubFetch(sampleRoutes());
    mount("/samples/..%2Fadmin");
    await screen.findByTestId("sample-not-found");
    expect(calls(mock, "/api/public/samples")).toHaveLength(0);
  });

  it("degrades when the sample endpoint fails", async () => {
    stubFetch([[/\/api\/public\/samples\/COST/, () => errJson(503, { code: "db" })]]);
    mount("/samples/COST");
    await screen.findByTestId("samples-unavailable");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("COST");
  });

  it("renders a listed-but-unbuilt ticker with every ledger column blank and reasoned", async () => {
    stubFetch(sampleRoutes());
    mount("/samples/JPM");
    await screen.findByTestId("sample-unbuilt");
    expect(screen.getAllByText(/no stored memo/)).toHaveLength(4);
    expect(screen.getByTestId("data-freshness")).toHaveTextContent("This sample has not been built yet.");
    expect(screen.queryByRole("heading", { name: "The full committee memo" })).not.toBeInTheDocument();
  });
  it("shows no research checks on the public memo (design W2b §9: public pages are unchanged)", async () => {
    // The public payload carries `quality` (strip_for_public keeps it); the
    // page renders the memo as it did before W2b.
    const memo = qualityMemo();
    stubFetch(sampleRoutes({ NVDA: { ...NVDA_SAMPLE, memo }, COST: makeSample(), JPM: unbuiltSample() }));
    const { container } = mount("/samples/NVDA");
    await screen.findByRole("heading", { name: "The full committee memo" });
    expect(container.querySelector('[data-testid="research-checks"]')).toBeNull();
    expect(container.querySelector("[data-claim-status]")).toBeNull();
    expect(container.querySelector('[data-testid="rating-reconciliation-note"]')).toBeNull();
    expect(container.querySelector('[data-testid="confidence-capped"]')).toBeNull();
    expect(container.textContent).not.toContain(QUALITY_EXPECT.withheld_point);
    expect(container.querySelector('[title^="How sure the PM is"]')).not.toBeNull();
    // The memo itself is there, figures unmarked.
    expect(container.textContent).toContain(QUALITY_EXPECT.fabricated_pm_figure);
  });
});
