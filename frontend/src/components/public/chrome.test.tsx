import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, within } from "@testing-library/react";
import FeatureComparison from "@/components/public/FeatureComparison";
import PublicNav from "@/components/public/PublicNav";
import PublicShell from "@/components/public/PublicShell";
import { allowanceText, formatCents, monthlyEquivalent, rowCells } from "@/components/public/allowance";
import { trialCta } from "@/components/public/Hero";
import { SIGNED_IN, SIGNED_OUT, featureMatrix, renderWithProviders, stubFetch } from "@/test/providers";

describe("PublicNav", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows Sign in / Sign up free to a signed-out visitor and Continue to app when signed in", () => {
    stubFetch();
    const v = renderWithProviders(<PublicNav />, { config: { auth_enabled: true }, auth: SIGNED_OUT });
    const banner = screen.getByRole("banner");
    expect(within(banner).getAllByRole("link", { name: "Sign in" })[0]).toHaveAttribute("href", "/sign-in?returnTo=%2Fapp");
    expect(within(banner).getAllByRole("link", { name: "Sign up free" })[0]).toHaveAttribute("href", "/sign-up?returnTo=%2Fapp");
    expect(within(banner).queryByRole("link", { name: "Continue to app" })).not.toBeInTheDocument();
    v.unmount();
    renderWithProviders(<PublicNav />, { config: { auth_enabled: true }, auth: SIGNED_IN });
    expect(screen.getAllByRole("link", { name: "Continue to app" })[0]).toHaveAttribute("href", "/app");
    expect(screen.queryByRole("link", { name: "Sign in" })).not.toBeInTheDocument();
  });

  it("renders no auth CTA while the provider is still loading", () => {
    stubFetch();
    renderWithProviders(<PublicNav />, { config: { auth_enabled: true }, auth: { ...SIGNED_OUT, status: "loading" } });
    expect(screen.queryByRole("link", { name: /Sign|Continue/ })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Pricing" })).toHaveAttribute("href", "/pricing");
  });

  it("has a keyboard-operable mobile menu with aria-expanded", () => {
    stubFetch();
    renderWithProviders(<PublicNav />);
    const btn = screen.getByRole("button", { name: "Open menu" });
    expect(btn).toHaveAttribute("aria-expanded", "false");
    expect(document.getElementById("public-mobile-menu")).toHaveAttribute("hidden");
    fireEvent.click(btn);
    expect(screen.getByRole("button", { name: "Close menu" })).toHaveAttribute("aria-expanded", "true");
    expect(document.getElementById("public-mobile-menu")).not.toHaveAttribute("hidden");
    fireEvent.click(within(document.getElementById("public-mobile-menu")!).getByRole("link", { name: "FAQ" }));
    expect(document.getElementById("public-mobile-menu")).toHaveAttribute("hidden");
  });
});

describe("PublicShell", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("provides the skip link, landmarks and the page title/description", () => {
    stubFetch();
    renderWithProviders(
      <PublicShell title="Test page" description="A description.">
        <h1>Hello</h1>
      </PublicShell>,
    );
    expect(screen.getByRole("link", { name: "Skip to content" })).toHaveAttribute("href", "#main");
    expect(screen.getByRole("main")).toHaveAttribute("id", "main");
    expect(screen.getByRole("banner")).toBeInTheDocument();
    expect(screen.getByRole("contentinfo")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Legal" })).toBeInTheDocument();
    expect(document.title).toBe("Test page — MarketMosaic");
    expect(document.querySelector('meta[name="description"]')?.getAttribute("content")).toBe("A description.");
  });
});

describe("FeatureComparison", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("renders one row per shipped feature from the matrix, and hides reserved ones", () => {
    stubFetch();
    renderWithProviders(<FeatureComparison />, { config: { features: featureMatrix() } });
    const rows = document.querySelectorAll("tbody tr[data-feature]");
    expect(rows).toHaveLength(10);
    expect(document.querySelector('tr[data-feature="chart_commentary"]')).toBeNull();
    expect(document.querySelector('tr[data-feature="fundamentals_explorer"]')).toBeNull();
    const chat = document.querySelector('tr[data-feature="pm_chat"]') as HTMLElement;
    expect(within(chat).getByText("10 per month")).toBeInTheDocument();
    expect(within(chat).getByText("300 per month")).toBeInTheDocument();
    expect(screen.queryByTestId("matrix-unknown")).not.toBeInTheDocument();
  });

  it("shows dashes and a notice when the matrix is unknown", () => {
    stubFetch();
    renderWithProviders(<FeatureComparison compact />, { config: { features: {} } });
    expect(document.querySelectorAll("tbody tr[data-feature]")).toHaveLength(5);
    expect(screen.getByTestId("matrix-unknown")).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBe(10);
  });
});

describe("allowance copy", () => {
  const entry = (over: Partial<ReturnType<typeof featureMatrix>["memo_view"]> = {}) => ({
    description: "", free: 3, pro: null, metered: true, period: "month", distinct_resources: false, ...over,
  });

  it("formats prices", () => {
    expect(formatCents(2999)).toBe("$29.99");
    expect(formatCents(29900)).toBe("$299");
    expect(formatCents(0)).toBe("$0");
    expect(monthlyEquivalent(29900)).toBe("$24.92");
    expect(formatCents(1000, "eur")).toBe("€10");
  });

  it("names every allowance shape honestly", () => {
    expect(allowanceText(3, entry({ distinct_resources: true }), "memo_view")).toBe("3 distinct companies' memos per month");
    expect(allowanceText(3, entry({ distinct_resources: true }), "other")).toBe("3 distinct companies per month");
    expect(allowanceText(20, entry(), "research_run")).toBe("20 per month");
    expect(allowanceText(null, entry(), "memo_view")).toBe("Unlimited");
    expect(allowanceText(true, entry(), "dcf")).toBe("Included");
    expect(allowanceText(false, entry(), "portfolio")).toBe("Not included");
    expect(allowanceText(0, entry(), "pm_chat")).toBe("Not included");
    expect(allowanceText("follows_memo", entry(), "dcf")).toBe("For companies whose memo you opened this month");
    expect(allowanceText(2, entry({ period: "week" }), "x")).toBe("2 per week");
    expect(rowCells({}, "memo_view")).toBeNull();
  });

  it("names the trial CTA only with a known trial length", () => {
    expect(trialCta(7)).toBe("Start 7-day Pro trial");
    expect(trialCta(null)).toBe("Start your Pro trial");
  });
});
