import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import Pricing from "@/pages/public/Pricing";
import { resetAnalyticsForTests } from "@/lib/analytics";
import { SIGNED_IN, SIGNED_OUT, featureMatrix, renderWithProviders, stubFetch } from "@/test/providers";

describe("Pricing", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    resetAnalyticsForTests();
  });

  it("renders prices, trial and allowances from config — never from a literal", () => {
    stubFetch();
    renderWithProviders(<Pricing />, {
      config: { auth_enabled: true, billing_enabled: true, features: featureMatrix(), trial_days: 7 },
      auth: SIGNED_OUT,
    });
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    const pro = document.querySelector('[data-plan="pro"]') as HTMLElement;
    const free = document.querySelector('[data-plan="free"]') as HTMLElement;
    expect(within(pro).getByText("$29.99")).toBeInTheDocument();
    expect(within(pro).getByText("$299")).toBeInTheDocument();
    expect(pro.textContent).toContain("$24.92 a month equivalent");
    expect(within(pro).getByText("7-day trial, no card")).toBeInTheDocument();
    expect(within(pro).getByRole("link", { name: "Start 7-day Pro trial" })).toHaveAttribute("href", "/sign-up?returnTo=%2Fapp");
    expect(within(free).getByRole("link", { name: "Sign up free" })).toHaveAttribute("href", "/sign-up?returnTo=%2Fapp");
    // Free allowances from the matrix, with the distinct-companies wording.
    expect(free.textContent).toContain("3 distinct companies' memos per month of stored research memos");
    expect(free.textContent).toContain("1 per month research run");
    expect(free.textContent).toContain("10 per month Ask-the-PM turns");
    expect(free.textContent).toContain("for companies whose memo you opened this month");
    expect(pro.textContent).toContain("20 per month research runs");
    expect(pro.textContent).toContain("300 per month Ask-the-PM turns");
    expect(pro.textContent).toContain("Unlimited stored research memos");
    // Period and trial-to-paid rules are stated.
    expect(screen.getByText(/per UTC calendar month and reset on the first of the month at 00:00 UTC/)).toBeInTheDocument();
    expect(screen.getAllByText(/at least 48 hours remain/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/no card required/)).toBeInTheDocument();
    expect(screen.queryByTestId("billing-closed")).not.toBeInTheDocument();
    // The comparison table carries the same cells.
    const row = document.querySelector('tr[data-feature="memo_view"]') as HTMLElement;
    expect(within(row).getByText("3 distinct companies' memos per month")).toBeInTheDocument();
    expect(within(row).getByText("Unlimited")).toBeInTheDocument();
    const dcf = document.querySelector('tr[data-feature="dcf"]') as HTMLElement;
    expect(within(dcf).getByText("For companies whose memo you opened this month")).toBeInTheDocument();
    expect(within(dcf).getByText("Included")).toBeInTheDocument();
    const portfolio = document.querySelector('tr[data-feature="portfolio"]') as HTMLElement;
    expect(within(portfolio).getByText("Not included")).toBeInTheDocument();
  });

  it("follows an override in the matrix and formats other prices", () => {
    stubFetch();
    renderWithProviders(<Pricing />, {
      config: {
        auth_enabled: true,
        billing_enabled: true,
        features: featureMatrix({ memo_view: { free: 5 }, pm_chat: { pro: 500 } }),
        trial_days: 14,
        prices: { monthly_cents: 1900, annual_cents: 19000, currency: "usd" },
      },
      auth: SIGNED_OUT,
    });
    expect(screen.getByText("5 distinct companies' memos per month")).toBeInTheDocument();
    expect(screen.getByText("500 per month")).toBeInTheDocument();
    expect(screen.getByText("$19")).toBeInTheDocument();
    expect(screen.getByText("$190")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Start 14-day Pro trial" })).toBeInTheDocument();
  });

  it("shows no numbers when the matrix is unknown, and says why", () => {
    stubFetch();
    renderWithProviders(<Pricing />, { config: { auth_enabled: true, features: {}, trial_days: null }, auth: SIGNED_OUT });
    expect(screen.getByTestId("matrix-unknown")).toBeInTheDocument();
    expect(screen.queryByText(/\d+ distinct companies/)).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Start your Pro trial" })).toBeInTheDocument();
    expect(screen.getByText(/Every new account starts with a Pro trial — no card required/)).toBeInTheDocument();
  });

  it("tells visitors when subscriptions are not open on this deployment", () => {
    stubFetch();
    renderWithProviders(<Pricing />, { config: { auth_enabled: true, billing_enabled: false, features: featureMatrix(), trial_days: 7 }, auth: SIGNED_OUT });
    expect(screen.getByTestId("billing-closed")).toBeInTheDocument();
  });

  it("routes a signed-in user to the account page and a wall-off visitor to the app", () => {
    stubFetch();
    const v = renderWithProviders(<Pricing />, { config: { auth_enabled: true, features: featureMatrix(), trial_days: 7 }, auth: SIGNED_IN });
    expect(screen.getByRole("link", { name: "Manage plan" })).toHaveAttribute("href", "/app/account");
    v.unmount();
    renderWithProviders(<Pricing />, { config: { auth_enabled: false, features: featureMatrix(), trial_days: 7 } });
    const pro = document.querySelector('[data-plan="pro"]') as HTMLElement;
    expect(within(pro).getByRole("link", { name: "Continue to app" })).toHaveAttribute("href", "/app");
  });
});
