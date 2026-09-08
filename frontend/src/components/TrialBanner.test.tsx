import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import TrialBanner from "@/components/TrialBanner";
import { resetAccountCache } from "@/auth/useAccount";
import { SIGNED_IN, freeAccount, makeAccount, okJson, renderWithProviders, stubFetch } from "@/test/providers";

describe("TrialBanner", () => {
  beforeEach(() => resetAccountCache());
  afterEach(() => vi.unstubAllGlobals());

  it("renders the exact trial end instant from /api/me, in UTC", async () => {
    stubFetch([["/api/me", () => okJson(makeAccount())]]);
    renderWithProviders(<TrialBanner />, { config: { auth_enabled: true, billing_enabled: true }, auth: SIGNED_IN });
    const banner = await screen.findByRole("status");
    expect(banner).toHaveAttribute("data-variant", "trial");
    expect(screen.getByTestId("trial-ends")).toHaveTextContent("September 15, 2026 at 14:03 UTC");
    expect(banner).toHaveTextContent("your account drops to Free");
    expect(screen.getByRole("link", { name: "Keep Pro" })).toHaveAttribute("href", "/app/account");
  });

  it("treats an explicit offset the same way", async () => {
    const acct = makeAccount();
    acct.plan.trial_ends_at = "2026-09-15T10:03:00-04:00";
    stubFetch([["/api/me", () => okJson(acct)]]);
    renderWithProviders(<TrialBanner />, { config: { auth_enabled: true }, auth: SIGNED_IN });
    await screen.findByRole("status");
    expect(screen.getByTestId("trial-ends")).toHaveTextContent("September 15, 2026 at 14:03 UTC");
    // Billing off: no upgrade link.
    expect(screen.queryByRole("link", { name: "Keep Pro" })).not.toBeInTheDocument();
  });

  it("renders nothing for a Free account without a warning", async () => {
    stubFetch([["/api/me", () => okJson(freeAccount())]]);
    renderWithProviders(<TrialBanner />, { config: { auth_enabled: true }, auth: SIGNED_IN });
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("renders the billing warning variant", async () => {
    const acct = makeAccount();
    acct.plan = { ...acct.plan, source: "grace", warning: "Payment failed — update your card", trial_ends_at: null };
    acct.billing.portal_available = true;
    stubFetch([["/api/me", () => okJson(acct)]]);
    renderWithProviders(<TrialBanner />, { config: { auth_enabled: true }, auth: SIGNED_IN });
    const banner = await screen.findByRole("status");
    expect(banner).toHaveAttribute("data-variant", "warning");
    expect(banner).toHaveTextContent("Payment failed — update your card");
    expect(screen.getByRole("link", { name: "Manage billing" })).toBeInTheDocument();
  });

  it("renders nothing and fetches nothing with the wall off", async () => {
    const mock = stubFetch([["/api/me", () => okJson(makeAccount())]]);
    renderWithProviders(<TrialBanner />, { config: { auth_enabled: false } });
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(mock.mock.calls.filter(([u]) => String(u).includes("/api/me"))).toHaveLength(0);
  });
});
