import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen } from "@testing-library/react";
import BillingSuccess, { POLL_DEADLINE_MS, subscriptionConfirmed } from "@/pages/BillingSuccess";
import { SIGNED_IN, calls, freeAccount, makeAccount, okJson, renderWithProviders, stubFetch } from "@/test/providers";
import type { Account } from "@/types";

function subscribed(): Account {
  const acct = makeAccount();
  acct.plan = { ...acct.plan, source: "subscription", trial_ends_at: null, period_end: "2026-10-08T14:03:00" };
  acct.billing = { has_subscription: true, stripe_status: "active", interval: "month", portal_available: true, billing_enabled: true };
  return acct;
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe("BillingSuccess", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("never claims payment from the URL: an unchanged account ends in the unconfirmed state with a reconcile offer", async () => {
    const mock = stubFetch([["/api/me", () => okJson(freeAccount())]]);
    renderWithProviders(<BillingSuccess />, {
      route: "/app/billing/success?session_id=cs_test_123&paid=true",
      config: { auth_enabled: true, billing_enabled: true },
      auth: SIGNED_IN,
    });
    expect(screen.getByRole("status")).toHaveTextContent("Confirming your subscription");
    await advance(POLL_DEADLINE_MS + 4000);
    expect(screen.queryByTestId("billing-confirmed")).not.toBeInTheDocument();
    expect(screen.getByTestId("billing-unconfirmed")).toHaveTextContent("We haven't received confirmation yet");
    expect(screen.getByRole("button", { name: "Check with Stripe now" })).toBeInTheDocument();
    // It polled more than once during the window.
    expect(calls(mock, "/api/me").length).toBeGreaterThan(2);
  });

  it("confirms only once /api/me reports a subscription", async () => {
    let n = 0;
    stubFetch([["/api/me", () => okJson(n++ < 2 ? freeAccount() : subscribed())]]);
    renderWithProviders(<BillingSuccess />, { route: "/app/billing/success", config: { auth_enabled: true }, auth: SIGNED_IN });
    await advance(100);
    expect(screen.queryByTestId("billing-confirmed")).not.toBeInTheDocument();
    await advance(7000);
    expect(screen.getByTestId("billing-confirmed")).toHaveTextContent("Your Pro subscription is active");
    expect(screen.getByRole("link", { name: "Go to the dashboard" })).toHaveAttribute("href", "/app");
  });

  it("reconcile applies the backend answer and can still end unconfirmed", async () => {
    let reconciled = 0;
    stubFetch([
      ["/api/billing/reconcile", () => okJson(reconciled++ === 0 ? freeAccount() : subscribed())],
      ["/api/me", () => okJson(freeAccount())],
    ]);
    renderWithProviders(<BillingSuccess />, { route: "/app/billing/success", config: { auth_enabled: true }, auth: SIGNED_IN });
    await advance(POLL_DEADLINE_MS + 4000);
    fireEvent.click(screen.getByRole("button", { name: "Check with Stripe now" }));
    await advance(10);
    expect(screen.getByTestId("billing-unconfirmed")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Check with Stripe now" }));
    await advance(10);
    expect(screen.getByTestId("billing-confirmed")).toBeInTheDocument();
  });

  it("keeps polling through a transient fetch failure", async () => {
    let n = 0;
    stubFetch([["/api/me", () => (n++ === 0 ? Promise.reject(new Error("blip")) : okJson(subscribed()))]]);
    renderWithProviders(<BillingSuccess />, { route: "/app/billing/success", config: { auth_enabled: true }, auth: SIGNED_IN });
    await advance(3500);
    expect(screen.getByTestId("billing-confirmed")).toBeInTheDocument();
  });

  describe("subscriptionConfirmed", () => {
    it("requires server-side subscription state", () => {
      expect(subscriptionConfirmed(null)).toBe(false);
      expect(subscriptionConfirmed(freeAccount())).toBe(false);
      expect(subscriptionConfirmed(makeAccount())).toBe(false); // trial is not a subscription
      expect(subscriptionConfirmed(subscribed())).toBe(true);
      const trialing = subscribed();
      trialing.plan.source = "trial";
      trialing.billing.stripe_status = "trialing";
      expect(subscriptionConfirmed(trialing)).toBe(true); // checkout during the local trial → Stripe "trialing" is expected
      const incomplete = subscribed();
      incomplete.plan.source = "default";
      incomplete.billing.stripe_status = "incomplete";
      expect(subscriptionConfirmed(incomplete)).toBe(false);
    });
  });
});
