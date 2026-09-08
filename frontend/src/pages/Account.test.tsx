import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, within } from "@testing-library/react";
import Account from "@/pages/Account";
import { resetAccountCache } from "@/auth/useAccount";
import { SIGNED_IN, calls, freeAccount, makeAccount, okJson, renderWithProviders, stubFetch } from "@/test/providers";

const USAGE = {
  period_key: "2026-09",
  features: {},
  history: [{ feature: "memo_view", resource_ref: "NVDA", created_at: "2026-09-08T09:00:00", status: "committed", quantity: 1 }],
};

describe("Account page", () => {
  beforeEach(() => resetAccountCache());
  afterEach(() => vi.unstubAllGlobals());

  it("shows the email from the client session, the plan, the exact trial end and the meters", async () => {
    stubFetch([
      ["/api/me/usage", () => okJson(USAGE)],
      ["/api/me", () => okJson(makeAccount())],
    ]);
    renderWithProviders(<Account />, { route: "/app/account", config: { auth_enabled: true, billing_enabled: true }, auth: SIGNED_IN });
    expect(await screen.findByTestId("plan-badge")).toHaveTextContent("Pro trial");
    expect(screen.getByText("stub@example.com")).toBeInTheDocument();
    expect(screen.getByTestId("trial-ends")).toHaveTextContent("September 15, 2026 at 14:03 UTC");
    const research = screen.getByTestId("meter-research_run");
    expect(research).toHaveTextContent("2 of 20 used");
    expect(within(research).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "10");
    expect(screen.getByTestId("meter-memo_view")).toHaveTextContent("Unlimited");
    expect(screen.getByRole("button", { name: /Upgrade to Pro — \$29\.99\/month/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Annual — \$299\/year/ })).toBeInTheDocument();
    // Trial-to-paid copy states when billing starts.
    expect(screen.getByText(/keeps your remaining trial days when more than 48 hours remain/)).toBeInTheDocument();
    expect(await screen.findByText("Recent activity")).toBeInTheDocument();
    expect(screen.getByText("NVDA")).toBeInTheDocument();
  });

  it("hides Manage billing when the portal is unavailable and shows it when it is", async () => {
    stubFetch([["/api/me", () => okJson(makeAccount())]]);
    const v = renderWithProviders(<Account />, { config: { auth_enabled: true, billing_enabled: true }, auth: SIGNED_IN });
    await screen.findByTestId("plan-badge");
    expect(screen.queryByRole("button", { name: "Manage billing" })).not.toBeInTheDocument();
    v.unmount();
    resetAccountCache();
    vi.unstubAllGlobals();

    const acct = makeAccount();
    acct.plan = { ...acct.plan, source: "subscription", trial_ends_at: null, period_end: "2026-10-08T14:03:00" };
    acct.billing = { has_subscription: true, stripe_status: "active", interval: "month", portal_available: true, billing_enabled: true };
    const mock = stubFetch([
      ["/api/billing/portal", () => okJson({ url: "https://billing.stripe.com/p/x" })],
      ["/api/me", () => okJson(acct)],
    ]);
    const assign = vi.fn();
    vi.stubGlobal("location", { ...window.location, assign });
    renderWithProviders(<Account />, { config: { auth_enabled: true, billing_enabled: true }, auth: SIGNED_IN });
    expect(await screen.findByTestId("plan-badge")).toHaveTextContent("Pro");
    expect(screen.getByText(/Current period renews on October 8, 2026 at 14:03 UTC/)).toBeInTheDocument();
    // Already subscribed: no upgrade buttons.
    expect(screen.queryByRole("button", { name: /Upgrade to Pro/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Manage billing" }));
    await new Promise((r) => setTimeout(r, 0));
    expect(calls(mock, "/api/billing/portal")).toHaveLength(1);
    expect(assign).toHaveBeenCalledWith("https://billing.stripe.com/p/x");
  });

  it("renders Free meters with follows-memo and not-included states, and no upgrade buttons when billing is off", async () => {
    stubFetch([["/api/me", () => okJson(freeAccount())]]);
    renderWithProviders(<Account />, { config: { auth_enabled: true, billing_enabled: false }, auth: SIGNED_IN });
    expect(await screen.findByTestId("plan-badge")).toHaveTextContent("Free");
    expect(screen.getByTestId("meter-memo_view")).toHaveTextContent("3 of 3 used");
    expect(screen.getByTestId("meter-dcf")).toHaveTextContent("Available for tickers whose memo you opened this month");
    expect(screen.getByTestId("meter-portfolio")).toHaveTextContent("Not included on your plan");
    expect(screen.queryByRole("button", { name: /Upgrade to Pro/ })).not.toBeInTheDocument();
    expect(screen.getByText("Paid plans are not open on this deployment yet.")).toBeInTheDocument();
  });

  it("shows the upgrade prompt for a locked nav item via ?upgrade=", async () => {
    stubFetch([["/api/me", () => okJson(freeAccount())]]);
    renderWithProviders(<Account />, { route: "/app/account?upgrade=portfolio", config: { auth_enabled: true }, auth: SIGNED_IN });
    expect(await screen.findByTestId("upgrade-prompt")).toHaveTextContent("Portfolio build is part of Pro");
  });

  it("starts checkout for the chosen interval and follows the returned URL", async () => {
    const mock = stubFetch([
      ["/api/billing/checkout", () => okJson({ url: "https://checkout.stripe.com/c/x" })],
      ["/api/me", () => okJson(freeAccount())],
    ]);
    const assign = vi.fn();
    vi.stubGlobal("location", { ...window.location, assign });
    renderWithProviders(<Account />, { config: { auth_enabled: true, billing_enabled: true }, auth: SIGNED_IN });
    await screen.findByTestId("plan-badge");
    fireEvent.click(screen.getByRole("button", { name: /Annual/ }));
    await new Promise((r) => setTimeout(r, 0));
    const [, init] = calls(mock, "/api/billing/checkout")[0];
    expect(JSON.parse(String(init?.body))).toEqual({ interval: "year" });
    expect(assign).toHaveBeenCalledWith("https://checkout.stripe.com/c/x");
  });

  it("explains that accounts are off when the wall is disabled", () => {
    stubFetch();
    renderWithProviders(<Account />, { config: { auth_enabled: false } });
    expect(screen.getByText(/Accounts are not enabled on this deployment/)).toBeInTheDocument();
  });
});
