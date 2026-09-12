import React, { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, isApiError } from "@/api/client";
import { useAuth } from "@/auth/AuthContext";
import { useConfig } from "@/auth/ConfigProvider";
import { invalidateAccount, useAccount } from "@/auth/useAccount";
import UpgradePrompt, { refusalForFeature } from "@/components/UpgradePrompt";
import UsageMeter from "@/components/UsageMeter";
import { track } from "@/lib/analytics";
import { featureCopy, formatExactUtc, formatShortUtc, freeHeadlineAllowances, planLabel } from "@/lib/entitlements";
import type { Account as AccountShape, BillingInterval, UsageResponse } from "@/types";

/**
 * /app/account — plan, exact trial end, usage meters, upgrade / manage
 * billing, sign out. The email comes from the auth provider's client
 * session (the backend does not store it). Every number is `/api/me`.
 */
const METER_ORDER = [
  "memo_view", "research_run", "pm_chat", "dcf", "comps",
  "portfolio", "macro", "track_record", "memo_history",
];

function centsToPrice(cents: number, currency: string): string {
  const amount = (cents / 100).toFixed(2).replace(/\.00$/, "");
  return currency.toLowerCase() === "usd" ? `$${amount}` : `${amount} ${currency.toUpperCase()}`;
}

export default function Account() {
  const { config } = useConfig();
  const auth = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const { account, loading, error, refresh } = useAccount();
  const [usage, setUsage] = useState<UsageResponse | null>(null);
  const [busy, setBusy] = useState<"checkout" | "portal" | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const upgradeFor = params.get("upgrade");

  useEffect(() => {
    if (!account) return;
    api.usage().then(setUsage).catch(() => setUsage(null));
  }, [account?.period_key, account?.user.id]);

  if (!config.auth_enabled) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Account</h1>
        <div className="card text-sm text-slate-300">
          Accounts are not enabled on this deployment. Every feature is open; nothing is metered.
        </div>
      </div>
    );
  }

  const startCheckout = async (interval: BillingInterval) => {
    setBusy("checkout");
    setActionError(null);
    track("checkout_started", { interval, plan: account?.plan.plan });
    try {
      const { url } = await api.checkout(interval);
      // Stripe-hosted Checkout. Coming back to /app/billing/success proves
      // nothing on its own; that page waits for /api/me to say "subscription".
      window.location.assign(url);
    } catch (e) {
      setActionError(describe(e));
      setBusy(null);
    }
  };

  const openPortal = async () => {
    setBusy("portal");
    setActionError(null);
    try {
      const { url } = await api.portal();
      window.location.assign(url);
    } catch (e) {
      setActionError(describe(e));
      setBusy(null);
    }
  };

  const signOut = async () => {
    await auth.signOut();
    navigate("/", { replace: true });
  };

  const plan = account?.plan;
  const billing = account?.billing;
  const isSubscribed = plan?.source === "subscription" || plan?.source === "grace";
  const showUpgrade = config.billing_enabled && account && !isSubscribed && plan?.plan !== "none";
  const prices = config.prices;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Account</h1>
        <div className="text-sm text-slate-400 mt-1">
          {auth.user?.email ? (
            <>
              Signed in as <span className="text-slate-200">{auth.user.email}</span>
              {auth.user.emailVerified ? null : <span className="text-warn-500"> (email not verified)</span>}
            </>
          ) : (
            "Signed in"
          )}
        </div>
      </div>

      {upgradeFor && <UpgradePrompt refusal={refusalForFeature(upgradeFor, plan?.plan ?? null)} />}

      {error && !account && (
        <div className="card-tight border-danger-500/40 text-danger-500 text-sm">
          {error}{" "}
          <button type="button" className="underline" onClick={() => void refresh()}>
            Retry
          </button>
        </div>
      )}
      {loading && !account && <div className="card text-sm text-slate-400">Loading your account…</div>}

      {account && plan && billing && (
        <>
          <section className="card space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <div className="section-title">Plan</div>
              <span className={`badge ${plan.plan === "pro" ? "border-accent-600/40 text-accent-500" : "border-ink-700 text-slate-300"}`} data-testid="plan-badge">
                {planLabel(plan.plan, plan.source)}
              </span>
              {billing.stripe_status && (
                <span className="text-xs text-slate-500">Stripe status: {billing.stripe_status}</span>
              )}
            </div>
            <PlanDetail plan={plan} entitlements={account.entitlements} trialDays={config.trial_days} />
            {plan.warning && <div className="text-sm text-warn-500">{plan.warning}</div>}

            {actionError && <div className="text-sm text-danger-500">{actionError}</div>}

            <div className="flex flex-wrap gap-2 pt-1">
              {showUpgrade && (
                <>
                  <button type="button" className="btn-primary text-sm" disabled={busy !== null} onClick={() => void startCheckout("month")}>
                    {busy === "checkout" ? "Opening checkout…" : `Upgrade to Pro — ${centsToPrice(prices.monthly_cents, prices.currency)}/month`}
                  </button>
                  <button type="button" className="btn-ghost text-sm" disabled={busy !== null} onClick={() => void startCheckout("year")}>
                    {`Annual — ${centsToPrice(prices.annual_cents, prices.currency)}/year`}
                  </button>
                </>
              )}
              {billing.portal_available && (
                <button type="button" className="btn-ghost text-sm" disabled={busy !== null} onClick={() => void openPortal()}>
                  {busy === "portal" ? "Opening billing…" : "Manage billing"}
                </button>
              )}
              <Link to="/pricing" className="text-sm text-accent-500 underline underline-offset-2 self-center">
                Compare plans
              </Link>
            </div>
            {showUpgrade && plan.source === "trial" && (
              <p className="text-xs text-slate-500">
                Upgrading during the trial keeps your remaining trial days when more than 48 hours remain; billing starts when the trial ends. With less time left, billing starts at checkout.
              </p>
            )}
            {!config.billing_enabled && (
              <p className="text-xs text-slate-500">Paid plans are not open on this deployment yet.</p>
            )}
          </section>

          <section className="card">
            <div className="flex items-center justify-between">
              <div className="section-title">Usage this month</div>
              <div className="text-xs text-slate-500">
                {account.period_key} · UTC calendar month
                {!account.usage_limits_enabled && " · limits not enforced"}
              </div>
            </div>
            <div className="divide-y divide-ink-800 mt-2">
              {METER_ORDER.filter((f) => account.entitlements[f]).map((f) => (
                <UsageMeter key={f} entitlement={account.entitlements[f]} />
              ))}
            </div>
            <p className="text-[11px] text-slate-500 mt-3 leading-snug">
              Stored-memo views count distinct tickers per month — opening the same memo twice is one view. DCF and comps on Free follow the memos you have opened.
            </p>
          </section>

          {usage && Array.isArray(usage.history) && usage.history.length > 0 && (
            <section className="card">
              <div className="section-title mb-2">Recent activity</div>
              <table className="w-full text-sm">
                <thead className="text-xs text-slate-500 border-b border-ink-700">
                  <tr>
                    <th className="text-left font-medium py-1">When (UTC)</th>
                    <th className="text-left font-medium">What</th>
                    <th className="text-left font-medium">Ticker</th>
                    <th className="text-left font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {usage.history.slice(0, 20).map((h, i) => (
                    <tr key={i} className="border-b border-ink-800">
                      <td className="py-1 text-slate-400">{formatShortUtc(h.created_at) ?? h.created_at}</td>
                      <td className="capitalize">{featureCopy(h.feature).singular}</td>
                      <td className="font-mono">{h.resource_ref || "—"}</td>
                      <td className="text-slate-400">{h.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
        </>
      )}

      <section className="card-tight flex items-center justify-between">
        <div className="text-sm text-slate-400">Signed in on this browser.</div>
        <button type="button" className="btn-ghost text-sm" onClick={() => void signOut()}>
          Sign out
        </button>
      </section>
    </div>
  );
}

interface PlanDetailProps {
  plan: AccountShape["plan"];
  entitlements: AccountShape["entitlements"];
  /** From `/api/public/config`; null when unknown, in which case the trial
   *  is described without a length (the end date is exact either way). */
  trialDays: number | null;
}

function PlanDetail({ plan, entitlements, trialDays }: PlanDetailProps) {
  if (plan.source === "trial" && plan.trial_ends_at) {
    return (
      <div className="text-sm text-slate-300">
        Your {trialDays ? `${trialDays}-day ` : ""}Pro trial ends on <span className="font-medium" data-testid="trial-ends">{formatExactUtc(plan.trial_ends_at)}</span>. No card is on file; afterwards the account drops to Free and nothing is deleted.
      </div>
    );
  }
  if (plan.source === "subscription" && plan.period_end) {
    return (
      <div className="text-sm text-slate-300">
        {plan.cancel_at_period_end
          ? `Pro ends on ${formatExactUtc(plan.period_end)} — your subscription will not renew.`
          : `Current period renews on ${formatExactUtc(plan.period_end)}.`}
      </div>
    );
  }
  if (plan.source === "grace" && plan.grace_until) {
    return (
      <div className="text-sm text-slate-300">
        Pro stays on until <span className="font-medium">{formatExactUtc(plan.grace_until)}</span> while the payment is retried.
      </div>
    );
  }
  if (plan.plan === "none") {
    return <div className="text-sm text-danger-500">This account is suspended. Contact support.</div>;
  }
  // The headline numbers are this user's own /api/me limits (overrides
  // included); when none is numeric the clause is dropped, never guessed.
  const headline = freeHeadlineAllowances(entitlements);
  return (
    <div className="text-sm text-slate-300" data-testid="free-summary">
      Free Explorer: browse the committee&apos;s stored work{headline ? ` — ${headline}` : ""}.
    </div>
  );
}

function describe(e: unknown): string {
  if (isApiError(e)) {
    if (e.code === "billing_unavailable") return "Billing is temporarily unavailable. Please try again in a few minutes.";
    if (e.code === "email_unverified") return "Please verify your email address before upgrading.";
    if (e.code === "already_subscribed") {
      invalidateAccount();
      return "You already have an active subscription — refreshing your plan.";
    }
    return e.detail || e.message;
  }
  return String(e);
}
