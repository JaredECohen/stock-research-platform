import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "@/api/client";
import { invalidateAccount } from "@/auth/useAccount";
import { track } from "@/lib/analytics";
import type { Account } from "@/types";

/**
 * /app/billing/success — where Stripe Checkout returns.
 *
 * Landing here is NOT proof of payment (anyone can type the URL, and the
 * webhook may not have arrived yet). We poll `GET /api/me` for up to 30s
 * and only say "active" once the backend's subscription state says so;
 * after that we offer `POST /api/billing/reconcile`, which asks Stripe
 * directly through the backend. Nothing from the URL is read.
 */
export const POLL_INTERVAL_MS = 3000;
export const POLL_DEADLINE_MS = 30_000;

type Phase = "checking" | "confirmed" | "unconfirmed" | "reconciling";

export function subscriptionConfirmed(acct: Account | null | undefined): boolean {
  if (!acct) return false;
  if (acct.plan.source === "subscription") return true;
  const s = acct.billing?.stripe_status;
  return acct.billing?.has_subscription === true && (s === "active" || s === "trialing");
}

export default function BillingSuccess() {
  const [phase, setPhase] = useState<Phase>("checking");
  const [account, setAccount] = useState<Account | null>(null);
  const [error, setError] = useState<string | null>(null);
  const confirmedOnce = useRef(false);

  const confirm = (acct: Account) => {
    setAccount(acct);
    if (subscriptionConfirmed(acct)) {
      setPhase("confirmed");
      if (!confirmedOnce.current) {
        confirmedOnce.current = true;
        invalidateAccount();
        track("checkout_completed", { interval: acct.billing.interval ?? undefined, plan: acct.plan.plan });
      }
      return true;
    }
    return false;
  };

  useEffect(() => {
    let cancelled = false;
    const started = Date.now();
    let timer: number | null = null;

    const tick = async () => {
      if (cancelled) return;
      try {
        const acct = await api.me();
        if (cancelled) return;
        if (confirm(acct)) return;
      } catch {
        // Keep polling — a blip should not end in "unconfirmed" early.
      }
      if (Date.now() - started >= POLL_DEADLINE_MS) {
        setPhase("unconfirmed");
        return;
      }
      timer = window.setTimeout(() => void tick(), POLL_INTERVAL_MS);
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, []);

  const reconcile = async () => {
    setPhase("reconciling");
    setError(null);
    try {
      const acct = await api.reconcile();
      if (!confirm(acct)) setPhase("unconfirmed");
    } catch (e) {
      setError((e as Error).message);
      setPhase("unconfirmed");
    }
  };

  return (
    <div className="max-w-xl space-y-4">
      <h1 className="text-2xl font-semibold">Subscription</h1>

      {phase === "checking" && (
        <div className="card flex items-center gap-3" role="status" aria-live="polite">
          <span aria-hidden className="inline-block h-4 w-4 rounded-full border-2 border-accent-500 border-t-transparent animate-spin" />
          <div className="text-sm text-slate-200">
            Confirming your subscription with the billing provider…
            <div className="text-xs text-slate-500 mt-0.5">This usually takes a few seconds.</div>
          </div>
        </div>
      )}

      {phase === "reconciling" && (
        <div className="card text-sm text-slate-200" role="status" aria-live="polite">
          Checking with Stripe directly…
        </div>
      )}

      {phase === "confirmed" && (
        <div className="card border-accent-600/40 bg-accent-600/[0.05]" role="status" aria-live="polite" data-testid="billing-confirmed">
          <div className="font-semibold text-accent-500">Your Pro subscription is active</div>
          <p className="text-sm text-slate-300 mt-1">
            {account?.billing.stripe_status === "trialing"
              ? "Your remaining trial days are kept; billing starts when the trial ends."
              : "Thank you. Pro features are unlocked now."}
          </p>
          <div className="flex gap-2 mt-3">
            <Link to="/app" className="btn-primary text-sm">Go to the dashboard</Link>
            <Link to="/app/account" className="btn-ghost text-sm">View your account</Link>
          </div>
        </div>
      )}

      {phase === "unconfirmed" && (
        <div className="card border-warn-500/40 bg-warn-500/5" role="status" aria-live="polite" data-testid="billing-unconfirmed">
          <div className="font-semibold text-warn-500">We haven&apos;t received confirmation yet</div>
          <p className="text-sm text-slate-300 mt-1 leading-relaxed">
            Stripe confirms payments by notifying us; that notification has not arrived. If you completed checkout, ask
            us to check with Stripe directly. If you cancelled, nothing was charged.
          </p>
          {error && <div className="text-sm text-danger-500 mt-2">{error}</div>}
          <div className="flex flex-wrap gap-2 mt-3">
            <button type="button" className="btn-primary text-sm" onClick={() => void reconcile()}>
              Check with Stripe now
            </button>
            <Link to="/app/account" className="btn-ghost text-sm">Back to account</Link>
          </div>
        </div>
      )}
    </div>
  );
}
