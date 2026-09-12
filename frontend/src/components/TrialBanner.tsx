import React from "react";
import { Link } from "react-router-dom";
import { useAccount } from "@/auth/useAccount";
import { useConfig } from "@/auth/ConfigProvider";
import { daysUntil, formatExactUtc } from "@/lib/entitlements";

/**
 * Shell banner for the trial and for billing warnings. The exact end
 * instant comes from `/api/me` (`plan.trial_ends_at`) — never computed
 * locally from "7 days" — and is shown as a date, not a countdown, so the
 * user knows precisely when Pro features fall back to Free.
 */
export default function TrialBanner() {
  const { config } = useConfig();
  const { account } = useAccount();
  if (!config.auth_enabled || !account) return null;

  const plan = account.plan;
  const canUpgrade = config.billing_enabled && plan.source !== "subscription";

  if (plan.source === "trial" && plan.trial_ends_at) {
    const ends = formatExactUtc(plan.trial_ends_at);
    const days = daysUntil(plan.trial_ends_at);
    return (
      <div
        className="card-tight border-accent-600/40 bg-accent-600/[0.05] mb-4 text-sm flex flex-wrap items-center gap-x-3 gap-y-1"
        role="status"
        aria-live="polite"
        data-variant="trial"
      >
        <span className="badge border-accent-600/40 text-accent-500">Pro trial</span>
        <span className="text-slate-200">
          Your Pro trial ends on <span className="font-medium" data-testid="trial-ends">{ends}</span>
          {days !== null && <span className="text-slate-500"> ({days === 0 ? "today" : days === 1 ? "1 day left" : `${days} days left`})</span>}.
          {" "}After that your account drops to Free — your memos and settings are kept.
        </span>
        {canUpgrade && (
          <Link to="/app/account" className="text-accent-500 underline underline-offset-2 ml-auto">
            Keep Pro
          </Link>
        )}
      </div>
    );
  }

  if (plan.warning) {
    return (
      <div
        className="card-tight border-warn-500/40 bg-warn-500/5 mb-4 text-sm flex flex-wrap items-center gap-x-3 gap-y-1"
        role="status"
        aria-live="polite"
        data-variant="warning"
      >
        <span className="text-warn-500 font-medium">{plan.warning}</span>
        {account.billing.portal_available && (
          <Link to="/app/account" className="text-accent-500 underline underline-offset-2 ml-auto">
            Manage billing
          </Link>
        )}
      </div>
    );
  }

  return null;
}
