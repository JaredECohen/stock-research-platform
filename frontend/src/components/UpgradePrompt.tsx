import React from "react";
import { Link } from "react-router-dom";
import { Lock } from "lucide-react";
import type { EntitlementRefusal } from "@/types";
import { useConfig } from "@/auth/ConfigProvider";
import { featureCopy, formatShortUtc } from "@/lib/entitlements";

/**
 * What the user sees instead of a generic error when the backend answers
 * 402. It says which allowance ran out (or which feature is Pro-only),
 * what the Pro plan changes about that, and when a quota resets — copy
 * built from the refusal body and the `features` matrix the backend
 * publishes, so it cannot disagree with what is enforced.
 */
interface Props {
  refusal: EntitlementRefusal;
  onDismiss?: () => void;
  /** Smaller variant for inline use inside a page section. */
  compact?: boolean;
}

export function refusalForFeature(feature: string, plan: string | null = null): EntitlementRefusal {
  return {
    code: "plan_required",
    feature,
    plan,
    used: null,
    limit: null,
    resets_at: null,
    upgrade_url: "/pricing",
    message: "",
  };
}

export default function UpgradePrompt({ refusal, onDismiss, compact }: Props) {
  const { config } = useConfig();
  const copy = featureCopy(refusal.feature, config.features);
  const isQuota = refusal.code === "quota_exceeded";
  const resets = formatShortUtc(refusal.resets_at);
  const onPro = refusal.plan === "pro";

  const title = isQuota
    ? `You've used ${refusal.used ?? "all"} of ${refusal.limit ?? "your"} ${copy.label} this month`
    : `${capitalize(copy.singular)} is part of Pro`;

  return (
    <div
      className={`card border-accent-600/40 bg-accent-600/[0.05] ${compact ? "p-4" : ""}`}
      role="status"
      aria-live="polite"
      data-testid="upgrade-prompt"
    >
      <div className="flex items-start gap-3">
        <div className="h-8 w-8 shrink-0 rounded-lg bg-accent-600/20 border border-accent-600/40 flex items-center justify-center text-accent-500">
          <Lock size={16} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-semibold">{title}</div>
          <p className="text-sm text-slate-300 mt-1 leading-relaxed">
            {onPro && isQuota
              ? `The Pro allowance for ${copy.label} resets ${resets ? `on ${resets}` : "at the start of next month"} (UTC). Nothing you have done is lost.`
              : copy.value}
          </p>
          {isQuota && !onPro && resets && (
            <p className="text-xs text-slate-500 mt-1">Your Free allowance resets on {resets} (UTC calendar month).</p>
          )}
          {refusal.message && !isQuota && (
            <p className="text-xs text-slate-500 mt-1">{refusal.message}</p>
          )}
          <div className="flex flex-wrap items-center gap-2 mt-3">
            {!onPro && (
              <Link to={refusal.upgrade_url || "/pricing"} className="btn-primary text-xs">
                See Pro plans
              </Link>
            )}
            <Link to="/app/account" className="btn-ghost text-xs">
              View your usage
            </Link>
            {onDismiss && (
              <button type="button" onClick={onDismiss} className="text-xs text-slate-400 hover:text-slate-200 ml-auto">
                Dismiss
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function capitalize(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}
