import React from "react";
import type { Entitlement } from "@/types";
import { featureCopy, formatShortUtc } from "@/lib/entitlements";

/**
 * One allowance as the backend reports it. Four shapes:
 *   not allowed        "Not included on your plan"
 *   follows memo       Free DCF/comps ride on memos opened this month
 *   unlimited          "Unlimited"
 *   metered            used / limit bar, reset date
 * The numbers are never computed here — `/api/me` is the meter.
 */
interface Props {
  entitlement: Entitlement;
  label?: string;
}

export default function UsageMeter({ entitlement, label }: Props) {
  const copy = featureCopy(entitlement.feature);
  const name = label || copy.label;
  const resets = formatShortUtc(entitlement.resets_at);

  let body: React.ReactNode;
  let pct: number | null = null;
  if (!entitlement.allowed) {
    body = <span className="text-slate-500">Not included on your plan</span>;
  } else if (entitlement.follows_memo) {
    body = <span className="text-slate-400">Available for tickers whose memo you opened this month</span>;
  } else if (entitlement.limit === null || entitlement.limit === undefined) {
    body = (
      <span className="text-slate-300">
        {entitlement.metered ? `${entitlement.used} used · ` : ""}
        <span className="text-accent-500">Unlimited</span>
      </span>
    );
  } else {
    const limit = Math.max(0, entitlement.limit);
    pct = limit === 0 ? 100 : Math.min(100, Math.round((entitlement.used / limit) * 100));
    body = (
      <span className="text-slate-300">
        <span className="font-mono">{entitlement.used}</span> of <span className="font-mono">{limit}</span> used
        {resets ? <span className="text-slate-500"> · resets {resets}</span> : null}
      </span>
    );
  }

  const exhausted = pct !== null && pct >= 100;

  return (
    <div className="py-2" data-testid={`meter-${entitlement.feature}`}>
      <div className="flex items-baseline justify-between gap-3">
        <div className="text-sm font-medium capitalize">{name}</div>
        <div className="text-xs">{body}</div>
      </div>
      {pct !== null && (
        <div
          className="mt-1.5 h-1.5 rounded-full bg-ink-700 overflow-hidden"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
          aria-label={`${name} used`}
        >
          <div
            className={`h-full rounded-full ${exhausted ? "bg-warn-500" : "bg-accent-600"}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      )}
    </div>
  );
}
