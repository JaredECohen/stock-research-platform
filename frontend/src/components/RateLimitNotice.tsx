import React, { useEffect, useState } from "react";
import { Clock } from "lucide-react";
import type { RateLimitRefusal } from "@/types";

/**
 * Shown on a 429. The user's input is NOT ours to drop: the parent keeps
 * it in state and passes `onRetry`, which this component enables once
 * `retry_after` seconds have elapsed. The countdown is cosmetic — the
 * backend re-checks the window on the retry.
 */
interface Props {
  refusal: RateLimitRefusal;
  onRetry: () => void;
  /** e.g. "Your message is kept in the box below." */
  preservedNote?: string;
  onDismiss?: () => void;
}

const SCOPE_LABEL: Record<string, string> = {
  ip: "from this network address",
  "user:llm_light": "for AI requests",
  "user:research": "for research runs",
  "user:data": "for data requests",
  "user:series": "for series requests",
  "user:checkout": "for checkout",
  concurrency: "for simultaneous requests",
};

function scopeLabel(scope: string): string {
  if (SCOPE_LABEL[scope]) return SCOPE_LABEL[scope];
  const base = scope.replace(/^(user|ip):/, "").replace(/_/g, " ");
  return base ? `for ${base} requests` : "";
}

export default function RateLimitNotice({ refusal, onRetry, preservedNote, onDismiss }: Props) {
  const [remaining, setRemaining] = useState(Math.max(0, Math.ceil(refusal.retry_after)));

  useEffect(() => {
    setRemaining(Math.max(0, Math.ceil(refusal.retry_after)));
  }, [refusal]);

  useEffect(() => {
    if (remaining <= 0) return;
    const id = window.setInterval(() => setRemaining((n) => (n > 0 ? n - 1 : 0)), 1000);
    return () => window.clearInterval(id);
  }, [remaining > 0]);

  const concurrent = refusal.code === "concurrency_limited";
  const title = concurrent ? "Another request of this kind is still running" : "Slow down a moment";

  return (
    <div className="card-tight border-warn-500/40 bg-warn-500/5" role="status" aria-live="polite" data-testid="rate-limit-notice">
      <div className="flex items-start gap-3">
        <Clock size={16} className="text-warn-500 mt-0.5 shrink-0" />
        <div className="flex-1 min-w-0 text-sm">
          <div className="font-medium text-warn-500">{title}</div>
          <p className="text-slate-300 mt-1 leading-relaxed">
            {concurrent
              ? "Wait for it to finish, then try again."
              : `Too many requests ${scopeLabel(refusal.scope)}.`}{" "}
            {remaining > 0 ? (
              <>
                You can retry in <span className="font-mono" data-testid="retry-countdown">{remaining}s</span>.
              </>
            ) : (
              "You can retry now."
            )}
          </p>
          {preservedNote && <p className="text-xs text-slate-500 mt-1">{preservedNote}</p>}
          <div className="flex items-center gap-2 mt-2">
            <button type="button" className="btn-ghost text-xs" onClick={onRetry} disabled={remaining > 0}>
              {remaining > 0 ? `Retry in ${remaining}s` : "Retry"}
            </button>
            {onDismiss && (
              <button type="button" onClick={onDismiss} className="text-xs text-slate-400 hover:text-slate-200">
                Dismiss
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
