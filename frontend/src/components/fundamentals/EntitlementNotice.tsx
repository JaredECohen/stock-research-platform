import React from "react";
import { Link } from "react-router-dom";
import { Lock } from "lucide-react";
import type { AppliedLimits, StructuredErrorDetail } from "@/types";

/**
 * The two ways a plan shapes a chart, each rendered with the exact
 * numbers the backend sent — never a generic failure and never a literal:
 *
 *   402 `plan_required`  the request asked for more companies × metrics
 *                        than the plan draws; `extra.limits` / `requested`
 *                        / `upgrade` carry what was asked, what the plan
 *                        allows, and what Pro unlocks.
 *   capped_by_plan       the URL asked for more years than the plan draws;
 *                        the chart still rendered at the ceiling.
 *
 * Neither appears on an ordinary Free page load: the default range is not
 * a cap and the backend does not report it as one.
 */
export interface EntitlementNoticeProps {
  refusal?: StructuredErrorDetail | null;
  capped?: { applied: AppliedLimits; requestedYears: number } | null;
  onDismiss?: () => void;
  className?: string;
}

interface ShapeLike {
  max_companies?: unknown;
  max_metrics?: unknown;
  max_years?: unknown;
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** "2 companies × 2 metrics × 5 years" / "… × full history". */
export function shapeText(shape: ShapeLike | null | undefined): string | null {
  if (!shape) return null;
  const c = num(shape.max_companies);
  const m = num(shape.max_metrics);
  if (c === null || m === null) return null;
  const y = num(shape.max_years);
  return `${c} companies × ${m} metrics × ${y === null ? "full history" : `${y} years`}`;
}

function requestedText(req: unknown): string | null {
  if (!req || typeof req !== "object") return null;
  const r = req as { companies?: unknown; metrics?: unknown };
  const c = num(r.companies);
  const m = num(r.metrics);
  if (c === null || m === null) return null;
  return `${c} ${c === 1 ? "company" : "companies"} × ${m} ${m === 1 ? "metric" : "metrics"}`;
}

function planName(plan: string | null | undefined): string {
  return plan === "pro" ? "Pro" : plan === "free" ? "The Free plan" : "This plan";
}

export default function EntitlementNotice({ refusal, capped, onDismiss, className = "" }: EntitlementNoticeProps) {
  if (!refusal && !capped) return null;

  let title: string;
  let body: React.ReactNode;
  let upgradeUrl = "/pricing";
  let onPro = false;

  if (refusal) {
    const extra = refusal.extra ?? {};
    const limits = shapeText(extra.limits as ShapeLike | undefined);
    const requested = requestedText(extra.requested);
    const upgrade = extra.upgrade as { plan?: string; limits?: ShapeLike; url?: string } | undefined;
    const pro = shapeText(upgrade?.limits);
    upgradeUrl = upgrade?.url || refusal.upgrade_url || "/pricing";
    onPro = refusal.plan === "pro";
    title = "This chart is bigger than the plan draws";
    body = (
      <>
        {limits && requested ? (
          <p>
            {planName(refusal.plan)} draws up to <strong>{limits}</strong>; this URL asks for <strong>{requested}</strong>. Remove a company or a
            metric to draw it.
          </p>
        ) : (
          <p>{refusal.message || "The request exceeds what the plan draws."}</p>
        )}
        {pro && !onPro && (
          <p className="mt-1">
            Pro draws up to <strong>{pro}</strong>.
          </p>
        )}
      </>
    );
  } else {
    const c = capped!;
    const ceiling = c.applied.max_years;
    title = "Range capped by the plan";
    body = (
      <p>
        This URL asks for <strong>{c.requestedYears} years</strong>; the plan draws up to{" "}
        <strong>{ceiling === null ? "the full history" : `${ceiling} years`}</strong>, so the chart shows that much. Pro draws the full stored history.
      </p>
    );
  }

  return (
    <div className={`card border-accent-600/40 bg-accent-600/[0.05] ${className}`} role="status" aria-live="polite" data-testid="entitlement-notice">
      <div className="flex items-start gap-3">
        <div className="h-8 w-8 shrink-0 rounded-lg bg-accent-600/20 border border-accent-600/40 flex items-center justify-center text-accent-500">
          <Lock size={16} aria-hidden="true" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-semibold">{title}</div>
          <div className="text-sm text-slate-300 mt-1 leading-relaxed">{body}</div>
          <div className="flex flex-wrap items-center gap-2 mt-3">
            {!onPro && (
              <Link to={upgradeUrl} className="btn-primary text-xs">
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
