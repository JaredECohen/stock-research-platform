import React, { useEffect, useId, useState } from "react";
import { Sparkles } from "lucide-react";
import { api, isApiError } from "@/api/client";
import { invalidateAccount } from "@/auth/useAccount";
import RateLimitNotice from "@/components/RateLimitNotice";
import UpgradePrompt from "@/components/UpgradePrompt";
import UsageMeter from "@/components/UsageMeter";
import { formatShortUtc } from "@/lib/entitlements";
import type { CommentaryResponse, Entitlement, EntitlementRefusal, RateLimitRefusal } from "@/types";

/**
 * "Explain this chart". The request carries only what the chart shows
 * (tickers, metrics, the years the server applied, and the fingerprint of
 * the displayed values), and the answer comes back in two sections that
 * are kept apart all the way to the markup — `<section aria-labelledby>`
 * each — because the research process never lets observed data and
 * interpretation blend:
 *
 *   Observed in the data   sentences recomputed from the displayed series
 *   From stored memos      excerpts attributed to a memo version, with a
 *                          stale flag when the memo predates the data
 *
 * A degraded answer (no model, anonymous visitor) still renders the
 * observed section and says nothing was charged. The usage meter is
 * `/api/me`'s number, refreshed after each charged answer; the button is
 * disabled with the reason when the allowance is exhausted.
 */
export interface CommentaryPanelProps {
  tickers: string[];
  metrics: string[];
  /** `limits.applied.max_years` of the displayed series (null = full history). */
  years: number | null;
  /** Fingerprint of the displayed series; null until a chart is drawn. */
  fingerprint: string | null;
  /** The viewer's `chart_commentary` entitlement when there is an account. */
  entitlement?: Entitlement | null;
  /** True when `/api/me` is live (wall on, signed in): refresh it after a charge. */
  accountEnabled?: boolean;
  className?: string;
}

const FOCUS = "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500";

function limitReason(e: Entitlement | null | undefined): string | null {
  if (!e) return null;
  if (!e.allowed) return "Chart commentary is not included on this plan.";
  if (typeof e.limit === "number") {
    const exhausted = (typeof e.remaining === "number" && e.remaining <= 0) || e.used >= e.limit;
    if (exhausted) {
      const resets = formatShortUtc(e.resets_at);
      return `You've used ${e.used} of ${e.limit} chart commentaries this month${resets ? `; the allowance resets ${resets} (UTC)` : ""}.`;
    }
  }
  return null;
}

export default function CommentaryPanel({ tickers, metrics, years, fingerprint, entitlement, accountEnabled = false, className = "" }: CommentaryPanelProps) {
  const headingId = useId();
  const reasonId = useId();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CommentaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refusal, setRefusal] = useState<EntitlementRefusal | null>(null);
  const [rateLimit, setRateLimit] = useState<RateLimitRefusal | null>(null);

  // The answer belongs to one fingerprint: when the chart changes, the
  // old commentary would describe data no longer on screen.
  useEffect(() => {
    setResult(null);
    setError(null);
    setRefusal(null);
    setRateLimit(null);
  }, [fingerprint]);

  const generate = async () => {
    if (!fingerprint) return;
    setLoading(true);
    setError(null);
    setRefusal(null);
    setRateLimit(null);
    const fp = fingerprint;
    try {
      const r = await api.fundamentalsCommentary({ tickers, metrics, years, fingerprint: fp });
      setResult(r);
      // A non-degraded answer was charged (or served from cache against a
      // reserved meter); the meter on this panel is `/api/me`'s number.
      if (accountEnabled && !r.degraded) invalidateAccount();
    } catch (e) {
      if (isApiError(e) && e.entitlement) setRefusal(e.entitlement);
      else if (isApiError(e) && e.rateLimit) setRateLimit(e.rateLimit);
      else if (isApiError(e) && e.status === 409) setError("The chart changed while the request was in flight. Draw it again, then ask for commentary.");
      else setError(isApiError(e) ? e.detail || e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const noChart = !fingerprint ? "Draw a chart first: commentary describes exactly what is on screen." : null;
  const exhausted = limitReason(entitlement);
  const quota = refusal?.code === "quota_exceeded" ? refusal.message || "This month's allowance is used up." : null;
  const reason = noChart ?? exhausted ?? quota;
  const disabled = loading || !!reason;

  const status = loading
    ? "Generating commentary…"
    : result
      ? result.degraded
        ? "Commentary unavailable; showing what the data says without a model."
        : `Commentary ready${result.cache_hit ? " (cached)" : ""}.`
      : "";

  return (
    <section aria-labelledby={headingId} className={`card space-y-3 ${className}`} data-testid="commentary-panel">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id={headingId} className="text-base font-semibold">
            Commentary
          </h2>
          <p className="text-xs text-slate-400 mt-0.5 leading-relaxed max-w-prose">
            Two sources, kept apart: what the displayed numbers show, and what stored memos say about these companies. Scenarios and
            observations, not recommendations.
          </p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <button
            type="button"
            onClick={() => void generate()}
            disabled={disabled}
            aria-describedby={reason ? reasonId : undefined}
            className={`btn-primary text-xs disabled:opacity-50 disabled:cursor-not-allowed ${FOCUS}`}
            data-testid="commentary-button"
          >
            <Sparkles size={14} aria-hidden="true" />
            {loading ? "Generating…" : "Explain this chart"}
          </button>
          {reason && (
            <p id={reasonId} className="text-xs text-slate-400 text-right max-w-xs" data-testid="commentary-reason">
              {reason}
            </p>
          )}
        </div>
      </div>

      {entitlement && <UsageMeter entitlement={entitlement} label="chart commentaries" />}

      <div role="status" aria-live="polite" className="text-xs text-slate-400 min-h-[1rem]" data-testid="commentary-status">
        {status}
      </div>

      {refusal && <UpgradePrompt refusal={refusal} compact onDismiss={() => setRefusal(null)} />}
      {rateLimit && (
        <RateLimitNotice refusal={rateLimit} onRetry={() => void generate()} onDismiss={() => setRateLimit(null)} preservedNote="Your chart and selection are unchanged." />
      )}
      {error && (
        <div className="card-tight border-danger-500/40 text-sm text-danger-500" role="alert">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          {result.degraded && (
            <div className="card-tight border-warn-500/40 bg-warn-500/5 text-sm text-slate-200" role="note" data-testid="commentary-degraded">
              <span className="text-warn-500 font-medium">AI commentary unavailable</span>
              {result.degraded_reason ? <>: {result.degraded_reason}</> : null}. Nothing was charged. The observations below are computed from the
              displayed data, without a model.
            </div>
          )}

          <section aria-labelledby={`${headingId}-observed`} data-testid="commentary-observed">
            <h3 id={`${headingId}-observed`} className="section-title mb-1.5">
              Observed in the data
            </h3>
            {result.observed.length === 0 ? (
              <p className="text-sm text-slate-500">No observations were produced for the displayed series.</p>
            ) : (
              <ul className="space-y-1.5 text-sm text-slate-200">
                {result.observed.map((item, i) => (
                  <li key={i}>
                    {item.text}
                    {item.refs.length > 0 && (
                      <span className="block text-[11px] text-slate-500 font-mono">
                        {item.refs.map((r) => `${r.ticker} ${r.metric} ${r.period}`).join(" · ")}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section aria-labelledby={`${headingId}-memo`} data-testid="commentary-memo">
            <h3 id={`${headingId}-memo`} className="section-title mb-1.5">
              From stored memos
            </h3>
            {result.memo_view.length === 0 ? (
              <p className="text-sm text-slate-500">No stored memo excerpts for the selected companies.</p>
            ) : (
              <ul className="space-y-1.5 text-sm text-slate-200">
                {result.memo_view.map((item, i) => {
                  const when = formatShortUtc(item.memo_generated_at);
                  return (
                    <li key={i}>
                      {item.text}
                      <span className="block text-[11px] text-slate-500">
                        <span className="font-mono">{item.ticker}</span> memo
                        {typeof item.memo_version === "number" ? ` v${item.memo_version}` : ""}
                        {when ? ` · generated ${when}` : ""}
                        {item.memo_stale && (
                          <>
                            {" "}
                            <span className="badge badge-mixed text-[10px]" title={item.memo_stale_reason ?? undefined}>
                              stale
                            </span>
                            {item.memo_stale_reason ? <span className="sr-only">: {item.memo_stale_reason}</span> : null}
                          </>
                        )}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>

          {result.caveats.length > 0 && (
            <section aria-labelledby={`${headingId}-caveats`} data-testid="commentary-caveats">
              <h3 id={`${headingId}-caveats`} className="section-title mb-1.5">
                Caveats
              </h3>
              <ul className="list-disc pl-5 space-y-1 text-xs text-slate-400">
                {result.caveats.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </section>
          )}

          <p className="text-[11px] text-slate-500">
            {result.cache_hit ? "Cached commentary" : "Generated"}
            {formatShortUtc(result.generated_at) ? ` ${formatShortUtc(result.generated_at)} (UTC)` : ""}
            {result.model ? ` · model: ${result.model}` : " · no model"}. Research and education only.
          </p>
        </div>
      )}
    </section>
  );
}
