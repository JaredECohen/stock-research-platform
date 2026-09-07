import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, Info, X } from "lucide-react";
import { api } from "@/api/client";
import type { ProvidersStatusResponse } from "@/types";

/**
 * Provider-credit / circuit-breaker health banner.
 *
 * Polls /api/providers/status and, only when the LLM layer is unhealthy,
 * tells the user that memos and chat may fall back to deterministic
 * sections. Healthy → renders nothing, so it never pushes page content.
 *
 * Two variants:
 *   - "degraded" (amber): llm.degraded, any open breaker, or live mode with
 *     no LLM key. The user's next memo will be partial and should know why.
 *   - "info" (blue): a recent failover with degraded=false. Output is still
 *     coming, just from a different provider, so it earns only a quiet note.
 *
 * Every field beyond the base payload is optional — the deployed backend may
 * lag this frontend — and a failed status fetch never surfaces as an error:
 * a broken banner about brokenness would be the worst possible outcome.
 */

const POLL_MS = 60_000;
// A failover older than this is history, not news; the poll cadence means
// the banner would otherwise linger for hours after a one-off retry.
const FAILOVER_RECENT_MS = 30 * 60_000;
const DISMISS_KEY = "mm_provider_health_dismissed";

const PROVIDER_LABEL: Record<string, string> = {
  openai: "OpenAI",
  anthropic: "Anthropic",
  gemini: "Gemini",
};

export function providerLabel(key: string): string {
  return PROVIDER_LABEL[key.toLowerCase()] || key;
}

export type HealthVariant = "degraded" | "info";

export interface HealthAssessment {
  variant: HealthVariant;
  title: string;
  reasons: string[];
  // Stable identity for the dismissal: a new reason produces a new key and
  // re-shows the banner even if the user dismissed the previous one.
  key: string;
}

function breakerSentence(provider: string, b: { failure_count: number; seconds_since_last_failure: number | null; cooldown_seconds: number }): string {
  const failures = `${b.failure_count} failure${b.failure_count === 1 ? "" : "s"}`;
  if (b.seconds_since_last_failure === null || !Number.isFinite(b.cooldown_seconds)) {
    return `${providerLabel(provider)} circuit breaker is open after ${failures} — retrying after the cooldown`;
  }
  const remaining = Math.max(0, Math.round(b.cooldown_seconds - b.seconds_since_last_failure));
  return `${providerLabel(provider)} circuit breaker is open after ${failures} — retrying in ~${remaining}s`;
}

function isRecent(iso: string | null): boolean {
  if (!iso) return false;
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return false;
  return Date.now() - t <= FAILOVER_RECENT_MS;
}

/** Pure derivation so the UI and tests share one definition of "unhealthy". */
export function assessHealth(status: ProvidersStatusResponse | null | undefined): HealthAssessment | null {
  if (!status) return null;
  const llm = status.llm;
  const reasons: string[] = [];

  if (llm?.degraded) {
    for (const r of llm.degradation_reasons || []) {
      if (r && !reasons.includes(r)) reasons.push(r);
    }
  }
  for (const [provider, b] of Object.entries(llm?.breakers || {})) {
    if (b && b.is_open) {
      const s = breakerSentence(provider, b);
      if (!reasons.includes(s)) reasons.push(s);
    }
  }
  if (status.mode === "live" && status.llm_configured === false) {
    reasons.push("No LLM API key is configured in live mode.");
  }
  // degraded=true with no explanation from the backend still deserves a line;
  // an empty bullet list would look like a rendering bug.
  if (llm?.degraded && reasons.length === 0) {
    reasons.push("The backend reports the AI provider as degraded.");
  }
  if (reasons.length > 0) {
    return {
      variant: "degraded",
      title: "AI analysis is degraded",
      reasons,
      key: `degraded|${reasons.join("|")}`,
    };
  }

  const fo = llm?.failover;
  if (fo && fo.count > 0 && fo.last_to && isRecent(fo.last_at)) {
    const from = fo.last_from ? providerLabel(fo.last_from) : "the primary provider";
    const why = fo.last_reason ? ` (${fo.last_reason})` : "";
    const line = `Using ${providerLabel(fo.last_to)} after ${from} failed${why}.`;
    return {
      variant: "info",
      title: "AI provider failover",
      reasons: [line],
      key: `info|${fo.last_to}|${fo.last_from || ""}|${fo.last_at || ""}`,
    };
  }
  return null;
}

function readDismissed(): string | null {
  try {
    return sessionStorage.getItem(DISMISS_KEY);
  } catch {
    return null;
  }
}

function writeDismissed(key: string): void {
  try {
    sessionStorage.setItem(DISMISS_KEY, key);
  } catch {
    // Private mode / quota — the in-memory state still hides it for this mount.
  }
}

export default function ProviderHealthBanner() {
  const [status, setStatus] = useState<ProvidersStatusResponse | null>(null);
  const [dismissedKey, setDismissedKey] = useState<string | null>(() => readDismissed());

  useEffect(() => {
    let alive = true;
    const load = () => {
      api
        .providersStatus()
        .then((s) => {
          if (alive) setStatus(s);
        })
        .catch(() => {
          // request() already traces the failed call through logEvent; the
          // banner itself must stay silent so a status outage can't add a
          // second, misleading error on top of whatever is actually broken.
        });
    };
    load();
    const timer = window.setInterval(load, POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  const health = assessHealth(status);
  if (!health || health.key === dismissedKey) return null;

  const dismiss = () => {
    writeDismissed(health.key);
    setDismissedKey(health.key);
  };

  const degraded = health.variant === "degraded";
  const tone = degraded
    ? "border-warn-500/40 bg-warn-500/10 text-warn-500"
    : "border-sky-500/40 bg-sky-500/10 text-sky-300";
  const Icon = degraded ? AlertTriangle : Info;

  return (
    <div
      role="status"
      aria-live="polite"
      data-variant={health.variant}
      className={`card-tight ${tone} mb-4 flex flex-col sm:flex-row sm:items-start gap-3`}
    >
      <Icon size={18} className="shrink-0 mt-0.5" aria-hidden="true" />
      <div className="flex-1 min-w-0 text-sm">
        <div className="font-semibold">{health.title}</div>
        <ul className="mt-1 space-y-0.5 text-slate-200 list-disc pl-4">
          {health.reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
        {degraded && (
          <div className="mt-1 text-slate-300">
            Memos and chat may fall back to deterministic sections until the provider recovers.
          </div>
        )}
        <Link to="/settings" className="inline-block mt-1.5 underline underline-offset-2 hover:opacity-80">
          Provider health details
        </Link>
      </div>
      <button
        type="button"
        onClick={dismiss}
        aria-label="Dismiss provider health notice"
        className="self-end sm:self-start shrink-0 rounded-md p-1 hover:bg-ink-700/60 focus:outline-none focus:ring-2 focus:ring-current"
      >
        <X size={16} aria-hidden="true" />
      </button>
    </div>
  );
}
