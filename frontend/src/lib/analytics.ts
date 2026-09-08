// FEAT-002 — first-party product analytics.
//
// Batches allowlisted events to `POST /api/public/events` (same flush
// pattern as lib/logger.ts). No third-party script, no fingerprinting:
// the only identifier is a random uuid kept in localStorage (`anon_id`),
// which the backend joins to a user only when a valid bearer token is on
// the request. The cookie page describes exactly this and nothing more.
//
// Infallible by design — a funnel event must never break a page.

/** Mirror of `ANALYTICS_EVENT_ALLOWLIST` in backend/app/auth/analytics.py.
 *  Unknown names are dropped client-side so a typo cannot reach the wire. */
export const ANALYTICS_EVENTS = [
  "landing_view",
  "sample_view",
  "sample_interact",
  "signup_started",
  "signup_completed",
  "trial_activated",
  "first_value",
  "pricing_view",
  "checkout_started",
  "checkout_completed",
  "trial_converted",
  "trial_expired",
  "subscription_renewed",
  "subscription_canceled",
  "downgraded",
  "quota_hit",
  "rate_limit_hit",
] as const;

export type AnalyticsEventName = (typeof ANALYTICS_EVENTS)[number];

const ALLOWED = new Set<string>(ANALYTICS_EVENTS);

/** Prop keys the backend keeps (`PROP_KEY_ALLOWLIST`); anything else is
 *  stripped here so nothing free-form leaves the browser. */
const PROP_KEYS = new Set([
  "ticker", "feature", "plan", "scope", "kind", "route", "method", "status",
  "interval", "source", "reason", "page", "path", "limit", "used", "window_seconds",
  "trial_source",
]);

export type AnalyticsProps = Record<string, string | number | boolean | null | undefined>;

interface QueuedEvent {
  name: AnalyticsEventName;
  ts: string;
  props: Record<string, string | number | boolean | null>;
}

const ANON_KEY = "mm_anon_id";
const FLUSH_INTERVAL_MS = 2000;
const MAX_BATCH = 50; // backend cap per request
const BASE = (import.meta.env.VITE_BACKEND_URL as string | undefined) || "";

let anonId: string | null = null;

function newId(): string {
  return (crypto.randomUUID?.() || `a-${Date.now()}-${Math.random().toString(36).slice(2)}`).slice(0, 36);
}

/** Stable per-browser anonymous id. Falls back to a per-page id when
 *  storage is unavailable (private mode, blocked site data). */
export function getAnonId(): string {
  if (anonId) return anonId;
  try {
    let v = localStorage.getItem(ANON_KEY);
    if (!v) {
      v = newId();
      localStorage.setItem(ANON_KEY, v);
    }
    anonId = v;
  } catch {
    anonId = newId();
  }
  return anonId;
}

let buffer: QueuedEvent[] = [];
let flushTimer: number | null = null;

function cleanProps(props?: AnalyticsProps): QueuedEvent["props"] {
  const out: QueuedEvent["props"] = {};
  if (!props) return out;
  for (const [k, v] of Object.entries(props)) {
    if (!PROP_KEYS.has(k) || v === undefined) continue;
    out[k] = typeof v === "string" ? v.slice(0, 200) : v;
  }
  return out;
}

async function post(batch: QueuedEvent[]): Promise<void> {
  try {
    await fetch(`${BASE}/api/public/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Anon-Id": getAnonId() },
      keepalive: true,
      body: JSON.stringify({ events: batch }),
    });
  } catch {
    // Drop silently — analytics never surfaces as a user-visible error.
  }
}

/** Send everything queued now. Exposed for tests and the pagehide hook. */
export async function flushAnalytics(): Promise<void> {
  if (flushTimer !== null) {
    window.clearTimeout(flushTimer);
    flushTimer = null;
  }
  while (buffer.length > 0) {
    const batch = buffer.splice(0, MAX_BATCH);
    await post(batch);
  }
}

function scheduleFlush() {
  if (flushTimer !== null) return;
  flushTimer = window.setTimeout(() => {
    flushTimer = null;
    void flushAnalytics();
  }, FLUSH_INTERVAL_MS);
}

/** Queue one event. Returns false when the name is not allowlisted. */
export function track(name: string, props?: AnalyticsProps): boolean {
  if (!ALLOWED.has(name)) return false;
  buffer.push({ name: name as AnalyticsEventName, ts: new Date().toISOString(), props: cleanProps(props) });
  if (buffer.length >= MAX_BATCH) {
    void flushAnalytics();
  } else {
    scheduleFlush();
  }
  return true;
}

/** Test hook: discard anything queued and cancel the timer. */
export function resetAnalyticsForTests(): void {
  buffer = [];
  if (flushTimer !== null) {
    window.clearTimeout(flushTimer);
    flushTimer = null;
  }
  anonId = null;
}

if (typeof window !== "undefined") {
  window.addEventListener("pagehide", () => {
    void flushAnalytics();
  });
}
