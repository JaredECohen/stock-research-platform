// Wave 8G — UI trace logger.
//
// Buffers user actions (route changes, API calls, clicks, errors)
// and flushes them to `POST /api/admin/ui-log` on a 1.5s tick. Queries:
//   GET /api/admin/ui-log?since_minutes=15
//   GET /api/admin/ui-log?session_id=<sid>
//
// Designed to be infallible — if the backend is down or the body is
// invalid, we drop the batch silently rather than stack errors on top
// of whatever already broke.

export type UIEvent = {
  kind: string;
  path?: string;
  method?: string;
  status_code?: number;
  duration_ms?: number;
  session_id?: string;
  ts?: string;
  payload?: Record<string, unknown>;
};

const SESSION_ID = (() => {
  // Stable per-tab session id — included on every event so we can
  // reconstruct a user's interaction sequence from the table.
  try {
    const k = "mm_session_id";
    let v = sessionStorage.getItem(k);
    if (!v) {
      v = (crypto.randomUUID?.() || `s-${Date.now()}-${Math.random().toString(36).slice(2)}`).slice(0, 36);
      sessionStorage.setItem(k, v);
    }
    return v;
  } catch {
    return `s-${Date.now()}`;
  }
})();

let buffer: UIEvent[] = [];
let flushTimer: number | null = null;
const FLUSH_INTERVAL_MS = 1500;
const MAX_BUFFER = 100;

async function flush(): Promise<void> {
  if (buffer.length === 0) return;
  const batch = buffer.splice(0, buffer.length);
  try {
    await fetch("/api/admin/ui-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // keepalive ensures the request survives a navigation away.
      keepalive: true,
      body: JSON.stringify({ events: batch }),
    });
  } catch {
    // Drop silently — never let logging cascade into user-visible errors.
  }
}

function scheduleFlush() {
  if (flushTimer !== null) return;
  flushTimer = window.setTimeout(() => {
    flushTimer = null;
    void flush();
  }, FLUSH_INTERVAL_MS);
}

export function logEvent(e: Omit<UIEvent, "session_id" | "ts">): void {
  if (buffer.length >= MAX_BUFFER) {
    // Pre-emptively flush when buffer fills so we don't stockpile.
    void flush();
  }
  buffer.push({
    ...e,
    session_id: SESSION_ID,
    ts: new Date().toISOString(),
  });
  scheduleFlush();
}

export function getSessionId(): string {
  return SESSION_ID;
}

// One-time installation of global error handlers + a flush-on-unload
// hook. Importing this module triggers it.
if (typeof window !== "undefined") {
  window.addEventListener("error", (ev) => {
    logEvent({
      kind: "error",
      payload: {
        message: ev.message,
        filename: ev.filename,
        lineno: ev.lineno,
        colno: ev.colno,
        stack: ev.error && (ev.error as Error).stack,
      },
    });
  });
  window.addEventListener("unhandledrejection", (ev) => {
    const reason = ev.reason as { message?: string; stack?: string } | undefined;
    logEvent({
      kind: "error",
      payload: {
        message: reason?.message || String(ev.reason),
        stack: reason?.stack,
        unhandled_rejection: true,
      },
    });
  });
  // Best-effort flush before the user navigates away. `keepalive` on the
  // fetch above lets the inflight POST complete after pagehide.
  window.addEventListener("pagehide", () => {
    void flush();
  });
}
