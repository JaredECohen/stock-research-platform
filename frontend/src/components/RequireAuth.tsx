import React, { useEffect, useState } from "react";
import { Navigate, Outlet, useLocation, useNavigate } from "react-router-dom";
import { api, AUTH_REQUIRED_EVENT, isApiError } from "@/api/client";
import { useAuth } from "@/auth/AuthContext";
import { useConfig } from "@/auth/ConfigProvider";
import { signInPath } from "@/auth/returnTo";

/**
 * Gate for the /app shell.
 *
 * UX only — the backend authorises every request — so the aim is to send a
 * signed-out visitor to sign-in with a way back, never to protect data.
 *   wall off     → render children (today's behaviour)
 *   loading      → skeleton
 *   signed_out   → /sign-in?returnTo=<current path>
 *   unavailable  → notice (backend fails closed with 503 too)
 *   signed_in    → POST /api/me/bootstrap once per session, then children
 *
 * A 401 from any API call (token expired mid-session) signs the local
 * session out and routes back through sign-in with the same returnTo.
 */
type BootState = "idle" | "pending" | "done" | "unavailable";

export default function RequireAuth({ children }: { children?: React.ReactNode }) {
  const { config } = useConfig();
  const auth = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [boot, setBoot] = useState<BootState>("idle");
  const here = location.pathname + location.search + location.hash;

  // Token rejected mid-session → clean sign-out, then sign-in with returnTo.
  useEffect(() => {
    if (!config.auth_enabled) return;
    const onAuthRequired = () => {
      void auth.signOut().finally(() => navigate(signInPath(here), { replace: true }));
    };
    window.addEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
    return () => window.removeEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
  }, [config.auth_enabled, auth, navigate, here]);

  useEffect(() => {
    if (!config.auth_enabled || auth.status !== "signed_in") return;
    const key = `mm_bootstrapped:${auth.user?.id || "anon"}`;
    let already = false;
    try {
      already = sessionStorage.getItem(key) === "1";
    } catch {}
    if (already) {
      setBoot("done");
      return;
    }
    let cancelled = false;
    setBoot("pending");
    api
      .bootstrap()
      .then(() => {
        try {
          sessionStorage.setItem(key, "1");
        } catch {}
        if (!cancelled) setBoot("done");
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        if (isApiError(e) && e.status === 503) {
          setBoot("unavailable");
        } else {
          // 401 is handled by the event above; anything else (a blip, a
          // backend that predates /api/me) must not lock the user out of
          // pages the backend will authorise on its own.
          setBoot("done");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [config.auth_enabled, auth.status, auth.user?.id]);

  const body = children ?? <Outlet />;

  if (!config.auth_enabled) return <>{body}</>;

  if (auth.status === "loading" || (auth.status === "signed_in" && (boot === "idle" || boot === "pending"))) {
    return <Skeleton />;
  }
  if (auth.status === "unavailable" || boot === "unavailable") {
    return <Unavailable />;
  }
  if (auth.status === "signed_out") {
    return <Navigate to={signInPath(here)} replace />;
  }
  return <>{body}</>;
}

function Skeleton() {
  return (
    <div className="min-h-screen flex items-center justify-center text-sm text-slate-400" role="status" aria-live="polite">
      <span
        aria-hidden
        className="inline-block h-4 w-4 mr-3 rounded-full border-2 border-accent-500 border-t-transparent animate-spin"
      />
      Loading your workspace…
    </div>
  );
}

function Unavailable() {
  return (
    <div className="min-h-screen flex items-center justify-center px-6">
      <div className="card max-w-md" role="alert">
        <div className="text-lg font-semibold">Sign-in is temporarily unavailable</div>
        <p className="text-sm text-slate-300 mt-2 leading-relaxed">
          The account service could not be reached, so the research workspace is paused rather than opened
          without a verified session. The public pages are still available. Please try again in a few minutes.
        </p>
        <a href="/" className="btn-ghost mt-4 inline-flex">
          Back to the home page
        </a>
      </div>
    </div>
  );
}
