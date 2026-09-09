import React from "react";
import { Link, type RouteObject } from "react-router-dom";
import { useAuth } from "@/auth/AuthContext";
import { useConfig } from "@/auth/ConfigProvider";

/**
 * Marketing site mount point (FEAT-002 slice S6 replaces this file).
 *
 * Contract with App.tsx: export `publicRoutes: RouteObject[]`, rendered
 * outside the /app shell (no RequireAuth, no Layout). The placeholder
 * Landing keeps the build and the router tests honest until the real
 * pages land; it deliberately makes no product claims.
 */
export function Landing() {
  const { config } = useConfig();
  const auth = useAuth();
  const signedIn = !config.auth_enabled || auth.status === "signed_in";
  return (
    <div className="min-h-screen flex items-center justify-center px-6">
      <main className="card max-w-lg text-center">
        <div className="text-xs uppercase tracking-widest text-accent-500 font-semibold">MarketMosaic</div>
        <h1 className="text-2xl font-semibold mt-1">The marketing site is coming</h1>
        <p className="text-sm text-slate-400 mt-2 leading-relaxed">
          Sample research, pricing and the methodology pages are on their way. The research workspace is available now.
        </p>
        <div className="flex flex-wrap justify-center gap-2 mt-4">
          {signedIn ? (
            <Link to="/app" className="btn-primary text-sm">Continue to app</Link>
          ) : (
            <>
              <Link to="/sign-in?returnTo=%2Fapp" className="btn-primary text-sm">Sign in</Link>
              <Link to="/sign-up?returnTo=%2Fapp" className="btn-ghost text-sm">Sign up free</Link>
            </>
          )}
        </div>
        <p className="text-[11px] text-slate-500 mt-6 leading-snug">Research and education only. Not personalized financial advice.</p>
      </main>
    </div>
  );
}

export const publicRoutes: RouteObject[] = [{ path: "/", element: <Landing /> }];
