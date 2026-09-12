import React from "react";
import { Navigate, useSearchParams } from "react-router-dom";
import { Activity } from "lucide-react";
import { useAuth } from "./AuthContext";
import { useClerkModule } from "./ClerkAuthProvider";
import { useConfig } from "./ConfigProvider";
import { safeReturnTo } from "./returnTo";

/**
 * /sign-in and /sign-up. Renders Clerk's hosted components when Clerk is
 * the provider, a single button for the stub, and bounces straight to
 * `returnTo` when there is nothing to sign into (wall off, or already in).
 * `returnTo` is sanitised — only /app paths survive.
 */
export default function SignInPage({ mode }: { mode: "sign-in" | "sign-up" }) {
  const [params] = useSearchParams();
  const returnTo = safeReturnTo(params.get("returnTo"));
  const { config } = useConfig();
  const auth = useAuth();
  const clerk = useClerkModule();

  if (!config.auth_enabled || auth.status === "signed_in") {
    return <Navigate to={returnTo} replace />;
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-10">
      <a href="/" className="flex items-center gap-2 mb-8">
        <div className="h-8 w-8 rounded-lg bg-accent-600/20 border border-accent-600/40 flex items-center justify-center text-accent-500">
          <Activity size={18} />
        </div>
        <div>
          <div className="text-base font-semibold tracking-tight">MarketMosaic</div>
          <div className="text-[11px] uppercase tracking-widest text-slate-500">AI Investment Committee</div>
        </div>
      </a>

      {auth.status === "unavailable" && (
        <div className="card max-w-md" role="alert">
          <div className="text-lg font-semibold">Sign-in is temporarily unavailable</div>
          <p className="text-sm text-slate-300 mt-2">The account service is not configured on this deployment.</p>
        </div>
      )}

      {auth.mode === "stub" && auth.status !== "unavailable" && (
        <div className="card max-w-md" data-testid="stub-sign-in">
          <div className="text-lg font-semibold">{mode === "sign-up" ? "Create an account" : "Sign in"}</div>
          <p className="text-sm text-slate-400 mt-1">
            Development stub — no real account is created. The API only honours this session when the login wall is off.
          </p>
          <button type="button" className="btn-primary mt-4" onClick={() => auth.stubSignIn?.()}>
            Continue as test user
          </button>
        </div>
      )}

      {auth.mode === "clerk" && auth.status !== "unavailable" && (
        clerk ? (
          mode === "sign-up" ? (
            <clerk.SignUp routing="path" path="/sign-up" signInUrl="/sign-in" forceRedirectUrl={returnTo} />
          ) : (
            <clerk.SignIn routing="path" path="/sign-in" signUpUrl="/sign-up" forceRedirectUrl={returnTo} />
          )
        ) : (
          <div className="text-sm text-slate-400" role="status">
            Loading sign-in…
          </div>
        )
      )}

      <p className="text-[11px] text-slate-500 mt-8 max-w-md text-center leading-snug">
        Research and education only. Not personalized financial advice.
      </p>
    </div>
  );
}
