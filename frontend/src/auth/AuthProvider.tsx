import React, { useEffect, useMemo } from "react";
import { setTokenProvider } from "@/api/client";
import { AuthContext, DISABLED_AUTH, useAuth } from "./AuthContext";
import { ClerkAuthProvider } from "./ClerkAuthProvider";
import { StubAuthProvider, isStubRequested } from "./StubAuthProvider";
import { useConfig } from "./ConfigProvider";

/**
 * Picks the provider from runtime config:
 *   auth off                         → DISABLED (today's app)
 *   VITE_AUTH_STUB set (dev / tests) → stub
 *   auth on + publishable key        → Clerk (lazy)
 *   auth on, no key                  → "unavailable": the backend fails
 *                                      closed too, so the shell says so
 *                                      instead of looping through sign-in.
 */
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const { config, loaded } = useConfig();

  if (!loaded) {
    // Config still in flight (≤2s): render nothing rather than mount the
    // wrong provider and remount when the flags arrive.
    return null;
  }

  if (!config.auth_enabled) {
    return (
      <AuthContext.Provider value={DISABLED_AUTH}>
        <TokenBridge>{children}</TokenBridge>
      </AuthContext.Provider>
    );
  }

  if (isStubRequested()) {
    return (
      <StubAuthProvider>
        <TokenBridge>{children}</TokenBridge>
      </StubAuthProvider>
    );
  }

  if (config.clerk_publishable_key) {
    return (
      <ClerkAuthProvider publishableKey={config.clerk_publishable_key}>
        <TokenBridge>{children}</TokenBridge>
      </ClerkAuthProvider>
    );
  }

  return (
    <AuthContext.Provider value={{ ...DISABLED_AUTH, status: "unavailable", mode: "clerk" }}>
      <TokenBridge>{children}</TokenBridge>
    </AuthContext.Provider>
  );
}

/** Hands the active provider's `getToken` to the API client so every
 *  request carries the bearer; clears it on sign-out / unmount.
 *
 *  Installed during render rather than in an effect on purpose: React runs
 *  effects child-first, so RequireAuth's bootstrap call (a descendant)
 *  would fire before an effect here had registered the provider and go
 *  out without a bearer — a 401, an auth-required event, and a sign-in
 *  loop. Assigning a module variable is idempotent, so a StrictMode
 *  double render is harmless. */
export function TokenBridge({ children }: { children: React.ReactNode }) {
  const { status, getToken } = useAuth();
  useMemo(() => setTokenProvider(status === "signed_in" ? getToken : null), [status, getToken]);
  useEffect(() => () => setTokenProvider(null), []);
  return <>{children}</>;
}
