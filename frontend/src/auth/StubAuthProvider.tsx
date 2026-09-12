import React, { useCallback, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AuthContext, type AuthContextValue, type AuthStatus } from "./AuthContext";
import { signInPath } from "./returnTo";

/**
 * Fake provider for Vitest and keyless local development.
 *
 * Driven by `VITE_AUTH_STUB=signed_in|signed_out` (or the `initialStatus`
 * prop in tests). The token is the literal "stub-token": a backend running
 * with AUTH_ENABLED=false ignores it, and one with the wall on rejects it
 * as unverifiable — the stub can never mint real access.
 */
export const STUB_TOKEN = "stub-token";
export const STUB_USER = { id: "user_stub", email: "stub@example.com", emailVerified: true };

export function stubStatusFromEnv(): AuthStatus {
  const raw = String(import.meta.env.VITE_AUTH_STUB || "").toLowerCase();
  return raw === "signed_in" ? "signed_in" : "signed_out";
}

export function isStubRequested(): boolean {
  const raw = String(import.meta.env.VITE_AUTH_STUB || "").toLowerCase();
  return raw === "signed_in" || raw === "signed_out";
}

interface Props {
  children: React.ReactNode;
  initialStatus?: "signed_in" | "signed_out";
}

export function StubAuthProvider({ children, initialStatus }: Props) {
  const navigate = useNavigate();
  const [status, setStatus] = useState<AuthStatus>(initialStatus ?? stubStatusFromEnv());

  const openSignIn = useCallback(
    (returnTo?: string) => {
      navigate(signInPath(returnTo || "/app"));
    },
    [navigate],
  );

  const value = useMemo<AuthContextValue>(
    () => ({
      status,
      mode: "stub",
      user: status === "signed_in" ? STUB_USER : null,
      getToken: async () => (status === "signed_in" ? STUB_TOKEN : null),
      signOut: async () => setStatus("signed_out"),
      openSignIn,
      stubSignIn: () => setStatus("signed_in"),
    }),
    [status, openSignIn],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
