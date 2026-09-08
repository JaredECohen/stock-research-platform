import { createContext, useContext } from "react";

/**
 * FEAT-002 — the one auth interface the app talks to.
 *
 * Providers (Clerk, stub) implement this; pages never import Clerk. Status:
 *   disabled     login wall off — every route renders as it did before FEAT-002
 *   loading      provider still initialising (Clerk script, session restore)
 *   signed_out   wall on, no session → RequireAuth sends to /sign-in
 *   signed_in    session present; `getToken` returns a bearer for the API
 *   unavailable  wall on but no provider can run (no publishable key).
 *                The backend fails closed too (503 auth_unavailable);
 *                marketing pages stay up, the shell shows a notice.
 */
export type AuthStatus = "disabled" | "loading" | "signed_out" | "signed_in" | "unavailable";

export interface AuthUser {
  id: string;
  /** From the provider's client session; the backend never stores it. */
  email: string | null;
  emailVerified: boolean;
}

export interface AuthContextValue {
  status: AuthStatus;
  mode: "disabled" | "clerk" | "stub";
  user: AuthUser | null;
  /** Bearer token for the API, or null when there is no session. */
  getToken: () => Promise<string | null>;
  signOut: () => Promise<void>;
  /** Navigate to the hosted sign-in flow and come back to `returnTo`. */
  openSignIn: (returnTo?: string) => void;
  /** Stub provider only: flip to signed_in from the /sign-in page. */
  stubSignIn?: () => void;
}

export const DISABLED_AUTH: AuthContextValue = {
  status: "disabled",
  mode: "disabled",
  user: null,
  getToken: async () => null,
  signOut: async () => {},
  openSignIn: () => {},
};

export const AuthContext = createContext<AuthContextValue>(DISABLED_AUTH);

export function useAuth(): AuthContextValue {
  return useContext(AuthContext);
}
