import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AuthContext, type AuthContextValue } from "./AuthContext";
import { signInPath } from "./returnTo";

/**
 * Clerk-backed provider. `@clerk/clerk-react` is `import()`ed here and
 * nowhere else, so a deployment with the wall off never ships Clerk in
 * its bundle, and the publishable key arrives at runtime from
 * /api/public/config — rotating it needs no rebuild.
 *
 * The bearer the API receives is the "marketmosaic" JWT template (email +
 * email_verified claims, RS256) — the backend's jwks.py verifies exactly
 * that template; the default session token would be rejected.
 */
export const CLERK_JWT_TEMPLATE = "marketmosaic";

type ClerkModule = typeof import("@clerk/clerk-react");

const ClerkModuleContext = createContext<ClerkModule | null>(null);

/** The lazily loaded module, for the sign-in page's `<SignIn/>` component. */
export function useClerkModule(): ClerkModule | null {
  return useContext(ClerkModuleContext);
}

interface Props {
  children: React.ReactNode;
  publishableKey: string;
}

export function ClerkAuthProvider({ children, publishableKey }: Props) {
  const [mod, setMod] = useState<ClerkModule | null>(null);
  const [failed, setFailed] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    import("@clerk/clerk-react")
      .then((m) => {
        if (!cancelled) setMod(m);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const pending = useMemo<AuthContextValue>(
    () => ({
      status: failed ? "unavailable" : "loading",
      mode: "clerk",
      user: null,
      getToken: async () => null,
      signOut: async () => {},
      openSignIn: (returnTo?: string) => navigate(signInPath(returnTo || "/app")),
    }),
    [failed, navigate],
  );

  if (!mod) {
    return <AuthContext.Provider value={pending}>{children}</AuthContext.Provider>;
  }

  const { ClerkProvider } = mod;
  return (
    <ClerkModuleContext.Provider value={mod}>
      <ClerkProvider
        publishableKey={publishableKey}
        signInUrl="/sign-in"
        signUpUrl="/sign-up"
        afterSignOutUrl="/"
      >
        <ClerkBridge mod={mod}>{children}</ClerkBridge>
      </ClerkProvider>
    </ClerkModuleContext.Provider>
  );
}

/** Translates Clerk's hooks into our AuthContext shape. Lives inside
 *  ClerkProvider so the hooks have a client to read. */
function ClerkBridge({ mod, children }: { mod: ClerkModule; children: React.ReactNode }) {
  const navigate = useNavigate();
  const auth = mod.useAuth();
  const { user } = mod.useUser();

  const value = useMemo<AuthContextValue>(() => {
    const status: AuthContextValue["status"] = !auth.isLoaded ? "loading" : auth.isSignedIn ? "signed_in" : "signed_out";
    const primary = user?.primaryEmailAddress ?? null;
    return {
      status,
      mode: "clerk",
      user:
        status === "signed_in" && user
          ? {
              id: user.id,
              email: primary?.emailAddress ?? null,
              emailVerified: primary?.verification?.status === "verified",
            }
          : null,
      getToken: async () => {
        if (!auth.isSignedIn) return null;
        try {
          return (await auth.getToken({ template: CLERK_JWT_TEMPLATE })) ?? null;
        } catch {
          // An expired session or a missing template: the API will answer
          // 401 and RequireAuth routes back to sign-in. Never log the token.
          return null;
        }
      },
      signOut: async () => {
        await auth.signOut();
      },
      openSignIn: (returnTo?: string) => navigate(signInPath(returnTo || "/app")),
    };
  }, [auth, user, navigate]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
