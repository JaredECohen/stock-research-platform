// FEAT-002 (S6) — small hooks shared by the public pages.
import { useEffect, useState } from "react";
import { useAuth } from "@/auth/AuthContext";
import { useConfig } from "@/auth/ConfigProvider";

/**
 * What the marketing chrome needs to know about the visitor. With the
 * login wall off there is nothing to sign into, so the visitor is treated
 * as signed in: CTAs read "Continue to app" and the sign-up buttons hide
 * (the sign-in page would only bounce them to /app anyway).
 */
export function useVisitor(): { authEnabled: boolean; signedIn: boolean; loading: boolean } {
  const { config } = useConfig();
  const auth = useAuth();
  const authEnabled = config.auth_enabled;
  return {
    authEnabled,
    signedIn: !authEnabled || auth.status === "signed_in",
    loading: authEnabled && auth.status === "loading",
  };
}

/** `prefers-reduced-motion`: charts skip their draw animation and the
 *  shell drops transitions (Tailwind's `motion-safe:` handles CSS; this is
 *  for recharts, which animates in JS). */
export function useReducedMotion(): boolean {
  const query = "(prefers-reduced-motion: reduce)";
  const [reduced, setReduced] = useState<boolean>(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
    return window.matchMedia(query).matches;
  });
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mql = window.matchMedia(query);
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    if (typeof mql.addEventListener === "function") {
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    }
    return undefined;
  }, []);
  return reduced;
}

const SITE_NAME = "MarketMosaic";

/**
 * Per-page `<title>` and meta description. The SPA has one index.html;
 * search engines and tab titles still deserve a page-specific title, and
 * the description meta is what a share preview falls back to. Restores
 * nothing on unmount: the next page sets its own.
 */
export function usePageMeta(title: string, description?: string): void {
  useEffect(() => {
    if (typeof document === "undefined") return;
    document.title = title ? `${title} — ${SITE_NAME}` : SITE_NAME;
    if (description) {
      let tag = document.querySelector<HTMLMetaElement>('meta[name="description"]');
      if (!tag) {
        tag = document.createElement("meta");
        tag.setAttribute("name", "description");
        document.head.appendChild(tag);
      }
      tag.setAttribute("content", description);
    }
  }, [title, description]);
}
