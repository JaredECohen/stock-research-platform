import React from "react";
import PublicNav from "./PublicNav";
import PublicFooter from "./PublicFooter";
import { FOCUS_RING } from "./ctas";
import { usePageMeta } from "./hooks";

interface Props {
  /** Tab title; the page's own <h1> is rendered by the caller. */
  title: string;
  description?: string;
  children: React.ReactNode;
}

/**
 * Landmarks for every public page: skip link → header/nav → <main id="main">
 * → footer. Pages render exactly one <h1> inside `children`.
 */
export default function PublicShell({ title, description, children }: Props) {
  usePageMeta(title, description);
  return (
    <div className="min-h-screen flex flex-col">
      <a
        href="#main"
        className={`sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 btn-primary ${FOCUS_RING}`}
      >
        Skip to content
      </a>
      <PublicNav />
      <main id="main" tabIndex={-1} className="flex-1 w-full max-w-6xl mx-auto px-4 sm:px-6 outline-none">
        {children}
      </main>
      <PublicFooter />
    </div>
  );
}
