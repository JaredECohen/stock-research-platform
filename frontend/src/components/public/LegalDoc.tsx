import React from "react";
import { AlertTriangle } from "lucide-react";
import { useConfig } from "@/auth/ConfigProvider";
import { Markdown } from "@/components/Markdown";
import billingTerms from "@/content/legal/billing-terms.md?raw";
import cookies from "@/content/legal/cookies.md?raw";
import privacy from "@/content/legal/privacy.md?raw";
import terms from "@/content/legal/terms.md?raw";

export type LegalDocId = "privacy" | "terms" | "billing-terms" | "cookies";

export const LEGAL_DOCS: Record<LegalDocId, { title: string; description: string; body: string }> = {
  privacy: { title: "Privacy policy", description: "What MarketMosaic stores about you, why, and for how long.", body: privacy },
  terms: { title: "Terms of service", description: "The terms under which MarketMosaic's research software is provided.", body: terms },
  "billing-terms": { title: "Billing terms", description: "How the Pro trial, subscription, renewal and cancellation work.", body: billingTerms },
  cookies: { title: "Cookies and storage", description: "Exactly what this site stores in your browser: an anonymous id, a session id, and the sign-in provider's cookies when accounts are enabled.", body: cookies },
};

/**
 * Visible until `legal_reviewed` is true in `/api/public/config`
 * (LEGAL_REVIEWED_AT set on the web service). A draft shipped as final
 * is worse than a labelled draft.
 */
export function DraftBanner() {
  const { config } = useConfig();
  if (config.legal_reviewed) return null;
  return (
    <div
      role="note"
      data-testid="legal-draft-banner"
      className="card-tight border-warn-500/50 bg-warn-500/10 text-sm text-slate-100 flex items-start gap-2"
    >
      <AlertTriangle size={16} className="text-warn-500 mt-0.5 shrink-0" aria-hidden />
      <div>
        <div className="font-semibold">Draft — pending owner legal review</div>
        <p className="text-slate-300 mt-0.5">
          This page describes what the software does today but has not yet been reviewed by the operator's legal counsel. Wording may change.
        </p>
      </div>
    </div>
  );
}

/**
 * Browser storage this site writes, by key, so the cookie page states
 * exactly what is stored. The keys mirror the code that writes them
 * (paths in `source`); `content/copy.test.ts` greps those files for each
 * key so the inventory cannot drift from the code.
 */
export const STORAGE_INVENTORY: Array<{ key: string; where: "localStorage" | "sessionStorage"; scope: "marketing" | "app"; purpose: string; source: string }> = [
  { key: "mm_anon_id", where: "localStorage", scope: "marketing", purpose: "Random anonymous id for product events", source: "src/lib/analytics.ts" },
  { key: "mm_session_id", where: "sessionStorage", scope: "marketing", purpose: "Per-tab id grouping interface logs", source: "src/lib/logger.ts" },
  { key: "mm_bootstrapped:<user id>", where: "sessionStorage", scope: "app", purpose: "Marks that the account was initialised this session", source: "src/components/RequireAuth.tsx" },
  { key: "mm_provider_health_dismissed", where: "sessionStorage", scope: "app", purpose: "Remembers a dismissed status banner", source: "src/components/ProviderHealthBanner.tsx" },
  { key: "screener:tab:v1, screener:factor:v1, screener:custom:v1", where: "localStorage", scope: "app", purpose: "Screener tab and saved rule set", source: "src/pages/Screener.tsx" },
];

/**
 * Cookies the sign-in provider sets on this domain. main.tsx mounts
 * AuthProvider around the whole App, so with accounts enabled clerk-js
 * runs on the marketing pages too and `__client_uat` exists for every
 * visitor. Kept out of STORAGE_INVENTORY because our code never writes
 * them (clerk-js does), so the source check in `content/copy.test.ts`
 * does not apply; and named here rather than in cookies.md because the
 * Markdown renderer's italics rule would mangle the leading underscores.
 */
export const PROVIDER_COOKIES: Array<{ name: string; when: string; purpose: string }> = [
  { name: "__client_uat", when: "Every visitor, signed in or not", purpose: "Records whether a Clerk session exists so the page can check without a network round-trip" },
  { name: "__session", when: "Once you sign in", purpose: "Carries your session token so the app can call the API as you" },
];

export function ProviderCookies() {
  return (
    <div className="overflow-x-auto rounded-xl border border-ink-700 mt-6">
      <table className="min-w-full text-sm">
        <caption className="text-left px-3 py-2 text-xs uppercase tracking-wider text-slate-400 bg-ink-900/60">
          Cookies set by the sign-in provider
        </caption>
        <thead className="text-xs uppercase tracking-wider text-slate-400">
          <tr>
            <th scope="col" className="text-left px-3 py-2">Cookie</th>
            <th scope="col" className="text-left px-3 py-2">Set for</th>
            <th scope="col" className="text-left px-3 py-2">Purpose</th>
          </tr>
        </thead>
        <tbody>
          {PROVIDER_COOKIES.map((c) => (
            <tr key={c.name} className="border-t border-ink-700 text-slate-300 align-top">
              <td className="px-3 py-2 font-mono text-xs text-slate-100">{c.name}</td>
              <td className="px-3 py-2 whitespace-nowrap">{c.when}</td>
              <td className="px-3 py-2">{c.purpose}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function StorageInventory() {
  return (
    <div className="overflow-x-auto rounded-xl border border-ink-700 mt-6">
      <table className="min-w-full text-sm">
        <caption className="text-left px-3 py-2 text-xs uppercase tracking-wider text-slate-400 bg-ink-900/60">
          Storage keys this site writes
        </caption>
        <thead className="text-xs uppercase tracking-wider text-slate-400">
          <tr>
            <th scope="col" className="text-left px-3 py-2">Key</th>
            <th scope="col" className="text-left px-3 py-2">Where</th>
            <th scope="col" className="text-left px-3 py-2">Pages</th>
            <th scope="col" className="text-left px-3 py-2">Purpose</th>
          </tr>
        </thead>
        <tbody>
          {STORAGE_INVENTORY.map((s) => (
            <tr key={s.key} className="border-t border-ink-700 text-slate-300 align-top">
              <td className="px-3 py-2 font-mono text-xs text-slate-100">{s.key}</td>
              <td className="px-3 py-2 whitespace-nowrap">{s.where}</td>
              <td className="px-3 py-2 whitespace-nowrap">{s.scope === "marketing" ? "Marketing and app" : "App only"}</td>
              <td className="px-3 py-2">{s.purpose}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function LegalDoc({ doc }: { doc: LegalDocId }) {
  const d = LEGAL_DOCS[doc];
  return (
    <article className="max-w-3xl">
      <div className="mb-4">
        <DraftBanner />
      </div>
      <Markdown text={d.body} />
      {doc === "cookies" ? (
        <>
          <StorageInventory />
          <ProviderCookies />
        </>
      ) : null}
    </article>
  );
}
