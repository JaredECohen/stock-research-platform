// Shared rendering rules for the Industry Analysis surface.
//
// The one rule that matters here: a value the backend did not compute
// renders as "n/a (reason)" and never as 0, "—", or a blank. The API is
// built to supply the reason (`reason`, `unpriced_reason`, `stale_reason`,
// `stats_unavailable_reason`), so a caller that has to invent one is a
// signal that the wrong field is being read.

import { NA } from "@/lib/format";
import type { IndustryAccess, IndustrySurface } from "@/types/industries";

export { NA };

export function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

/** "n/a (reason)". The reason is required so no caller can forget it. */
export function na(reason: string): string {
  return `${NA} (${reason})`;
}

/** A percent return with two decimals and an explicit sign: "+1.23%". */
export function fmtPctSigned(v: unknown, reason = "not computed"): string {
  if (!isNum(v)) return na(reason);
  const s = `${(v * 100).toFixed(2)}%`;
  return v > 0 ? `+${s}` : s;
}

/** A share (0–1) as a whole percent: "80%". */
export function fmtShare(v: unknown, reason = "not computed"): string {
  if (!isNum(v)) return na(reason);
  return `${Math.round(v * 100)}%`;
}

/** Market cap, compact: "$2.9T". */
export function fmtCap(v: unknown, reason = "not on file"): string {
  if (!isNum(v)) return na(reason);
  const abs = Math.abs(v);
  const [div, suffix] = abs >= 1e12 ? [1e12, "T"] : abs >= 1e9 ? [1e9, "B"] : abs >= 1e6 ? [1e6, "M"] : [1, ""];
  return `$${(v / div).toFixed(suffix ? 2 : 0)}${suffix}`;
}

export function fmtPrice(v: unknown, reason = "no close"): string {
  return isNum(v) ? `$${v.toFixed(2)}` : na(reason);
}

/**
 * An ISO timestamp as a plain date-time, or the reason it is absent.
 *
 * Deliberately not a relative "3 days ago": the reader is comparing an
 * as-of with a period key and a benchmark window, and a relative string
 * hides which.
 *
 * The API serialises naive timestamps — the backend's clock seams are
 * `datetime.utcnow()`, so a value with no offset IS UTC. `new Date()`
 * would read it as the VIEWER's local time and then `toISOString()`
 * would shift it: a 21:00 UTC as-of printed as "2026-09-05 01:00 UTC"
 * for a reader four hours behind, which is a false claim about when the
 * market closed. Naive strings are therefore stamped UTC explicitly and
 * an offset-bearing string is left to `Date` to convert.
 */
export function fmtDateTime(v: string | null | undefined, reason = "not recorded"): string {
  if (!v) return na(reason);
  const hasZone = /(Z|[+-]\d{2}:?\d{2})$/.test(v);
  const d = new Date(hasZone ? v : `${v}Z`);
  if (Number.isNaN(d.getTime())) return v;
  const iso = d.toISOString();
  return `${iso.slice(0, 10)} ${iso.slice(11, 16)} UTC`;
}

export function fmtDate(v: string | null | undefined, reason = "not recorded"): string {
  if (!v) return na(reason);
  return v.slice(0, 10);
}

/** snake_case / dotted ids → human text, for keys the API does not label
 *  itself. The map is spelled out rather than upper-cased wholesale so
 *  "kpis" reads as "KPIs" and "n" stays the sample size it is. */
const ACRONYMS: Record<string, string> = {
  kpi: "KPI",
  kpis: "KPIs",
  ev: "EV",
  ebitda: "EBITDA",
  pe: "PE",
  ttm: "TTM",
  yoy: "YoY",
  roic: "ROIC",
  ew: "EW",
  mcw: "MCW",
  qtd: "QTD",
  ytd: "YTD",
  gics: "GICS",
  llm: "LLM",
  id: "ID",
  us: "US",
  n: "n",
};

export function humanize(id: string): string {
  const words = String(id).replace(/[._]/g, " ").trim().split(/\s+/);
  return words
    .map((w, i) => {
      const low = w.toLowerCase();
      if (low in ACRONYMS) return ACRONYMS[low];
      if (/^\d/.test(w)) return w.toUpperCase();
      return i === 0 ? w.charAt(0).toUpperCase() + w.slice(1) : w;
    })
    .join(" ");
}

/** Horizon ids the analytics layer emits (`1w`, `1m`, `qtd`, `ytd`, `1y`). */
export function horizonLabel(id: string): string {
  return String(id).toUpperCase();
}

/** A degradation label (`analyst_narrative:llm_unavailable`,
 *  `performance:insufficient_sample`, `companies:events:prior_period:2026-W35`)
 *  as a sentence. The raw label is always kept beside it: these strings are
 *  the backend's vocabulary and an operator greps for them. */
export function degradationText(label: string): string {
  const parts = String(label).split(":");
  const head = humanize(parts[0] ?? label);
  const rest = parts.slice(1).filter(Boolean);
  if (rest.length === 0) return head;
  // Mid-sentence, so the clause after the colon is not capitalised —
  // unless `humanize` produced an acronym, which is not a sentence case
  // to undo ("LLM unavailable", not "lLM unavailable").
  const clause = (p: string): string => {
    const text = humanize(p);
    const first = text.split(" ")[0];
    return first === first.toUpperCase() && first.length > 1 ? text : text.charAt(0).toLowerCase() + text.slice(1);
  };
  return `${head}: ${rest.map(clause).join(" — ")}`;
}

// ---------------------------------------------------------------------------
// Access
// ---------------------------------------------------------------------------

/** Why a surface is not being shown. `sign_in` and `plan` are different
 *  states with different remedies, and neither is an error. */
export type IndustryGate = { reason: "sign_in" | "plan"; tier: string; surface: string } | null;

/**
 * Whether THIS client should refuse to fetch `surface`.
 *
 * Deliberately conservative — it only claims a gate the deployment
 * actually has:
 *
 *   * nothing is gated while `access.enforced` is false (the login wall
 *     is off; every surface answers to everyone and the block says so);
 *   * a `public` surface is never gated;
 *   * signed out on a Pro surface → `sign_in`, and the page must not
 *     fetch: an anonymous read of a Pro route is a guaranteed 401, and
 *     the shared 401 handler would bounce the viewer to sign-in from a
 *     page they were only browsing;
 *   * signed in with the `industry_analysis` entitlement explicitly
 *     refused → `plan`;
 *   * anything else (entitlement unknown, account still loading) → no
 *     client-side gate. The server authorises every call regardless of
 *     what the UI believes; guessing "denied" here would hide a surface
 *     the viewer has paid for.
 */
export function gateFor(
  surface: IndustrySurface | string,
  access: IndustryAccess | null | undefined,
  opts: { signedIn: boolean; entitlementAllowed?: boolean | null },
): IndustryGate {
  if (!access || !access.enforced) return null;
  const tier = access.surfaces?.[surface] ?? (surface === access.surface ? access.tier : "pro");
  if (tier === "public") return null;
  if (!opts.signedIn) return { reason: "sign_in", tier, surface };
  if (opts.entitlementAllowed === false) return { reason: "plan", tier, surface };
  return null;
}
