// Shared rendering rules for the Industry Analysis surface.
//
// The one rule that matters here: a value the backend did not compute
// renders as "n/a (reason)" and never as 0, "—", or a blank. The API is
// built to supply the reason (`reason`, `unpriced_reason`, `stale_reason`,
// `stats_unavailable_reason`), so a caller that has to invent one is a
// signal that the wrong field is being read.

import { NA } from "@/lib/format";
import {
  INDUSTRY_SAMPLE_FLOOR_STATES,
  type IndustryAccess,
  type IndustrySampleFloor,
  type IndustrySampleFloorState,
  type IndustrySurface,
} from "@/types/industries";

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

/** A share (0–1) as a whole percent: "80%". For a headline coverage
 *  figure, where a decimal point buys nothing. NOT for a constituent
 *  weight — see `fmtPct`. */
export function fmtShare(v: unknown, reason = "not computed"): string {
  if (!isNum(v)) return na(reason);
  return `${Math.round(v * 100)}%`;
}

/**
 * A share (0–1) as an unsigned percent with two decimals: "4.66%".
 *
 * The precision is the point. A real industry group spreads its market
 * cap over dozens of names, so members sit at 0.3% of it, and
 * `fmtShare`'s `Math.round` prints those as "0%" — a nonzero quantity
 * rendered as nothing, in a column that then visibly fails to sum. A
 * weight too small to show even here prints "<0.01%", which is a
 * statement about the DISPLAY; a value that really is zero still prints
 * "0%", because a real zero is not a missing value.
 */
export function fmtPct(v: unknown, reason = "not computed"): string {
  if (!isNum(v)) return na(reason);
  const pct = v * 100;
  if (pct === 0) return "0%";
  if (Math.abs(pct) < 0.01) return pct > 0 ? "<0.01%" : ">-0.01%";
  return `${pct.toFixed(2)}%`;
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

// ---------------------------------------------------------------------------
// Units
// ---------------------------------------------------------------------------

/**
 * What KIND of quantity a number is, decided from the path it sits at.
 *
 * The wire carries no units: `0.058221` is a 1-month return, `0.046569`
 * is a market-cap weight and `4.0` is a sample size, and only the key
 * path says which. Every generic renderer on this surface (the facts
 * view, the changes table) asks this one function, so the same quantity
 * cannot print as "+5.82%" in one card and "0.058221" in the next.
 *
 * Two rules, both learned from bugs this replaced:
 *
 *   * **`_` separates like `.` does.** `ret_1m` is a return and
 *     `weight_mcw` is a weight; a test that required a whole dotted
 *     segment matched neither and printed both as raw fractions beside
 *     the same numbers rendered as percents.
 *   * **The quantity is named ABOVE the delta wrappers.** The changes
 *     shape turns `returns.1m.n` into `returns.1m.n.{from,to,change}`,
 *     so a rule that reads the leaf sees `to`, misses the count, and
 *     prints a sample of 4 names as "+400.00%". The wrappers are
 *     stripped before the leaf is read.
 *
 * Anything the path does not claim is `plain` and prints as it arrived —
 * guessing a unit is how a page states something the server never said.
 */
export type ValueUnit = "return" | "share" | "money" | "price" | "count" | "plain";

/** Signed, because these move both ways and the sign is the news. */
const RETURN_TOKENS = new Set(["return", "returns", "ret", "relative", "growth"]);
/** Unsigned 0–1 fractions: a breadth reading of "+75%" reads as a move. */
const SHARE_TOKENS = new Set([
  "weight",
  "weighted",
  "share",
  "pct",
  "percent",
  "coverage",
  "breadth",
  "dispersion",
  "margin",
  "margins",
  "yield",
]);
/** Counts, ids and window sizes — never percents, however deep inside a
 *  returns block they sit. Matched against the LEAF's own tokens, so
 *  `n_mcw`, `benchmark_n` and `window_sessions` are all counts. */
const COUNT_TOKENS = new Set([
  "n",
  "count",
  "sessions",
  "days",
  "order",
  "id",
  "version",
  "year",
  "limit",
  "budget",
  "attempts",
  "sample",
  "hash",
]);
/** The delta/value wrappers the API puts BELOW the quantity's name. */
const WRAPPER_LEAVES = new Set(["from", "to", "change", "delta", "value"]);
const MONEY_LEAVES = new Set(["market_cap", "enterprise_value"]);
const PRICE_LEAVES = new Set(["last_close", "close", "price"]);

function tokens(segment: string): string[] {
  return segment.toLowerCase().split(/[_\s]+/).filter(Boolean);
}

export function unitFor(path: string): ValueUnit {
  const segs = String(path).split(".").filter(Boolean);
  while (segs.length > 1 && WRAPPER_LEAVES.has(segs[segs.length - 1].toLowerCase())) segs.pop();
  const leaf = (segs[segs.length - 1] ?? "").toLowerCase();
  if (MONEY_LEAVES.has(leaf)) return "money";
  if (PRICE_LEAVES.has(leaf)) return "price";
  if (tokens(leaf).some((t) => COUNT_TOKENS.has(t))) return "count";
  const all = segs.flatMap(tokens);
  if (all.some((t) => RETURN_TOKENS.has(t))) return "return";
  if (all.some((t) => SHARE_TOKENS.has(t))) return "share";
  return "plain";
}

/** A number rendered in the unit its path claims. A `plain` number is
 *  printed as it arrived rather than dressed up. */
export function fmtByPath(path: string, v: number): string {
  switch (unitFor(path)) {
    case "return":
      return fmtPctSigned(v);
    case "share":
      return fmtPct(v);
    case "money":
      return fmtCap(v);
    case "price":
      return fmtPrice(v);
    default:
      return String(v);
  }
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
// The sample floor
// ---------------------------------------------------------------------------

/**
 * `coverage.sample_floor` off a report, or null when the edition predates
 * the field (a stats row written before it existed keeps its old sample
 * block, and a page that invented a state for one would be claiming
 * something the server never said).
 *
 * Validated rather than cast: the state drives which badge is shown, and
 * an unrecognised one must fall through to "not recorded" rather than
 * render as a third, silently wrong colour.
 */
export function sampleFloorOf(coverage: unknown): IndustrySampleFloor | null {
  if (typeof coverage !== "object" || coverage === null) return null;
  const raw = (coverage as Record<string, unknown>).sample_floor;
  if (typeof raw !== "object" || raw === null) return null;
  const v = raw as Record<string, unknown>;
  const state = v.state;
  if (typeof state !== "string" || !(INDUSTRY_SAMPLE_FLOOR_STATES as readonly string[]).includes(state)) {
    return null;
  }
  if (typeof v.explanation !== "string" || v.explanation === "") return null;
  return {
    state: state as IndustrySampleFloorState,
    structural: v.structural === true,
    clears_with_warm_up: v.clears_with_warm_up === true,
    // A count the response did not carry is null, never 0 — a renderer
    // that printed "0 of 0" would be inventing a coverage figure.
    min_sample: isNum(v.min_sample) ? v.min_sample : null,
    n_constituents: isNum(v.n_constituents) ? v.n_constituents : null,
    n_with_prices: isNum(v.n_with_prices) ? v.n_with_prices : null,
    priced_short_by: isNum(v.priced_short_by) ? v.priced_short_by : null,
    constituents_short_by: isNum(v.constituents_short_by) ? v.constituents_short_by : null,
    explanation: v.explanation,
  };
}

/**
 * The short line above the server's explanation — in the reader's words,
 * never the enum. The two short states get DIFFERENT headlines because
 * that is the whole point: one is a week that will catch up, the other is
 * a group this universe cannot report on at all.
 */
export function sampleFloorHeadline(floor: IndustrySampleFloor): string {
  switch (floor.state) {
    case "universe_too_small":
      return "This universe is too small to cover this industry";
    case "prices_not_warmed":
      return "Not enough prices yet this week";
    default:
      return "Enough priced companies to report";
  }
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
