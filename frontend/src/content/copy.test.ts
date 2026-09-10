// Copy review for the marketing site (FEAT-002 S6). Two guarantees:
//
//   1. No invented social proof or outcome claims anywhere a visitor can
//      read — not in the pages, the components, the FAQ, the legal drafts
//      or index.html. The product is research and education software and
//      the copy must never suggest otherwise.
//   2. The cookie page's storage inventory names keys that the code
//      actually writes, so the disclosure cannot drift from the code.
//
// Files are read through Vite's `?raw` glob so the test needs no Node
// typings; test files are excluded by the glob's negative pattern.
import { describe, expect, it } from "vitest";
import indexHtml from "../../index.html?raw";
import { FAQ } from "@/content/faq";
import { STORAGE_INVENTORY } from "@/components/public/LegalDoc";

const PUBLIC_SOURCES = import.meta.glob(
  ["/src/pages/public/**/*.{ts,tsx}", "/src/components/public/**/*.{ts,tsx}", "/src/content/**/*.{ts,md}", "!**/*.test.*"],
  { query: "?raw", import: "default", eager: true },
) as Record<string, string>;

/** Source of the files the storage inventory points at. */
const STORAGE_SOURCES = import.meta.glob(["/src/lib/*.ts", "/src/components/*.tsx", "/src/pages/*.tsx", "!**/*.test.*"], {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

const FILES: Record<string, string> = { ...PUBLIC_SOURCES, "/index.html": indexHtml };

/** Phrases that would be a fabricated claim on this site. Each is a
 *  regex so a near-miss ("Trusted by", "trusted-by") is caught too. */
const FORBIDDEN: Array<[string, RegExp]> = [
  ["testimonials", /testimonial/i],
  ["'trusted by' social proof", /trusted\s+by/i],
  ["'join N' social proof", /\bjoin\s+(thousands|hundreds|millions|\d)/i],
  ["subscriber / user counts", /\b\d[\d,.]*\s*[kKmM]?\+?\s+(users|subscribers|customers|investors|traders|members|analysts)\b/i],
  ["'loved by' / 'used by' claims", /\b(loved|used|chosen)\s+by\b/i],
  ["beat the market", /beats?\s+the\s+market/i],
  ["outperform", /outperform/i],
  ["guaranteed", /guarantee/i],
  ["proven", /\bproven\b/i],
  ["annualised / average returns", /\b(annuali[sz]ed|average|historical)\s+returns?\b/i],
  ["percentage returns", /\d+(\.\d+)?%\s+(return|gain|alpha|profit|upside\s+on\s+average)/i],
  ["'returns of'", /\breturns?\s+of\s+\d/i],
  ["made money / profits", /\b(made|make|making)\s+(money|profits?)\b/i],
  ["win / success / hit rate", /\b(win|success|hit)\s+rate\b/i],
  ["star ratings", /\b[45](\.\d)?\s*stars?\b/i],
  ["'as seen in/on'", /\bas\s+seen\s+(in|on)\b/i],
  ["quoted testimonial with attribution", /[“"][^”"]{30,}[”"]\s*[—–-]\s*[A-Z][a-z]+\s+[A-Z]\./],
  ["'investors like you'", /investors\s+like\s+you/i],
  ["buy / sell recommendations", /\b(strong\s+buy|strong\s+sell|buy\s+now|sell\s+now)\b/i],
  ["'recommended stocks'", /recommended\s+stocks?/i],
];

describe("marketing copy makes no fabricated claims", () => {
  it("scans every public page, component, content file and index.html", () => {
    const names = Object.keys(FILES);
    expect(names.length).toBeGreaterThan(20);
    expect(names.some((n) => n.endsWith("faq.ts"))).toBe(true);
    expect(names.some((n) => n.endsWith("privacy.md"))).toBe(true);
    expect(names.some((n) => n.endsWith("Landing.tsx"))).toBe(true);
    expect(names.some((n) => /\.test\./.test(n))).toBe(false);
    const hits: string[] = [];
    for (const [file, text] of Object.entries(FILES)) {
      for (const [label, re] of FORBIDDEN) {
        const m = re.exec(text);
        if (m) hits.push(`${file}: ${label} → "${m[0]}"`);
      }
    }
    expect(hits).toEqual([]);
  });

  it("keeps the research-and-education framing in the shared copy", () => {
    const ctas = FILES["/src/components/public/ctas.ts"];
    expect(ctas).toMatch(/investment research and education only/);
    expect(ctas).toMatch(/not a recommendation/);
    const advice = FAQ.find((f) => f.id === "advice");
    expect(advice?.answer).toMatch(/^No\./);
    expect(advice?.answer).toMatch(/makes no claim about how any security or strategy will perform/);
  });

  it("writes no allowance number into the FAQ (numbers come from the backend matrix)", () => {
    for (const f of FAQ) {
      // "48 hours" (Stripe's trial_end minimum) and the reset time are the
      // only numerals an answer may carry.
      const stripped = f.answer.replace(/48 hours/g, "").replace(/00:00 UTC/g, "");
      expect(stripped, f.id).not.toMatch(/\b\d+\b/);
    }
  });
});

describe("storage inventory matches the code", () => {
  it("every key the cookie page lists is written by the file it names", () => {
    for (const item of STORAGE_INVENTORY) {
      const src = STORAGE_SOURCES[`/${item.source}`];
      expect(src, `${item.source} should be readable`).toBeTypeOf("string");
      for (const key of item.key.split(",").map((k) => k.trim().replace(/:<user id>$/, ""))) {
        expect(src, `${item.source} should contain ${key}`).toContain(key);
      }
      expect(src).toContain(item.where);
    }
  });

  it("the marketing pages store nothing beyond the anon id and the session id", () => {
    const marketing = STORAGE_INVENTORY.filter((s) => s.scope === "marketing");
    expect(marketing.map((m) => [m.key, m.where])).toEqual([
      ["mm_anon_id", "localStorage"],
      ["mm_session_id", "sessionStorage"],
    ]);
    // And no public page or component touches storage itself.
    for (const [file, text] of Object.entries(PUBLIC_SOURCES)) {
      if (!file.includes("/pages/public/") && !file.includes("/components/public/")) continue;
      expect(text, file).not.toMatch(/localStorage\.|sessionStorage\.|document\.cookie/);
    }
  });
});
