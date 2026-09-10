// index.html / robots.txt / sitemap.xml are static files Vite ships as-is;
// a string test is the only kind that can pin them. Read via `?raw` so no
// Node typings are needed.
import { describe, expect, it } from "vitest";
import html from "../../../index.html?raw";
import robots from "../../../public/robots.txt?raw";
import sitemap from "../../../public/sitemap.xml?raw";
import { publicRoutes } from "@/pages/public";
import { SITE_ORIGIN } from "@/components/public/ctas";

describe("index.html meta", () => {
  it("carries title, description, Open Graph, Twitter and a canonical", () => {
    expect(html).toMatch(/<title>MarketMosaic — Your AI Investment Committee<\/title>/);
    expect(html).toMatch(/<meta\s+name="description"\s+content="[^"]{60,}"/);
    expect(html).toMatch(/property="og:title"/);
    expect(html).toMatch(/property="og:description"/);
    expect(html).toMatch(/property="og:type"\s+content="website"/);
    expect(html).toMatch(/property="og:url"\s+content="https:\/\/marketmosaic\.ai\/"/);
    expect(html).toMatch(/name="twitter:card"\s+content="summary"/);
    expect(html).toMatch(/<link\s+rel="canonical"\s+href="https:\/\/marketmosaic\.ai\/"/);
    expect(html).toMatch(/name="viewport"\s+content="width=device-width, initial-scale=1\.0"/);
    expect(html).toContain('lang="en"');
    // No third-party scripts on the marketing shell (fonts are stylesheets).
    const scripts = html.match(/<script[^>]*src="([^"]+)"/g) || [];
    expect(scripts).toEqual(['<script type="module" src="/src/main.tsx"']);
  });

  it("describes the product as research, not advice or performance", () => {
    expect(html).toMatch(/Research and education only/);
    expect(html).not.toMatch(/outperform|guarantee|returns/i);
  });
});

describe("robots.txt and sitemap.xml", () => {
  it("indexes the marketing pages and blocks the app, the API and sign-in", () => {
    expect(robots).toMatch(/^User-agent: \*$/m);
    expect(robots).toMatch(/^Allow: \/$/m);
    for (const p of ["/app", "/api/", "/sign-in", "/sign-up"]) expect(robots).toContain(`Disallow: ${p}`);
    expect(robots).toContain(`Sitemap: ${SITE_ORIGIN}/sitemap.xml`);
  });

  it("lists exactly the public routes (samples expanded to the default tickers)", () => {
    const locs: string[] = Array.from(sitemap.matchAll(/<loc>([^<]+)<\/loc>/g), (m: RegExpMatchArray) => m[1]);
    expect(locs.length).toBeGreaterThan(0);
    const routePaths = publicRoutes.map((r) => r.path as string).filter((p) => !p.startsWith("/sign-"));
    const matches = (loc: string) => {
      const p = loc.replace(SITE_ORIGIN, "") || "/";
      return routePaths.some((rp) => (rp === "/samples/:ticker" ? /^\/samples\/[A-Z.-]+$/.test(p) : rp === p));
    };
    for (const loc of locs) {
      expect(loc.startsWith(SITE_ORIGIN), loc).toBe(true);
      expect(matches(loc), loc).toBe(true);
    }
    for (const rp of routePaths.filter((p) => p !== "/samples/:ticker")) {
      expect(locs, rp).toContain(`${SITE_ORIGIN}${rp === "/" ? "/" : rp}`);
    }
    for (const t of ["NVDA", "COST", "JPM"]) expect(locs).toContain(`${SITE_ORIGIN}/samples/${t}`);
  });
});
