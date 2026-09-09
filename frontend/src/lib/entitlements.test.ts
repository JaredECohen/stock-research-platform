import { describe, expect, it } from "vitest";
import { featureCopy, freeAllowanceSentence, freeHeadlineAllowances, proAllowanceSentence } from "@/lib/entitlements";
import { ent, featureMatrix } from "@/test/providers";
import returnedSource from "./entitlements.ts?raw";

describe("entitlement copy is derived from the backend matrix", () => {
  it("phrases every allowance shape from the matrix", () => {
    const m = featureMatrix();
    expect(proAllowanceSentence("research_run", m, "research runs")).toBe("Pro includes 20 research runs a month.");
    expect(proAllowanceSentence("memo_view", m, "stored memo views")).toBe("Pro has no monthly cap on stored memo views (distinct tickers).");
    expect(proAllowanceSentence("portfolio", m, "portfolio builds")).toBe("Portfolio builds are part of Pro.");
    expect(proAllowanceSentence("macro", m, "macro analysis")).toBe("Macro analysis is part of Pro.");
    expect(freeAllowanceSentence("dcf", m, "DCF models")).toBe("On Free, a ticker is available once its memo has been opened this month.");
    expect(freeAllowanceSentence("pm_chat", m, "Ask-the-PM turns")).toBe("Free includes 10 Ask-the-PM turns a month.");
    // Not on Free at all: nothing to say on the Free side.
    expect(freeAllowanceSentence("portfolio", m, "portfolio builds")).toBeNull();
  });

  it("follows overrides and says nothing without a matrix", () => {
    const m = featureMatrix({ research_run: { pro: 0 }, chart_commentary: { pro: 250 } });
    expect(proAllowanceSentence("chart_commentary", m, "chart commentaries")).toBe("Pro includes 250 chart commentaries a month.");
    expect(proAllowanceSentence("research_run", m, "research runs")).toBe("Research runs are not included in Pro.");
    expect(proAllowanceSentence("research_run", undefined, "research runs")).toBeNull();
    expect(proAllowanceSentence("research_run", {}, "research runs")).toBeNull();
    expect(featureCopy("pm_chat").value).not.toMatch(/\d/);
    expect(featureCopy("pm_chat", m).value).toContain("Pro includes 300 Ask-the-PM turns a month.");
  });

  it("builds the Free headline from the user's own limits", () => {
    const e = {
      memo_view: ent("memo_view", { limit: 3 }),
      research_run: ent("research_run", { limit: 1 }),
      pm_chat: ent("pm_chat", { limit: 10 }),
    };
    expect(freeHeadlineAllowances(e)).toBe("up to 3 stored memo views, 1 research run and 10 Ask-the-PM turns a month");
    expect(freeHeadlineAllowances({ pm_chat: ent("pm_chat", { limit: 5 }) })).toBe("up to 5 Ask-the-PM turns a month");
    expect(freeHeadlineAllowances({ pm_chat: ent("pm_chat", { limit: null }) })).toBeNull();
    expect(freeHeadlineAllowances({ pm_chat: ent("pm_chat", { limit: 10, allowed: false }) })).toBeNull();
    expect(freeHeadlineAllowances(undefined)).toBeNull();
  });

  it("keeps the copy module free of literal allowance numbers", () => {
    // Guard against the drift the adversarial review caught: any digit in
    // a string literal here would be a number the backend may not enforce.
    // Comments carry worked examples ("1 research run"), so scan code only.
    const code = returnedSource.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");
    // Template interpolations (`${e.limit}`) are values from the backend,
    // not literals; the lone "0" is padStart for the clock.
    const literals = (code.match(/"[^"\n]*"|`[^`\n]*`/g) ?? []).map((s) => s.replace(/\$\{[^}]*\}/g, ""));
    const withDigits = literals.filter((s) => /\d/.test(s) && !/^"@\//.test(s) && s !== '"0"');
    expect(withDigits).toEqual([]);
  });
});
