import React from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import FactsView, { OMITTED_KEYS, isRate } from "@/components/industries/FactsView";
import {
  degradationText,
  fmtCap,
  fmtDateTime,
  fmtPct,
  fmtPctSigned,
  fmtShare,
  humanize,
  na,
  sampleFloorHeadline,
  sampleFloorOf,
  unitFor,
} from "@/components/industries/format";
import * as fx from "@/test/fixtures/industry";

// The generic facts renderer is what stands between a backend that says
// "missing, and here is why" and a page that shows a blank. Every test
// here is a way that could go wrong.

// Owner decision 2026-09-24: nothing public shows the licensed taxonomy —
// no third-party brand, no taxonomy code. The page is built on the
// captured wire fixture, so the fixture is walked here (the mirror of the
// backend's `test_industry_public_surface`), and the facts renderer — the
// one component that prints whatever keys a section carries — is checked
// on every section of the captured edition.
const BRAND = /(?<![A-Za-z])gics(?![A-Za-z])/i;
const LONG_CODE = /(?<![\w$.,])\d{6}(?:\d{2})?(?![\w%]|[.,]\d)/;
const BARE_SHORT_CODE = /^\d{2}$|^\d{4}$/;
const YEAR = /^(?:19|20)\d\d$/;
const PAREN_CODE = /\(\d{4}\)|\bgroup \d{4}\b|Industry Group Analyst \d{4}/;

function wireLeaks(node: unknown, path = "$", out: string[] = []): string[] {
  if (Array.isArray(node)) {
    node.forEach((v, i) => wireLeaks(v, `${path}[${i}]`, out));
  } else if (node && typeof node === "object") {
    for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
      if (BRAND.test(k)) out.push(`${path}: key ${k}`);
      if (BARE_SHORT_CODE.test(k) && !YEAR.test(k)) out.push(`${path}: code key ${k}`);
      if (k === "provider_industry") continue;
      wireLeaks(v, `${path}.${k}`, out);
    }
  } else if (typeof node === "string") {
    if (BRAND.test(node)) out.push(`${path}: brand in ${node.slice(0, 80)}`);
    if (LONG_CODE.test(node)) out.push(`${path}: 6/8-digit code in ${node.slice(0, 80)}`);
    if (PAREN_CODE.test(node)) out.push(`${path}: group code form in ${node.slice(0, 80)}`);
    if (BARE_SHORT_CODE.test(node.trim()) && !YEAR.test(node.trim())) out.push(`${path}: bare code ${node}`);
  }
  return out;
}

describe("the captured wire fixture is public data", () => {
  it("carries no third-party brand and no taxonomy code anywhere, meta included", () => {
    expect(wireLeaks(fx.WIRE_META)).toEqual([]);
    for (const [name, body] of Object.entries({
      taxonomy: fx.taxonomy,
      report: fx.report,
      warmingUpReport: fx.warmingUpReport,
      universeShortReport: fx.universeShortReport,
      notUpdatedReport: fx.notUpdatedReport,
      noAgenticDetail: fx.noAgenticDetail,
      companies: fx.companies,
      history: fx.history,
      changes: fx.changes,
    })) {
      expect([name, wireLeaks(body)]).toEqual([name, []]);
    }
  });

  it("addresses every group by slug and names it by label", () => {
    for (const s of fx.taxonomy.sectors) {
      expect(s.code).toMatch(/^[a-z][a-z0-9-]+$/);
      for (const g of s.industry_groups) {
        expect(g.code).toMatch(/^[a-z][a-z0-9-]+$/);
        expect(g.sector_code).toBe(s.code);
      }
    }
    expect(fx.report.code).toBe(fx.CODE);
    expect(fx.CODE).toMatch(/^[a-z][a-z0-9-]+$/);
  });

  it("renders every section's facts without printing a code or the brand", () => {
    for (const [name, section] of Object.entries(fx.report.payload.sections)) {
      const { unmount } = render(<FactsView facts={(section as { facts: Record<string, unknown> }).facts} />);
      const text = document.body.textContent ?? "";
      expect([name, BRAND.test(text), LONG_CODE.test(text), PAREN_CODE.test(text)]).toEqual([name, false, false, false]);
      unmount();
    }
  });
});

describe("FactsView", () => {
  it("renders a {value: null, reason} cell as n/a with the server's reason", () => {
    render(<FactsView facts={{ our_forecast: { value: null, reason: "no licensed consensus tape" } }} />);
    expect(screen.getByTestId("facts-view")).toHaveTextContent("n/a (no licensed consensus tape)");
  });

  it("reports a bare null rather than dropping it, and says the reason is missing", () => {
    render(<FactsView facts={{ macro_regime: null }} />);
    expect(screen.getByTestId("facts-view")).toHaveTextContent("n/a (no reason recorded)");
  });

  it("says an empty list is empty instead of rendering nothing", () => {
    render(<FactsView facts={{ cohort_filing_themes: [] }} />);
    expect(screen.getByTestId("facts-view")).toHaveTextContent("n/a (none on this edition)");
  });

  it("counts what a long list dropped rather than stopping silently", () => {
    render(<FactsView facts={{ items: Array.from({ length: 20 }, (_, i) => `row ${i}`) }} />);
    expect(screen.getByTestId("facts-view")).toHaveTextContent("+8 more not shown");
  });

  it("keeps zero as zero — a real zero is not a missing value", () => {
    render(<FactsView facts={{ n_priced: 0 }} />);
    const view = screen.getByTestId("facts-view");
    expect(view).toHaveTextContent("0");
    expect(view).not.toHaveTextContent("n/a");
  });

  it("skips only the keys the header already carries verbatim, and says which", () => {
    render(<FactsView facts={{ attribution: "x", mapping_caveat: "y", n: 3 }} />);
    expect(OMITTED_KEYS).toEqual(["attribution", "mapping_caveat", "disclaimer"]);
    expect(screen.getByTestId("facts-view")).not.toHaveTextContent("x");
    expect(screen.getByTestId("facts-view")).toHaveTextContent("3");
  });

  it("says a section recorded no observed data at all", () => {
    render(<FactsView facts={{}} />);
    expect(screen.getByTestId("facts-empty")).toHaveTextContent("recorded no observed data");
  });

  it("renders a real section from the captured edition without losing its keys", () => {
    const facts = fx.report.payload.sections.performance.facts as Record<string, unknown>;
    render(<FactsView facts={facts} />);
    const view = screen.getByTestId("facts-view");
    expect(view).toHaveTextContent("Returns");
    expect(view).toHaveTextContent("Benchmark relative");
    expect(view).toHaveTextContent("Sample");
  });

  it("gives every number in a returns block the same unit", () => {
    render(
      <FactsView
        facts={{ returns: { "1m": { equal_weight: 0.010765, median: 0.009186, n: 4 } } }}
      />,
    );
    const view = screen.getByTestId("facts-view");
    // The failure this pins: "+1.08%" printed directly above "0.009186".
    expect(view).toHaveTextContent("+1.08%");
    expect(view).toHaveTextContent("+0.92%");
    // …while the sample size inside the same block stays a count.
    expect(view).toHaveTextContent(/n:\s*4/);
    expect(view).not.toHaveTextContent("+400.00%");
  });

  it("keeps a sample size a count inside the delta shape the API actually ships", () => {
    // The regression: `facts_delta` names the quantity ONE LEVEL ABOVE
    // the number — `returns.1m.n` becomes `returns.1m.n.{from,to,change}`
    // — so a rule that read the leaf saw `to`, missed the count, and
    // printed a priced sample of 4 names as "+400.00%" directly above the
    // same row in the diff table printing the honest 4.
    const facts = fx.report.payload.sections.what_changed.facts as Record<string, unknown>;
    const delta = facts.facts_delta as Record<string, { from: unknown; to: unknown; change: unknown }>;
    expect(delta["returns.1m.n"]).toEqual({ from: 0, to: 4, change: 4 });
    render(<FactsView facts={facts} />);
    const view = screen.getByTestId("facts-view");
    expect(view).not.toHaveTextContent("+400.00%");
    expect(view).not.toHaveTextContent("0.009186");
    // …and the rate sitting in the same block is still a percent.
    expect(view).toHaveTextContent("+0.92%");
  });

  it("gives a company's return and weight the same units the table gives them", () => {
    // `leaders.ret_1m` and `per_ticker.weight_mcw` are separated by `_`,
    // not `.`, and a rule that only split on `.` printed them as raw
    // fractions on the Companies tab while the table above printed the
    // same numbers as percents.
    const facts = fx.report.payload.sections.companies.facts as Record<string, unknown>;
    const leader = (facts.leaders as Array<Record<string, number>>)[0];
    render(<FactsView facts={facts} />);
    const view = screen.getByTestId("facts-view");
    expect(view).toHaveTextContent(fmtPctSigned(leader.ret_1m));
    expect(view).toHaveTextContent(fmtPct(leader.weight_mcw));
    expect(view).not.toHaveTextContent(String(leader.ret_1m));
    expect(view).not.toHaveTextContent(String(leader.weight_mcw));
  });

  it("renders a price and a market cap as money, like the table does", () => {
    const row = (fx.report.payload.sections.companies.facts.per_ticker as Array<Record<string, unknown>>).find(
      (r) => typeof r.last_close === "number",
    )!;
    render(<FactsView facts={{ per_ticker: [row] }} />);
    const view = screen.getByTestId("facts-view");
    expect(view).toHaveTextContent(fmtCap(row.market_cap));
    expect(view).toHaveTextContent(`$${(row.last_close as number).toFixed(2)}`);
  });

  it("renders a nested timestamp as the UTC instant, like the header does", () => {
    render(<FactsView facts={{ as_of: "2026-09-04T21:00:00" }} />);
    expect(screen.getByTestId("facts-view")).toHaveTextContent("2026-09-04 21:00 UTC");
  });
});

describe("isRate", () => {
  it("reads the path, not the leaf key", () => {
    expect(isRate("returns.1m.equal_weight")).toBe(true);
    expect(isRate("returns.1m.median")).toBe(true);
    expect(isRate("benchmark_relative.universe_ew.1m.value")).toBe(true);
    expect(isRate("breadth.1m.pct_positive")).toBe(true);
  });

  it("keeps sample sizes and identifiers as plain numbers wherever they sit", () => {
    expect(isRate("returns.1m.n")).toBe(false);
    expect(isRate("returns.1m.n_mcw")).toBe(false);
    expect(isRate("benchmark_relative.universe_ew.1m.benchmark_n")).toBe(false);
    expect(isRate("sample.n_constituents")).toBe(false);
    expect(isRate("metadata.version")).toBe(false);
  });

  it("does not turn an unrelated number into a percent", () => {
    expect(isRate("statistics.inputs_hash")).toBe(false);
    expect(isRate("valuation.ev_ebitda.median")).toBe(false);
  });

  it("reads `_` as a separator, so a return and a weight are not raw fractions", () => {
    expect(unitFor("companies.leaders.ret_1m")).toBe("return");
    expect(unitFor("companies.laggards.weight_mcw")).toBe("share");
    expect(unitFor("companies.largest.weight_mcw")).toBe("share");
    expect(isRate("companies.leaders.ret_1m")).toBe(true);
    expect(isRate("companies.per_ticker.weight_mcw")).toBe(true);
  });

  it("looks above the delta wrappers for the name of the quantity", () => {
    expect(unitFor("facts_delta.returns.1m.n.to")).toBe("count");
    expect(unitFor("facts_delta.returns.1m.n.change")).toBe("count");
    expect(unitFor("facts_delta.returns.1m.n_mcw.from")).toBe("count");
    expect(unitFor("facts_delta.returns.1m.equal_weight.to")).toBe("return");
  });

  it("keeps window sizes counts even inside a percent-bearing block", () => {
    expect(unitFor("statistics.breadth.above_50d_mean.window_sessions")).toBe("count");
    expect(unitFor("statistics.method.breadth_mean_window.max_span_days")).toBe("count");
    expect(unitFor("statistics.breadth.above_50d_mean.share")).toBe("share");
  });

  it("separates a signed return from an unsigned share", () => {
    expect(unitFor("performance.returns.1m.median")).toBe("return");
    expect(unitFor("statistics.sample.coverage")).toBe("share");
    expect(unitFor("statistics.dispersion.stdev")).toBe("share");
    expect(unitFor("companies.per_ticker.market_cap")).toBe("money");
    expect(unitFor("companies.per_ticker.last_close")).toBe("price");
    expect(unitFor("overview.research_priority")).toBe("plain");
  });
});

describe("format helpers", () => {
  it("signs a percent and refuses to invent one", () => {
    expect(fmtPctSigned(0.0123)).toBe("+1.23%");
    expect(fmtPctSigned(-0.031)).toBe("-3.10%");
    expect(fmtPctSigned(null, "no prices")).toBe("n/a (no prices)");
  });

  it("formats a share and a market cap, with a reason when absent", () => {
    expect(fmtShare(0.8)).toBe("80%");
    expect(fmtShare(undefined, "not computed")).toBe("n/a (not computed)");
    expect(fmtCap(2.9e12)).toBe("$2.90T");
    expect(fmtCap(null)).toBe("n/a (not on file)");
  });

  it("keeps a small weight visible instead of rounding it to nothing", () => {
    // `fmtShare` is right for a coverage headline and wrong here: a
    // constituent at 0.3% of the group's market cap is not 0%.
    expect(fmtShare(0.003)).toBe("0%");
    expect(fmtPct(0.003)).toBe("0.30%");
    expect(fmtPct(0.046569)).toBe("4.66%");
    // Too small to show at this precision is a statement about the
    // DISPLAY; a real zero still prints as one.
    expect(fmtPct(0.00000004)).toBe("<0.01%");
    expect(fmtPct(0)).toBe("0%");
    expect(fmtPct(null, "no market cap on file")).toBe("n/a (no market cap on file)");
  });

  it("stamps a naive timestamp UTC instead of shifting it by the viewer's offset", () => {
    expect(fmtDateTime("2026-09-04T21:00:00")).toBe("2026-09-04 21:00 UTC");
    expect(fmtDateTime("2026-09-04T21:00:00Z")).toBe("2026-09-04 21:00 UTC");
    expect(fmtDateTime("2026-09-04T23:00:00+02:00")).toBe("2026-09-04 21:00 UTC");
    expect(fmtDateTime(null)).toBe("n/a (not recorded)");
  });

  it("keeps the industry's acronyms upper-case when humanising a key", () => {
    expect(humanize("core_kpis")).toBe("Core KPIs");
    expect(humanize("ev_ebitda")).toBe("EV EBITDA");
    expect(humanize("benchmark_relative")).toBe("Benchmark relative");
  });

  it("turns a degradation label into a sentence without losing the label", () => {
    expect(degradationText("analyst_narrative:llm_unavailable")).toBe("Analyst narrative: LLM unavailable");
    expect(degradationText("performance:insufficient_sample")).toBe("Performance: insufficient sample");
  });

  it("requires a reason for every n/a", () => {
    expect(na("no prices")).toBe("n/a (no prices)");
  });

  it("reads the sample-floor block off a real coverage block, and refuses a shape it does not recognise", () => {
    const real = sampleFloorOf(fx.universeShortReport.coverage);
    expect(real?.state).toBe("universe_too_small");
    expect(real?.structural).toBe(true);
    expect(real?.explanation).toBe(
      (fx.universeShortReport.coverage as { sample_floor: { explanation: string } }).sample_floor.explanation,
    );

    expect(sampleFloorOf(undefined)).toBeNull();
    expect(sampleFloorOf({})).toBeNull();
    // A state this build does not know must fall through to "not
    // recorded" rather than pick one of the three badges at random.
    expect(sampleFloorOf({ sample_floor: { state: "something_new", explanation: "x" } })).toBeNull();
    // …and a state with no sentence is not renderable either.
    expect(sampleFloorOf({ sample_floor: { state: "met", explanation: "" } })).toBeNull();
  });

  it("gives each sample-floor state its own headline, and never prints the enum", () => {
    const headlines = fx.INDUSTRY_FLOOR_SAMPLES.map((f) => sampleFloorHeadline(f));
    expect(new Set(headlines).size).toBe(headlines.length);
    for (const [i, text] of headlines.entries()) {
      expect(text).not.toContain(fx.INDUSTRY_FLOOR_SAMPLES[i].state);
      expect(text).not.toContain("_");
    }
  });
});
