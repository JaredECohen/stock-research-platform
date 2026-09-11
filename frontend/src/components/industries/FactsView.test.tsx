import React from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import FactsView, { OMITTED_KEYS, isRate } from "@/components/industries/FactsView";
import { degradationText, fmtCap, fmtDateTime, fmtPctSigned, fmtShare, humanize, na } from "@/components/industries/format";
import * as fx from "@/test/fixtures/industry";

// The generic facts renderer is what stands between a backend that says
// "missing, and here is why" and a page that shows a blank. Every test
// here is a way that could go wrong.

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
});
