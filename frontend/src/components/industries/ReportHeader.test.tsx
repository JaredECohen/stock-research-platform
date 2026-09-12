import React from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import ReportHeader from "@/components/industries/ReportHeader";
import * as fx from "@/test/fixtures/industry";

describe("ReportHeader", () => {
  it("names the edition, its period, its as-of and the taxonomy it was written against", () => {
    render(<ReportHeader report={fx.report} />);
    const header = screen.getByTestId("industry-report-header");
    expect(header).toHaveTextContent(`Edition v${fx.report.version}`);
    expect(header).toHaveTextContent(fx.report.period_key);
    expect(header).toHaveTextContent(fx.report.taxonomy_version);
  });

  it("renders a naive as-of as the UTC instant the backend recorded, not the viewer's local time", () => {
    render(<ReportHeader report={fx.report} />);
    // `2026-09-04T21:00:00` has no offset and IS UTC (the backend's clock
    // seam is utcnow). Reading it as local time and restamping it UTC
    // would move a Friday close by the viewer's offset.
    expect(screen.getByTestId("report-as-of")).toHaveTextContent("2026-09-04 21:00 UTC");
  });

  it("keeps membership and price coverage as separate numbers", () => {
    render(<ReportHeader report={fx.report} />);
    const coverage = screen.getByTestId("report-coverage");
    const c = fx.report.coverage as Record<string, number>;
    expect(coverage).toHaveTextContent(`${c.n_constituents} classified constituents`);
    expect(screen.getByTestId("coverage-priced")).toHaveTextContent(`${c.n_with_prices} of ${c.n_constituents}`);
  });

  it("names every excluded constituent with the reason it was excluded", () => {
    render(<ReportHeader report={fx.report} />);
    const excluded = (fx.report.coverage as { excluded: Array<{ ticker: string; reason: string }> }).excluded;
    expect(excluded.length).toBeGreaterThan(0);
    for (const e of excluded) {
      expect(screen.getByTestId("coverage-excluded")).toHaveTextContent(`${e.ticker} (${e.reason})`);
    }
  });

  it("prints the weighting and the breadth window the statistics row recorded", () => {
    render(<ReportHeader report={fx.report} />);
    const method = fx.report.stats!.method as { weighting: string[] };
    expect(screen.getByTestId("method-weighting")).toHaveTextContent(method.weighting.join(" and "));
  });

  it("lists every benchmark definition, including one the run could not build", () => {
    render(<ReportHeader report={fx.report} />);
    const defs = screen.getByTestId("benchmark-definitions");
    const benchmarks = (fx.report.payload.sections.performance.facts as {
      benchmarks: Array<{ id: string; definition: string; available?: boolean }>;
    }).benchmarks;
    for (const b of benchmarks) expect(defs).toHaveTextContent(b.id);
    const missing = benchmarks.find((b) => b.available === false);
    if (missing) expect(screen.getByTestId(`benchmark-${missing.id}`)).toHaveTextContent("not available this period");
  });

  it("shows no stale badge on a fresh edition", () => {
    render(<ReportHeader report={fx.report} />);
    expect(screen.queryByTestId("badge-stale")).not.toBeInTheDocument();
    expect(screen.queryByTestId("last-attempt")).not.toBeInTheDocument();
  });

  it("badges a stale edition with the store's reason and the attempt that failed", () => {
    const stale = fx.staleReport();
    render(<ReportHeader report={stale} />);
    expect(screen.getByTestId("badge-stale")).toHaveTextContent(stale.stale_reason!);
    const attempt = screen.getByTestId("last-attempt");
    expect(attempt).toHaveTextContent(
      `attempt ${stale.last_attempt!.attempts} of ${stale.last_attempt!.max_attempts}`,
    );
    expect(attempt).toHaveTextContent(stale.last_attempt!.error_type);
    expect(attempt).toHaveTextContent("has not been replaced");
  });

  it("counts the degradations and lists their raw labels for an operator", () => {
    const degraded = fx.degradedReport();
    render(<ReportHeader report={degraded} />);
    expect(screen.getByTestId("badge-degraded")).toHaveTextContent(`Degraded (${degraded.degraded.length})`);
    for (const label of degraded.degraded) expect(screen.getByTestId("degraded-list")).toHaveTextContent(label);
  });

  it("carries the report's attribution and the research-map label verbatim", () => {
    render(
      <ReportHeader
        report={fx.report}
        taxonomyAttribution={fx.taxonomy.attribution}
        securityReferenceCaveat={fx.companies.security_reference_caveat}
      />,
    );
    const block = screen.getByTestId("report-attribution");
    expect(block).toHaveTextContent(fx.report.attribution);
    expect(screen.getByTestId("security-reference-caveat")).toHaveTextContent(
      fx.companies.security_reference_caveat,
    );
  });

  it("prints the taxonomy's attribution only when it is a different claim", () => {
    // On the captured edition the report row's `attribution` falls back to
    // the registry's own string, so the two are the same sentence and the
    // page must print it once.
    const view = render(<ReportHeader report={fx.report} taxonomyAttribution={fx.report.attribution} />);
    expect(screen.queryByTestId("taxonomy-attribution")).not.toBeInTheDocument();
    view.unmount();

    const r = fx.clone(fx.report);
    r.attribution = "Original MarketMosaic analyst research; the descriptive text is not licensed GICS content.";
    render(<ReportHeader report={r} taxonomyAttribution={fx.taxonomy.attribution} />);
    expect(screen.getByTestId("taxonomy-attribution")).toHaveTextContent(fx.taxonomy.attribution);
  });

  it("prints the research-and-education disclaimer the edition carries", () => {
    render(<ReportHeader report={fx.report} />);
    expect(screen.getByTestId("report-disclaimer")).toHaveTextContent(fx.report.disclaimer);
    expect(fx.report.disclaimer.toLowerCase()).toContain("research and education only");
  });

  it("prints the server's reason for a missing statistics row instead of inventing its own", () => {
    // `stats_unavailable_reason` exists so no caller has to guess. With
    // `stats` null the header used to say "weighting not recorded on the
    // statistics row" and "breadth window not recorded" — two reasons it
    // made up — while dropping the one the API supplied.
    const r = fx.clone(fx.report);
    r.stats = null;
    r.stats_unavailable_reason = "the group was below min_sample (2 of 3 priced) for 2026-W36";
    render(<ReportHeader report={r} />);
    expect(screen.getByTestId("stats-unavailable")).toHaveTextContent(r.stats_unavailable_reason);
    expect(screen.getByTestId("method-weighting")).toHaveTextContent(`n/a (${r.stats_unavailable_reason})`);
    expect(screen.getByTestId("report-coverage")).not.toHaveTextContent("weighting not recorded on the statistics row");
    expect(screen.getByTestId("report-coverage")).not.toHaveTextContent("breadth window not recorded");
  });

  it("says nothing about a statistics row that is present", () => {
    render(<ReportHeader report={fx.report} />);
    expect(fx.report.stats).not.toBeNull();
    expect(screen.queryByTestId("stats-unavailable")).not.toBeInTheDocument();
  });

  it("shows the errors behind the degradation labels rather than only the labels", () => {
    const r = fx.degradedReport();
    r.errors = [
      { section: "themes", type: "ThemesUnavailable", message: "no cohort filings in the period" },
      "cross_industry: the snapshot for 2026-W36 was not written",
    ];
    render(<ReportHeader report={r} />);
    const errors = screen.getByTestId("report-errors");
    expect(errors).toHaveTextContent("2 errors recorded on this run");
    expect(errors).toHaveTextContent("themes: ThemesUnavailable: no cohort filings in the period");
    expect(errors).toHaveTextContent("the snapshot for 2026-W36 was not written");
  });

  it("prints what the edition cost, with the call count, rather than omitting a zero", () => {
    render(<ReportHeader report={fx.report} />);
    // The captured edition ran no LLM call, so $0.0000 is the true cost —
    // leaving it out would read as "not measured".
    expect(fx.report.generation.llm_calls).toBe(0);
    expect(screen.getByTestId("llm-cost")).toHaveTextContent("LLM cost $0.0000 over 0 calls");

    const r = fx.clone(fx.report);
    r.llm_cost_usd = null;
    render(<ReportHeader report={r} />);
    expect(screen.getAllByTestId("llm-cost")[1]).toHaveTextContent("n/a (cost not recorded on this edition)");
  });

  it("says a missing coverage figure is missing rather than printing a zero", () => {
    const r = fx.clone(fx.report);
    r.coverage = {};
    render(<ReportHeader report={r} />);
    expect(screen.getByTestId("report-coverage")).toHaveTextContent("n/a (membership not recorded)");
    expect(screen.getByTestId("coverage-priced")).toHaveTextContent("n/a (price coverage not recorded)");
  });

  // --- below the sample floor, and WHICH kind of below -------------------
  //
  // All three editions below are captures, not hand-built states: the
  // healthy week, the same group's first week (enough members, no prices
  // yet) and a group whose membership is itself under the floor.

  it("says a healthy group is above the floor rather than staying silent about it", () => {
    render(<ReportHeader report={fx.report} />);
    expect(screen.queryByTestId("badge-sample-floor")).not.toBeInTheDocument();
    const line = screen.getByTestId("coverage-sample-floor");
    expect(line).toHaveTextContent("Enough priced companies to report");
    expect(line).toHaveTextContent(
      (fx.report.coverage as { sample_floor: { explanation: string } }).sample_floor.explanation,
    );
  });

  it("calls a week short of prices a warm-up, not a limit of the universe", () => {
    render(<ReportHeader report={fx.warmingUpReport} />);
    const floor = (fx.warmingUpReport.coverage as { sample_floor: Record<string, unknown> }).sample_floor;
    expect(floor.state).toBe("prices_not_warmed");
    expect(screen.getByTestId("badge-sample-floor")).toHaveTextContent("Not enough prices yet this week");
    const line = screen.getByTestId("coverage-sample-floor");
    expect(line).toHaveTextContent(String(floor.explanation));
    expect(line).toHaveTextContent("without changing the universe");
    // The enum is never what the reader sees.
    expect(line).not.toHaveTextContent("prices_not_warmed");
    expect(line).not.toHaveTextContent("insufficient_sample");
  });

  it("calls a group the universe cannot cover exactly that, not 'not ready yet'", () => {
    render(<ReportHeader report={fx.universeShortReport} />);
    const floor = (fx.universeShortReport.coverage as { sample_floor: Record<string, unknown> }).sample_floor;
    expect(floor.state).toBe("universe_too_small");
    expect(screen.getByTestId("badge-sample-floor")).toHaveTextContent(
      "This universe is too small to cover this industry",
    );
    const line = screen.getByTestId("coverage-sample-floor");
    expect(line).toHaveTextContent(String(floor.explanation));
    expect(line).toHaveTextContent("no amount of price warm-up can cover it");
    expect(line).not.toHaveTextContent("universe_too_small");
  });

  it("gives the two short states different words, not one label with a different colour", () => {
    const { unmount } = render(<ReportHeader report={fx.warmingUpReport} />);
    const warming = screen.getByTestId("coverage-sample-floor").textContent ?? "";
    unmount();
    render(<ReportHeader report={fx.universeShortReport} />);
    const structural = screen.getByTestId("coverage-sample-floor").textContent ?? "";
    expect(warming).not.toBe("");
    expect(structural).not.toBe(warming);
  });

  it("says an edition with no sample-floor state has none, rather than assuming it is fine", () => {
    const r = fx.clone(fx.report);
    r.coverage = { n_constituents: 5, n_with_prices: 4 };
    render(<ReportHeader report={r} />);
    expect(screen.getByTestId("coverage-sample-floor")).toHaveTextContent(
      "n/a (this edition recorded no sample-floor state)",
    );
    expect(screen.queryByTestId("badge-sample-floor")).not.toBeInTheDocument();
  });
});
