import React from "react";
import { describe, expect, it } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import CompaniesTable from "@/components/industries/CompaniesTable";
import * as fx from "@/test/fixtures/industry";

// The table is where membership and price coverage are easiest to
// conflate, and where the provenance of a group assignment has to be
// visible rather than assumed. Both are pinned here.

const mapped = fx.companies.items.find((i) => i.classification.source === "research_map")!;
const derived = fx.companies.items.find((i) => i.classification.source === "provider_alias")!;
const unpriced = fx.companies.items.find((i) => !i.priced)!;

describe("CompaniesTable", () => {
  it("captions the table with membership AND price coverage as separate numbers", () => {
    render(<CompaniesTable companies={fx.companies} />);
    const caption = screen.getByTestId("companies-caption");
    expect(caption).toHaveTextContent(`${fx.companies.count} classified constituents`);
    expect(caption).toHaveTextContent(`${fx.companies.n_priced} priced`);
    // The fixture is the case that matters: the two numbers differ.
    expect(fx.companies.n_priced).toBeLessThan(fx.companies.count);
  });

  it("exposes a sortable header per column with aria-sort", () => {
    render(<CompaniesTable companies={fx.companies} />);
    const headers = screen.getAllByRole("columnheader");
    expect(headers.length).toBeGreaterThan(0);
    for (const h of headers) expect(h).toHaveAttribute("aria-sort");
    // Exactly one column is sorted at a time.
    expect(headers.filter((h) => h.getAttribute("aria-sort") !== "none")).toHaveLength(1);
  });

  it("toggles the sort direction on the active column and moves it on a new one", () => {
    render(<CompaniesTable companies={fx.companies} />);
    const tickerHeader = screen.getByRole("columnheader", { name: /Ticker/ });
    fireEvent.click(within(tickerHeader).getByRole("button"));
    expect(tickerHeader).toHaveAttribute("aria-sort", "ascending");
    fireEvent.click(within(tickerHeader).getByRole("button"));
    expect(tickerHeader).toHaveAttribute("aria-sort", "descending");
    expect(screen.getByRole("columnheader", { name: /Market cap/ })).toHaveAttribute("aria-sort", "none");
  });

  it("shows the sub-industry name and code from the classification row", () => {
    render(<CompaniesTable companies={fx.companies} />);
    const cell = screen.getByTestId(`sub-industry-${mapped.ticker}`);
    expect(cell).toHaveTextContent(mapped.sub_industry_name!);
    expect(cell).toHaveTextContent(mapped.sub_industry_code!);
  });

  it("says n/a with a reason for a row that has no sub-industry, rather than a blank", () => {
    render(<CompaniesTable companies={fx.companies} />);
    expect(derived.sub_industry_name).toBeNull();
    expect(screen.getByTestId(`sub-industry-${derived.ticker}`)).toHaveTextContent("n/a (no sub-industry on this row)");
  });

  it("badges the source and carries the API's own label verbatim as the title", () => {
    render(<CompaniesTable companies={fx.companies} />);
    const research = screen.getByTestId(`source-badge-${mapped.ticker}`);
    expect(research).toHaveTextContent("Research map");
    expect(research).toHaveAttribute("title", mapped.classification.source_label);
    expect(research.getAttribute("title")).toContain("not licensed issuer GICS mapping");

    const provider = screen.getByTestId(`source-badge-${derived.ticker}`);
    expect(provider).toHaveTextContent("Provider-derived");
    expect(provider).toHaveAttribute("title", derived.classification.source_label);
  });

  it("gives an unpriced name the server's reason, never a zero", () => {
    render(<CompaniesTable companies={fx.companies} />);
    const row = screen.getByTestId(`company-row-${unpriced.ticker}`);
    expect(within(row).getByTestId(`unpriced-${unpriced.ticker}`)).toHaveTextContent(
      `n/a (${unpriced.unpriced_reason})`,
    );
    expect(row.textContent).not.toContain("$0.00");
    expect(row.textContent).not.toContain("+0.00%");
  });

  it("prints the caveats the response carries, including the research-map label", () => {
    render(<CompaniesTable companies={fx.companies} />);
    const caveats = screen.getByTestId("companies-caveats");
    expect(caveats).toHaveTextContent(fx.companies.mapping_caveat);
    expect(caveats).toHaveTextContent(fx.companies.security_reference_caveat);
  });

  it("counts the members a capped page dropped", () => {
    const capped = fx.clone(fx.companies);
    capped.limit = 1;
    capped.items = capped.items.slice(0, 1);
    capped.truncated = fx.companies.count - 1;
    render(<CompaniesTable companies={capped} />);
    expect(screen.getByTestId("companies-caption")).toHaveTextContent(
      `${capped.truncated} further members not shown`,
    );
    // The coverage numbers still describe the whole membership.
    expect(screen.getByTestId("companies-caption")).toHaveTextContent(`${fx.companies.count} classified constituents`);
  });

  it("says so when the edition's own priced count disagrees with this read", () => {
    // Two reads of the same quantity that landed on different weeks: the
    // priced count here is the one the group's OTHER captured edition
    // recorded (the week before its prices warmed up), not an invented
    // number. The page shows both and names which one is reconciled with
    // the excluded list, rather than picking silently.
    const edition = fx.warmingUpReport.payload.sections.companies.facts.n_priced as number;
    expect(edition).not.toBe(fx.companies.n_priced);
    render(<CompaniesTable companies={fx.companies} editionNPriced={edition} />);
    const note = screen.getByTestId("coverage-disagreement");
    expect(note).toHaveTextContent(`the edition's own companies facts count ${edition} priced`);
    expect(note).toHaveTextContent(`this membership read counts ${fx.companies.n_priced} of ${fx.companies.count}`);
  });

  it("stays quiet when the two counts agree", () => {
    render(<CompaniesTable companies={fx.companies} editionNPriced={fx.companies.n_priced} />);
    expect(screen.queryByTestId("coverage-disagreement")).not.toBeInTheDocument();
  });

  it("shows a small market-cap weight instead of rounding it to 0%", () => {
    // A real group spreads its cap over dozens of names, so members sit
    // at a few tenths of a percent. The captured group has five members
    // and none that small, so this is the case the fixture cannot make.
    const thin = fx.clone(fx.companies);
    thin.items[0].weight_mcw = 0.003;
    thin.items[0].priced = true;
    render(<CompaniesTable companies={thin} />);
    const row = screen.getByTestId(`company-row-${thin.items[0].ticker}`);
    expect(row).toHaveTextContent("0.30%");
    expect(row).not.toHaveTextContent(/(^|[^.\d])0%/);
  });

  it("prints the server's reason when there is no statistics row, not a zero", () => {
    // `n_priced` is 0 here because nothing priced anything, not because
    // no member had a price — and the API says which.
    const none = fx.clone(fx.companies);
    none.stats = null;
    none.n_priced = 0;
    none.stats_unavailable_reason = "no statistics row for 2026-W36: the group is below min_sample";
    render(<CompaniesTable companies={none} />);
    const caption = screen.getByTestId("companies-caption");
    expect(caption).toHaveTextContent(`n/a (${none.stats_unavailable_reason})`);
    expect(caption).not.toHaveTextContent("0 priced");
  });

  it("says the group has no classified members rather than rendering an empty table", () => {
    const empty = fx.clone(fx.companies);
    empty.items = [];
    render(<CompaniesTable companies={empty} />);
    expect(screen.getByTestId("companies-empty")).toHaveTextContent("no company is classified into this group");
  });
});
