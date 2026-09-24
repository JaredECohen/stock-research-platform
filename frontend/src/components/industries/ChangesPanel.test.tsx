import React from "react";
import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import ChangesPanel from "@/components/industries/ChangesPanel";
import FactsView from "@/components/industries/FactsView";
import * as fx from "@/test/fixtures/industry";

describe("ChangesPanel", () => {
  it("names both editions and whether they are adjacent", () => {
    render(<ChangesPanel changes={fx.changes} />);
    const basis = screen.getByTestId("changes-basis");
    expect(basis).toHaveTextContent(`v${(fx.changes.from as { version: number }).version}`);
    expect(basis).toHaveTextContent(`v${(fx.changes.to as { version: number }).version}`);
    expect(basis).toHaveTextContent("the edition this one replaced");
  });

  it("never differences a fact that is missing on one side — it prints the reason", () => {
    render(<ChangesPanel changes={fx.changes} />);
    const [name, delta] = Object.entries(fx.changes.facts_delta).find(([, d]) => d.delta === null)!;
    const row = screen.getByTestId(`delta-${name}`);
    expect(within(row).getByText(`n/a (${delta.reason})`)).toBeInTheDocument();
    // The give-away failure: a missing side rendered as a zero move.
    expect(row.textContent).not.toMatch(/(^|\s)(\+?0(\.00)?%?)\s*$/);
  });

  it("counts both what moved and what could not be differenced", () => {
    render(<ChangesPanel changes={fx.changes} />);
    const deltas = Object.values(fx.changes.facts_delta);
    const moved = deltas.filter((d) => typeof d.delta === "number" && d.delta !== 0).length;
    const unmeasurable = deltas.filter((d) => typeof d.delta !== "number").length;
    expect(screen.getByTestId("changes-basis")).toHaveTextContent(
      `${moved} fact${moved === 1 ? "" : "s"} moved`,
    );
    expect(screen.getByTestId("changes-basis")).toHaveTextContent(`${unmeasurable} could not be differenced`);
    expect(unmeasurable).toBeGreaterThan(0);
  });

  it("reports membership additions and removals, saying 'none' rather than showing nothing", () => {
    render(<ChangesPanel changes={fx.changes} />);
    const block = screen.getByTestId("changes-constituents");
    expect(block).toHaveTextContent("Added: n/a (none)");
    expect(block).toHaveTextContent("Removed: n/a (none)");
  });

  it("shows a real membership change when there is one", () => {
    const c = fx.clone(fx.changes);
    c.constituents = { added: ["ARM"], removed: ["INTC"], n_from: 5, n_to: 5 };
    render(<ChangesPanel changes={c} />);
    expect(screen.getByTestId("changes-constituents")).toHaveTextContent("Added: ARM");
    expect(screen.getByTestId("changes-constituents")).toHaveTextContent("Removed: INTC");
  });

  it("keeps the analyst's what-changed line in the interpretation block, labelled", () => {
    const c = fx.clone(fx.changes);
    c.analyst_view = { ...c.analyst_view, what_changed: "Breadth narrowed.", reasons: { ...c.analyst_view.reasons, what_changed: null } };
    render(<ChangesPanel changes={c} />);
    const block = screen.getByTestId("changes-analyst-view");
    expect(block).toHaveTextContent("Analyst interpretation");
    expect(screen.getByTestId("changes-what-changed")).toHaveTextContent("Breadth narrowed.");
  });

  it("prints n/a with the server's reason for a line a template wrote, never the template's text", () => {
    // The captured edition's what-changed line was template-filled: the
    // server sent null and its reason (owner decision 1).
    const view = fx.changes.analyst_view;
    expect(view.what_changed).toBeNull();
    expect(view.reasons.what_changed).toBeTruthy();
    render(<ChangesPanel changes={fx.changes} />);
    expect(screen.getByTestId("changes-what-changed")).toHaveTextContent(`n/a (${view.reasons.what_changed})`);
    expect(screen.getByTestId("changes-what-changed")).not.toHaveTextContent("this edition wrote no what-changed line");

    const c = fx.clone(fx.changes);
    c.analyst_view = { ...c.analyst_view, to: null, reasons: { ...c.analyst_view.reasons, to: "template-filled in this edition" } };
    render(<ChangesPanel changes={c} />);
    expect(screen.getAllByTestId("changes-view-to")[1]).toHaveTextContent("n/a (template-filled in this edition)");
  });

  it("agrees with the facts view about which facts are percents", () => {
    // Both cards sit on the same tab, and they used to disagree: this
    // table printed `returns.1m.n` as the count 4 while the facts view
    // two cards above printed the same fact as "+400.00%". They share
    // `format.unitFor` now, and this is the check that they still do.
    const c = fx.clone(fx.changes);
    const counts = Object.keys(c.facts_delta).filter((k) => /(^|\.)n(_[a-z]+)?$/.test(k));
    expect(counts.length).toBeGreaterThan(0);
    render(<ChangesPanel changes={c} />);
    const table = screen.getByTestId("changes-table");
    expect(table).not.toHaveTextContent("400.00%");
    for (const key of counts) {
      const row = screen.getByTestId(`delta-${key}`);
      const to = c.facts_delta[key].to;
      if (typeof to === "number") expect(row).toHaveTextContent(String(to));
    }

    // The same facts, rendered by the other card, in the same units.
    const view = render(
      <FactsView facts={fx.report.payload.sections.what_changed.facts as Record<string, unknown>} />,
    );
    expect(view.getByTestId("facts-view")).not.toHaveTextContent("400.00%");
  });

  it("says so when neither edition stored a comparable statistics row", () => {
    const c = fx.clone(fx.changes);
    c.facts_delta = {};
    render(<ChangesPanel changes={c} />);
    expect(screen.getByTestId("changes-table")).toHaveTextContent("neither edition stored a comparable statistics row");
  });
});
