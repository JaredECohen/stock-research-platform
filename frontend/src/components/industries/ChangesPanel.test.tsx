import React from "react";
import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import ChangesPanel from "@/components/industries/ChangesPanel";
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
    render(<ChangesPanel changes={fx.changes} />);
    const block = screen.getByTestId("changes-analyst-view");
    expect(block).toHaveTextContent("Analyst interpretation");
    expect(block).toHaveTextContent(String((fx.changes.analyst_view as { what_changed: string }).what_changed));
  });

  it("says so when neither edition stored a comparable statistics row", () => {
    const c = fx.clone(fx.changes);
    c.facts_delta = {};
    render(<ChangesPanel changes={c} />);
    expect(screen.getByTestId("changes-table")).toHaveTextContent("neither edition stored a comparable statistics row");
  });
});
