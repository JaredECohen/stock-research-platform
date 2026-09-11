import React from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import ReportTabs, { FACTS_HEADING, INTERPRETATION_HEADING } from "@/components/industries/ReportTabs";
import { INDUSTRY_FACTS_ONLY_SECTIONS } from "@/types/industries";
import * as fx from "@/test/fixtures/industry";

const ORDER = fx.report.payload.section_order!;

function mount(section = ORDER[0], over: Partial<Parameters<typeof ReportTabs>[0]> = {}) {
  const onSelect = vi.fn();
  render(<ReportTabs report={fx.report} section={section} onSelect={onSelect} {...over} />);
  return { onSelect };
}

describe("ReportTabs", () => {
  it("renders one tab per section the edition declares, in the edition's own order", () => {
    mount();
    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.getAttribute("id"))).toEqual(ORDER.map((s) => `industry-tab-${s}`));
    // Thirteen today — asserted against the payload, not a literal, so a
    // fourteenth section does not need a change here.
    expect(tabs).toHaveLength(Object.keys(fx.report.payload.sections).length);
  });

  it("keeps observed data and analyst interpretation under headings that name them", () => {
    mount("performance");
    expect(screen.getByTestId("heading-facts")).toHaveTextContent(FACTS_HEADING);
    expect(screen.getByTestId("heading-interpretation")).toHaveTextContent(INTERPRETATION_HEADING);
    // Two separate regions, not one blended block.
    expect(screen.getByTestId("facts-view")).toBeInTheDocument();
    expect(screen.getByTestId("interpretation-view")).toBeInTheDocument();
  });

  it("says a facts-only section has no analyst layer by design, not by failure", () => {
    const factsOnly = ORDER.find((s) => INDUSTRY_FACTS_ONLY_SECTIONS.includes(s))!;
    mount(factsOnly);
    expect(fx.report.payload.sections[factsOnly].interpretation).toBeNull();
    expect(screen.getByTestId("interpretation-absent")).toHaveTextContent("observed data only");
  });

  it("moves selection with Left/Right and wraps at both ends", () => {
    const { onSelect } = mount(ORDER[0]);
    const tablist = screen.getByRole("tablist");

    fireEvent.keyDown(tablist, { key: "ArrowRight" });
    expect(onSelect).toHaveBeenLastCalledWith(ORDER[1]);
    fireEvent.keyDown(tablist, { key: "ArrowLeft" });
    expect(onSelect).toHaveBeenLastCalledWith(ORDER[ORDER.length - 1]);
    fireEvent.keyDown(tablist, { key: "End" });
    expect(onSelect).toHaveBeenLastCalledWith(ORDER[ORDER.length - 1]);
    fireEvent.keyDown(tablist, { key: "Home" });
    expect(onSelect).toHaveBeenLastCalledWith(ORDER[0]);
  });

  it("is a roving tabindex — Tab reaches the strip once, not thirteen times", () => {
    mount(ORDER[2]);
    const tabs = screen.getAllByRole("tab");
    expect(tabs.filter((t) => t.getAttribute("tabindex") === "0")).toHaveLength(1);
    expect(screen.getByTestId(`tab-${ORDER[2]}`)).toHaveAttribute("tabindex", "0");
  });

  it("wires each tab to its panel", () => {
    mount(ORDER[1]);
    const tab = screen.getByTestId(`tab-${ORDER[1]}`);
    const panel = screen.getByRole("tabpanel");
    expect(tab).toHaveAttribute("aria-controls", panel.id);
    expect(panel).toHaveAttribute("aria-labelledby", tab.id);
  });

  it("renders an extra (the companies table) inside the observed-data block", () => {
    mount("companies", { extras: { companies: <div data-testid="extra-companies">table</div> } });
    expect(screen.getByTestId("extra-companies")).toBeInTheDocument();
  });

  it("states the absence when the order names a section the payload lacks", () => {
    const r = fx.clone(fx.report);
    r.payload.section_order = [...ORDER, "future_section"];
    render(<ReportTabs report={r} section="future_section" onSelect={() => {}} />);
    expect(screen.getByTestId("section-missing")).toHaveTextContent("carries no");
  });

  it("falls back to the first section when the URL names one this edition does not have", () => {
    mount("nonsense");
    expect(screen.getByTestId(`tab-${ORDER[0]}`)).toHaveAttribute("aria-selected", "true");
  });

  it("labels every claim with who wrote the interpretation", () => {
    mount(ORDER[0]);
    expect(screen.getByTestId("interpretation-provenance")).toHaveTextContent(
      fx.report.payload.narrative_by_section![ORDER[0]] === "llm" ? "industry analyst model" : "deterministic template",
    );
  });

  it("shows each claim's basis and falsifier when the claims are opened", () => {
    mount("drivers");
    const claims = screen.getByTestId("interpretation-claims");
    expect(within(claims).getByText(/Basis:/)).toBeInTheDocument();
    expect(within(claims).getAllByText(/Falsifier:/).length).toBeGreaterThan(0);
  });
});
