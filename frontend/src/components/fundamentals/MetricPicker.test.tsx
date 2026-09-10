import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import MetricPicker, { familyLabel } from "@/components/fundamentals/MetricPicker";
import { makeSpec } from "@/test/fixtures/fundamentals";

const CATALOG = [
  makeSpec("revenue", { family: "income" }),
  makeSpec("net_income", { family: "income", sign_note: "negative when the company reports a loss" }),
  makeSpec("gross_margin", { family: "margins", kind: "derived", formula_text: "gross_profit / revenue" }),
];

describe("MetricPicker", () => {
  it("says the catalog is loading until it arrives", () => {
    render(<MetricPicker catalog={null} selected={[]} onChange={() => {}} max={4} />);
    expect(screen.getByText("Loading the metric catalog…")).toBeInTheDocument();
    expect(screen.queryAllByRole("checkbox")).toHaveLength(0);
  });

  it("groups by family and appends a ticked metric in selection order", () => {
    const onChange = vi.fn();
    render(<MetricPicker catalog={CATALOG} selected={["gross_margin"]} onChange={onChange} max={4} />);
    expect(familyLabel("cash_flow")).toBe("Cash flow");
    expect(screen.getByText("Income")).toBeInTheDocument();
    expect(screen.getByText("Margins")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: /^Revenue/ }));
    // Selection order is preserved (it drives dash pattern and axis side),
    // so the new metric goes last rather than being sorted into the catalog.
    expect(onChange).toHaveBeenCalledWith(["gross_margin", "revenue"]);
    fireEvent.click(screen.getByRole("checkbox", { name: /^Gross margin/ }));
    expect(onChange).toHaveBeenLastCalledWith([]);
  });

  it("disables only the unticked options at the ceiling and says why", () => {
    render(<MetricPicker catalog={CATALOG} selected={["revenue", "net_income"]} onChange={() => {}} max={2} maxSource="plan" />);
    expect(screen.getByRole("checkbox", { name: /^Gross margin/ })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: /^Revenue/ })).toBeEnabled();
    expect(screen.getByTestId("metric-max")).toHaveTextContent("Up to 2 metrics per chart on this plan. Untick one to choose another.");
  });

  it("describes each option to the screen reader with its unit, kind, formula and sign note", () => {
    render(<MetricPicker catalog={CATALOG} selected={[]} onChange={() => {}} max={4} />);
    const box = screen.getByRole("checkbox", { name: /^Net income/ });
    const desc = document.getElementById(box.getAttribute("aria-describedby")!);
    expect(desc).toHaveTextContent("negative when the company reports a loss");
    const margin = screen.getByRole("checkbox", { name: /^Gross margin/ });
    expect(document.getElementById(margin.getAttribute("aria-describedby")!)).toHaveTextContent("percent, derived. gross_profit / revenue");
  });
});
