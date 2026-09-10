import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import ViewModeControl from "@/components/fundamentals/ViewModeControl";
import type { LayoutResult } from "@/lib/fundamentals/layout";

const AVAILABILITY: LayoutResult["availability"] = {
  "dual-axis": { enabled: false, reason: "dual axis is unreadable on narrow screens" },
  "small-multiples": { enabled: true, reason: null },
  indexed: { enabled: false, reason: "indexing a percent series is misleading" },
};

describe("ViewModeControl", () => {
  it("offers every view mode as one radio group and reports the choice", () => {
    const onChange = vi.fn();
    render(<ViewModeControl view="auto" onChange={onChange} />);
    expect(screen.getAllByRole("radio").map((r) => r.getAttribute("value"))).toEqual(["auto", "dual-axis", "small-multiples", "indexed", "table"]);
    expect(screen.getByRole("radio", { name: "Auto" })).toBeChecked();
    // Without a layout report nothing is disabled: the engine has not spoken.
    for (const r of screen.getAllByRole("radio")) expect(r).toBeEnabled();
    fireEvent.click(screen.getByRole("radio", { name: "Table" }));
    expect(onChange).toHaveBeenCalledWith("table");
  });

  it("keeps a forbidden mode visible but disabled, with the engine's reason for pointer and screen reader", () => {
    render(<ViewModeControl view="small-multiples" onChange={() => {}} availability={AVAILABILITY} />);
    const dual = screen.getByRole("radio", { name: /Dual axis/ });
    expect(dual).toBeDisabled();
    expect(dual.closest("label")).toHaveAttribute("title", "dual axis is unreadable on narrow screens");
    expect(dual.closest("label")).toHaveTextContent("(unavailable: dual axis is unreadable on narrow screens)");
    expect(screen.getByRole("radio", { name: /Indexed/ })).toBeDisabled();
    expect(screen.getByRole("radio", { name: "Small multiples" })).toBeChecked();
    // Auto and Table are always reachable: they never depend on the series.
    expect(screen.getByRole("radio", { name: "Auto" })).toBeEnabled();
    expect(screen.getByRole("radio", { name: "Table" })).toBeEnabled();
  });
});
