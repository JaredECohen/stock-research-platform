import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import TimeRangeControl from "@/components/fundamentals/TimeRangeControl";

describe("TimeRangeControl", () => {
  it("offers 5 years, 10 years and Max as one radio group and reports the choice", () => {
    const onChange = vi.fn();
    render(<TimeRangeControl years={5} onChange={onChange} />);
    const radios = screen.getAllByRole("radio");
    expect(radios.map((r) => r.getAttribute("value"))).toEqual(["5", "10", "max"]);
    expect(screen.getByRole("radio", { name: "5 years" })).toBeChecked();
    fireEvent.click(screen.getByRole("radio", { name: "10 years" }));
    expect(onChange).toHaveBeenCalledWith(10);
    fireEvent.click(screen.getByRole("radio", { name: "Max" }));
    expect(onChange).toHaveBeenLastCalledWith(null);
    // Nothing has been drawn yet, so nothing is claimed about the range.
    expect(screen.queryByTestId("range-applied")).toBeNull();
  });

  it("states what the last response drew as a fact, and a cap only as a cap", () => {
    const { rerender } = render(<TimeRangeControl years={null} onChange={() => {}} appliedYears={5} />);
    // Free plan on "Max": the ceiling is the plan's range, not a nag.
    expect(screen.getByTestId("range-applied")).toHaveTextContent("Drawn: the last 5 fiscal years (the most this plan draws).");
    rerender(<TimeRangeControl years={5} onChange={() => {}} appliedYears={5} />);
    expect(screen.getByTestId("range-applied")).toHaveTextContent("Drawn: the last 5 fiscal years.");
    rerender(<TimeRangeControl years={null} onChange={() => {}} appliedYears={null} />);
    expect(screen.getByTestId("range-applied")).toHaveTextContent("Drawn: every fiscal year on record.");
    rerender(<TimeRangeControl years={10} onChange={() => {}} appliedYears={5} capped />);
    expect(screen.getByTestId("range-applied")).toHaveTextContent("Drawn 5 years: the most this plan draws.");
  });
});
