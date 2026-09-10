import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import CompanySelector from "@/components/fundamentals/CompanySelector";
import type { CompanyOut } from "@/types";

const UNIVERSE: CompanyOut[] = [{ ticker: "AAPL", company_name: "Apple", exchange: "NASDAQ", sector: "Tech", industry: "Hardware", universe_tier: "auto_analysis" }];

describe("CompanySelector", () => {
  it("adds a typed ticker upper-cased and de-duplicated", () => {
    const onChange = vi.fn();
    render(<CompanySelector tickers={["MSFT"]} onChange={onChange} universe={UNIVERSE} max={5} />);
    const input = screen.getByLabelText("Add a company");
    fireEvent.change(input, { target: { value: "aapl" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onChange).toHaveBeenCalledWith(["MSFT", "AAPL"]);
    onChange.mockClear();
    fireEvent.change(input, { target: { value: "msft" } });
    fireEvent.keyDown(input, { key: "Enter" });
    // Already on the chart: nothing to write.
    expect(onChange).not.toHaveBeenCalled();
  });

  it("removes a chip by its button or with Backspace on it", () => {
    const onChange = vi.fn();
    render(<CompanySelector tickers={["AAPL", "MSFT"]} onChange={onChange} universe={UNIVERSE} max={5} />);
    fireEvent.click(screen.getByRole("button", { name: "Remove AAPL" }));
    expect(onChange).toHaveBeenCalledWith(["MSFT"]);
    fireEvent.keyDown(screen.getByRole("button", { name: "Remove MSFT" }), { key: "Backspace" });
    expect(onChange).toHaveBeenLastCalledWith(["AAPL"]);
  });

  it("replaces the picker with the reason at the plan's ceiling", () => {
    render(<CompanySelector tickers={["AAPL", "MSFT"]} onChange={() => {}} universe={UNIVERSE} max={2} maxSource="plan" />);
    expect(screen.queryByLabelText("Add a company")).toBeNull();
    expect(screen.getByTestId("company-max")).toHaveTextContent("Up to 2 companies per chart on this plan. Remove one to add another.");
    // The chips stay removable so the reader can make room.
    expect(screen.getByRole("button", { name: "Remove AAPL" })).toBeEnabled();
  });
});
