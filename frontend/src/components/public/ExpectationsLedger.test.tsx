import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import ExpectationsLedger from "@/components/public/ExpectationsLedger";
import { emptyLedger, makeLedger } from "@/test/fixtures/sample";

describe("ExpectationsLedger", () => {
  it("renders four columns with observed cells styled apart from interpretation", () => {
    render(<ExpectationsLedger ledger={makeLedger()} />);
    expect(screen.getByRole("heading", { level: 2, name: "Expectations ledger" })).toBeInTheDocument();
    for (const name of ["Reported consensus", "Management guidance", "Price-implied", "Our forecast"]) {
      expect(screen.getByRole("heading", { level: 3, name })).toBeInTheDocument();
    }
    const guidance = screen.getByRole("region", { name: "Management guidance" });
    const cell = guidance.querySelector("[data-basis]") as HTMLElement;
    expect(cell).toHaveAttribute("data-basis", "observed");
    expect(within(cell).getByText("Observed")).toBeInTheDocument();
    expect(cell.className).toContain("border-sky-400");
    expect(guidance.textContent).toContain("6-7%");
    expect(guidance.textContent).toContain("7-8%");
    expect(within(guidance).getByText("raised")).toBeInTheDocument();

    const ours = screen.getByRole("region", { name: "Our forecast" });
    const ourCell = ours.querySelector("[data-basis]") as HTMLElement;
    expect(ourCell).toHaveAttribute("data-basis", "interpretation");
    expect(ourCell.className).toContain("border-accent-500");
    expect(within(ours).getAllByText("Interpretation")).toHaveLength(2);
    expect(within(ours).getByRole("listitem")).toHaveTextContent("US renewal rate prints below 88% for two quarters");

    // Numbers: price as currency, upside as a signed percentage, never a bare 0.
    const price = screen.getByRole("region", { name: "Price-implied" });
    expect(within(price).getByText("$900.00")).toBeInTheDocument();
    expect(within(price).getByText("+2.0%")).toBeInTheDocument();
    expect(within(price).getByText("as of 2026-06-01")).toBeInTheDocument();
    expect(screen.getByLabelText("Legend")).toHaveTextContent("Observed — quoted data");
  });

  it("shows every blank leg with its reason and never as a zero", () => {
    render(
      <ExpectationsLedger
        ledger={makeLedger({
          management_guidance: { status: "not_captured", items: [], reason: "not captured" },
          price_implied: { status: "n/a", items: [], reason: "no DCF base-case upside was computed for this memo" },
          our_forecast: { status: "not_captured", items: [], reason: "the committee did not commit a view on this memo" },
        })}
        headingLevel={3}
      />,
    );
    expect(screen.getByRole("heading", { level: 3, name: "Expectations ledger" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Management guidance" })).toHaveTextContent("Not captured — not captured");
    expect(screen.getByRole("region", { name: "Price-implied" })).toHaveTextContent("n/a — no DCF base-case upside was computed for this memo");
    expect(screen.getByRole("region", { name: "Our forecast" })).toHaveTextContent("Not captured — the committee did not commit a view on this memo");
    expect(screen.getByRole("region", { name: "Price-implied" }).textContent).not.toMatch(/\$0|0\.0%/);
    expect(document.querySelectorAll('[data-status="available"]')).toHaveLength(1);
  });

  it("renders the unbuilt ledger with all four reasons", () => {
    render(<ExpectationsLedger ledger={emptyLedger()} />);
    expect(screen.getAllByText(/— no stored memo/)).toHaveLength(4);
  });
});
