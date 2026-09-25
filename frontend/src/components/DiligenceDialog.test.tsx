import { describe, expect, it } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import DiligenceDialog from "@/components/DiligenceDialog";
import { UNAVAILABLE_TEXT } from "@/lib/memoSections";
import { presentedMemo } from "@/test/fixtures/memoSections";

// MSFT's round 1 (captured presenter output): the PM asked the valuation
// analyst and the earnings analyst a follow-up each. The valuation answer
// was a template (its view is hidden in the grid too), so the presenter
// blanked it; the earnings answer is the analyst's own.
function mountMsft() {
  const rounds = presentedMemo("msft_live").round_findings ?? [];
  render(<DiligenceDialog rounds={rounds} />);
}

function questionBlock(question: RegExp): HTMLElement {
  return screen.getByText(question).closest("div.rounded") as HTMLElement;
}

describe("DiligenceDialog", () => {
  it("shows a placeholder, not the template's text or confidence, for a blanked answer", () => {
    mountMsft();
    const block = questionBlock(/Synthetic PM question 40 for the msft_live fixture/);
    const toggle = within(block).getByRole("button");
    expect(toggle).toHaveTextContent("answer unavailable");
    fireEvent.click(toggle);
    expect(within(block).getByText(UNAVAILABLE_TEXT)).toBeInTheDocument();
    expect(within(block).getByTestId("unavailable-reason")).toHaveTextContent(
      "not written by an analyst in this run",
    );
    // The hidden answer's confidence belongs to the template.
    expect(within(block).queryByText(/specialist confidence/)).not.toBeInTheDocument();
    expect(within(block).getAllByText(UNAVAILABLE_TEXT)).toHaveLength(1);
  });

  it("still renders an analyst's real answer with its confidence", () => {
    mountMsft();
    const block = questionBlock(/Synthetic PM question 42 for the msft_live fixture/);
    const toggle = within(block).getByRole("button");
    expect(toggle).toHaveTextContent("show answer");
    fireEvent.click(toggle);
    expect(within(block).getByText(/Synthetic analyst point 3 for the msft_live fixture/)).toBeInTheDocument();
    expect(within(block).getByText("specialist confidence: 65%")).toBeInTheDocument();
    expect(within(block).queryByTestId("unavailable-section")).not.toBeInTheDocument();
  });
});
