import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import CheckedText, { placeable } from "@/components/memo/CheckedText";
import { claimsFor } from "@/lib/memoQuality";
import { QUALITY_EXPECT, qualityMemo } from "@/test/fixtures/memoQuality";
import type { NumberClaim, NumberClaimStatus } from "@/types";

function claim(
  text: string,
  raw: string,
  status: NumberClaimStatus = "untraceable",
  overrides: Partial<NumberClaim> = {},
): NumberClaim {
  const start = text.indexOf(raw);
  return {
    field: "final_pm_view",
    start,
    end: start + raw.length,
    raw,
    value: null,
    unit: "",
    status,
    source_refs: [],
    ...overrides,
  };
}

function marks(container: HTMLElement) {
  return Array.from(container.querySelectorAll("[data-claim-status]")).map((el) => [
    el.textContent,
    el.getAttribute("data-claim-status"),
  ]);
}

describe("CheckedText", () => {
  const TEXT = "Revenue of $123.45B, a 37.7% margin and 18.5% growth.";

  it("splits the text at a claim's offsets and marks only the figure", () => {
    const { container } = render(<CheckedText text={TEXT} claims={[claim(TEXT, "$123.45B")]} />);
    expect(container.textContent).toBe(TEXT);
    expect(marks(container)).toEqual([["$123.45B", "untraceable"]]);
    const span = container.querySelector("[data-claim-status]") as HTMLElement;
    expect(span.title).toBe("Not found in the data this memo's analysts were given");
  });

  it("marks several claims in text order, whatever order they arrive in", () => {
    const claims = [claim(TEXT, "18.5%", "assumption"), claim(TEXT, "$123.45B"), claim(TEXT, "37.7%", "mis_anchored")];
    const { container } = render(<CheckedText text={TEXT} claims={claims} />);
    expect(container.textContent).toBe(TEXT);
    expect(marks(container)).toEqual([
      ["$123.45B", "untraceable"],
      ["37.7%", "mis_anchored"],
      ["18.5%", "assumption"],
    ]);
    const assumption = container.querySelector('[data-claim-status="assumption"]') as HTMLElement;
    expect(assumption.title).toMatch(/^PM assumption/);
  });

  it("skips a claim whose offsets no longer index its raw figure, and never throws", () => {
    const good = claim(TEXT, "37.7%");
    const bad: NumberClaim[] = [
      { ...claim(TEXT, "$123.45B"), start: 0, end: 8 },                // slice != raw (stale offset)
      { ...claim(TEXT, "18.5%"), end: TEXT.length + 5 },               // runs past the end
      { ...claim(TEXT, "18.5%"), start: -3 },                          // negative
      { ...claim(TEXT, "18.5%"), start: 40.5 },                        // not an integer
      { ...claim(TEXT, "37.7%"), start: good.start + 1, end: good.end + 1, raw: TEXT.slice(good.start + 1, good.end + 1) }, // overlaps
      { ...claim(TEXT, "18.5%"), end: claim(TEXT, "18.5%").start },    // empty
    ];
    const { container } = render(<CheckedText text={TEXT} claims={[good, ...bad]} />);
    expect(container.textContent).toBe(TEXT);
    expect(marks(container)).toEqual([["37.7%", "untraceable"]]);
  });

  it("renders the placeholder a presenter substituted without marks (stale offsets)", () => {
    const placeholder = "Unavailable in this version.";
    const { container } = render(<CheckedText text={placeholder} claims={[claim(TEXT, "$123.45B")]} />);
    expect(container.textContent).toBe(placeholder);
    expect(marks(container)).toEqual([]);
  });

  it("renders plain text for no claims, null claims and a null text", () => {
    expect(render(<CheckedText text={TEXT} />).container.innerHTML).toBe(TEXT);
    expect(render(<CheckedText text={TEXT} claims={null} />).container.innerHTML).toBe(TEXT);
    expect(render(<CheckedText text={null} claims={[claim(TEXT, "$123.45B")]} />).container.innerHTML).toBe("");
  });

  it("marks the captured memo's fabricated PM figure at its stored offsets", () => {
    const memo = qualityMemo();
    const claims = claimsFor(memo, "final_pm_view");
    expect(placeable(memo.final_pm_view, claims).map((c) => c.raw)).toContain(QUALITY_EXPECT.fabricated_pm_figure);
    const { container } = render(<CheckedText text={memo.final_pm_view} claims={claims} />);
    expect(container.textContent).toBe(memo.final_pm_view);
    expect(marks(container)).toEqual([
      [QUALITY_EXPECT.fabricated_pm_figure, "untraceable"],
      [QUALITY_EXPECT.declared_assumption, "assumption"],
    ]);
  });
});
