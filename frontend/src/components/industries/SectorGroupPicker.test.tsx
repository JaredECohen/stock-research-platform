import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import SectorGroupPicker, { optionId } from "@/components/industries/SectorGroupPicker";
import * as fx from "@/test/fixtures/industry";

// The picker is the only way into a 25-group list, so the two failures
// that matter are (1) a listbox a keyboard cannot drive and (2) a wide
// list rendered on a phone. Both are asserted here against the real
// taxonomy the API served.

function mountWide(value: string | null = null) {
  const onSelect = vi.fn();
  render(<SectorGroupPicker taxonomy={fx.taxonomy} value={value} onSelect={onSelect} />);
  return { onSelect, listbox: screen.getByRole("listbox") };
}

/** jsdom has no matchMedia; installing one lets the component take the
 *  narrow branch exactly as a phone would. */
function stubMatchMedia(matches: boolean) {
  vi.stubGlobal(
    "matchMedia",
    (query: string) =>
      ({
        matches,
        media: query,
        onchange: null,
        addEventListener: () => {},
        removeEventListener: () => {},
        addListener: () => {},
        removeListener: () => {},
        dispatchEvent: () => false,
      }) as unknown as MediaQueryList,
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("SectorGroupPicker — wide", () => {
  it("renders every group in the response, grouped by sector", () => {
    const { listbox } = mountWide();
    expect(within(listbox).getAllByRole("option")).toHaveLength(fx.allGroups.length);
    expect(within(listbox).getAllByRole("group")).toHaveLength(fx.taxonomy.sectors.length);
    const sector = fx.taxonomy.sectors[0];
    expect(within(listbox).getByRole("group", { name: `${sector.code} ${sector.name}` })).toBeInTheDocument();
  });

  it("moves aria-activedescendant with the arrow keys and Home/End", () => {
    const { listbox } = mountWide();
    const codes = fx.allGroups.map((g) => g.code);

    expect(listbox).toHaveAttribute("aria-activedescendant", optionId(codes[0]));
    fireEvent.keyDown(listbox, { key: "ArrowDown" });
    expect(listbox).toHaveAttribute("aria-activedescendant", optionId(codes[1]));
    fireEvent.keyDown(listbox, { key: "ArrowUp" });
    expect(listbox).toHaveAttribute("aria-activedescendant", optionId(codes[0]));
    fireEvent.keyDown(listbox, { key: "End" });
    expect(listbox).toHaveAttribute("aria-activedescendant", optionId(codes[codes.length - 1]));
    fireEvent.keyDown(listbox, { key: "Home" });
    expect(listbox).toHaveAttribute("aria-activedescendant", optionId(codes[0]));
  });

  it("does not run off either end of the list", () => {
    const { listbox } = mountWide();
    const codes = fx.allGroups.map((g) => g.code);
    fireEvent.keyDown(listbox, { key: "ArrowUp" });
    expect(listbox).toHaveAttribute("aria-activedescendant", optionId(codes[0]));
    fireEvent.keyDown(listbox, { key: "End" });
    fireEvent.keyDown(listbox, { key: "ArrowDown" });
    expect(listbox).toHaveAttribute("aria-activedescendant", optionId(codes[codes.length - 1]));
  });

  it("selects the active option with Enter and with Space", () => {
    const { listbox, onSelect } = mountWide();
    const codes = fx.allGroups.map((g) => g.code);
    fireEvent.keyDown(listbox, { key: "ArrowDown" });
    fireEvent.keyDown(listbox, { key: "Enter" });
    expect(onSelect).toHaveBeenCalledWith(codes[1]);
    fireEvent.keyDown(listbox, { key: " " });
    expect(onSelect).toHaveBeenLastCalledWith(codes[1]);
  });

  it("marks the selected group and starts the cursor there on a deep link", () => {
    const target = fx.groupWithEdition.code;
    const { listbox } = mountWide(target);
    expect(listbox).toHaveAttribute("aria-activedescendant", optionId(target));
    expect(screen.getByRole("option", { selected: true })).toHaveAttribute("id", optionId(target));
  });

  it("says what a group's edition is, or that it has none, on every row", () => {
    mountWide();
    const withEdition = fx.groupWithEdition;
    const row = document.getElementById(optionId(withEdition.code))!;
    expect(row.textContent).toContain(`v${withEdition.latest_report!.version}`);

    const without = fx.allGroups.find((g) => g.latest_report === null);
    if (without) {
      expect(document.getElementById(optionId(without.code))!.textContent).toContain("no published edition yet");
    }
  });
});

describe("SectorGroupPicker — narrow", () => {
  it("renders the platform select with an optgroup per sector", () => {
    stubMatchMedia(true);
    render(<SectorGroupPicker taxonomy={fx.taxonomy} value={null} onSelect={() => {}} />);

    const select = screen.getByTestId("industry-picker-select");
    expect(select.tagName).toBe("SELECT");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    expect(select.querySelectorAll("optgroup")).toHaveLength(fx.taxonomy.sectors.length);
    // The placeholder is the extra option.
    expect(select.querySelectorAll("option")).toHaveLength(fx.allGroups.length + 1);
  });

  it("selecting an option calls back with the code", () => {
    stubMatchMedia(true);
    const onSelect = vi.fn();
    render(<SectorGroupPicker taxonomy={fx.taxonomy} value={null} onSelect={onSelect} />);
    fireEvent.change(screen.getByTestId("industry-picker-select"), { target: { value: fx.groupWithEdition.code } });
    expect(onSelect).toHaveBeenCalledWith(fx.groupWithEdition.code);
  });
});

describe("SectorGroupPicker — groups this universe cannot cover", () => {
  it("marks a structurally short group where the reader chooses what to open", () => {
    const short = fx.allGroups.find((g) => !g.universe_coverage.coverable)!;
    expect(short.universe_coverage.constituents_short_by).toBeGreaterThan(0);
    // The floor the verdict was taken against is the deployment's, and it
    // travels with the response rather than being assumed here.
    expect(short.universe_coverage.min_sample).toBe(fx.MIN_SAMPLE);
    mountWide();
    const note = screen.getByTestId(`not-coverable-${short.code}`);
    expect(note).toHaveTextContent("Too few companies in this universe to report on");
    // The server's sentence, verbatim — the picker composes none of it.
    expect(note).toHaveTextContent(short.universe_coverage.explanation);
  });

  it("leaves a coverable group unmarked, so the mark means something", () => {
    const ok = fx.allGroups.find((g) => g.universe_coverage.coverable)!;
    mountWide();
    expect(screen.queryByTestId(`not-coverable-${ok.code}`)).not.toBeInTheDocument();
  });

  it("carries the same mark into the phone rendering", () => {
    stubMatchMedia(true);
    const short = fx.allGroups.find((g) => !g.universe_coverage.coverable)!;
    render(<SectorGroupPicker taxonomy={fx.taxonomy} value={null} onSelect={() => {}} />);
    const option = screen.getByTestId("industry-picker-select").querySelector(`option[value="${short.code}"]`);
    expect(option?.textContent).toContain("too few to report on");
  });
});

describe("SectorGroupPicker — empty taxonomy", () => {
  it("says the taxonomy has no groups rather than rendering an empty box", () => {
    const empty = fx.clone(fx.taxonomy);
    empty.sectors = [];
    render(<SectorGroupPicker taxonomy={empty} value={null} onSelect={() => {}} />);
    expect(screen.getByTestId("industry-picker-empty")).toHaveTextContent("no industry groups");
  });
});
