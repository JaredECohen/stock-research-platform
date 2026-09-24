import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import TrackRecord from "@/pages/TrackRecord";
import { api } from "@/api/client";
import type { TrackRecordOut } from "@/types/trackRecord";
// NOT hand-written: `GET /api/admin/track-record` as the backend served it
// after the real eligibility sweep over a fixed seed
// (`backend/app/scripts/capture_track_record_fixture.py`), and
// `test_track_record_fixture_contract.py` fails when it drifts.
import wire from "@/test/fixtures/trackRecord.wire.json";

const BY_HORIZON: Record<number, TrackRecordOut> = {
  90: wire.established_90d as TrackRecordOut,
  30: wire.provisional_30d as TrackRecordOut,
  365: wire.empty_365d as TrackRecordOut,
};

function mount(bodies: Record<number, TrackRecordOut> = BY_HORIZON) {
  const spy = vi.spyOn(api, "trackRecord").mockImplementation(async (params) => {
    const body = bodies[params?.horizon_days ?? 90];
    if (!body) throw new Error(`no fixture for ${params?.horizon_days}`);
    return body;
  });
  render(
    <MemoryRouter>
      <TrackRecord />
    </MemoryRouter>,
  );
  return spy;
}

async function pick(horizon: number) {
  fireEvent.change(screen.getByLabelText("Horizon (days)"), { target: { value: String(horizon) } });
  await waitFor(() => expect(screen.queryByText("Loading…")).toBeNull());
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("TrackRecord (W6 provisional record)", () => {
  it("shows the provisional banner with the 30-day coverage numbers", async () => {
    mount();
    await screen.findByText("Memos evaluated");
    await pick(30);
    const banner = await screen.findByRole("status");
    const p = wire.provisional_30d.provisional;
    expect(banner.textContent).toContain(`${p.companies} companies`);
    expect(banner.textContent).toContain(`${p.directional} directional`);
    expect(banner.textContent).toContain(`${wire.provisional_30d.coverage.universe_companies}-company universe`);
    expect(banner.textContent).toContain(`at least ${p.min_companies} companies and ${p.min_directional}`);
    expect(banner.textContent).toMatch(/not independent calls/);
  });

  it("drops the banner once the 90-day record clears the thresholds", async () => {
    mount();
    await screen.findByText("Memos evaluated");
    expect(wire.established_90d.provisional.is_provisional).toBe(false);
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("puts SPY-relative alpha beside the absolute hit rate", async () => {
    mount();
    const hit = (await screen.findByText("Thesis hit rate")).parentElement as HTMLElement;
    expect(within(hit).getByText("61%")).toBeTruthy();
    expect(within(hit).getByText(/always-Bullish would score 67%/)).toBeTruthy();
    const beat = screen.getByText("Beat SPY").parentElement as HTMLElement;
    expect(within(beat).getByText("50%")).toBeTruthy();
    expect(within(beat).getByText(/median alpha \+1\.5% · per-company -1\.2%/)).toBeTruthy();
    // Cards 2 and 3 are neighbours: alpha is next to the hit rate.
    expect(hit.nextElementSibling).toBe(beat);
  });

  it("says what it does not count, and that nothing was deleted", async () => {
    mount();
    const note = await screen.findByLabelText("Outcomes not counted");
    expect(note.textContent).toContain("kept, not deleted");
    expect(note.textContent).toContain(
      "3 outcomes on demo-mode memos copied from a development machine on 2026-05-04",
    );
    expect(note.textContent).toContain("1 outcomes on memos with no recorded generation mode");
  });

  it("discloses rating sources and late-evaluation candidates", async () => {
    mount();
    await screen.findByText("Memos evaluated");
    expect(screen.getByText(/Rated by: deterministic keyword PM 72 · LLM PM 36/)).toBeTruthy();
    expect(screen.getByText(/not the LLM\s+committee/)).toBeTruthy();
    expect(screen.getByText(/1 of the 90-day outcomes were evaluated/)).toBeTruthy();
    const table = screen.getByRole("table", { name: "Coverage by horizon" });
    expect(within(table).getAllByRole("row")).toHaveLength(1 + wire.established_90d.coverage.horizons.length);
  });

  it("shows the empty state for a horizon with no outcomes", async () => {
    mount();
    await screen.findByText("Memos evaluated");
    await pick(365);
    expect(await screen.findByText(/No outcomes for this filter yet/)).toBeTruthy();
    expect(screen.queryByText("Memos evaluated")).toBeNull();
  });

  it("falls back to the raw code for an exclusion reason it has no label for", async () => {
    const body = structuredClone(wire.established_90d) as TrackRecordOut;
    body.eligibility.excluded_by_reason = { some_future_reason_v2: 4 };
    body.eligibility.excluded = 4;
    body.eligibility.unclassified = 2;
    mount({ ...BY_HORIZON, 90: body });
    const note = await screen.findByLabelText("Outcomes not counted");
    expect(note.textContent).toContain("4 outcomes on some_future_reason_v2");
    expect(note.textContent).toContain("2 awaiting classification");
  });
});
