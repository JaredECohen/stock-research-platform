import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import MemoVersionTimeline from "@/components/MemoVersionTimeline";
import { api } from "@/api/client";

// The rows as `GET /api/stocks/{t}/memos` serves them since W2a
// (`routes_stocks._history_row`): a version whose confidence the presenter
// hid carries `confidence_score: null` with `confidence_available: false`;
// a clean one the number with `true`; an unreadable stored row the raw
// number with `null`.
type HistoryRow = Awaited<ReturnType<typeof api.memoHistory>>[number] & {
  confidence_available?: boolean | null;
};

function row(overrides: Partial<HistoryRow>): HistoryRow {
  return {
    version: 1,
    trigger: "full_reanalysis",
    parent_version: null,
    generated_at: "2026-09-20T12:00:00Z",
    revision_log: [],
    rating_label: "Neutral",
    confidence_score: 62,
    confidence_available: true,
    ...overrides,
  };
}

function mount(rows: HistoryRow[]) {
  vi.spyOn(api, "memoHistory").mockResolvedValue(rows);
  render(<MemoVersionTimeline ticker="GOOGL" />);
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("MemoVersionTimeline", () => {
  it("says 'confidence n/a' for a version whose confidence is unavailable", async () => {
    mount([
      row({ version: 2, rating_label: "Bullish", confidence_score: null, confidence_available: false }),
      row({ version: 1 }),
    ]);
    expect(await screen.findByText("Bullish · confidence n/a")).toBeInTheDocument();
    expect(screen.getByText("Neutral · 62")).toBeInTheDocument();
  });

  it("shows the raw number for a row the server could not present", async () => {
    mount([row({ confidence_score: 58.6, confidence_available: null })]);
    expect(await screen.findByText("Neutral · 59")).toBeInTheDocument();
  });

  it("keeps rows from servers that pre-date the field", async () => {
    const legacy = row({ confidence_score: 70 });
    delete legacy.confidence_available;
    mount([legacy]);
    expect(await screen.findByText("Neutral · 70")).toBeInTheDocument();
    expect(screen.queryByText(/confidence n\/a/)).not.toBeInTheDocument();
  });
});
