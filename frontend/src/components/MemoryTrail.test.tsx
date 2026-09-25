import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import MemoryTrail from "@/components/MemoryTrail";
import { api } from "@/api/client";

// The two shapes `GET /api/stocks/{t}/memory` serves
// (`routes_stocks.get_stock_memory`): the legacy file trail in off/shadow,
// and in inject the learning ledger (`learning/context.public_trail`), whose
// `historical_context` is the coverage note the W7 critique requires on the
// page. The keys are identical; `path` tells them apart.
type Memory = Awaited<ReturnType<typeof api.stockMemory>>;

const COVERAGE =
  "Learning ledger for ZZLRN: 1 lesson(s), 1 tested against later benchmark-relative outcomes, " +
  "and 1 filing observation(s). Lessons are provisional hypotheses, not investment advice; " +
  "each memo's current evidence takes precedence.";

function ledger(overrides: Partial<Memory> = {}): Memory {
  return {
    ticker: "ZZLRN",
    path: "database: learning ledger",
    entry_count: 2,
    historical_context: COVERAGE,
    entries: [
      {
        date: "2026-08-01",
        trigger: "filing observation",
        body: "What's new in 10-Q filed 2026-08-01: bookings up.",
        structured_facts: null,
      },
      {
        date: "2026-06-01",
        trigger: "provisional hypothesis · supported 4 of 4 later outcomes",
        body:
          "When bookings accelerate, expect the stock to outperform the benchmark over 90 days." +
          "\n\nProvisional hypothesis — not investment advice.",
        structured_facts: null,
      },
    ],
    ...overrides,
  } as Memory;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("MemoryTrail", () => {
  it("shows the ledger's coverage note beside its entries", async () => {
    vi.spyOn(api, "stockMemory").mockResolvedValue(ledger());
    render(<MemoryTrail ticker="ZZLRN" />);
    expect(await screen.findByText(COVERAGE)).toBeInTheDocument();
    expect(
      screen.getByText("provisional hypothesis · supported 4 of 4 later outcomes"),
    ).toBeInTheDocument();
  });

  it("shows the coverage note even when the ledger has no entries yet", async () => {
    vi.spyOn(api, "stockMemory").mockResolvedValue(
      ledger({ entry_count: 0, entries: [] }),
    );
    render(<MemoryTrail ticker="ZZLRN" />);
    expect(await screen.findByText(COVERAGE)).toBeInTheDocument();
  });

  it("keeps the legacy trail's condensed context hidden, as before", async () => {
    vi.spyOn(api, "stockMemory").mockResolvedValue(
      ledger({
        path: "/app/memory/companies/ZZLRN.md",
        historical_context: "- 2026-01-02: old condensed takeaway",
        entries: [
          {
            date: "2026-05-01",
            trigger: "earnings",
            body: "Legacy file entry.",
            structured_facts: null,
          },
        ],
      }),
    );
    render(<MemoryTrail ticker="ZZLRN" />);
    expect(await screen.findByText("Legacy file entry.")).toBeInTheDocument();
    expect(screen.queryByText(/old condensed takeaway/)).toBeNull();
  });
});
