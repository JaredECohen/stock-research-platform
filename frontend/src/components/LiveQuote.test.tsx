import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import LiveQuote, { NEXT_OPEN_SLACK_MS, POLL_MS, parseUtc } from "@/components/LiveQuote";
import { eodClose, liveAtClose, liveOpen, staleQuote, unavailable } from "@/test/fixtures/quotes";
import type { QuotesOut } from "@/types/quotes";

// Every body is the CAPTURED `/api/quotes` output (see test/fixtures/quotes.ts).
const apiMock = vi.hoisted(() => ({ getQuotes: vi.fn() }));
vi.mock("@/api/client", () => ({ api: apiMock }));

async function flush() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function renderWith(body: QuotesOut, props: Partial<React.ComponentProps<typeof LiveQuote>> = {}) {
  apiMock.getQuotes.mockResolvedValue(body);
  const view = render(<LiveQuote ticker={body.quotes[0].ticker.toLowerCase()} {...props} />);
  await flush();
  return view;
}

function chipText(): string {
  return (screen.getByTestId("live-quote").textContent ?? "").replace(/\s+/g, " ");
}

describe("LiveQuote", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("live and open: price, change, ET time and the delay disclosure", async () => {
    await renderWith(liveOpen);
    expect(apiMock.getQuotes).toHaveBeenCalledWith(["QTLIVE"]);
    const text = chipText();
    expect(text).toContain("$123.45");
    expect(text).toContain("+2.45 (+2.02%)");
    expect(text).toContain("10:44 AM ET");
    expect(text).toContain("may be delayed up to 15 min");
    expect(screen.getByTestId("live-quote").dataset.source).toBe("live");
  });

  it("live and open, but timed in an earlier session: shows that day, not a bare time", async () => {
    // Just after the open a delayed feed can still answer with Tuesday's close.
    const prior: QuotesOut = {
      ...liveOpen,
      quotes: [{ ...liveOpen.quotes[0], price_time: "2026-09-22T20:00:00Z", as_of: "2026-09-22T20:00:00Z" }],
    };
    await renderWith(prior);
    expect(liveOpen.market.session_date).toBe("2026-09-23");
    expect(chipText()).toContain("· Tue Sep 22, 4:00 PM ET · may be delayed up to 15 min");
  });

  it("live after the close: 'At close' with the session's date and 4:00 PM ET", async () => {
    await renderWith(liveAtClose);
    expect(chipText()).toContain("$88.10 · At close, Wed Sep 23, 4:00 PM ET");
    expect(chipText()).not.toContain("may be delayed");
  });

  it("says 'last quote', not 'At close', when the provider time is before the close", async () => {
    const early: QuotesOut = {
      ...liveAtClose,
      quotes: [{ ...liveAtClose.quotes[0], price_time: "2026-09-23T19:40:00Z", as_of: "2026-09-23T19:40:00Z" }],
    };
    await renderWith(early);
    expect(chipText()).toContain("last quote Wed Sep 23, 3:40 PM ET");
    expect(chipText()).not.toContain("At close");
  });

  it("stale: amber, with the last quote time and 'live quote unavailable'", async () => {
    await renderWith(staleQuote);
    const chip = screen.getByTestId("live-quote");
    expect(chipText()).toContain("$64.20 · last quote 10:24 AM ET, live quote unavailable");
    expect(chip.querySelector(".text-warn-500")).not.toBeNull();
    expect(chip.dataset.source).toBe("stale");
  });

  it("eod_close: the stored close and its date, labelled as no live quote", async () => {
    await renderWith(eodClose);
    expect(chipText()).toContain("$45.67 · last close, Tue Sep 22 (no live quote)");
  });

  it("unavailable renders nothing, and so does a failed request", async () => {
    const { container } = await renderWith(unavailable);
    expect(container).toBeEmptyDOMElement();
    apiMock.getQuotes.mockRejectedValue(new Error("boom"));
    const onQuote = vi.fn();
    const failed = render(<LiveQuote ticker="QTLIVE" onQuote={onQuote} />);
    await flush();
    expect(failed.container).toBeEmptyDOMElement();
    expect(onQuote).toHaveBeenCalledWith(null);
  });

  it("shows drift since the memo from price_at_memo (naive UTC memo time read as UTC)", async () => {
    await renderWith(liveOpen, { priceAtMemo: 100, memoAt: "2026-09-03T14:00:00" });
    expect(chipText()).toContain("Since memo: +23.5% (memo price $100.00, Sep 3)");
    expect(parseUtc("2026-09-03T14:00:00").toISOString()).toBe("2026-09-03T14:00:00.000Z");
  });

  it("polls every 5 minutes only while the market is open", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-23T14:45:00Z"));
    await renderWith(liveOpen);
    expect(apiMock.getQuotes).toHaveBeenCalledTimes(1);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_MS);
    });
    expect(apiMock.getQuotes).toHaveBeenCalledTimes(2);
  });

  it("does not poll while closed, and refetches at the next open", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-23T21:00:00Z"));
    await renderWith(liveAtClose);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_MS * 3);
    });
    expect(apiMock.getQuotes).toHaveBeenCalledTimes(1);
    // next_open is 2026-09-24T13:30:00Z: 16.5 hours later, plus the slack.
    const untilOpen = Date.parse(liveAtClose.market.next_open) - Date.now() + NEXT_OPEN_SLACK_MS;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(untilOpen);
    });
    expect(apiMock.getQuotes).toHaveBeenCalledTimes(2);
  });

  it("refetches when the tab becomes visible again", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-23T21:00:00Z"));
    await renderWith(liveAtClose);
    vi.setSystemTime(new Date("2026-09-23T21:05:00Z"));
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"));
      await Promise.resolve();
    });
    expect(apiMock.getQuotes).toHaveBeenCalledTimes(2);
  });

  it("does not poll while the tab is hidden, even with the market open", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-23T14:45:00Z"));
    const visibility = vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
    try {
      await renderWith(liveOpen);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_MS * 2);
      });
      expect(apiMock.getQuotes).toHaveBeenCalledTimes(1);
      // Becoming visible again is what refetches.
      visibility.mockReturnValue("visible");
      await act(async () => {
        document.dispatchEvent(new Event("visibilitychange"));
        await Promise.resolve();
      });
      expect(apiMock.getQuotes).toHaveBeenCalledTimes(2);
    } finally {
      visibility.mockRestore();
    }
  });

  it("throttles visibility refetches to one a minute, and ignores the tab being hidden", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-23T21:00:00Z"));
    await renderWith(liveAtClose);
    vi.setSystemTime(new Date("2026-09-23T21:00:30Z"));  // 30 s after the fetch
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"));
      await Promise.resolve();
    });
    expect(apiMock.getQuotes).toHaveBeenCalledTimes(1);
    const visibility = vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
    try {
      vi.setSystemTime(new Date("2026-09-23T21:05:00Z"));
      await act(async () => {
        document.dispatchEvent(new Event("visibilitychange"));
        await Promise.resolve();
      });
      expect(apiMock.getQuotes).toHaveBeenCalledTimes(1);
    } finally {
      visibility.mockRestore();
    }
  });

  it("reports the quote to its parent", async () => {
    const onQuote = vi.fn();
    await renderWith(staleQuote, { onQuote });
    expect(onQuote).toHaveBeenCalledWith(staleQuote.quotes[0]);
  });
});
