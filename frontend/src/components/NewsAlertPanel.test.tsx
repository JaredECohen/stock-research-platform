import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import NewsAlertPanel from "@/components/NewsAlertPanel";
import type { NewsAlert } from "@/types";

// The news agent stores a date-only published date as a bare "YYYY-MM-DD"
// (N34 normalisation; the Gemini prompt asks for ISO dates). `new Date()`
// reads that form as UTC midnight, the previous evening west of UTC, so
// the panel showed those stories a day early for US viewers. Pin a US zone
// so the check means the same thing on a UTC CI runner.
// Node re-reads TZ when it is assigned; the app's tsconfig has no Node
// types, so reach the env through globalThis.
const env = (globalThis as unknown as { process: { env: Record<string, string | undefined> } })
  .process.env;
const savedTz = env.TZ;
beforeAll(() => {
  env.TZ = "America/New_York";
});
afterAll(() => {
  if (savedTz === undefined) delete env.TZ;
  else env.TZ = savedTz;
});

function alert(published_at: string): NewsAlert {
  return {
    ticker: "BK",
    title: "BNY reports third-quarter results",
    url: "https://www.reuters.com/x",
    severity: "material",
    published_at,
    source: "gemini",
  };
}

function shown(published_at: string): string | null {
  const { container, unmount } = render(<NewsAlertPanel alerts={[alert(published_at)]} />);
  const text = container.querySelector("li span.opacity-70")?.textContent ?? null;
  unmount();
  return text;
}

describe("NewsAlertPanel dates", () => {
  it("shows a bare ISO date as that calendar day, not the day before", () => {
    const expected = new Date(2026, 8, 20).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    });
    expect(shown("2026-09-20")).toBe(expected);
  });

  it("still converts a timestamped instant", () => {
    const expected = new Date("2026-09-21T15:00:00Z").toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    });
    expect(shown("2026-09-21T15:00:00Z")).toBe(expected);
  });

  it("falls back to the raw text for an unreadable date", () => {
    render(<NewsAlertPanel alerts={[alert("unknown")]} />);
    expect(screen.getByText("unknown")).toBeTruthy();
  });
});
