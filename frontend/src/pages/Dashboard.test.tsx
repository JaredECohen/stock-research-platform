import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import Dashboard from "@/pages/Dashboard";
import { okJson, renderWithProviders, stubFetch } from "@/test/providers";

describe("Dashboard links", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("points every link at the /app shell so nothing bounces through a legacy redirect", async () => {
    stubFetch([
      [
        "/api/screener",
        () =>
          okJson({
            theme: null,
            generated_at: "",
            total: 1,
            rows: [{ rank: 1, ticker: "NVDA", company_name: "NVIDIA", sector: "Technology", pm_score: 88, one_line_thesis: "x", rating_label: "Bullish" }],
          }),
      ],
    ]);
    renderWithProviders(<Dashboard />, { route: "/app" });
    await screen.findByRole("link", { name: "NVDA" });
    const hrefs = screen.getAllByRole("link").map((a) => a.getAttribute("href") || "");
    expect(hrefs.length).toBeGreaterThan(5);
    for (const h of hrefs) expect(h.startsWith("/app/")).toBe(true);
    expect(screen.getByRole("link", { name: "NVDA" })).toHaveAttribute("href", "/app/research?ticker=NVDA");
    expect(hrefs.some((h) => h.startsWith("/app/chat?q="))).toBe(true);
  });
});
