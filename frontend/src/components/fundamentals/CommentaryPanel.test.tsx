import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import CommentaryPanel from "@/components/fundamentals/CommentaryPanel";
import { makeCommentary } from "@/test/fixtures/fundamentals";
import { calls, ent, errJson, okJson, renderWithProviders, stubFetch } from "@/test/providers";

const SELECTION = { tickers: ["AAPL"], metrics: ["revenue"], years: 5 as number | null, fingerprint: "fp-aapl-revenue" };

afterEach(() => vi.unstubAllGlobals());

describe("CommentaryPanel", () => {
  it("disables the button with the reason when the monthly allowance is used up", () => {
    stubFetch();
    renderWithProviders(
      <CommentaryPanel {...SELECTION} entitlement={ent("chart_commentary", { limit: 5, used: 5, remaining: 0, metered: true })} accountEnabled />,
    );
    const button = screen.getByRole("button", { name: "Explain this chart" });
    expect(button).toBeDisabled();
    const reason = screen.getByTestId("commentary-reason");
    expect(reason).toHaveTextContent("You've used 5 of 5 chart commentaries this month; the allowance resets Oct 1, 2026 (UTC).");
    expect(button).toHaveAttribute("aria-describedby", reason.id);
    expect(screen.getByTestId("meter-chart_commentary")).toHaveTextContent("5 of 5 used");
  });

  it("disables the button until a chart is drawn", () => {
    stubFetch();
    renderWithProviders(<CommentaryPanel {...SELECTION} fingerprint={null} />);
    expect(screen.getByRole("button", { name: "Explain this chart" })).toBeDisabled();
    expect(screen.getByTestId("commentary-reason")).toHaveTextContent("Draw a chart first");
  });

  it("sends exactly the displayed selection and renders the two sections under distinct headings", async () => {
    const mock = stubFetch([["/api/fundamentals/commentary", () => okJson(makeCommentary({ memo_view: [{ ...makeCommentary().memo_view[0], memo_stale: true, memo_stale_reason: "memo predates FY2024" }] }))]]);
    renderWithProviders(<CommentaryPanel {...SELECTION} />);
    fireEvent.click(screen.getByRole("button", { name: "Explain this chart" }));
    expect(screen.getByTestId("commentary-status")).toHaveTextContent("Generating commentary…");
    await screen.findByRole("heading", { level: 3, name: "Observed in the data" });
    expect(screen.getByRole("heading", { level: 3, name: "From stored memos" })).toBeInTheDocument();
    expect(JSON.parse(String(calls(mock, "/api/fundamentals/commentary")[0][1]?.body))).toEqual(SELECTION);
    // Observed items cite their points; memo items name the version and the stale flag.
    expect(screen.getByText("AAPL revenue FY2020 · AAPL revenue FY2024")).toBeInTheDocument();
    expect(screen.getByText("The stored memo sees services mix as the margin lever.")).toHaveTextContent("AAPL memo v14 · generated Aug 30, 2026");
    expect(screen.getByText("stale")).toHaveAttribute("title", "memo predates FY2024");
    expect(screen.getByRole("heading", { level: 3, name: "Caveats" })).toBeInTheDocument();
    expect(screen.getByTestId("commentary-status")).toHaveTextContent("Commentary ready.");
  });

  it("shows the upgrade prompt on a 402 quota refusal and keeps the button disabled", async () => {
    stubFetch([
      [
        "/api/fundamentals/commentary",
        () =>
          errJson(402, {
            code: "quota_exceeded",
            feature: "chart_commentary",
            plan: "free",
            used: 5,
            limit: 5,
            resets_at: "2026-10-01T00:00:00",
            upgrade_url: "/pricing",
            message: "Monthly chart commentary allowance used.",
          }),
      ],
    ]);
    renderWithProviders(<CommentaryPanel {...SELECTION} />, { config: { auth_enabled: true } });
    fireEvent.click(screen.getByRole("button", { name: "Explain this chart" }));
    expect(await screen.findByTestId("upgrade-prompt")).toHaveTextContent("You've used 5 of 5 chart commentaries this month");
    expect(screen.getByRole("button", { name: "Explain this chart" })).toBeDisabled();
    expect(screen.getByTestId("commentary-reason")).toHaveTextContent("Monthly chart commentary allowance used.");
  });

  it("keeps the selection and offers a retry on a 429", async () => {
    let n = 0;
    stubFetch([
      [
        "/api/fundamentals/commentary",
        () => (n++ === 0 ? errJson(429, { code: "concurrency_limited", scope: "concurrency", retry_after: 0, message: "Two commentaries are already in flight." }) : okJson(makeCommentary())),
      ],
    ]);
    renderWithProviders(<CommentaryPanel {...SELECTION} />);
    fireEvent.click(screen.getByRole("button", { name: "Explain this chart" }));
    expect(await screen.findByText("Your chart and selection are unchanged.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Retry/ }));
    await screen.findByRole("heading", { level: 3, name: "Observed in the data" });
    expect(n).toBe(2);
  });

  it("clears the answer when the displayed chart changes", async () => {
    stubFetch([["/api/fundamentals/commentary", () => okJson(makeCommentary())]]);
    const view = renderWithProviders(<CommentaryPanel {...SELECTION} />);
    fireEvent.click(screen.getByRole("button", { name: "Explain this chart" }));
    await screen.findByRole("heading", { level: 3, name: "Observed in the data" });
    view.rerender(<CommentaryPanel {...SELECTION} fingerprint="fp-other" />);
    await waitFor(() => expect(screen.queryByRole("heading", { level: 3, name: "Observed in the data" })).toBeNull());
  });
});
