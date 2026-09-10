import { describe, expect, it } from "vitest";
import { screen, within } from "@testing-library/react";
import SeriesStateNotice from "@/components/fundamentals/SeriesStateNotice";
import { makeSeries } from "@/test/fixtures/fundamentals";
import { renderWithProviders } from "@/test/providers";

const LABELS = { revenue: "Revenue", gross_margin: "Gross margin" };

describe("SeriesStateNotice", () => {
  it("renders nothing when every series is drawn, current and observed", () => {
    const { container } = renderWithProviders(<SeriesStateNotice unavailable={[]} series={[makeSeries("AAPL", "revenue", [1, 2, 3, 4, 5])]} />);
    expect(container.querySelector('[data-testid="series-state"]')).toBeNull();
  });

  it("names a not-backfilled company with the server's remedy and the research CTA", () => {
    renderWithProviders(
      <SeriesStateNotice
        unavailable={[{ ticker: "NEWCO", reason: "not_backfilled", remedy: "Run research on it — the memo job backfills its statement history." }]}
        series={[]}
      />,
    );
    const row = screen.getByTestId("unavailable-NEWCO");
    expect(row).toHaveTextContent("NEWCO — history not loaded.");
    expect(row).toHaveTextContent("the memo job backfills its statement history");
    expect(within(row).getByRole("link", { name: "Run research on NEWCO" })).toHaveAttribute("href", "/app/research?ticker=NEWCO");
  });

  it("still names an unavailable company whose reason has no research remedy", () => {
    renderWithProviders(<SeriesStateNotice unavailable={[{ ticker: "ZZZZ", reason: "unknown_ticker" }]} series={[]} />);
    const row = screen.getByTestId("unavailable-ZZZZ");
    // An unknown wire reason is shown, never swallowed, and there is no CTA
    // because research is not the fix for a ticker the platform cannot find.
    expect(row).toHaveTextContent("ZZZZ — unknown ticker.");
    expect(within(row).queryByRole("link")).toBeNull();
  });

  it("labels stale and estimated series in text, with the reason and the count", () => {
    renderWithProviders(
      <SeriesStateNotice
        unavailable={[]}
        series={[
          makeSeries("MSFT", "gross_margin", [0.6, 0.6, 0.6, 0.6, 0.6], { stale: true, stale_reason: "last fiscal period FY2022 is older than 15 months" }),
          makeSeries("AAPL", "revenue", [1, { v: 2, estimated: true }, { v: 3, estimated: true }, 4, 5]),
        ]}
        metricLabels={LABELS}
      />,
    );
    expect(screen.getByTestId("stale-badge")).toHaveTextContent("MSFT Gross margin: last fiscal period FY2022 is older than 15 months. Drawn at reduced opacity.");
    expect(screen.getByTestId("estimated-badge")).toHaveTextContent("AAPL Revenue: 2 values computed with a documented fallback");
  });

  it("lists the server's warnings verbatim", () => {
    renderWithProviders(<SeriesStateNotice unavailable={[]} series={[]} warnings={["1 row labelled 2024Q3 was ignored in an annual request."]} />);
    expect(screen.getByTestId("series-warnings")).toHaveTextContent("1 row labelled 2024Q3 was ignored in an annual request.");
  });
});
