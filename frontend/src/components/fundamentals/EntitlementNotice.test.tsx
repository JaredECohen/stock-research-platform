import { describe, expect, it } from "vitest";
import { screen, within } from "@testing-library/react";
import EntitlementNotice, { shapeText } from "@/components/fundamentals/EntitlementNotice";
import { renderWithProviders } from "@/test/providers";

describe("EntitlementNotice", () => {
  it("formats a plan shape with full history or a year ceiling", () => {
    expect(shapeText({ max_companies: 2, max_metrics: 2, max_years: 5 })).toBe("2 companies × 2 metrics × 5 years");
    expect(shapeText({ max_companies: 5, max_metrics: 4, max_years: null })).toBe("5 companies × 4 metrics × full history");
    expect(shapeText({ max_companies: "x" })).toBeNull();
  });

  it("renders the 402 body's limits, request and upgrade shape verbatim", () => {
    renderWithProviders(
      <EntitlementNotice
        refusal={{
          code: "plan_required",
          message: "ignored when the extras are present",
          feature: "fundamentals_explorer",
          plan: "free",
          upgrade_url: "/pricing",
          extra: {
            limits: { max_companies: 2, max_metrics: 2, max_years: 5 },
            requested: { companies: 3, metrics: 1, years: null },
            upgrade: { plan: "pro", limits: { max_companies: 5, max_metrics: 4, max_years: null }, url: "/pricing?from=fundamentals" },
          },
        }}
      />,
    );
    const notice = screen.getByTestId("entitlement-notice");
    expect(notice).toHaveTextContent("The Free plan draws up to 2 companies × 2 metrics × 5 years; this URL asks for 3 companies × 1 metric.");
    expect(notice).toHaveTextContent("Pro draws up to 5 companies × 4 metrics × full history.");
    expect(within(notice).getByRole("link", { name: "See Pro plans" })).toHaveAttribute("href", "/pricing?from=fundamentals");
    expect(notice).not.toHaveTextContent("ignored when");
  });

  it("falls back to the message when the extras are missing, and hides the upgrade link on Pro", () => {
    renderWithProviders(<EntitlementNotice refusal={{ code: "plan_required", message: "Too big.", plan: "pro" }} />);
    expect(screen.getByTestId("entitlement-notice")).toHaveTextContent("Too big.");
    expect(screen.queryByRole("link", { name: "See Pro plans" })).toBeNull();
  });

  it("explains a capped range with the plan ceiling and the years asked for", () => {
    renderWithProviders(<EntitlementNotice capped={{ applied: { max_companies: 2, max_metrics: 2, max_years: 5 }, requestedYears: 10 }} />);
    expect(screen.getByTestId("entitlement-notice")).toHaveTextContent("This URL asks for 10 years; the plan draws up to 5 years");
  });

  it("renders nothing without a refusal or a cap", () => {
    const { container } = renderWithProviders(<EntitlementNotice />);
    expect(container.querySelector('[data-testid="entitlement-notice"]')).toBeNull();
  });
});
