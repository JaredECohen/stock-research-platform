import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import Legal from "@/pages/public/Legal";
import { STORAGE_INVENTORY } from "@/components/public/LegalDoc";
import { renderWithProviders, stubFetch } from "@/test/providers";

describe("Legal pages", () => {
  afterEach(() => vi.unstubAllGlobals());

  it.each([
    ["privacy", "Privacy policy"],
    ["terms", "Terms of service"],
    ["billing-terms", "Billing terms"],
    ["cookies", "Cookies and storage"],
  ] as const)("/%s shows the draft banner until legal_reviewed is true", (doc, title) => {
    stubFetch();
    const v = renderWithProviders(<Legal doc={doc} />, { config: { legal_reviewed: false } });
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(title);
    expect(screen.getByTestId("legal-draft-banner")).toHaveTextContent("Draft — pending owner legal review");
    expect(document.title).toBe(`${title} — MarketMosaic`);
    v.unmount();
    renderWithProviders(<Legal doc={doc} />, { config: { legal_reviewed: true } });
    expect(screen.queryByTestId("legal-draft-banner")).not.toBeInTheDocument();
  });

  it("renders the markdown body as headings and lists, escaped", () => {
    stubFetch();
    renderWithProviders(<Legal doc="terms" />);
    expect(screen.getByRole("heading", { level: 2, name: "2. Not advice" })).toBeInTheDocument();
    expect(screen.getByText(/not a registered investment adviser/)).toBeInTheDocument();
  });

  it("the cookie page lists exactly what is stored: the anon id, the session id, no third-party scripts", () => {
    stubFetch();
    renderWithProviders(<Legal doc="cookies" />);
    expect(screen.getByText(/loads no third-party analytics scripts/)).toBeInTheDocument();
    expect(screen.getByText("mm_anon_id")).toBeInTheDocument();
    expect(screen.getByText("mm_session_id")).toBeInTheDocument();
    const marketing = STORAGE_INVENTORY.filter((s) => s.scope === "marketing").map((s) => s.key);
    expect(marketing).toEqual(["mm_anon_id", "mm_session_id"]);
    expect(screen.getByRole("table", { name: "Storage keys this site writes" })).toBeInTheDocument();
  });
});
