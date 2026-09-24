import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "@/App";
import { AuthProvider } from "@/auth/AuthProvider";
import { ConfigProvider } from "@/auth/ConfigProvider";
import FAQPage from "@/pages/public/FAQPage";
import Methodology from "@/pages/public/Methodology";
import { publicRoutes } from "@/pages/public";
import { FAQ } from "@/content/faq";
import { resetAnalyticsForTests } from "@/lib/analytics";
import { renderWithProviders, stubFetch } from "@/test/providers";
import { sampleRoutes } from "@/test/fixtures/sample";

vi.mock("recharts", () => {
  const Noop = () => null;
  return { Bar: Noop, BarChart: Noop, CartesianGrid: Noop, Line: Noop, LineChart: Noop, ResponsiveContainer: Noop, Tooltip: Noop, XAxis: Noop, YAxis: Noop };
});

describe("publicRoutes", () => {
  it("exports every public path App.tsx mounts, and nothing under /app", () => {
    const paths = publicRoutes.map((r) => r.path).sort();
    expect(paths).toEqual(
      ["/", "/billing-terms", "/cookies", "/faq", "/methodology", "/pricing", "/privacy", "/samples", "/samples/:ticker", "/sign-in/*", "/sign-up/*", "/terms"].sort(),
    );
    expect(paths.some((p) => p?.startsWith("/app"))).toBe(false);
  });
});

describe("Methodology", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("states the four questions, two mandates, four ledger columns, observed vs interpretation and the blank rule", () => {
    stubFetch();
    renderWithProviders(<Methodology />);
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    const questions = screen.getByRole("heading", { name: "Four questions" }).closest("section")!;
    for (const q of ["What will happen?", "Who captures the economic benefit?", "What is already priced in?", "What evidence will reveal the gap?"]) {
      expect(within(questions).getByRole("heading", { name: q })).toBeInTheDocument();
    }
    expect(screen.getByRole("heading", { name: "Two mandates, kept apart" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Long-term compounders" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Mispriced inflections" })).toBeInTheDocument();
    const ledger = screen.getByRole("heading", { name: "The expectations ledger" }).closest("section")!;
    for (const c of ["Reported consensus", "Management guidance", "Price-implied", "Our forecast"]) {
      expect(within(ledger).getByText(c)).toBeInTheDocument();
    }
    expect(screen.getByRole("heading", { name: "Observed data versus interpretation" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Blank means not obtained, not zero" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "See the ledger on a real sample" })).toHaveAttribute("href", "/samples");
  });

  it("describes industry coverage in our own words, never the licensed classification's brand", () => {
    // Owner decision 2026-09-24: no third-party classification branding on
    // any public page. Letters on neither side, so "biologics" is not it.
    stubFetch();
    renderWithProviders(<Methodology />);
    expect(document.body.textContent).not.toMatch(/(?<![A-Za-z])gics(?![A-Za-z])/i);
  });
});

describe("FAQ page", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("renders every entry as a keyboard-operable disclosure with an anchor id", () => {
    stubFetch();
    renderWithProviders(<FAQPage />);
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    for (const e of FAQ) {
      const details = document.getElementById(e.id) as HTMLDetailsElement;
      expect(details).not.toBeNull();
      expect(details.tagName).toBe("DETAILS");
      expect(within(details).getByText(e.question)).toBeInTheDocument();
    }
    const first = document.getElementById(FAQ[0].id) as HTMLDetailsElement;
    expect(first.open).toBe(false);
    fireEvent.click(within(first).getByText(FAQ[0].question));
    expect(first.open).toBe(true);
  });
});

describe("App integration", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    resetAnalyticsForTests();
  });

  const mountApp = (route: string, auth_enabled: boolean) =>
    render(
      <MemoryRouter initialEntries={[route]}>
        <ConfigProvider initial={{ auth_enabled }}>
          <AuthProvider>
            <App />
          </AuthProvider>
        </ConfigProvider>
      </MemoryRouter>,
    );

  it.each(["/pricing", "/methodology", "/faq", "/privacy", "/terms", "/billing-terms", "/cookies", "/samples"])(
    "mounts %s through App outside the /app shell, signed out with the wall on",
    async (route) => {
      vi.stubEnv("VITE_AUTH_STUB", "signed_out");
      stubFetch(sampleRoutes());
      mountApp(route, true);
      expect(await screen.findByRole("main")).toHaveAttribute("id", "main");
      expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
      expect(screen.queryByRole("button", { name: "Account menu" })).not.toBeInTheDocument();
      expect(screen.queryByTestId("stub-sign-in")).not.toBeInTheDocument();
    },
  );

  it("mounts the landing page at / and the sign-up page at /sign-up", async () => {
    vi.stubEnv("VITE_AUTH_STUB", "signed_out");
    stubFetch(sampleRoutes());
    const v = mountApp("/", true);
    expect(await screen.findByRole("heading", { level: 1 })).toHaveTextContent("Research that shows its work.");
    v.unmount();
    stubFetch(sampleRoutes());
    mountApp("/sign-up?returnTo=%2Fapp", true);
    expect(await screen.findByTestId("stub-sign-in")).toBeInTheDocument();
  });
});
