import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "@/App";
import { AuthProvider } from "@/auth/AuthProvider";
import { ConfigProvider } from "@/auth/ConfigProvider";
import { resetAccountCache } from "@/auth/useAccount";
import Fundamentals from "@/pages/Fundamentals";
import { makeCommentary, makeSeries, makeSeriesResponse, makeSpec, type PointInput } from "@/test/fixtures/fundamentals";
import { LocationSpy, SIGNED_IN, calls, ent, errJson, makeAccount, okJson, renderWithProviders, stubFetch, type Responder } from "@/test/providers";
import type { AuthContextValue } from "@/auth/AuthContext";
import type { MetricSeries, PublicConfig } from "@/types";

// The page renders the real chart engine. jsdom has no layout, so
// ResponsiveContainer measures 0×0 and recharts draws nothing — the
// `role="img"` wrapper, its label and the resolved layout mode are what
// these tests assert on. ResizeObserver is stubbed because recharts
// constructs one on mount.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

function stubMatchMedia(narrow: boolean) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn((query: string) => ({
      matches: query.includes("max-width") ? narrow : false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  );
}

const CATALOG = {
  catalog_version: "2026.09.1",
  as_of: "2026-09-08T12:00:00Z",
  metrics: [
    makeSpec("revenue", { family: "income" }),
    makeSpec("gross_margin", { family: "margins", kind: "derived", formula_text: "gross_profit / revenue" }),
    makeSpec("free_cash_flow", { family: "cash_flow", kind: "derived" }),
    makeSpec("fcf_after_sbc", { family: "cash_flow", kind: "derived" }),
    makeSpec("pe_ttm", { family: "valuation", kind: "market" }),
  ],
};

/** Hand-picked values per (ticker, metric): one gap, one estimate, one stale series. */
const VALUES: Record<string, { points: PointInput[]; stale?: boolean }> = {
  "AAPL:revenue": { points: [274.5e9, 365.8e9, 394.3e9, 383.3e9, 391.0e9] },
  "MSFT:revenue": { points: [143.0e9, 168.1e9, null, 211.9e9, 245.1e9] },
  "AAPL:gross_margin": { points: [0.382, 0.418, { v: 0.433, estimated: true }, 0.441, 0.462] },
  "MSFT:gross_margin": { points: [0.679, 0.689, 0.684, 0.688, 0.697], stale: true },
};

function seriesFor(tickers: string[], metrics: string[]): MetricSeries[] {
  const out: MetricSeries[] = [];
  for (const m of metrics) {
    for (const t of tickers) {
      const v = VALUES[`${t}:${m}`] ?? { points: [1, 2, 3, 4, 5] };
      out.push(makeSeries(t, m, v.points, { stale: v.stale }));
    }
  }
  return out;
}

/** Answers like the backend does — including its `{companies, metrics,
 *  years}` spelling of the applied limits, which the client normalises. */
function seriesResponder(over: (body: { tickers: string[]; metrics: string[]; years?: number }) => Record<string, unknown> = () => ({})): Responder {
  return (_url, init) => {
    const body = JSON.parse(String(init?.body)) as { tickers: string[]; metrics: string[]; years?: number };
    const base = makeSeriesResponse({ series: seriesFor(body.tickers, body.metrics) });
    return okJson({
      ...base,
      periods: ["FY2020", "FY2021", "FY2022", "FY2023", "FY2024"],
      warnings: [],
      fingerprint: `fp-${body.tickers.join("")}`,
      limits: { applied: { companies: 2, metrics: 2, years: 5 }, capped_by_plan: false },
      ...over(body),
    });
  };
}

const catalogRoute: [string, Responder] = ["/api/fundamentals/catalog", () => okJson(CATALOG)];

function mount(route: string, opts: { config?: Partial<PublicConfig>; auth?: AuthContextValue } = {}) {
  return renderWithProviders(<Fundamentals />, { route, path: "/app/fundamentals", ...opts });
}

const location = () => screen.getByTestId("location").textContent;

function focusOrder(): HTMLElement[] {
  return Array.from(document.querySelectorAll<HTMLElement>('input, button, a[href], [tabindex="0"]')).filter((el) => !el.hasAttribute("disabled"));
}

describe("Fundamentals page", () => {
  beforeEach(() => {
    resetAccountCache();
    vi.stubGlobal("ResizeObserver", ResizeObserverStub);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("normalises the URL after the catalog loads and requests exactly that selection", async () => {
    const mock = stubFetch([catalogRoute, ["/api/fundamentals/series", seriesResponder()]]);
    mount("/app/fundamentals?t=aapl,AAPL&m=revenue,bogus");
    await screen.findByTestId("fundamentals-chart");
    expect(location()).toBe("/app/fundamentals?t=AAPL&m=revenue");
    const series = calls(mock, "/api/fundamentals/series");
    expect(series).toHaveLength(1);
    // The default range sends no `years`; the backend picks the plan's range.
    expect(JSON.parse(String(series[0][1]?.body))).toEqual({ tickers: ["AAPL"], metrics: ["revenue"] });
    expect(screen.getByTestId("chip-AAPL")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /^Revenue/ })).toBeChecked();
    // A Free-shaped default load reports applied years=5 and NO cap: no upgrade notice.
    expect(screen.getByTestId("range-applied")).toHaveTextContent("Drawn: the last 5 fiscal years (the most this plan draws).");
    expect(screen.queryByTestId("entitlement-notice")).toBeNull();
  });

  it("shows the plan's exact limits from a 402 and keeps the selection", async () => {
    stubFetch([
      catalogRoute,
      [
        "/api/fundamentals/series",
        () =>
          errJson(402, {
            code: "plan_required",
            message: "Free Explorer draws up to 2 companies × 2 metrics × 5 years; this request asks for 3 companies × 2 metrics.",
            feature: "fundamentals_explorer",
            plan: "free",
            upgrade_url: "/pricing",
            extra: {
              limits: { max_companies: 2, max_metrics: 2, max_years: 5 },
              requested: { companies: 3, metrics: 2, years: null },
              upgrade: { plan: "pro", limits: { max_companies: 5, max_metrics: 4, max_years: null }, url: "/pricing" },
            },
          }),
      ],
    ]);
    mount("/app/fundamentals?t=AAPL,MSFT,NVDA&m=revenue,gross_margin");
    const notice = await screen.findByTestId("entitlement-notice");
    expect(notice).toHaveTextContent("2 companies × 2 metrics × 5 years");
    expect(notice).toHaveTextContent("this URL asks for 3 companies × 2 metrics");
    expect(notice).toHaveTextContent("Pro draws up to 5 companies × 4 metrics × full history");
    expect(within(notice).getByRole("link", { name: "See Pro plans" })).toHaveAttribute("href", "/pricing");
    expect(screen.queryByTestId("fundamentals-chart")).toBeNull();
    for (const t of ["AAPL", "MSFT", "NVDA"]) expect(screen.getByTestId(`chip-${t}`)).toBeInTheDocument();
    expect(location()).toBe("/app/fundamentals?t=AAPL,MSFT,NVDA&m=revenue,gross_margin");
  });

  it("shows the capped-range notice only when the URL asks for more years than the plan draws", async () => {
    stubFetch([catalogRoute, ["/api/fundamentals/series", seriesResponder((b) => ({ limits: { applied: { companies: 2, metrics: 2, years: 5 }, capped_by_plan: b.years === 10 } }))]]);
    mount("/app/fundamentals?t=AAPL&m=revenue&y=10");
    const notice = await screen.findByTestId("entitlement-notice");
    expect(notice).toHaveTextContent("This URL asks for 10 years; the plan draws up to 5 years");
    expect(screen.getByTestId("fundamentals-chart")).toBeInTheDocument();
    expect(screen.getByTestId("range-applied")).toHaveTextContent("Drawn 5 years: the most this plan draws.");
  });

  it("keeps every input on a 429 and offers a retry", async () => {
    let n = 0;
    const mock = stubFetch([
      catalogRoute,
      [
        "/api/fundamentals/series",
        (url, init) =>
          n++ === 0
            ? errJson(429, { code: "rate_limited", scope: "ip", retry_after: 0, window_seconds: 60, message: "Too many series requests." })
            : seriesResponder()(url, init),
      ],
    ]);
    mount("/app/fundamentals?t=AAPL,MSFT&m=revenue&y=10&v=table");
    await screen.findByText("Your companies, metrics and range are unchanged — retry when the timer ends.");
    expect(location()).toBe("/app/fundamentals?t=AAPL,MSFT&m=revenue&y=10&v=table");
    expect(screen.getByTestId("chip-AAPL")).toBeInTheDocument();
    expect(screen.getByTestId("chip-MSFT")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /^Revenue/ })).toBeChecked();
    expect(screen.getByRole("radio", { name: "10 years" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Table" })).toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: /Retry/ }));
    await screen.findByTestId("fundamentals-chart");
    expect(calls(mock, "/api/fundamentals/series")).toHaveLength(2);
    expect(JSON.parse(String(calls(mock, "/api/fundamentals/series")[1][1]?.body))).toEqual({ tickers: ["AAPL", "MSFT"], metrics: ["revenue"], years: 10 });
  });

  it("runs the commentary flow: status, two labelled sections, and the meter moves", async () => {
    let meCalls = 0;
    const me: Responder = () => {
      meCalls += 1;
      const used = meCalls === 1 ? 1 : 2;
      return okJson(
        makeAccount({
          entitlements: { ...makeAccount().entitlements, chart_commentary: ent("chart_commentary", { limit: 5, used, remaining: 5 - used, metered: true }) },
        }),
      );
    };
    const mock = stubFetch([
      catalogRoute,
      ["/api/fundamentals/series", seriesResponder()],
      ["/api/fundamentals/commentary", () => okJson(makeCommentary())],
      ["/api/me", me],
    ]);
    mount("/app/fundamentals?t=AAPL,MSFT&m=revenue,gross_margin", { config: { auth_enabled: true }, auth: SIGNED_IN });
    await screen.findByTestId("fundamentals-chart");
    expect(await screen.findByTestId("meter-chart_commentary")).toHaveTextContent("1 of 5 used");

    const button = screen.getByRole("button", { name: "Explain this chart" });
    expect(button).toBeEnabled();
    fireEvent.click(button);
    const status = screen.getByTestId("commentary-status");
    expect(status).toHaveAttribute("aria-live", "polite");
    expect(status).toHaveTextContent("Generating commentary…");

    const observed = await screen.findByRole("heading", { level: 3, name: "Observed in the data" });
    const memo = screen.getByRole("heading", { level: 3, name: "From stored memos" });
    expect(observed.closest("section")).not.toBe(memo.closest("section"));
    expect(observed.closest("section")).toHaveAttribute("aria-labelledby", observed.id);
    expect(memo.closest("section")).toHaveAttribute("aria-labelledby", memo.id);
    expect(screen.getByText("AAPL revenue grew from $274.5B (FY2020) to $391.0B (FY2024).")).toBeInTheDocument();
    expect(screen.getByText("The stored memo sees services mix as the margin lever.")).toBeInTheDocument();
    expect(status).toHaveTextContent("Commentary ready.");
    expect(screen.queryByTestId("commentary-degraded")).toBeNull();

    // The request is exactly the displayed chart: selection order, the years
    // the server applied, and the series fingerprint.
    const [, init] = calls(mock, "/api/fundamentals/commentary")[0];
    expect(JSON.parse(String(init?.body))).toEqual({ tickers: ["AAPL", "MSFT"], metrics: ["revenue", "gross_margin"], years: 5, fingerprint: "fp-AAPLMSFT" });
    await waitFor(() => expect(screen.getByTestId("meter-chart_commentary")).toHaveTextContent("2 of 5 used"));
  });

  it("renders the degraded banner with the reason and still shows the observed section", async () => {
    stubFetch([
      catalogRoute,
      ["/api/fundamentals/series", seriesResponder()],
      [
        "/api/fundamentals/commentary",
        () => okJson(makeCommentary({ commentary_id: null, degraded: true, degraded_reason: "commentary requires an account", memo_view: [], model: null })),
      ],
    ]);
    mount("/app/fundamentals?t=AAPL&m=revenue");
    await screen.findByTestId("fundamentals-chart");
    fireEvent.click(screen.getByRole("button", { name: "Explain this chart" }));
    const banner = await screen.findByTestId("commentary-degraded");
    expect(banner).toHaveTextContent("AI commentary unavailable: commentary requires an account. Nothing was charged.");
    expect(screen.getByRole("heading", { level: 3, name: "Observed in the data" })).toBeInTheDocument();
    expect(screen.getByText("No stored memo excerpts for the selected companies.")).toBeInTheDocument();
    expect(screen.getByTestId("commentary-status")).toHaveTextContent("Commentary unavailable");
  });

  it("disables commentary with the reason when the allowance is used up", async () => {
    stubFetch([
      catalogRoute,
      ["/api/fundamentals/series", seriesResponder()],
      ["/api/me", () => okJson(makeAccount({ entitlements: { ...makeAccount().entitlements, chart_commentary: ent("chart_commentary", { limit: 5, used: 5, remaining: 0, metered: true }) } }))],
    ]);
    mount("/app/fundamentals?t=AAPL&m=revenue", { config: { auth_enabled: true }, auth: SIGNED_IN });
    await screen.findByTestId("fundamentals-chart");
    await screen.findByTestId("meter-chart_commentary");
    expect(screen.getByRole("button", { name: "Explain this chart" })).toBeDisabled();
    expect(screen.getByTestId("commentary-reason")).toHaveTextContent("You've used 5 of 5 chart commentaries this month");
  });

  it("shows the research CTA for a company with no stored history", async () => {
    stubFetch([
      catalogRoute,
      [
        "/api/fundamentals/series",
        seriesResponder(() => ({
          series: [],
          unavailable: [{ ticker: "NEWCO", reason: "not_backfilled", remedy: "No financial history is stored for this company yet. Run research on it — the memo job backfills its statement history." }],
        })),
      ],
    ]);
    mount("/app/fundamentals?t=NEWCO&m=revenue");
    const row = await screen.findByTestId("unavailable-NEWCO");
    expect(row).toHaveTextContent("NEWCO — history not loaded.");
    expect(row).toHaveTextContent("the memo job backfills its statement history");
    expect(within(row).getByRole("link", { name: "Run research on NEWCO" })).toHaveAttribute("href", "/app/research?ticker=NEWCO");
    expect(screen.queryByTestId("fundamentals-chart")).toBeNull();
    expect(screen.getByRole("button", { name: "Explain this chart" })).toBeDisabled();
  });

  it("shows the research CTA, not a Retry, when every ticker is unknown to the platform (404 unknown_tickers)", async () => {
    stubFetch([
      catalogRoute,
      [
        "/api/fundamentals/series",
        () =>
          errJson(404, {
            code: "unknown_tickers",
            message: "None of the requested tickers is known to the platform: NEWCO. No financial history is stored for this company yet. Run research on it — the memo job backfills its statement history.",
            feature: "fundamentals_explorer",
            extra: { tickers: ["NEWCO"] },
          }),
      ],
    ]);
    mount("/app/fundamentals?t=NEWCO&m=revenue");
    const row = await screen.findByTestId("unavailable-NEWCO");
    expect(row).toHaveTextContent("NEWCO — history not loaded.");
    expect(row).toHaveTextContent("Run research on it — the memo job adds the company and backfills its statement history.");
    expect(within(row).getByRole("link", { name: "Run research on NEWCO" })).toHaveAttribute("href", "/app/research?ticker=NEWCO");
    // The remedy is a research run: retrying this request would 404 again.
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
    expect(screen.queryByTestId("fundamentals-chart")).toBeNull();
    // Inputs stay so the reader can change the selection instead.
    expect(screen.getByTestId("chip-NEWCO")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /^Revenue/ })).toBeChecked();
  });

  it("falls back to the requested tickers when the 404 echoes none, and clears the CTA once a known company is added", async () => {
    const mock = stubFetch([
      catalogRoute,
      [/\/api\/stocks$/, () => okJson([{ ticker: "AAPL", company_name: "Apple", sector: "Tech", industry: null, universe_tier: "auto_analysis" }])],
      [
        "/api/fundamentals/series",
        (url, init) => {
          const body = JSON.parse(String(init?.body)) as { tickers: string[] };
          if (body.tickers.every((t) => t === "NEWCO" || t === "ZZZZ")) {
            return errJson(404, { code: "unknown_tickers", message: "None of the requested tickers is known to the platform: NEWCO, ZZZZ.", feature: "fundamentals_explorer" });
          }
          return seriesResponder(() => ({
            series: seriesFor(["AAPL"], body.tickers.length ? ["revenue"] : []),
            unavailable: [{ ticker: "NEWCO", reason: "not_backfilled" }, { ticker: "ZZZZ", reason: "not_backfilled" }],
          }))(url, init);
        },
      ],
    ]);
    mount("/app/fundamentals?t=NEWCO,ZZZZ&m=revenue");
    // No `extra.tickers` on the wire: every requested ticker gets its own CTA.
    const first = await screen.findByTestId("unavailable-NEWCO");
    expect(within(first).getByRole("link", { name: "Run research on NEWCO" })).toBeInTheDocument();
    expect(within(screen.getByTestId("unavailable-ZZZZ")).getByRole("link", { name: "Run research on ZZZZ" })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();

    const input = await screen.findByPlaceholderText("Add a company…");
    fireEvent.change(input, { target: { value: "AAPL" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(location()).toBe("/app/fundamentals?t=NEWCO,ZZZZ,AAPL&m=revenue"));
    await screen.findByTestId("fundamentals-chart");
    expect(calls(mock, "/api/fundamentals/series")).toHaveLength(2);
    // The server now reports the two unknowns itself; the 404-derived rows are gone, not doubled.
    expect(screen.getAllByTestId("unavailable-NEWCO")).toHaveLength(1);
    expect(screen.getAllByTestId("unavailable-ZZZZ")).toHaveLength(1);
    expect(screen.queryByText(/not on the platform yet/)).toBeNull();
  });

  it("labels stale and estimated series in the notice and the legend", async () => {
    stubFetch([catalogRoute, ["/api/fundamentals/series", seriesResponder()]]);
    mount("/app/fundamentals?t=AAPL,MSFT&m=revenue,gross_margin");
    await screen.findByTestId("fundamentals-chart");
    const stale = screen.getAllByTestId("stale-badge");
    expect(stale).toHaveLength(1);
    expect(stale[0]).toHaveTextContent("MSFT Gross margin: last fiscal period FY2022 is older than 15 months");
    const estimated = screen.getAllByTestId("estimated-badge");
    expect(estimated).toHaveLength(1);
    expect(estimated[0]).toHaveTextContent("AAPL Gross margin: 1 value computed with a documented fallback");
    const legend = screen.getByRole("list", { name: "Series" });
    expect(within(legend).getByText("1 estimated")).toBeInTheDocument();
    expect(within(legend).getByText("stale")).toBeInTheDocument();
    // The screen-reader change sentence for a percent series reads in points.
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain("AAPL Gross margin rose 8.0 points from 38.2% (FY2020) to 46.2% (FY2024)");
  });

  it("offers the table even when every series is all-missing, so reasons stay visible", async () => {
    stubFetch([
      catalogRoute,
      ["/api/fundamentals/series", seriesResponder(() => ({ series: [makeSeries("AAPL", "pe_ttm", Array(5).fill({ v: null, reason: "no_price" }))] }))],
    ]);
    mount("/app/fundamentals?t=AAPL&m=pe_ttm");
    await screen.findByTestId("fundamentals-chart");
    expect(screen.getByTestId("chart-empty")).toHaveTextContent("No observed points to draw");
    fireEvent.click(screen.getByRole("button", { name: "View as table" }));
    expect(within(screen.getByRole("table")).getAllByText("n/a (no price at period end)")).toHaveLength(5);
  });

  it("reaches the controls, chart and table toggle in order from the keyboard", async () => {
    // A Pro-shaped limit keeps the picker on screen (at the plan's maximum
    // it gives way to the "remove one to add another" note).
    stubFetch([catalogRoute, ["/api/fundamentals/series", seriesResponder(() => ({ limits: { applied: { companies: 5, metrics: 4, years: null }, capped_by_plan: false } }))]]);
    mount("/app/fundamentals?t=AAPL,MSFT&m=revenue,gross_margin");
    await screen.findByTestId("fundamentals-chart");
    const order = focusOrder();
    const at = (el: HTMLElement) => order.indexOf(el);
    const company = await screen.findByLabelText("Add a company");
    const metric = screen.getAllByRole("checkbox")[0];
    const range = within(screen.getByTestId("time-range")).getAllByRole("radio")[0];
    const view = within(screen.getByTestId("view-mode")).getAllByRole("radio")[0];
    const chart = screen.getByRole("img");
    const toggle = screen.getByRole("button", { name: "View as table" });
    for (const el of [company, metric, range, view, chart, toggle]) expect(at(el)).toBeGreaterThanOrEqual(0);
    expect(at(company)).toBeLessThan(at(metric));
    expect(at(metric)).toBeLessThan(at(range));
    expect(at(range)).toBeLessThan(at(view));
    expect(at(view)).toBeLessThan(at(chart));
    expect(at(view)).toBeLessThan(at(toggle));
    expect(chart).toHaveAttribute("tabindex", "0");
    // Chips are removable from the keyboard.
    fireEvent.keyDown(screen.getByRole("button", { name: "Remove MSFT" }), { key: "Backspace" });
    await waitFor(() => expect(location()).toBe("/app/fundamentals?t=AAPL&m=revenue,gross_margin"));
  });

  it("collapses dual axis to small multiples on a narrow viewport", async () => {
    stubFetch([catalogRoute, ["/api/fundamentals/series", seriesResponder()]]);
    const wide = mount("/app/fundamentals?t=AAPL,MSFT&m=revenue,gross_margin");
    expect(await screen.findByTestId("fundamentals-chart")).toHaveAttribute("data-resolved-mode", "dual-axis");
    wide.unmount();

    stubMatchMedia(true);
    mount("/app/fundamentals?t=AAPL,MSFT&m=revenue,gross_margin");
    expect(await screen.findByTestId("fundamentals-chart")).toHaveAttribute("data-resolved-mode", "small-multiples");
    const dual = screen.getByRole("radio", { name: /Dual axis/ });
    expect(dual).toBeDisabled();
    expect(dual.closest("label")).toHaveTextContent("dual axis is unreadable on narrow screens");
  });

  it("switches the view through the URL and disables modes the layout forbids", async () => {
    stubFetch([catalogRoute, ["/api/fundamentals/series", seriesResponder()]]);
    mount("/app/fundamentals?t=AAPL,MSFT&m=revenue,gross_margin");
    await screen.findByTestId("fundamentals-chart");
    // Indexing a percent series is misleading: the engine forbids it.
    const indexed = screen.getByRole("radio", { name: /Indexed/ });
    expect(indexed).toBeDisabled();
    expect(indexed.closest("label")).toHaveTextContent("indexing a percent series is misleading");
    fireEvent.click(screen.getByRole("radio", { name: "Small multiples" }));
    await waitFor(() => expect(location()).toBe("/app/fundamentals?t=AAPL,MSFT&m=revenue,gross_margin&v=small-multiples"));
    expect(screen.getByTestId("fundamentals-chart")).toHaveAttribute("data-resolved-mode", "small-multiples");
    fireEvent.click(screen.getByRole("radio", { name: "Table" }));
    expect(await screen.findByRole("table")).toBeInTheDocument();
  });

  it("guides an empty page with example charts and adds a company through the picker", async () => {
    stubFetch([catalogRoute, ["/api/fundamentals/series", seriesResponder()], [/\/api\/stocks$/, () => okJson([{ ticker: "AAPL", company_name: "Apple", sector: "Tech", industry: null, universe_tier: "auto_analysis" }])]]);
    mount("/app/fundamentals", { config: { sample_tickers: ["NVDA", "AMD"] } });
    const guide = await screen.findByTestId("fundamentals-empty");
    expect(within(guide).getByRole("link", { name: /Revenue and gross margin — NVDA vs AMD/ })).toHaveAttribute("href", "/app/fundamentals?t=NVDA,AMD&m=revenue,gross_margin");
    const input = await screen.findByPlaceholderText("Add a company…");
    fireEvent.change(input, { target: { value: "AAPL" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(location()).toBe("/app/fundamentals?t=AAPL"));
    expect(screen.getByTestId("fundamentals-no-metrics")).toHaveTextContent("Pick at least one metric to draw AAPL.");
    fireEvent.click(screen.getByRole("checkbox", { name: /^Revenue/ }));
    await waitFor(() => expect(location()).toBe("/app/fundamentals?t=AAPL&m=revenue"));
    await screen.findByTestId("fundamentals-chart");
  });
});

describe("Fundamentals routing", () => {
  beforeEach(() => {
    resetAccountCache();
    vi.stubGlobal("ResizeObserver", ResizeObserverStub);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("redirects the legacy /fundamentals path and lists the page in the nav", async () => {
    stubFetch([catalogRoute, ["/api/fundamentals/series", seriesResponder()]]);
    render(
      <MemoryRouter initialEntries={["/fundamentals?t=AAPL&m=revenue"]}>
        <ConfigProvider initial={{ auth_enabled: false }}>
          <AuthProvider>
            <LocationSpy />
            <App />
          </AuthProvider>
        </ConfigProvider>
      </MemoryRouter>,
    );
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(location()).toBe("/app/fundamentals?t=AAPL&m=revenue");
    expect(screen.getByRole("link", { name: "Fundamentals" })).toHaveAttribute("href", "/app/fundamentals");
    expect(screen.getByRole("heading", { level: 1, name: "Fundamentals" })).toBeInTheDocument();
  });
});
