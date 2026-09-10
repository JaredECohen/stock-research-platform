import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import Scorecard from "@/pages/Scorecard";
import { normaliseEvaluationResponse, scorecardExportUrl } from "@/api/client";
import { resetAccountCache } from "@/auth/useAccount";
import {
  ALL_CAVEATS,
  AS_OF,
  VERSION,
  makeDetail,
  makeFF6Result,
  makeHistory,
  makeLassoResult,
  makeQuintileResult,
  makeSpec,
  makeUniverse,
} from "@/test/fixtures/scorecard";
import { errJson, okJson, renderWithProviders, stubFetch, type Responder } from "@/test/providers";

// The page renders the real recharts components inside ResponsiveContainer;
// jsdom has no layout, so the chart draws nothing and the tests assert on
// the `role="img"` wrapper. ResizeObserver and matchMedia are stubbed
// because recharts and the reduced-motion hook construct them on mount.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

function stubMatchMedia() {
  vi.stubGlobal(
    "matchMedia",
    vi.fn((query: string) => ({
      matches: false,
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

// ---------------------------------------------------------------------------
// Wire fixtures shaped as the backend serialises them (schemas/scorecard.py),
// i.e. with the bookkeeping the client mirror does not carry.
// ---------------------------------------------------------------------------

const UNIVERSE_WIRE = { ...makeUniverse(5), spec_hash: "9f1c2e7a", is_month_end: true, scored: 7, insufficient: 1, sort_by: "overall_score", order: "desc", stale: false };

const SPEC_WIRE = { ...makeSpec(), source: "registry", score_scale: "0-100, 50 = z of 0 (the sector or universe-fallback mean); percentiles are rank-based" };

const DETAIL_WIRE = {
  ...makeDetail(),
  company_name: "Costco Wholesale",
  is_month_end: true,
  spec_hash: "9f1c2e7a",
  inputs_hash: "abc123",
  history: makeHistory().points.map((p) => ({ ...p, is_month_end: true, overall_z: null })),
  context: { price_stale: false },
  notes: ["price context: cached 252-day series (unadjusted FMP closes)"],
};

const PRICE_DEPTH = "the price store holds 32 month-ends; the sample cannot be deeper than that";
const DEFERRED = { book_to_market: "book equity per share is not stored for the universe; deferred to fs-v2" };
const NOTE = "double_lasso: insufficient — 14 months in panel, 30 required";

/** `ScorecardEvaluationOut`: evaluations keyed by kind, caveats at the top
 *  level, the quintile bucket table under the backend's names. */
function evaluationWire(params: Record<string, unknown> = { min_leg: 15, min_months: 30, min_obs: 2500, price_store_depth: PRICE_DEPTH }) {
  const q = makeQuintileResult();
  const { quintile_table, caveats: _qc, ...qRest } = q;
  const quantile_table = quintile_table.map((b) => ({ q: b.q, mean_ret: b.mean_ret, n_months: b.n }));
  const { caveats: _fc, ...ffRest } = makeFF6Result();
  const { caveats: _lc, ...lassoRest } = makeLassoResult();
  const base = { run_id: "run_eval_1", created_at: "2026-09-01T04:05:00", sample_start: "2024-02-29", sample_end: "2026-07-31" };
  return {
    version_key: VERSION,
    evaluations: {
      quintile_ls: { ...base, kind: "quintile_ls", n_obs: 30, params, result: { ...qRest, quantile_table, caveats: ALL_CAVEATS } },
      ff6_regression: { ...base, kind: "ff6_regression", n_obs: 30, params, result: { ...ffRest, reasons: [] } },
      double_lasso: { ...base, kind: "double_lasso", n_obs: 4680, params: { ...params, controls_deferred: DEFERRED }, result: { ...lassoRest, reasons: [] } },
    },
    caveats: ALL_CAVEATS,
    note: NOTE,
  };
}

const universeRoute: [RegExp, Responder] = [/\/api\/scorecard(\?|$)/, () => okJson(UNIVERSE_WIRE)];
const specRoute: [string, Responder] = ["/api/scorecard/spec", () => okJson(SPEC_WIRE)];
const detailRoute: [string, Responder] = ["/api/scorecard/COST", () => okJson(DETAIL_WIRE)];
const evaluationRoute: [string, Responder] = ["/api/scorecard/evaluation", () => okJson(evaluationWire())];

function mount(route = "/app/scorecard") {
  return renderWithProviders(<Scorecard />, { route, path: "/app/scorecard" });
}

const location = () => screen.getByTestId("location").textContent;

function bodyRows() {
  const table = screen.getByTestId("universe-table");
  return within(table.querySelector("tbody") as HTMLElement).getAllByRole("row");
}

function tickers() {
  return bodyRows().map((r) => r.getAttribute("data-ticker"));
}

describe("Scorecard page", () => {
  beforeEach(() => {
    localStorage.clear();
    resetAccountCache();
    vi.stubGlobal("ResizeObserver", ResizeObserverStub);
    stubMatchMedia();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("tabs and URL persistence", () => {
    it("opens on the Universe tab, writes the chosen tab to the URL and remembers it for a bare visit", async () => {
      stubFetch([specRoute, evaluationRoute, universeRoute]);
      const view = mount();
      await screen.findByTestId("universe-table");
      expect(screen.getByTestId("tab-universe")).toHaveAttribute("aria-selected", "true");
      expect(location()).toBe("/app/scorecard");

      fireEvent.click(screen.getByTestId("tab-evaluation"));
      expect(location()).toBe("/app/scorecard?tab=evaluation");
      expect(screen.getByTestId("tab-evaluation")).toHaveAttribute("aria-selected", "true");
      await screen.findByTestId("scorecard-evaluation");
      view.unmount();

      // A bare URL restores the last tab from localStorage. Await the
      // evaluation so its chart mounts before the globals are unstubbed.
      stubFetch([specRoute, evaluationRoute, universeRoute]);
      mount();
      expect(screen.getByTestId("tab-evaluation")).toHaveAttribute("aria-selected", "true");
      await screen.findByTestId("scorecard-evaluation");
    });

    it("restores the tab and ticker from the URL", async () => {
      stubFetch([specRoute, detailRoute, universeRoute]);
      mount("/app/scorecard?tab=ticker&ticker=cost");
      expect(screen.getByTestId("tab-ticker")).toHaveAttribute("aria-selected", "true");
      await screen.findByTestId("scorecard-panel");
      expect(screen.getByTestId("tab-universe")).toHaveAttribute("aria-selected", "false");
    });

    it("ignores an unknown tab value and falls back to Universe", async () => {
      stubFetch([specRoute, universeRoute]);
      mount("/app/scorecard?tab=bogus");
      await screen.findByTestId("universe-table");
      expect(screen.getByTestId("tab-universe")).toHaveAttribute("aria-selected", "true");
    });
  });

  describe("universe tab", () => {
    it("loads the universe once with the route ceiling and never sends an unlisted sort key", async () => {
      const mock = stubFetch([specRoute, universeRoute]);
      mount();
      await screen.findByTestId("universe-table");
      const calls = mock.mock.calls.map(([u]) => String(u)).filter((u) => /\/api\/scorecard(\?|$)/.test(u));
      expect(calls).toHaveLength(1);
      expect(calls[0]).toBe("/api/scorecard?limit=600");
      expect(screen.getByTestId("universe-run")).toHaveTextContent("run_2026-08-31_fs-v1");
      expect(screen.getByTestId("universe-run")).toHaveTextContent("7 scored · 1 insufficient");
      expect(screen.getByTestId("spec-label")).toHaveTextContent("Spec fs-v1 · 0-100, 50 = z of 0 (the sector or universe-fallback mean); percentiles are rank-based");
    });

    it("sorts and filters the mocked universe client-side, keeping unscored names last", async () => {
      stubFetch([specRoute, universeRoute]);
      mount();
      await screen.findByTestId("universe-table");
      expect(bodyRows()).toHaveLength(8);
      // Rank ascending by default; the unranked name sorts last.
      expect(tickers()[0]).toBe("COST");
      expect(tickers()[7]).toBe("NEWCO");

      const scoreHeader = screen.getByRole("columnheader", { name: /^Score/ });
      fireEvent.click(within(scoreHeader).getByRole("button"));
      expect(scoreHeader).toHaveAttribute("aria-sort", "descending");
      const expected = [...UNIVERSE_WIRE.rows]
        .filter((r) => typeof r.overall_score === "number")
        .sort((a, b) => (b.overall_score as number) - (a.overall_score as number))
        .map((r) => r.ticker);
      expect(tickers().slice(0, expected.length)).toEqual(expected);
      expect(tickers()[7]).toBe("NEWCO");

      fireEvent.change(screen.getByTestId("sector-filter"), { target: { value: "Financial Services" } });
      const financials = UNIVERSE_WIRE.rows.filter((r) => r.sector === "Financial Services").length;
      expect(bodyRows()).toHaveLength(financials);
      expect(tickers()).toContain("JPM");
      expect(screen.getByTestId("universe-count")).toHaveTextContent(`${financials} of 8 names · fs-v1 · as of 2026-08-31`);
    });

    it("says n/a (not scored) for a masked family and names both causes in the tooltip", async () => {
      stubFetch([specRoute, universeRoute]);
      mount();
      await screen.findByTestId("universe-table");
      const jpm = screen.getByTestId("row-JPM");
      const leverage = jpm.querySelector('[data-family="leverage"]') as HTMLElement;
      expect(leverage).toHaveTextContent("n/a (not scored)");
      const title = (leverage.querySelector("[title]") as HTMLElement).getAttribute("title") ?? "";
      expect(title).toMatch(/masked for its sector/);
      expect(title).toMatch(/fewer than half of its inputs/);
    });

    it("links the CSV and JSON exports for the displayed run under contract v1 with no token in the URL", async () => {
      stubFetch([specRoute, universeRoute]);
      mount();
      await screen.findByTestId("universe-table");
      const csv = screen.getByTestId("export-link").getAttribute("href") ?? "";
      expect(csv).toBe(scorecardExportUrl({ format: "csv", version: VERSION, as_of: AS_OF }));
      expect(csv).toBe("/api/scorecard/export?format=csv&contract=v1&version=fs-v1&as_of=2026-08-31");
      expect(csv).toContain("contract=v1");
      expect(csv).not.toMatch(/token|bearer|authorization/i);
      const json = screen.getByTestId("export-json-link").getAttribute("href") ?? "";
      expect(json).toContain("format=json");
      expect(json).toContain("contract=v1");
      expect(json).not.toMatch(/token|bearer|authorization/i);
    });

    it("opens a ticker from a row, writing tab and ticker to the URL", async () => {
      stubFetch([specRoute, detailRoute, universeRoute]);
      mount();
      await screen.findByTestId("universe-table");
      fireEvent.click(within(screen.getByTestId("row-COST")).getByRole("button", { name: "COST" }));
      expect(location()).toBe("/app/scorecard?tab=ticker&ticker=COST");
      await screen.findByTestId("scorecard-panel");
    });

    it("shows the empty universe state on a plain 404 (no run yet)", async () => {
      stubFetch([specRoute, [/\/api\/scorecard(\?|$)/, () => errJson(404, "no scorecard run on file (version active)")]]);
      mount();
      await screen.findByTestId("universe-empty");
      expect(screen.getByTestId("universe-empty")).toHaveTextContent("n/a (no scorecard run yet)");
      expect(screen.queryByTestId("export-link")).toBeNull();
    });

    it("renders the methodology from /api/scorecard/spec", async () => {
      stubFetch([specRoute, universeRoute]);
      mount();
      await screen.findByTestId("universe-table");
      const spec = await screen.findByTestId("spec");
      expect(spec).toHaveTextContent("spec 9f1c2e7a · served from registry");
      expect(within(spec).getByTestId("spec-score-scale")).toHaveTextContent(SPEC_WIRE.score_scale);
      expect(spec).toHaveTextContent("fcf_yield");
      expect(spec).toHaveTextContent("Financial Services, Real Estate");
    });
  });

  describe("ticker tab", () => {
    it("shows the observed/model-read panel, the worker notes verbatim and a history chart built from the detail row", async () => {
      stubFetch([specRoute, detailRoute, universeRoute]);
      mount("/app/scorecard?tab=ticker&ticker=COST");
      await screen.findByTestId("scorecard-panel");
      expect(screen.getByTestId("observed-header")).toHaveTextContent("Observed");
      expect(screen.getByTestId("model-read-header")).toHaveTextContent("Model read (fs-v1)");
      expect(screen.getByTestId("feature-table")).toBeInTheDocument();
      expect(screen.getByTestId("ticker-notes")).toHaveTextContent("price context: cached 252-day series (unadjusted FMP closes)");
      const img = screen.getByRole("img");
      expect(img.getAttribute("aria-label")).toMatch(/^COST universe percentile \(fs-v1\), 2023-09-30 to 2026-08-31, 36 month-ends, 2 missing points/);

      fireEvent.change(screen.getByTestId("history-metric"), { target: { value: "sector_percentile" } });
      expect(screen.getByRole("img").getAttribute("aria-label")).toMatch(/^COST sector percentile \(fs-v1\)/);
    });

    it("asks for a ticker when none is in the URL", async () => {
      stubFetch([specRoute, universeRoute]);
      mount("/app/scorecard?tab=ticker");
      expect(screen.getByTestId("ticker-empty")).toBeInTheDocument();
      expect(screen.queryByTestId("scorecard-panel")).toBeNull();
    });

    it("says n/a with the reason when no run has scored the ticker", async () => {
      stubFetch([specRoute, ["/api/scorecard/ZZZZ", () => errJson(404, "no scorecard for ZZZZ (version active)")], universeRoute]);
      mount("/app/scorecard?tab=ticker&ticker=ZZZZ");
      await screen.findByTestId("ticker-missing");
      expect(screen.getByTestId("ticker-missing")).toHaveTextContent("n/a (no scorecard for ZZZZ)");
      expect(screen.getByRole("link", { name: "Open research" })).toHaveAttribute("href", "/app/research?ticker=ZZZZ");
    });
  });

  describe("evaluation tab", () => {
    it("renders the caveats verbatim on every card, the LASSO verdict, the note and the minimums from params", async () => {
      stubFetch([specRoute, evaluationRoute, universeRoute]);
      mount("/app/scorecard?tab=evaluation");
      await screen.findByTestId("scorecard-evaluation");

      // Top-level caveats and the per-card lists all carry the backend's exact sentences.
      const top = within(screen.getByTestId("evaluation-caveats")).getAllByRole("listitem");
      expect(top.map((li) => li.textContent)).toEqual(ALL_CAVEATS);
      for (const kind of ["quintile_ls", "ff6_regression", "double_lasso"]) {
        const items = within(screen.getByTestId(`caveats-${kind}`)).getAllByRole("listitem");
        expect(items.map((li) => li.textContent)).toEqual(ALL_CAVEATS);
      }
      expect(screen.getByTestId("evaluation-note")).toHaveTextContent(NOTE);

      const lasso = screen.getByTestId("eval-double_lasso");
      expect(lasso).toHaveAttribute("data-verdict", "independent");
      expect(within(lasso).getByTestId("lasso-verdict")).toHaveTextContent("Independent information");

      // The quintile table survives the quantile_table → quintile_table rename.
      const qt = within(screen.getByTestId("eval-quintile_ls")).getByTestId("quintile-table");
      expect(within(qt.querySelector("tbody") as HTMLElement).getAllByRole("row")).toHaveLength(5);

      // Minimums come from params (30 months / 2,500 obs are NOT the client rules).
      const mins = screen.getByTestId("evaluation-minimums");
      expect(within(mins).getByTestId("minimum-min_leg")).toHaveTextContent("15 names");
      expect(within(mins).getByTestId("minimum-min_months")).toHaveTextContent("30 months");
      expect(within(mins).getByTestId("minimum-min_obs")).toHaveTextContent("2,500 observations");
      expect(within(mins).getByTestId("minimum-min_obs")).toHaveAttribute("data-source", "params");
      expect(mins).toHaveTextContent("recorded by the worker");
      expect(within(mins).getByTestId("price-store-depth")).toHaveTextContent(PRICE_DEPTH);
      expect(within(mins).getByTestId("controls-deferred")).toHaveTextContent("book_to_market — book equity per share is not stored for the universe; deferred to fs-v2");
    });

    it("falls back to the documented client rules only when the row lacks the minimums, and says so", async () => {
      stubFetch([specRoute, ["/api/scorecard/evaluation", () => okJson(evaluationWire({}))], universeRoute]);
      mount("/app/scorecard?tab=evaluation");
      await screen.findByTestId("scorecard-evaluation");
      const mins = screen.getByTestId("evaluation-minimums");
      expect(within(mins).getByTestId("minimum-min_leg")).toHaveTextContent("15 names");
      expect(within(mins).getByTestId("minimum-min_months")).toHaveTextContent("24 months");
      expect(within(mins).getByTestId("minimum-min_obs")).toHaveTextContent("2,000 observations");
      expect(within(mins).getByTestId("minimum-min_months")).toHaveAttribute("data-source", "client-rule");
      expect(mins).toHaveTextContent("fs-v1 client rule");
      expect(screen.queryByTestId("price-store-depth")).toBeNull();
    });

    it("shows the not-run cards when no evaluation exists yet", async () => {
      stubFetch([specRoute, ["/api/scorecard/evaluation", () => okJson({ version_key: VERSION, evaluations: {}, caveats: ALL_CAVEATS, note: "no evaluation has run for this version yet; nothing here is a result" })], universeRoute]);
      mount("/app/scorecard?tab=evaluation");
      await screen.findByTestId("scorecard-evaluation");
      expect(screen.getByTestId("eval-quintile_ls")).toHaveAttribute("data-state", "not-run");
      expect(screen.getByTestId("evaluation-note")).toHaveTextContent("no evaluation has run for this version yet; nothing here is a result");
    });
  });

  describe("refusals", () => {
    it("renders the upgrade prompt on a 402 plan_required", async () => {
      stubFetch([
        specRoute,
        [/\/api\/scorecard(\?|$)/, () => errJson(402, { code: "plan_required", message: "The scorecard is part of Pro.", feature: "scorecard", plan: "free", upgrade_url: "/pricing" })],
      ]);
      mount();
      await screen.findByTestId("upgrade-prompt");
      expect(screen.getByTestId("upgrade-prompt")).toHaveTextContent("Scorecard is part of Pro");
      expect(screen.getByRole("link", { name: "See Pro plans" })).toHaveAttribute("href", "/pricing");
    });

    it("renders a quiet not-enabled state on a 404 feature_disabled", async () => {
      stubFetch([
        specRoute,
        [/\/api\/scorecard(\?|$)/, () => errJson(404, { code: "feature_disabled", message: "the fundamental scorecard is not enabled on this deployment", feature: "scorecard" })],
      ]);
      mount();
      await screen.findByTestId("scorecard-disabled");
      expect(screen.getByTestId("scorecard-disabled")).toHaveTextContent("the fundamental scorecard is not enabled on this deployment");
      expect(screen.queryByRole("tablist")).toBeNull();
      expect(screen.queryByRole("alert")).toBeNull();
    });

    it("offers a retry on a 429 without losing the tab", async () => {
      let n = 0;
      stubFetch([
        specRoute,
        [
          /\/api\/scorecard(\?|$)/,
          // retry_after 0: the notice disables its Retry button while a countdown runs.
          () => (n++ === 0 ? errJson(429, { code: "rate_limited", message: "slow down", scope: "data", retry_after: 0, window_seconds: 60 }) : okJson(UNIVERSE_WIRE)),
        ],
      ]);
      mount();
      const retry = await screen.findByRole("button", { name: /retry/i });
      fireEvent.click(retry);
      await screen.findByTestId("universe-table");
      expect(screen.getByTestId("tab-universe")).toHaveAttribute("aria-selected", "true");
    });
  });
});

describe("normaliseEvaluationResponse", () => {
  it("folds the by-kind wire shape into the array the component renders, without inventing caveats", () => {
    const out = normaliseEvaluationResponse(evaluationWire());
    expect(out.version_key).toBe(VERSION);
    expect(out.note).toBe(NOTE);
    expect(out.caveats).toEqual(ALL_CAVEATS);
    expect(out.evaluations.map((e) => e.kind).sort()).toEqual(["double_lasso", "ff6_regression", "quintile_ls"]);
    const q = out.evaluations.find((e) => e.kind === "quintile_ls");
    expect(q?.kind === "quintile_ls" && q.result.quintile_table.map((b) => b.n)).toEqual(makeQuintileResult().quintile_table.map((b) => b.n));
    const ff = out.evaluations.find((e) => e.kind === "ff6_regression");
    // No caveats on the row → the response's list, verbatim.
    expect(ff?.result.caveats).toEqual(ALL_CAVEATS);
    // Backend-only keys pass through untouched.
    expect((ff?.result as unknown as { reasons: string[] }).reasons).toEqual([]);
  });

  it("tolerates an empty or malformed body", () => {
    expect(normaliseEvaluationResponse(null)).toEqual({ version_key: "", evaluations: [], caveats: [], note: "" });
    expect(normaliseEvaluationResponse({ evaluations: [{ nope: true }], caveats: "x" }).evaluations).toEqual([]);
  });

  it("keeps a result's own caveats when the row carries them", () => {
    const wire = evaluationWire();
    wire.evaluations.ff6_regression.result = { ...wire.evaluations.ff6_regression.result, caveats: ["only this one"] } as typeof wire.evaluations.ff6_regression.result;
    const ff = normaliseEvaluationResponse(wire).evaluations.find((e) => e.kind === "ff6_regression");
    expect(ff?.result.caveats).toEqual(["only this one"]);
  });
});
