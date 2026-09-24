import React from "react";
import { describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import IndustryAnalysis from "@/pages/IndustryAnalysis";
import { AuthContext, DISABLED_AUTH, type AuthContextValue } from "@/auth/AuthContext";
import { ConfigContext, DEFAULT_CONFIG } from "@/auth/ConfigProvider";
import { resetAccountCache } from "@/auth/useAccount";
import {
  LocationSpy,
  SIGNED_IN,
  SIGNED_OUT,
  calls,
  errJson,
  makeAccount,
  okJson,
  stubFetch,
  type Matcher,
  type Responder,
} from "@/test/providers";
import type { PublicConfig } from "@/types";
import * as fx from "@/test/fixtures/industry";
import type { IndustryReport, IndustryTaxonomy } from "@/types/industries";

// What this file is defending, in the order the failures would bite:
//
//  * a page view that generates. Every test asserts the fetch spy saw no
//    POST or PUT — the weekly worker owns generation and a reader must
//    never start one;
//  * a gated surface fetched anyway. Under the wall an anonymous read of
//    a Pro route is a guaranteed 401 and the client's shared handler
//    signs the session out, so the gate must prevent the CALL, not just
//    hide the result;
//  * a URL that stops being the state — a picked group, a chosen edition
//    and a chosen section all have to survive a copy-paste;
//  * a failed refresh rendering as a fresh week;
//  * observed data and analyst interpretation sharing a heading.

const READY: Array<[Matcher, Responder]> = [
  ["/api/industries/taxonomy", () => okJson(fx.taxonomy)],
  [/\/api\/industries\/[^/]+\/report/, () => okJson(fx.report)],
  [/\/api\/industries\/[^/]+\/companies/, () => okJson(fx.companies)],
  [/\/api\/industries\/[^/]+\/history/, () => okJson(fx.history)],
  [/\/api\/industries\/[^/]+\/changes/, () => okJson(fx.changes)],
];

interface MountOptions {
  auth?: AuthContextValue;
  config?: Partial<PublicConfig>;
}

/** Both industry routes in one tree, because the index → detail hop is
 *  a real navigation this page performs and a single-route harness would
 *  assert on a location nothing rendered. */
function renderPage(route: string, opts: MountOptions = {}) {
  const config = { ...DEFAULT_CONFIG, ...(opts.config ?? {}) };
  return render(
    <MemoryRouter initialEntries={[route]}>
      <ConfigContext.Provider value={{ config, loaded: true, fallback: false }}>
        <AuthContext.Provider value={opts.auth ?? DISABLED_AUTH}>
          <LocationSpy />
          <Routes>
            <Route path="/app/industries" element={<IndustryAnalysis />} />
            <Route path="/app/industries/:code" element={<IndustryAnalysis />} />
          </Routes>
        </AuthContext.Provider>
      </ConfigContext.Provider>
    </MemoryRouter>,
  );
}

function mount(route: string, routes: Array<[Matcher, Responder]> = READY, opts: MountOptions = {}) {
  resetAccountCache();
  const fetchMock = stubFetch(routes);
  const view = renderPage(route, opts);
  return { fetchMock, view };
}

function mountIndex(routes: Array<[Matcher, Responder]> = READY) {
  return mount("/app/industries", routes);
}

async function settle() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

const location = () => screen.getByTestId("location").textContent;

// Telemetry POSTs are not writes this page makes: `lib/logger` flushes on a
// module-level 1.5 s timer, so a flush started by an EARLIER test can land
// inside this one's spy. Counting it made "never writes" fail on timer
// phase alone. `stubFetch` exempts the same two URLs.
const TELEMETRY = ["/api/admin/ui-log", "/api/public/events"];

/** The one assertion every test makes: the page reads, it never writes. */
function expectNoWrites(fetchMock: ReturnType<typeof stubFetch>) {
  const writes = fetchMock.mock.calls
    .map(([input, init]) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
      const method = String((init as RequestInit | undefined)?.method ?? "GET").toUpperCase();
      return { url, method };
    })
    .filter(({ url, method }) => method !== "GET" && method !== "HEAD" && !TELEMETRY.some((t) => url.includes(t)));
  expect(writes.map(({ method, url }) => `${method} ${url}`)).toEqual([]);
}

describe("IndustryAnalysis — index", () => {
  it("renders the picker from the taxonomy and never hardcodes a count", async () => {
    const { fetchMock } = mountIndex();
    await settle();

    expect(screen.getByTestId("industry-picker")).toBeInTheDocument();
    // Every group in the response is an option; the number comes from the
    // fixture, never from a literal in this test or in the page.
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(fx.allGroups.length);
    expect(screen.getByTestId("taxonomy-counts")).toHaveTextContent(
      `${fx.taxonomy.node_counts.industry_group} groups`,
    );
    expect(screen.getByText(fx.taxonomy.attribution)).toBeInTheDocument();
    expectNoWrites(fetchMock);
  });

  it("tells an operator how many groups this universe can never cover, in one place", async () => {
    mountIndex();
    await settle();

    const limit = fx.taxonomy.universe_coverage;
    expect(limit.not_coverable).toBeGreaterThan(0);
    const card = screen.getByTestId("coverage-limit");
    expect(card).toHaveTextContent(
      `${limit.not_coverable} of ${limit.groups} groups cannot be covered by the current universe`,
    );
    // What it would take is the SERVER's sentence, printed verbatim. The
    // page composing one here is how it came to tell an operator the only
    // remedy was adding companies — which is wrong whenever the universe
    // already holds companies no group counts, and only the server can
    // tell. The wording is asserted against the response, never typed out.
    expect(limit.explanation).toContain("classified constituents");
    expect(screen.getByTestId("coverage-limit-why")).toHaveTextContent(limit.explanation);
    expect(screen.getByTestId("taxonomy-counts")).not.toHaveTextContent(
      "The weekly price warm-up cannot change that",
    );
  });

  it("does not invent a remedy when the server sent the count without one", async () => {
    const t = fx.clone(fx.taxonomy);
    t.universe_coverage.explanation = "";
    mountIndex([["/api/industries/taxonomy", () => okJson(t)]]);
    await settle();
    expect(screen.getByTestId("coverage-limit-why")).toHaveTextContent(
      "n/a (this deployment did not say what the limit would take to lift)",
    );
  });

  it("says a deployment that reported no coverage limit reported none, rather than showing 0", async () => {
    const t = fx.clone(fx.taxonomy);
    delete (t as { universe_coverage?: unknown }).universe_coverage;
    mountIndex([["/api/industries/taxonomy", () => okJson(t)]]);
    await settle();
    expect(screen.getByTestId("coverage-limit")).toHaveTextContent(
      "n/a (this deployment did not report a coverage limit)",
    );
  });

  it("picking a group puts it in the URL and fetches only reads", async () => {
    const { fetchMock } = mountIndex();
    await settle();

    fireEvent.click(screen.getByText(fx.groupWithEdition.name));
    await settle();

    expect(location()).toBe(`/app/industries/${fx.groupWithEdition.code}`);
    expectNoWrites(fetchMock);
  });

  it("shows the 503 taxonomy state with the operator's remedy, not an error", async () => {
    const { fetchMock } = mountIndex([["/api/industries/taxonomy", () => errJson(503, fx.notImportedDetail)]]);
    await settle();

    const card = screen.getByTestId("taxonomy-not-imported");
    expect(card).toHaveTextContent("has not been imported");
    expect(card).toHaveTextContent("import_gics_taxonomy");
    expect(screen.queryByTestId("industry-picker")).not.toBeInTheDocument();
    expectNoWrites(fetchMock);
  });
});

describe("IndustryAnalysis — one group", () => {
  it("renders a deep link without interaction, with the report and its companies", async () => {
    const { fetchMock } = mount(`/app/industries/${fx.CODE}`);
    await settle();

    expect(screen.getByTestId("industry-report-header")).toHaveTextContent(fx.report.name);
    expect(screen.getByTestId("report-as-of")).toHaveTextContent("2026-09-04");
    // The companies table rides the Companies tab; the report header is
    // enough to prove the deep link rendered.
    expect(calls(fetchMock, `/api/industries/${fx.CODE}/report`)).toHaveLength(1);
    expectNoWrites(fetchMock);
  });

  it("separates observed data from analyst interpretation on every interpreted section", async () => {
    mount(`/app/industries/${fx.CODE}`);
    await settle();

    expect(screen.getByTestId("heading-facts")).toHaveTextContent("Observed data");
    expect(screen.getByTestId("heading-interpretation")).toHaveTextContent("Analyst interpretation");
    expect(screen.getByTestId("interpretation-view")).toHaveTextContent(
      fx.report.payload.sections.overview.interpretation!.text.slice(0, 40),
    );
  });

  it("moves between sections with the arrow keys and keeps the section in the URL", async () => {
    mount(`/app/industries/${fx.CODE}`);
    await settle();

    const order = fx.report.payload.section_order!;
    const first = screen.getByTestId(`tab-${order[0]}`);
    expect(first).toHaveAttribute("aria-selected", "true");
    expect(first).toHaveAttribute("tabindex", "0");
    expect(screen.getByTestId(`tab-${order[1]}`)).toHaveAttribute("tabindex", "-1");

    fireEvent.keyDown(screen.getByRole("tablist"), { key: "ArrowRight" });
    await settle();
    expect(screen.getByTestId(`tab-${order[1]}`)).toHaveAttribute("aria-selected", "true");
    expect(location()).toContain(`tab=${order[1]}`);

    fireEvent.keyDown(screen.getByRole("tablist"), { key: "End" });
    await settle();
    expect(screen.getByTestId(`tab-${order[order.length - 1]}`)).toHaveAttribute("aria-selected", "true");
  });

  it("renders the companies table under the Companies tab with its provenance column", async () => {
    mount(`/app/industries/${fx.CODE}?tab=companies`);
    await settle();

    const table = screen.getByTestId("companies-table");
    const mapped = fx.companies.items.find((i) => i.classification.source === "research_map")!;
    expect(within(table).getByTestId(`source-badge-${mapped.ticker}`)).toHaveTextContent("Research map");
    expect(within(table).getByTestId(`sub-industry-${mapped.ticker}`)).toHaveTextContent(mapped.sub_industry_name!);
  });

  it("surfaces a disagreement between the edition's priced count and the membership read", async () => {
    // The race is between two READS, and that is how it is staged: the
    // published edition, and `/companies` as the same endpoint answered
    // in the earlier week (it prices from the latest statistics row, so
    // it counted none). Both bodies are captures — neither is edited into
    // disagreeing, because an edition whose companies facts contradict
    // its own coverage block is one no producer can emit.
    const raced = fx.racedReads();
    const edition = raced.report.payload.sections.companies.facts;
    expect(edition.n_priced).not.toBe(raced.companies.n_priced);
    // Each body still agrees with itself.
    expect(edition.n_priced).toBe(raced.report.coverage.n_with_prices);
    expect(raced.companies.n_priced).toBe(raced.companies.items.filter((i) => i.priced).length);

    mount(`/app/industries/${fx.CODE}?tab=companies`, [
      ["/api/industries/taxonomy", () => okJson(fx.taxonomy)],
      [/\/report/, () => okJson(raced.report)],
      [/\/companies/, () => okJson(raced.companies)],
      [/\/history/, () => okJson(fx.history)],
    ]);
    await settle();
    // Two observed numbers for the same quantity, from two reads. The
    // page prints both and names the reconciled one instead of choosing.
    expect(screen.getByTestId("coverage-disagreement")).toHaveTextContent(
      `the edition's own companies facts count ${edition.n_priced} priced`,
    );
    expect(screen.getByTestId("coverage-disagreement")).toHaveTextContent(
      `this membership read counts ${raced.companies.n_priced} of ${raced.companies.count}`,
    );
  });

  it("words a structurally short group differently from a warm-up short one, end to end", async () => {
    // Both editions are captures of the real weekly path. The reader must
    // be able to tell "come back next week" from "this universe does not
    // hold enough of this industry" without knowing any status enum.
    const serve = (report: typeof fx.report) => [
      ["/api/industries/taxonomy", () => okJson(fx.taxonomy)],
      [/\/report/, () => okJson(report)],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ] as Array<[Matcher, Responder]>;

    const { view } = mount(`/app/industries/${fx.SHORT_CODE}`, serve(fx.universeShortReport));
    await settle();
    const structural = screen.getByTestId("coverage-sample-floor").textContent ?? "";
    expect(structural).toContain("This universe is too small to cover this industry");
    expect(structural).toContain("more classified constituents");
    // Not "the universe would have to add N companies": companies already
    // in this universe that no group counts close the same gap, and the
    // page must not send an operator to widen it on the server's behalf.
    expect(structural).not.toContain("the universe would have to add");
    expect(structural).not.toContain("insufficient_sample");
    view.unmount();

    mount(`/app/industries/${fx.CODE}`, serve(fx.warmingUpReport));
    await settle();
    const warming = screen.getByTestId("coverage-sample-floor").textContent ?? "";
    expect(warming).toContain("Not enough prices yet this week");
    expect(warming).toContain("without changing the universe");
    expect(warming).not.toBe(structural);
  });

  it("shows a stale edition's content with the failed attempt named", async () => {
    const stale = fx.staleReport();
    mount(`/app/industries/${fx.CODE}`, [
      ["/api/industries/taxonomy", () => okJson(fx.taxonomy)],
      [/\/report/, () => okJson(stale)],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ]);
    await settle();

    expect(screen.getByTestId("badge-stale")).toHaveTextContent(stale.stale_reason!);
    const attempt = screen.getByTestId("last-attempt");
    expect(attempt).toHaveTextContent("ReportRejected");
    expect(attempt).toHaveTextContent("the analyst draft did not pass validation (1 problem)");
    // The edition itself is still on the page — a failed week does not
    // blank it.
    expect(screen.getByTestId("industry-report-tabs")).toBeInTheDocument();
  });

  it("lists every degradation the edition carries", async () => {
    const degraded = fx.degradedReport();
    mount(`/app/industries/${fx.CODE}`, [
      ["/api/industries/taxonomy", () => okJson(fx.taxonomy)],
      [/\/report/, () => okJson(degraded)],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ]);
    await settle();

    expect(screen.getByTestId("badge-degraded")).toHaveTextContent(`Degraded (${degraded.degraded.length})`);
    const list = screen.getByTestId("degraded-list");
    for (const label of degraded.degraded) expect(list).toHaveTextContent(label);
  });

  it("renders the 404 'no edition yet' state with the last attempt", async () => {
    mount(`/app/industries/${fx.EMPTY_CODE}`, [
      ["/api/industries/taxonomy", () => okJson(fx.taxonomy)],
      [
        /\/report/,
        () =>
          errJson(404, {
            ...fx.noReportDetail,
            last_attempt: {
              job_id: 9,
              status: "failed",
              outcome: "failed",
              at: "2026-09-06T07:00:00",
              period_key: "2026-W36",
              attempts: 3,
              max_attempts: 3,
              error_type: "ReportRejected",
              error_message: "the analyst draft did not pass validation (1 problem)",
              report_id: null,
              source: "weekly_cron",
            },
          }),
      ],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ]);
    await settle();

    expect(fx.noReportDetail.reason).toBe("no_validated_analyst_edition");
    expect(screen.getByTestId("industry-missing")).toHaveTextContent("No analyst edition yet");
    expect(screen.getByTestId("missing-no-analyst")).toHaveTextContent(
      "No analyst-written edition has been published for this group yet.",
    );
    // Nothing withheld for this group: no audit-only sentence.
    expect(screen.getByTestId("missing-no-analyst")).not.toHaveTextContent("audit only");
    expect(screen.getByTestId("missing-last-attempt")).toHaveTextContent("ReportRejected");
    // The group still has members; "no edition" is not "nothing here".
    expect(screen.getByTestId("companies-table")).toBeInTheDocument();
  });

  // --- owner decision 1: template editions are never displayed ----------

  it("a group whose only edition is audit-only says so, and counts it", async () => {
    const detail = fx.noAgenticDetail;
    expect(detail.withheld_editions).toBeGreaterThan(0);
    mount(`/app/industries/${fx.WIRE_META.no_agentic_code}`, [
      ["/api/industries/taxonomy", () => okJson(fx.taxonomy)],
      [/\/report/, () => errJson(404, detail)],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ]);
    await settle();

    const empty = screen.getByTestId("missing-no-analyst");
    expect(empty).toHaveTextContent("No analyst-written edition has been published for this group yet.");
    expect(empty).toHaveTextContent(
      `${detail.withheld_editions} edition${detail.withheld_editions === 1 ? "" : "s"} without a validated analyst narrative`,
    );
    expect(empty).toHaveTextContent("kept for audit only and not shown");
    expect(screen.getByTestId("missing-last-attempt")).toHaveTextContent("no validated analyst edition");
    expect(screen.queryByTestId("industry-report-tabs")).not.toBeInTheDocument();
    expect(screen.getByTestId("companies-table")).toBeInTheDocument();
  });

  it("an edition number kept for audit only renders as withheld, not as missing", async () => {
    mount(`/app/industries/${fx.CODE}?version=3`, [
      ["/api/industries/taxonomy", () => okJson(fx.taxonomy)],
      [
        /\/report/,
        () =>
          errJson(404, {
            code: "edition_withheld",
            message: "edition 3 is kept for audit only and is not published",
            version: 3,
          }),
      ],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ]);
    await settle();

    expect(screen.getByTestId("industry-edition-withheld")).toHaveTextContent("This edition is kept for audit only.");
    expect(screen.queryByTestId("industry-missing")).not.toBeInTheDocument();
  });

  it("a group a newer week did not update shows its last analyst edition under a banner", async () => {
    mount(`/app/industries/${fx.SHORT_CODE}`, [
      ["/api/industries/taxonomy", () => okJson(fx.taxonomy)],
      [/\/report/, () => okJson(fx.notUpdatedReport)],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ]);
    await settle();

    expect(screen.getByTestId("not-updated-banner")).toHaveTextContent(
      `The ${fx.OUTAGE_PERIOD} refresh did not produce a validated analyst edition.`,
    );
    // The edition's content stays: a week without an analyst edition does not blank the page.
    expect(screen.getByTestId("industry-report-tabs")).toBeInTheDocument();
  });

  it("shows a loading state while the report is in flight", async () => {
    mount(`/app/industries/${fx.CODE}`, [
      ["/api/industries/taxonomy", () => okJson(fx.taxonomy)],
      [/\/report/, () => new Promise<Partial<Response>>(() => {})],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ]);
    await settle();

    expect(screen.getAllByTestId("industry-loading")[0]).toHaveAttribute("aria-busy", "true");
  });

  it("history selection changes the version param and refetches that edition", async () => {
    const prior = fx.priorEdition();
    const seen: string[] = [];
    const { fetchMock } = mount(`/app/industries/${fx.CODE}`, [
      ["/api/industries/taxonomy", () => okJson(fx.taxonomy)],
      [
        /\/report/,
        (url) => {
          seen.push(url);
          return okJson(url.includes(`version=${prior.version}`) ? prior : fx.report);
        },
      ],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ]);
    await settle();

    fireEvent.change(screen.getByTestId("history-select"), { target: { value: String(prior.version) } });
    await settle();

    expect(location()).toContain(`version=${prior.version}`);
    await waitFor(() => expect(seen.some((u) => u.includes(`version=${prior.version}`))).toBe(true));
    expect(screen.getByTestId("industry-report-header")).toHaveTextContent(`Edition v${prior.version}`);
    expectNoWrites(fetchMock);
  });

  it("never shows one edition's numbers under another edition's heading", async () => {
    const prior = fx.priorEdition();
    let release: (() => void) | null = null;
    mount(`/app/industries/${fx.CODE}`, [
      ["/api/industries/taxonomy", () => okJson(fx.taxonomy)],
      [
        /\/report/,
        (url) =>
          url.includes(`version=${prior.version}`)
            ? new Promise<Partial<Response>>((resolve) => {
                release = () => resolve(okJson(prior));
              })
            : okJson(fx.report),
      ],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ]);
    await settle();
    expect(screen.getByTestId("industry-report-header")).toHaveTextContent(`Edition v${fx.report.version}`);

    // Ask for the older edition; while it is in flight the page must not
    // keep rendering the newer one under the new URL.
    fireEvent.change(screen.getByTestId("history-select"), { target: { value: String(prior.version) } });
    await settle();
    expect(screen.queryByTestId("industry-report-header")).not.toBeInTheDocument();
    expect(screen.getAllByTestId("industry-loading")[0]).toBeInTheDocument();

    release!();
    await settle();
    expect(screen.getByTestId("industry-report-header")).toHaveTextContent(`Edition v${prior.version}`);
  });

  it("fetches the diff only when the what-changed tab is open", async () => {
    const { fetchMock } = mount(`/app/industries/${fx.CODE}`);
    await settle();
    expect(calls(fetchMock, "/changes")).toHaveLength(0);

    fireEvent.click(screen.getByTestId("tab-what_changed"));
    await settle();
    expect(calls(fetchMock, "/changes")).toHaveLength(1);
    expect(screen.getByTestId("changes-table")).toBeInTheDocument();
    expectNoWrites(fetchMock);
  });
});

describe("IndustryAnalysis — access", () => {
  it("does not fetch the report when the latest edition is Pro and nobody is signed in", async () => {
    const { fetchMock } = mount(`/app/industries/${fx.CODE}`, [
      ["/api/industries/taxonomy", () => okJson(fx.gatedTaxonomy())],
      [/\/report/, () => okJson(fx.report)],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ], { auth: SIGNED_OUT, config: { auth_enabled: true } });
    await settle();

    expect(screen.getByTestId("industry-gate-latest")).toHaveAttribute("data-gate-reason", "sign_in");
    expect(calls(fetchMock, "/report")).toHaveLength(0);
    expect(calls(fetchMock, "/companies")).toHaveLength(0);
    expect(calls(fetchMock, "/history")).toHaveLength(0);
    // The taxonomy still answers — it is how the page learned the tier.
    expect(calls(fetchMock, "/api/industries/taxonomy")).toHaveLength(1);
    expectNoWrites(fetchMock);
  });

  it("gates history alone when only history is Pro, and still serves the public edition", async () => {
    const t: IndustryTaxonomy = fx.clone(fx.taxonomy);
    t.access = { ...t.access, enforced: true, surfaces: { ...t.access.surfaces, latest: "public", history: "pro" } };
    const { fetchMock } = mount(`/app/industries/${fx.CODE}`, [
      ["/api/industries/taxonomy", () => okJson(t)],
      [/\/report/, () => okJson(fx.report)],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ], { auth: SIGNED_OUT, config: { auth_enabled: true } });
    await settle();

    expect(screen.getByTestId("industry-gate-history")).toBeInTheDocument();
    expect(calls(fetchMock, "/history")).toHaveLength(0);
    expect(calls(fetchMock, "/report")).toHaveLength(1);
    expect(screen.getByTestId("industry-report-header")).toBeInTheDocument();
  });

  it("reads every surface for a signed-in account that holds the entitlement", async () => {
    resetAccountCache();
    const t = fx.gatedTaxonomy();
    const fetchMock = stubFetch([
      ["/api/me", () => okJson(makeAccount())],
      ["/api/industries/taxonomy", () => okJson(t)],
      [/\/report/, () => okJson(fx.report)],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ]);
    renderPage(`/app/industries/${fx.CODE}`, { auth: SIGNED_IN, config: { auth_enabled: true } });
    await settle();

    expect(screen.queryByTestId("industry-gate-latest")).not.toBeInTheDocument();
    expect(calls(fetchMock, "/report")).toHaveLength(1);
    expectNoWrites(fetchMock);
  });

  it("shows the plan gate, without fetching, when the entitlement is explicitly refused", async () => {
    resetAccountCache();
    const account = makeAccount();
    account.entitlements.industry_analysis = {
      feature: "industry_analysis",
      allowed: false,
      limit: null,
      used: 0,
      remaining: null,
      resets_at: null,
    };
    const fetchMock = stubFetch([
      ["/api/me", () => okJson(account)],
      ["/api/industries/taxonomy", () => okJson(fx.gatedTaxonomy())],
      [/\/report/, () => okJson(fx.report)],
      [/\/history/, () => okJson(fx.history)],
    ]);
    renderPage(`/app/industries/${fx.CODE}`, { auth: SIGNED_IN, config: { auth_enabled: true } });
    await settle();
    await waitFor(() => expect(screen.getByTestId("industry-gate-latest")).toHaveAttribute("data-gate-reason", "plan"));
    expect(calls(fetchMock, "/report")).toHaveLength(0);
  });

  it("does not tell a reader the edition is public on a page that is hiding it", async () => {
    // The gate card stands WHERE the report would be. A card that says
    // "the latest published edition below is public" under a policy that
    // just refused it states the opposite of the policy it read.
    const { fetchMock } = mount(`/app/industries/${fx.CODE}`, [
      ["/api/industries/taxonomy", () => okJson(fx.gatedTaxonomy())],
      [/\/report/, () => okJson(fx.report)],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ], { auth: SIGNED_OUT, config: { auth_enabled: true } });
    await settle();

    for (const card of screen.getAllByTestId(/^industry-gate-/)) {
      expect(card).not.toHaveTextContent(/edition below is public/);
      expect(card).not.toHaveTextContent(/published edition itself is still shown below/);
    }
    expect(screen.queryByTestId("industry-report-header")).not.toBeInTheDocument();
    expectNoWrites(fetchMock);
  });

  it("does point at the edition when it is only history that is gated", async () => {
    const t: IndustryTaxonomy = fx.clone(fx.taxonomy);
    t.access = { ...t.access, enforced: true, surfaces: { ...t.access.surfaces, latest: "public", history: "pro" } };
    mount(`/app/industries/${fx.CODE}`, [
      ["/api/industries/taxonomy", () => okJson(t)],
      [/\/report/, () => okJson(fx.report)],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ], { auth: SIGNED_OUT, config: { auth_enabled: true } });
    await settle();

    expect(screen.getByTestId("industry-gate-history")).toHaveTextContent(
      /published edition itself is still shown below/,
    );
    expect(screen.getByTestId("industry-report-header")).toBeInTheDocument();
  });

  it("gates nothing while the login wall is off, whatever the tiers say", async () => {
    const { fetchMock } = mount(`/app/industries/${fx.CODE}`, [
      // `enforced: false` with Pro tiers is today's default deployment.
      ["/api/industries/taxonomy", () => okJson(fx.taxonomy)],
      [/\/report/, () => okJson(fx.report)],
      [/\/companies/, () => okJson(fx.companies)],
      [/\/history/, () => okJson(fx.history)],
    ]);
    await settle();

    expect(fx.taxonomy.access.surfaces.history).toBe("pro");
    expect(fx.taxonomy.access.enforced).toBe(false);
    expect(screen.queryByTestId("industry-gate-history")).not.toBeInTheDocument();
    expect(calls(fetchMock, "/history")).toHaveLength(1);
  });
});

describe("IndustryAnalysis — the page never generates", () => {
  it("exposes no mutating industry method on the API client", async () => {
    const { api } = await import("@/api/client");
    const industryMethods = Object.keys(api).filter((k) => k.toLowerCase().startsWith("industry"));
    expect(industryMethods.sort()).toEqual([
      "industryChanges",
      "industryCompanies",
      "industryHistory",
      "industryReport",
      "industryTaxonomy",
    ]);
  });

  it("makes no write on mount, on navigation between sections, or on an edition change", async () => {
    const report: IndustryReport = fx.report;
    const { fetchMock } = mount(`/app/industries/${report.code}`);
    await settle();
    fireEvent.click(screen.getByTestId("tab-performance"));
    await settle();
    fireEvent.click(screen.getByTestId("tab-companies"));
    await settle();
    fireEvent.change(screen.getByTestId("history-select"), { target: { value: "1" } });
    await settle();

    expectNoWrites(fetchMock);
    expect(vi.mocked(fetchMock).mock.calls.length).toBeGreaterThan(0);
  });
});
