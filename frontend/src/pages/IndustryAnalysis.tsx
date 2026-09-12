import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api, isApiError } from "@/api/client";
import { useAuth } from "@/auth/AuthContext";
import { useAccount } from "@/auth/useAccount";
import AccessGate from "@/components/industries/AccessGate";
import ChangesPanel from "@/components/industries/ChangesPanel";
import CompaniesTable from "@/components/industries/CompaniesTable";
import HistoryPicker from "@/components/industries/HistoryPicker";
import ReportHeader from "@/components/industries/ReportHeader";
import ReportTabs from "@/components/industries/ReportTabs";
import SectorGroupPicker from "@/components/industries/SectorGroupPicker";
import { fmtDate, gateFor, na, type IndustryGate } from "@/components/industries/format";
import type {
  IndustryChanges,
  IndustryCompanies,
  IndustryHistory,
  IndustryLastAttempt,
  IndustryReport,
  IndustryTaxonomy,
} from "@/types/industries";

/**
 * FEAT-003 — /app/industries and /app/industries/:code.
 *
 * A reading surface over what the Sunday worker persisted. **Nothing on
 * this page generates anything**: there is no regenerate button, no
 * mutation, and no method on the client that could queue one. A page
 * view is a handful of row fetches — which is what lets the latest
 * edition be public at all.
 *
 * The URL is the state: `/app/industries/4530?version=2&tab=performance`
 * is a citable reference to one edition's one section, which is the
 * point of publishing editions rather than a live page.
 *
 * What the page refuses to do:
 *
 *   * **fetch a surface the deployment has gated.** History and changes
 *     are Pro once the login wall is on; an anonymous read of them is a
 *     guaranteed 401, and the client's shared 401 handler would sign the
 *     session out and bounce the viewer to sign-in from a page they were
 *     only browsing. The gate renders from `access.surfaces`, which
 *     `/taxonomy` carries in every configuration;
 *   * **round a missing number to zero.** Every absent value on this
 *     page arrives with the server's reason and renders as "n/a
 *     (reason)";
 *   * **let a failed refresh look like a fresh week.** A stale edition
 *     keeps its content and gains a badge naming the attempt that failed.
 */

const FEATURE = "industry_analysis";

type Failure =
  | { state: "missing"; detail: string; lastAttempt?: IndustryLastAttempt | null }
  | { state: "not_imported"; detail: string; remedy?: string }
  | { state: "error"; detail: string; status?: number };

type Resource<T> = { state: "loading" } | ({ state: "ok"; data: T }) | Failure;

function classify(e: unknown): Failure {
  if (isApiError(e)) {
    const s = e.structured as (Record<string, unknown> & { remedy?: string; last_attempt?: IndustryLastAttempt }) | undefined;
    if (e.code === "taxonomy_not_imported") {
      return { state: "not_imported", detail: e.detail || e.message, remedy: s?.remedy };
    }
    if (e.status === 404) {
      return { state: "missing", detail: e.detail || e.message, lastAttempt: s?.last_attempt ?? null };
    }
    return { state: "error", detail: e.detail || e.message, status: e.status };
  }
  return { state: "error", detail: e instanceof Error ? e.message : String(e) };
}

/**
 * Load `loader()` whenever `key` changes and `enabled` is true.
 *
 * Disabled means the page decided NOT to call — a gate, or no group
 * selected — and the resource reads as null so nothing renders a spinner
 * for a request that will never be made.
 *
 * The result carries the `key` it was loaded for and a result for a
 * different key is never returned. Effects run after paint, so between
 * the render that changes the group (or the edition) and the effect that
 * starts the new fetch there is one frame in which the state still holds
 * the PREVIOUS group's report — a frame in which this page would put one
 * industry's numbers under another industry's name. That frame reads as
 * loading instead.
 */
function useResource<T>(enabled: boolean, key: string, loader: () => Promise<T>): Resource<T> | null {
  const [res, setRes] = useState<(Resource<T> & { key: string }) | null>(null);
  const seq = useRef(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  useEffect(() => {
    if (!enabled) {
      setRes(null);
      return;
    }
    const s = ++seq.current;
    setRes({ key, state: "loading" });
    loaderRef
      .current()
      .then((data) => {
        if (s === seq.current) setRes({ key, state: "ok", data });
      })
      .catch((e: unknown) => {
        if (s === seq.current) setRes({ key, ...classify(e) });
      });
  }, [enabled, key]);

  if (!enabled) return null;
  return res && res.key === key ? res : { state: "loading" };
}

function Loading({ what }: { what: string }) {
  return (
    <div className="card-tight text-sm text-slate-400" role="status" aria-busy="true" data-testid="industry-loading">
      Loading {what}…
    </div>
  );
}

function ErrorCard({ failure, onRetry }: { failure: Failure; onRetry?: () => void }) {
  if (failure.state === "not_imported") {
    return (
      <div className="card text-sm space-y-1" role="status" data-testid="taxonomy-not-imported">
        <div className="section-title">Industry taxonomy not imported</div>
        <p className="text-slate-400">{failure.detail}</p>
        {failure.remedy && (
          <p className="text-xs text-slate-500">
            An operator fixes this with: <span className="font-mono">{failure.remedy}</span>
          </p>
        )}
      </div>
    );
  }
  if (failure.state === "missing") {
    return (
      <div className="card text-sm space-y-1" role="status" data-testid="industry-missing">
        <div className="section-title">No published edition yet</div>
        <p className="text-slate-400">{failure.detail}</p>
        {failure.lastAttempt ? (
          <p className="text-xs text-slate-500" data-testid="missing-last-attempt">
            Last attempt: {failure.lastAttempt.status} ({failure.lastAttempt.error_type || "no error type"}
            {failure.lastAttempt.error_message ? ` — ${failure.lastAttempt.error_message}` : ""}), attempt{" "}
            {failure.lastAttempt.attempts} of {failure.lastAttempt.max_attempts}.
          </p>
        ) : (
          <p className="text-xs text-slate-500">
            {na("no generation attempt is on file for this group")}. The weekly worker writes the first edition on its
            next run.
          </p>
        )}
      </div>
    );
  }
  return (
    <div className="card-tight border-danger-500/40 text-sm" role="alert" data-testid="industry-error">
      <span className="text-danger-500">{failure.detail}</span>
      {onRetry && (
        <>
          {" "}
          <button type="button" onClick={onRetry} className="underline text-accent-500">
            Retry
          </button>
        </>
      )}
    </div>
  );
}

export default function IndustryAnalysis() {
  const { code = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const auth = useAuth();
  const { account } = useAccount();

  const version = params.get("version") || "latest";
  const tab = params.get("tab") || "overview";
  const [nonce, setNonce] = useState(0);
  const retry = useCallback(() => setNonce((n) => n + 1), []);

  const taxonomyRes = useResource<IndustryTaxonomy>(true, `taxonomy:${nonce}`, () => api.industryTaxonomy());
  const taxonomy = taxonomyRes?.state === "ok" ? taxonomyRes.data : null;
  const access = taxonomy?.access ?? null;

  const signedIn = auth.status === "signed_in";
  const entitlement = account?.entitlements?.[FEATURE];
  const gate = useCallback(
    (surface: string): IndustryGate =>
      gateFor(surface, access, { signedIn, entitlementAllowed: entitlement ? entitlement.allowed : null }),
    [access, signedIn, entitlement],
  );

  const latestGate = gate("latest");
  const historyGate = gate("history");
  const changesGate = gate("changes");

  const wantsReport = !!code && !!taxonomy && !latestGate;
  const reportRes = useResource<IndustryReport>(wantsReport, `report:${code}:${version}:${nonce}`, () =>
    api.industryReport(code, version),
  );
  const companiesRes = useResource<IndustryCompanies>(wantsReport, `companies:${code}:${nonce}`, () =>
    api.industryCompanies(code),
  );
  const historyRes = useResource<IndustryHistory>(!!code && !!taxonomy && !historyGate, `history:${code}:${nonce}`, () =>
    api.industryHistory(code),
  );
  // Fetched only when the reader opens the tab that shows it: a diff is
  // two more row reads, and most visits never ask for one.
  const changesRes = useResource<IndustryChanges>(
    !!code && !!taxonomy && !changesGate && tab === "what_changed" && version === "latest",
    `changes:${code}:${nonce}`,
    () => api.industryChanges(code),
  );

  const report = reportRes?.state === "ok" ? reportRes.data : null;
  const companies = companiesRes?.state === "ok" ? companiesRes.data : null;

  const setParam = useCallback(
    (key: string, value: string | null) => {
      const next = new URLSearchParams(params);
      if (value === null || value === "" || (key === "version" && value === "latest") || (key === "tab" && value === "overview")) {
        next.delete(key);
      } else {
        next.set(key, value);
      }
      setParams(next, { replace: false });
    },
    [params, setParams],
  );

  const selectGroup = useCallback(
    (next: string) => {
      // A new group means a new edition series: the version in the URL
      // belonged to the group being left and would silently ask for a
      // different week's report.
      navigate(`/app/industries/${encodeURIComponent(next)}`);
    },
    [navigate],
  );

  const extras = useMemo(() => {
    const out: Record<string, React.ReactNode> = {};
    // Only on the published edition: `/companies` prices its rows from
    // the LATEST statistics row, so comparing its count with an older
    // edition's would be comparing two different weeks and calling the
    // difference a disagreement.
    const editionNPriced = report?.is_latest_good
      ? report.payload?.sections?.companies?.facts?.n_priced
      : undefined;
    out.companies = companiesRes ? (
      companiesRes.state === "ok" ? (
        <CompaniesTable
          companies={companiesRes.data}
          editionNPriced={typeof editionNPriced === "number" ? editionNPriced : null}
        />
      ) : companiesRes.state === "loading" ? (
        <Loading what="constituents" />
      ) : (
        <ErrorCard failure={companiesRes} onRetry={retry} />
      )
    ) : null;
    out.what_changed = changesGate ? (
      <AccessGate gate={changesGate} what="the edition-to-edition diff" latestAvailable={!latestGate} />
    ) : version !== "latest" ? (
      <p className="text-xs text-slate-400" data-testid="changes-only-on-latest">
        {na("the diff is served against the published edition; clear the version to see it")}
      </p>
    ) : changesRes ? (
      changesRes.state === "ok" ? (
        <ChangesPanel changes={changesRes.data} />
      ) : changesRes.state === "loading" ? (
        <Loading what="the edition diff" />
      ) : (
        <ErrorCard failure={changesRes} onRetry={retry} />
      )
    ) : null;
    return out;
  }, [companiesRes, changesRes, changesGate, latestGate, report, version, retry]);

  // ---- taxonomy-level states ------------------------------------------
  if (!taxonomyRes || taxonomyRes.state === "loading") return <Loading what="the industry taxonomy" />;
  if (taxonomyRes.state !== "ok") return <ErrorCard failure={taxonomyRes} onRetry={retry} />;

  const picker = (
    <SectorGroupPicker taxonomy={taxonomyRes.data} value={code || null} onSelect={selectGroup} />
  );

  if (!code) {
    const counts = taxonomyRes.data.node_counts ?? {};
    const reports = taxonomyRes.data.reports ?? {};
    const limit = taxonomyRes.data.universe_coverage ?? null;
    return (
      <div className="space-y-4" data-testid="industry-index">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Industry Analysis</h1>
          <p className="text-xs text-slate-400 mt-1">
            Weekly editions per GICS industry group, written against the group's own mandate. Choose a group to read the
            latest published edition.
          </p>
        </div>
        <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 text-xs" data-testid="taxonomy-counts">
          <div className="card-tight">
            <dt className="text-slate-500 uppercase tracking-widest text-[10px]">Taxonomy</dt>
            <dd className="text-slate-200">
              {taxonomyRes.data.taxonomy_version.key} · effective{" "}
              {fmtDate(taxonomyRes.data.taxonomy_version.effective_from, "date not recorded")}
            </dd>
          </div>
          <div className="card-tight">
            <dt className="text-slate-500 uppercase tracking-widest text-[10px]">Structure</dt>
            <dd className="text-slate-200">
              {counts.sector ?? na("sectors not counted")} sectors · {counts.industry_group ?? na("groups not counted")}{" "}
              groups · {counts.industry ?? na("industries not counted")} industries ·{" "}
              {counts.sub_industry ?? na("sub-industries not counted")} sub-industries
            </dd>
          </div>
          <div className="card-tight">
            <dt className="text-slate-500 uppercase tracking-widest text-[10px]">Editions published</dt>
            <dd className="text-slate-200">
              {reports.groups_with_a_published_edition ?? 0} of {reports.groups ?? 0} groups
            </dd>
          </div>
          {/* The structural limit of this universe, in one place. Without
              it the groups it can never cover render as "not ready yet"
              on every page, week after week, and nobody learns that the
              universe is the thing that has to change.

              The second line is the SERVER's sentence, printed verbatim.
              The page composed one here once and it said the only remedy
              was adding companies — which is false whenever the universe
              already holds companies no group counts. Only the server
              knows that, so only the server writes the remedy. */}
          <div className="card-tight">
            <dt className="text-slate-500 uppercase tracking-widest text-[10px]">Coverage limit</dt>
            {limit ? (
              <>
                <dd className={limit.not_coverable > 0 ? "text-warn-500" : "text-slate-200"} data-testid="coverage-limit">
                  {limit.not_coverable} of {limit.groups} groups cannot be covered by the current universe
                </dd>
                <dd className="text-slate-500 mt-1" data-testid="coverage-limit-why">
                  {limit.explanation || na("this deployment did not say what the limit would take to lift")}
                </dd>
              </>
            ) : (
              <dd className="text-slate-400" data-testid="coverage-limit">
                {na("this deployment did not report a coverage limit")}
              </dd>
            )}
          </div>
        </dl>
        {latestGate && <AccessGate gate={latestGate} what="industry reports" />}
        {picker}
        <p className="text-[11px] leading-snug text-slate-500">{taxonomyRes.data.attribution}</p>
        <p className="text-[11px] leading-snug text-slate-500">{taxonomyRes.data.disclaimer}</p>
      </div>
    );
  }

  // ---- one group -------------------------------------------------------
  return (
    <div className="grid gap-6 lg:grid-cols-[18rem_minmax(0,1fr)]" data-testid="industry-detail">
      {/* `min-w-0` on both columns: a grid track's default `auto` minimum
          is its content's intrinsic width, so one wide table or one long
          unbroken string stretches the whole page instead of scrolling
          inside its own container. */}
      <div className="space-y-4 min-w-0">
        {picker}
        {historyGate ? (
          <AccessGate gate={historyGate} what="edition history" latestAvailable={!latestGate} />
        ) : historyRes?.state === "ok" ? (
          <HistoryPicker
            history={historyRes.data}
            value={version}
            onSelect={(v) => setParam("version", v)}
          />
        ) : historyRes?.state === "loading" ? (
          <Loading what="edition history" />
        ) : historyRes ? (
          <ErrorCard failure={historyRes} onRetry={retry} />
        ) : null}
      </div>

      <div className="space-y-4 min-w-0">
        {latestGate ? (
          <AccessGate gate={latestGate} what="industry reports" />
        ) : !reportRes || reportRes.state === "loading" ? (
          <Loading what="the industry report" />
        ) : reportRes.state !== "ok" ? (
          <>
            <ErrorCard failure={reportRes} onRetry={retry} />
            {/* A group with no edition still has members, and the
                membership read has already happened. Showing it is the
                difference between "nothing here" and "here is the group,
                the analysis has not been written yet". */}
            {reportRes.state === "missing" && companies && <CompaniesTable companies={companies} />}
          </>
        ) : (
          <>
            <ReportHeader
              report={report as IndustryReport}
              taxonomyAttribution={taxonomyRes.data.attribution}
              securityReferenceCaveat={companies?.security_reference_caveat ?? ""}
            />
            <ReportTabs
              report={report as IndustryReport}
              section={tab}
              onSelect={(s) => setParam("tab", s)}
              extras={extras}
            />
          </>
        )}
      </div>
    </div>
  );
}
