import React, { useCallback, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, historyFromDetail, isApiError, scorecardExportUrl, type ScorecardExportFile } from "@/api/client";
import { useConfig } from "@/auth/ConfigProvider";
import RateLimitNotice from "@/components/RateLimitNotice";
import TickerPicker from "@/components/TickerPicker";
import UpgradePrompt from "@/components/UpgradePrompt";
import ScorecardEvaluation from "@/components/scorecard/ScorecardEvaluation";
import ScorecardHistoryChart, { type HistoryMetric } from "@/components/scorecard/ScorecardHistoryChart";
import ScorecardPanel from "@/components/scorecard/ScorecardPanel";
import ScorecardUniverseTable from "@/components/scorecard/ScorecardUniverseTable";
import { familyLabel, na } from "@/components/scorecard/format";
import { SCORECARD_CLIENT_RULES, SCORECARD_EXPORT_CONTRACT, SCORECARD_SCORE_SCALE_LONG, evalParam } from "@/types/scorecard";
import type { CompanyOut, EntitlementRefusal, RateLimitRefusal, ScorecardDetail, ScorecardEvaluationResponse, ScorecardSpec, ScorecardUniverse } from "@/types";

/**
 * Phase 6 — /app/scorecard. Three views over what the worker persisted
 * (nothing here computes a score):
 *
 *   Universe   — the latest run's cross-section (sort/filter client-side,
 *                CSV/JSON export under the frozen v1 contract);
 *   Ticker     — one name's observed inputs beside the model read, with
 *                its month-end history lifted from the same detail row;
 *   Evaluation — the three worker evaluations with their caveats and the
 *                minimums they were run with, verbatim.
 *
 * The URL is the state (`?tab=`, `?ticker=`) so a view can be shared; the
 * last tab is also remembered per browser for a bare `/app/scorecard`.
 * Every non-happy state is explicit: 402 renders the upgrade prompt, a
 * 404 `feature_disabled` a quiet "not enabled" card, a plain 404 the
 * "no run yet" state, 429 a retry notice. Research and education only —
 * a rank is a research queue, not a buy or sell list.
 *
 * Export: with the login wall off the export is a plain link (the route
 * is unrestricted); with it on a navigation cannot carry the bearer and
 * a token may never ride in the URL, so the page fetches the file through
 * the API client and hands the viewer an object-URL download instead.
 */

export const SCORECARD_TABS = ["universe", "ticker", "evaluation"] as const;
export type ScorecardTab = (typeof SCORECARD_TABS)[number];
const TAB_LABEL: Record<ScorecardTab, string> = { universe: "Universe", ticker: "Ticker", evaluation: "Evaluation" };
const TAB_BLURB: Record<ScorecardTab, string> = {
  universe: "Every name in the curated universe on the latest run, ranked by the model read of its reported fundamentals.",
  ticker: "One name: the observed inputs beside the model read, with the month-end history of its rank.",
  evaluation: "Does the score predict anything? The worker's quintile spread, factor regression and double-selection LASSO, with their caveats.",
};

// localStorage key (versioned so the schema can change later).
const LS_TAB = "scorecard:tab:v1";
// The route's ceiling; the curated universe is 100–600 names and the
// count is never hardcoded lower than that.
const UNIVERSE_LIMIT = 600;
const HISTORY_MONTHS = 36;

function isTab(v: string | null | undefined): v is ScorecardTab {
  return !!v && (SCORECARD_TABS as readonly string[]).includes(v);
}

function loadTab(): ScorecardTab {
  try {
    const raw = window.localStorage.getItem(LS_TAB);
    return isTab(raw) ? raw : "universe";
  } catch {
    return "universe";
  }
}

function saveTab(tab: ScorecardTab): void {
  try {
    window.localStorage.setItem(LS_TAB, tab);
  } catch {
    // Ignore quota / SSR.
  }
}

// ---------------------------------------------------------------------------
// One small state machine per resource so each tab says exactly why it has
// nothing to show. `key` identifies what the result is for (the ticker, a
// constant) so a stale detail never renders under a new ticker.
// ---------------------------------------------------------------------------

type Failure =
  | { state: "missing"; detail: string }
  | { state: "disabled"; detail: string }
  | { state: "plan"; refusal: EntitlementRefusal }
  | { state: "rate"; refusal: RateLimitRefusal }
  | { state: "error"; detail: string; status?: number };

type Loaded<T> = { key: string } & ({ state: "loading" } | { state: "ok"; data: T } | Failure);

function classify(e: unknown): Failure {
  if (isApiError(e)) {
    if (e.code === "feature_disabled") return { state: "disabled", detail: e.detail || e.message };
    if (e.entitlement) return { state: "plan", refusal: e.entitlement };
    if (e.rateLimit) return { state: "rate", refusal: e.rateLimit };
    if (e.status === 404) return { state: "missing", detail: e.detail || e.message };
    return { state: "error", detail: e.detail || e.message, status: e.status };
  }
  return { state: "error", detail: e instanceof Error ? e.message : String(e) };
}

/** Load `loader()` once per `key` while `enabled`; keep the last result
 *  when the tab that needs it is hidden so a return visit does not refetch. */
function useResource<T>(enabled: boolean, key: string, loader: () => Promise<T>): [Loaded<T> | null, () => void] {
  const [result, setResult] = useState<Loaded<T> | null>(null);
  const [nonce, setNonce] = useState(0);
  const seq = useRef(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  const loadedKey = useRef<string | null>(null);
  const loadedNonce = useRef(0);

  useEffect(() => {
    if (!enabled) return;
    if (loadedKey.current === key && loadedNonce.current === nonce) return;
    loadedKey.current = key;
    loadedNonce.current = nonce;
    const s = ++seq.current;
    setResult({ key, state: "loading" });
    loaderRef
      .current()
      .then((data) => {
        if (s === seq.current) setResult({ key, state: "ok", data });
      })
      .catch((e: unknown) => {
        if (s === seq.current) setResult({ key, ...classify(e) });
      });
  }, [enabled, key, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return [result, reload];
}

/** Renders the non-ok states of a resource; null when there is data. */
function ResourceState({ res, onRetry, missing }: { res: Loaded<unknown> | null; onRetry: () => void; missing: React.ReactNode }) {
  if (!res || res.state === "loading") {
    return (
      <div className="card-tight text-sm text-slate-400" aria-busy="true" role="status" data-testid="scorecard-loading">
        Loading…
      </div>
    );
  }
  switch (res.state) {
    case "ok":
      return null;
    case "plan":
      return <UpgradePrompt refusal={res.refusal} />;
    case "rate":
      return <RateLimitNotice refusal={res.refusal} onRetry={onRetry} preservedNote="Your tab and ticker are unchanged — retry when the timer ends." />;
    case "disabled":
      return <DisabledCard detail={res.detail} />;
    case "missing":
      return <>{missing}</>;
    case "error":
      return (
        <div className="card-tight border-danger-500/40 text-sm" role="alert">
          <span className="text-danger-500">{res.detail}</span>{" "}
          <button type="button" onClick={onRetry} className="underline text-accent-500">
            Retry
          </button>
        </div>
      );
  }
}

function DisabledCard({ detail }: { detail: string }) {
  return (
    <div className="card text-sm text-slate-300" role="status" data-testid="scorecard-disabled">
      <div className="section-title mb-1">Scorecard not enabled</div>
      <p className="text-slate-400">{detail || "The fundamental scorecard is not enabled on this deployment."}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Evaluation helpers: the minimums a card prints come from the row's
// `params` when the worker recorded them and from the documented fs-v1
// client rules only when the row lacks them — and the page says which.
// ---------------------------------------------------------------------------

function paramsOf(evaluation: ScorecardEvaluationResponse): Record<string, unknown> {
  // Every kind shares `params_common`; the first row is representative.
  return evaluation.evaluations[0]?.params ?? {};
}

/** The first row whose params carry a string at `key`; the worker writes
 *  `controls_deferred` on the LASSO row only, `price_store_depth` on all. */
function paramFromAnyRow(evaluation: ScorecardEvaluationResponse, key: string): unknown {
  for (const row of evaluation.evaluations) {
    const v = row.params?.[key];
    if (v !== undefined && v !== null) return v;
  }
  return undefined;
}

function Minimums({ evaluation }: { evaluation: ScorecardEvaluationResponse }) {
  const params = paramsOf(evaluation);
  const items: Array<[label: string, key: "min_leg" | "min_months" | "min_obs", fallback: number, unit: string]> = [
    ["quintile leg", "min_leg", SCORECARD_CLIENT_RULES.evalMinLeg, "names"],
    ["regression / LASSO sample", "min_months", SCORECARD_CLIENT_RULES.evalMinMonths, "months"],
    ["LASSO panel", "min_obs", SCORECARD_CLIENT_RULES.lassoMinObs, "observations"],
  ];
  const depthRaw = paramFromAnyRow(evaluation, "price_store_depth");
  const depth = typeof depthRaw === "string" ? depthRaw : null;
  const deferredRaw = paramFromAnyRow(evaluation, "controls_deferred");
  const deferred = deferredRaw && typeof deferredRaw === "object" ? (deferredRaw as Record<string, unknown>) : null;
  const deferredEntries = deferred ? Object.entries(deferred).filter(([, v]) => typeof v === "string") : [];
  return (
    <div className="card-tight text-xs space-y-2" data-testid="evaluation-minimums">
      <div className="text-slate-400 uppercase tracking-widest text-[10px]">Minimums the evaluation was run with</div>
      <ul className="space-y-0.5 text-slate-300">
        {items.map(([label, key, fallback, unit]) => {
          const recorded = typeof params[key] === "number" && Number.isFinite(params[key] as number);
          const value = evalParam(params, key, fallback);
          return (
            <li key={key} data-testid={`minimum-${key}`} data-source={recorded ? "params" : "client-rule"}>
              <span className="font-mono text-slate-200">
                {value.toLocaleString("en-US")} {unit}
              </span>{" "}
              per {label} —{" "}
              <span className="text-slate-500">{recorded ? "recorded by the worker in the evaluation's params" : "fs-v1 client rule; the row does not record it"}</span>
            </li>
          );
        })}
      </ul>
      {depth && (
        <p className="text-slate-300" data-testid="price-store-depth">
          <span className="text-slate-500">Price store depth: </span>
          {depth}
        </p>
      )}
      {deferredEntries.length > 0 && (
        <div data-testid="controls-deferred">
          <span className="text-slate-500">LASSO controls deferred (verbatim): </span>
          <ul className="list-disc pl-4 text-slate-300">
            {deferredEntries.map(([name, why]) => (
              <li key={name}>
                <span className="font-mono">{name}</span> — {String(why)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/**
 * The response carries the caveats once at the top level and the worker
 * writes the same sentences on every result. The page prints the
 * response's list once, verbatim, and — when that list covers every
 * card's caveats — the cards point up instead of repeating it. A card
 * carrying a sentence the response lacks keeps its own verbatim list, so
 * nothing is paraphrased and nothing is hidden.
 */
function caveatsCoverCards(evaluation: ScorecardEvaluationResponse): boolean {
  const top = new Set(evaluation.caveats ?? []);
  if (top.size === 0) return false;
  return evaluation.evaluations.every((r) => (r.result.caveats ?? []).every((c) => top.has(c)));
}

function ResponseCaveats({ evaluation }: { evaluation: ScorecardEvaluationResponse }) {
  const caveats = evaluation.caveats ?? [];
  if (caveats.length === 0) return null;
  const covers = caveatsCoverCards(evaluation);
  return (
    <div className="card-tight text-xs" data-testid="evaluation-caveats" data-covers-cards={covers ? "true" : "false"}>
      <div className="text-slate-400 uppercase tracking-widest text-[10px] mb-1">Caveats (verbatim from the evaluation)</div>
      <ul className="list-disc pl-4 text-slate-300 space-y-0.5">
        {caveats.map((c) => (
          <li key={c}>{c}</li>
        ))}
      </ul>
      {!covers && evaluation.evaluations.length > 0 ? (
        <p className="mt-1 text-slate-500" data-testid="evaluation-caveats-more">
          A card below carries a caveat this list does not; each card keeps its own verbatim list.
        </p>
      ) : null}
    </div>
  );
}

/** Hand the viewer a fetched file. The object URL is revoked on the next
 *  tick — some browsers abort a download whose URL is revoked before the
 *  click has been processed. */
function saveFile(file: ScorecardExportFile): void {
  const url = URL.createObjectURL(file.blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = file.filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  try {
    a.click();
  } finally {
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

type ExportFormat = "csv" | "json";

/** Under the wall: fetch the export with the bearer, then download it.
 *  A refusal renders exactly as it would for any other read (402 → the
 *  upgrade prompt). A 401 is the one exception: on a deployment that sets
 *  `SCORECARD_EXPORT_TOKEN` the route wants that token and ignores the
 *  bearer, so the client fetches with `unauthorized: "throw"` — the
 *  refusal prints verbatim with a retry and the session is left alone
 *  (the shared 401 path would sign the viewer out through RequireAuth). */
function ExportButtons({ run }: { run: ScorecardUniverse }) {
  const [busy, setBusy] = useState<ExportFormat | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [failed, setFailed] = useState<(Failure & { key: string; format: ExportFormat }) | null>(null);

  const download = useCallback(
    async (format: ExportFormat) => {
      setBusy(format);
      setSaved(null);
      setFailed(null);
      try {
        const file = await api.scorecardExport({ format, version: run.version_key, as_of: run.as_of });
        saveFile(file);
        setSaved(file.filename);
      } catch (e) {
        setFailed({ key: `export-${format}`, format, ...classify(e) });
      } finally {
        setBusy(null);
      }
    },
    [run.version_key, run.as_of],
  );

  return (
    <div className="flex flex-wrap items-center gap-2" data-testid="export-buttons">
      {(["csv", "json"] as const).map((format) => (
        <button
          key={format}
          type="button"
          onClick={() => void download(format)}
          disabled={busy !== null}
          className="btn-ghost !py-1 !px-2 text-xs"
          data-testid={`export-${format}-button`}
          title={`Frozen column order, contract ${SCORECARD_EXPORT_CONTRACT}; fetched with your session, saved as a file`}
        >
          {busy === format ? "Fetching…" : `Export ${format.toUpperCase()} (contract ${SCORECARD_EXPORT_CONTRACT})`}
        </button>
      ))}
      {saved && (
        <span className="text-slate-500" data-testid="export-saved">
          saved {saved}
        </span>
      )}
      {failed && (
        <div className="basis-full space-y-1">
          <ResourceState res={failed} onRetry={() => void download(failed.format)} missing={<span className="text-slate-400">{na(failed.state === "missing" ? failed.detail : "export unavailable")}</span>} />
          {failed.state === "error" && failed.status === 401 && (
            <p className="text-[11px] text-slate-500" data-testid="export-token-gated">
              The export route refused this request; no file was saved and your session is unchanged. A deployment that sets SCORECARD_EXPORT_TOKEN answers
              every browser export this way — that token is for downstream systems to present as a bearer header, never for this page.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

/** `normalization.winsor_pct` is served as a FRACTION of the distribution
 *  (fs-v1: 0.025 — `scorecard_spec.NormalizationParams`, emitted verbatim
 *  by `spec_view`), while the sentence prints the two percentile bounds.
 *  This is the only place the number is scaled; the rounding trims float
 *  noise (0.025 × 100 must read 2.5, not 2.5000000000000004). null when
 *  the spec carried no finite value, so the page says n/a instead of
 *  printing NaN%. */
export function winsorBoundsPct(fraction: unknown): [lower: string, upper: string] | null {
  if (typeof fraction !== "number" || !Number.isFinite(fraction)) return null;
  const lo = Number((fraction * 100).toFixed(4));
  const hi = Number((100 - fraction * 100).toFixed(4));
  return [`${lo}%`, `${hi}%`];
}

function SpecBlock({ spec }: { spec: ScorecardSpec }) {
  const norm = spec.normalization;
  const winsor = norm ? winsorBoundsPct(norm.winsor_pct) : null;
  return (
    <details className="card-tight text-xs" data-testid="spec">
      <summary className="cursor-pointer text-slate-300">
        Methodology <span className="badge-neutral ml-1">{spec.version_key}</span>
        <span className="text-slate-500 ml-2">
          spec {spec.spec_hash}
          {spec.source ? ` · served from ${spec.source}` : ""}
        </span>
      </summary>
      <div className="mt-2 space-y-2 text-slate-300">
        <p data-testid="spec-score-scale">{spec.score_scale || SCORECARD_SCORE_SCALE_LONG}</p>
        {norm && (
          <p className="text-slate-400" data-testid="spec-normalization">
            Winsorised at {winsor ? `${winsor[0]} / ${winsor[1]}` : na("winsor bound not in the served spec")}, {norm.sector_neutral ? "sector-neutral" : "universe"} z
            (minimum sector n {norm.min_sector_n}, else universe z), clipped at ±{norm.clip_z}; percentiles are rank-based.
          </p>
        )}
        <div className="overflow-x-auto">
          <table className="min-w-full text-[11px]">
            <thead className="uppercase tracking-wider text-slate-500">
              <tr>
                <th scope="col" className="text-left px-2 py-1">
                  Family
                </th>
                <th scope="col" className="text-left px-2 py-1">
                  Feature
                </th>
                <th scope="col" className="text-left px-2 py-1">
                  Formula
                </th>
                <th scope="col" className="text-left px-2 py-1">
                  Direction
                </th>
                <th scope="col" className="text-left px-2 py-1">
                  Not applicable to
                </th>
              </tr>
            </thead>
            <tbody>
              {(spec.families ?? []).flatMap((fam) =>
                (fam.features ?? []).map((f) => (
                  <tr key={`${fam.name}-${f.name}`} className="border-t border-ink-700/60">
                    <td className="px-2 py-1 text-slate-400 whitespace-nowrap">{familyLabel(String(fam.name))}</td>
                    <td className="px-2 py-1 text-slate-200 whitespace-nowrap">{f.name}</td>
                    <td className="px-2 py-1 font-mono text-slate-400">{f.formula}</td>
                    <td className="px-2 py-1 text-slate-400 whitespace-nowrap">{f.sign === -1 ? "lower is better" : "higher is better"}</td>
                    <td className="px-2 py-1 text-slate-500">{f.applicability?.exclude_sectors?.length ? f.applicability.exclude_sectors.join(", ") : "—"}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      </div>
    </details>
  );
}

// ---------------------------------------------------------------------------

export default function Scorecard() {
  const { config } = useConfig();
  // With the wall on the plain links would go out without the bearer
  // (a 401 at best); the buttons fetch through the client instead.
  const wallOn = config.auth_enabled;
  const [params, setParams] = useSearchParams();
  const urlTab = params.get("tab");
  const tab: ScorecardTab = isTab(urlTab) ? urlTab : loadTab();
  const ticker = (params.get("ticker") || "").trim().toUpperCase();
  useEffect(() => saveTab(tab), [tab]);

  const update = useCallback(
    (patch: { tab?: ScorecardTab; ticker?: string | null }) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (patch.tab) next.set("tab", patch.tab);
          if (patch.ticker !== undefined) {
            const t = (patch.ticker || "").trim().toUpperCase();
            if (t) next.set("ticker", t);
            else next.delete("ticker");
          }
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );

  const [universe, reloadUniverse] = useResource(true, "universe", () => api.scorecardUniverse({ limit: UNIVERSE_LIMIT }));
  const [spec] = useResource(true, "spec", () => api.scorecardSpec());
  const [evaluation, reloadEvaluation] = useResource(tab === "evaluation", "evaluation", () => api.scorecardEvaluation());
  const [detail, reloadDetail] = useResource(tab === "ticker" && !!ticker, ticker, () => api.scorecard(ticker, { months: HISTORY_MONTHS }));
  const [stocks] = useResource(tab === "ticker", "stocks", () => api.listStocks());
  const [metric, setMetric] = useState<HistoryMetric>("universe_percentile");

  // The kill switch answers every scorecard route the same way, so the
  // first read is enough to know the page is not there.
  if (universe?.state === "disabled") {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Fundamental Factor Scorecard</h1>
        <DisabledCard detail={universe.detail} />
      </div>
    );
  }

  const run = universe?.state === "ok" ? universe.data : null;
  const csvHref = run && !wallOn ? scorecardExportUrl({ format: "csv", version: run.version_key, as_of: run.as_of }) : undefined;
  const jsonHref = run && !wallOn ? scorecardExportUrl({ format: "json", version: run.version_key, as_of: run.as_of }) : undefined;
  const specData = spec?.state === "ok" ? spec.data : null;
  const currentDetail: ScorecardDetail | null = detail?.state === "ok" && detail.key === ticker ? detail.data : null;
  const stockUniverse: CompanyOut[] = stocks?.state === "ok" ? stocks.data : [];

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Fundamental Factor Scorecard</h1>
        <p className="text-slate-400 text-sm mt-1 max-w-prose">
          A versioned, sector-neutral read of reported fundamentals across the curated universe. Scores run 0–100 where 50 = z of 0 (sector-neutral
          mean); observed inputs are always shown apart from the model read, and anything the run could not compute is n/a with its reason, never
          zero. Research and education only — a rank is a research queue, not a recommendation.
        </p>
        <p className="text-xs text-slate-500 mt-1" data-testid="spec-label">
          {specData ? (
            <>
              Spec <span className="badge-neutral">{specData.version_key}</span> · {specData.score_scale || SCORECARD_SCORE_SCALE_LONG}
            </>
          ) : (
            <>Spec {na("methodology not loaded yet")}</>
          )}
        </p>
      </div>

      <div className="card-tight flex flex-wrap gap-2" role="tablist" aria-label="Scorecard views">
        {SCORECARD_TABS.map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            aria-selected={tab === t}
            aria-controls={`scorecard-panel-${t}`}
            data-testid={`tab-${t}`}
            onClick={() => update({ tab: t })}
            className={`px-3 py-1.5 rounded-md text-sm border ${tab === t ? "border-accent-600 text-accent-500 bg-accent-600/15" : "border-ink-700 text-slate-300 hover:bg-ink-800"}`}
          >
            {TAB_LABEL[t]}
          </button>
        ))}
        <span className="text-xs text-slate-500 self-center ml-2">{TAB_BLURB[tab]}</span>
      </div>

      <div role="tabpanel" id={`scorecard-panel-${tab}`} aria-label={TAB_LABEL[tab]} className="space-y-4">
        {tab === "universe" && (
          <>
            {run && (
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400" data-testid="universe-run">
                <span>
                  Run <span className="font-mono">{run.run_id}</span> · as of {run.as_of}
                  {run.is_month_end ? " (month-end)" : ""}
                  {typeof run.scored === "number" ? ` · ${run.scored} scored` : ""}
                  {typeof run.insufficient === "number" ? ` · ${run.insufficient} insufficient` : ""}
                  {run.stale && (
                    <span className="badge-mixed ml-2" data-testid="universe-stale">
                      stale
                    </span>
                  )}
                </span>
                {jsonHref && (
                  <a href={jsonHref} className="underline text-accent-500" data-testid="export-json-link" title={`Frozen column order, contract ${SCORECARD_EXPORT_CONTRACT}`}>
                    Export JSON (contract {SCORECARD_EXPORT_CONTRACT})
                  </a>
                )}
                {wallOn && <ExportButtons run={run} />}
              </div>
            )}
            {run ? (
              <ScorecardUniverseTable universe={run} onSelect={(t) => update({ tab: "ticker", ticker: t })} exportHref={csvHref} />
            ) : universe?.state === "missing" ? (
              <ScorecardUniverseTable universe={null} />
            ) : (
              <ResourceState res={universe} onRetry={reloadUniverse} missing={null} />
            )}
            {run && (
              <p className="text-[11px] text-slate-500">
                Exports stream the run under contract {SCORECARD_EXPORT_CONTRACT} (fixed column order; a new order is a new contract). The URL carries no credential:
                {wallOn ? " the page fetches the file with your session and saves it locally;" : " the link is a plain download;"} a deployment that requires an export
                token expects downstream systems to present it as a bearer header.
              </p>
            )}
            {specData && <SpecBlock spec={specData} />}
          </>
        )}

        {tab === "ticker" && (
          <>
            <div className="max-w-sm">
              <TickerPicker
                value={ticker}
                onChange={(t) => update({ ticker: t })}
                universe={stockUniverse}
                loading={stocks?.state === "loading"}
                placeholder="Ticker (e.g. COST)"
              />
            </div>
            {!ticker && (
              <p className="text-sm text-slate-400" data-testid="ticker-empty">
                Pick a name, or open one from the Universe tab. The view shows the observed inputs beside the {specData?.version_key ?? "fs-v1"} model read and the
                month-end history of its rank.
              </p>
            )}
            {ticker && !currentDetail && (
              <ResourceState
                res={detail && detail.key === ticker ? detail : null}
                onRetry={reloadDetail}
                missing={
                  <div className="card-tight text-sm text-slate-400" role="status" data-testid="ticker-missing">
                    {na(`no scorecard for ${ticker}`)}. No succeeded run has scored this name — it may be outside the curated universe or lack the annual statements the
                    spec needs. <Link to={`/app/research?ticker=${encodeURIComponent(ticker)}`} className="underline text-accent-500">Open research</Link>
                  </div>
                }
              />
            )}
            {currentDetail && (
              <>
                <ScorecardPanel scorecard={currentDetail} />
                {Array.isArray(currentDetail.notes) && currentDetail.notes.length > 0 && (
                  <div className="card-tight text-xs" data-testid="ticker-notes">
                    <div className="text-slate-400 uppercase tracking-widest text-[10px] mb-1">Worker notes (verbatim)</div>
                    <ul className="list-disc pl-4 text-slate-300 space-y-0.5">
                      {currentDetail.notes.map((n) => (
                        <li key={n}>{n}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <div className="card space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="section-title">Month-end history</div>
                    <label className="text-xs text-slate-400 flex items-center gap-2">
                      Series
                      <select className="input !py-1 text-xs" value={metric} onChange={(e) => setMetric(e.target.value as HistoryMetric)} data-testid="history-metric">
                        <option value="universe_percentile">Universe percentile</option>
                        <option value="sector_percentile">Sector percentile</option>
                        <option value="overall_score">Overall score</option>
                      </select>
                    </label>
                  </div>
                  <ScorecardHistoryChart history={historyFromDetail(currentDetail)} metric={metric} />
                </div>
                <p className="text-xs text-slate-500">
                  <Link to={`/app/research?ticker=${encodeURIComponent(ticker)}`} className="underline text-accent-500">
                    Open the {ticker} memo
                  </Link>{" "}
                  — the memo carries this summary and flags when its narrative disagrees with the quant read.
                </p>
              </>
            )}
          </>
        )}

        {tab === "evaluation" && (
          <>
            {evaluation?.state === "ok" ? (
              <>
                <Minimums evaluation={evaluation.data} />
                {evaluation.data.note && (
                  <p className="card-tight border-warn-500/40 text-xs text-slate-200" role="note" data-testid="evaluation-note">
                    {evaluation.data.note}
                  </p>
                )}
                <ResponseCaveats evaluation={evaluation.data} />
                <ScorecardEvaluation evaluation={evaluation.data} caveatsRenderedAbove={caveatsCoverCards(evaluation.data)} />
              </>
            ) : (
              <ResourceState res={evaluation} onRetry={reloadEvaluation} missing={<ScorecardEvaluation evaluation={null} />} />
            )}
          </>
        )}
      </div>
    </div>
  );
}
