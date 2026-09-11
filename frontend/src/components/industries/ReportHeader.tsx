import React from "react";
import { AlertTriangle, Clock } from "lucide-react";
import type { IndustryReport } from "@/types/industries";
import { degradationText, fmtDate, fmtDateTime, fmtShare, humanize, isNum, na } from "./format";


/**
 * Everything a reader needs before they read a number: which taxonomy
 * edition this is, what it is as of, how much of the group it could
 * actually price, how the statistics were weighted and against what
 * benchmarks — and, when the last refresh failed, that it did and why.
 *
 * Two things this deliberately does NOT do:
 *
 *   * collapse membership into coverage. `n_constituents` is who is in
 *     the group; `n_with_prices` is how many of them had a price series.
 *     Both are printed, with the excluded names' reasons;
 *   * soften a failed refresh. A stale edition keeps its content — that
 *     is the point of keeping the last good one — but the badge names
 *     the attempt, its error and how many tries it has left, because a
 *     page that quietly serves last week's numbers as this week's is the
 *     failure mode this feature has to avoid.
 */

function Badge({
  tone,
  icon: Icon,
  children,
  testId,
}: {
  tone: "warn" | "mixed";
  icon: typeof Clock;
  children: React.ReactNode;
  testId: string;
}) {
  const cls = tone === "warn" ? "border-warn-500/50 text-warn-500" : "border-ink-700 text-slate-300";
  return (
    <span className={`badge inline-flex items-center gap-1 ${cls}`} data-testid={testId}>
      <Icon size={12} aria-hidden />
      {children}
    </span>
  );
}

/** The benchmark definitions the statistics row recorded, verbatim. A
 *  benchmark the run could not build is listed with `available: false`
 *  rather than dropped — its absence is why a relative number is missing. */
function benchmarks(report: IndustryReport): Array<{ id: string; definition: string; available: boolean; n?: number }> {
  const facts = (report.payload?.sections?.performance?.facts ?? {}) as Record<string, unknown>;
  const raw = Array.isArray(facts.benchmarks) ? (facts.benchmarks as Array<Record<string, unknown>>) : [];
  return raw.map((b) => ({
    id: String(b.id ?? ""),
    definition: String(b.definition ?? ""),
    available: b.available !== false,
    n: isNum(b.n) ? b.n : undefined,
  }));
}

/** One recorded error, as a line. The shape is the writer's — usually
 *  `{section, type, message}` — and an unrecognised one is printed whole
 *  rather than summarised away. */
function errorText(e: unknown): string {
  if (typeof e === "string") return e;
  if (e && typeof e === "object") {
    const r = e as Record<string, unknown>;
    const parts = [r.section, r.type ?? r.error_type, r.message ?? r.error_message].filter(Boolean).map(String);
    if (parts.length > 0) return parts.join(": ");
  }
  return JSON.stringify(e);
}

export interface ReportHeaderProps {
  report: IndustryReport;
  /** `GET /api/industries/taxonomy`'s attribution — the MSCI / S&P line
   *  for the codes and names. The report's own `attribution` is the
   *  analyst mandate's ("original research, not licensed GICS content"),
   *  a different claim, so both are printed rather than one standing in
   *  for the other. */
  taxonomyAttribution?: string;
  /** `GET /api/industries/{code}/companies.security_reference_caveat` —
   *  the research map author's own label for the symbol crosswalk.
   *  Rendered VERBATIM: it is the rights statement that lets the symbol
   *  mapping be shown at all, and a paraphrase is not it. */
  securityReferenceCaveat?: string;
  className?: string;
}

export default function ReportHeader({
  report,
  taxonomyAttribution = "",
  securityReferenceCaveat = "",
  className = "",
}: ReportHeaderProps) {
  const coverage = (report.coverage ?? {}) as Record<string, unknown>;
  const nMembers = coverage.n_constituents;
  const nPriced = coverage.n_with_prices;
  const excluded = Array.isArray(coverage.excluded) ? (coverage.excluded as Array<Record<string, unknown>>) : [];
  // The ONE reason a reader gets for an edition written without a
  // statistics row. Every field below that would otherwise invent its own
  // ("weighting not recorded on the statistics row") defers to it: the
  // API supplies `stats_unavailable_reason` precisely so no caller has to
  // guess, and an invented reason is a claim the server never made.
  const statsReason = report.stats
    ? null
    : report.stats_unavailable_reason || "no statistics row on this edition, and no reason recorded";
  const method = (report.stats?.method ?? {}) as Record<string, unknown>;
  const weighting = Array.isArray(method.weighting) ? (method.weighting as string[]) : [];
  const breadth = (method.breadth_mean_window ?? {}) as Record<string, unknown>;
  const bms = benchmarks(report);
  const llmCalls = (report.generation as Record<string, unknown>)?.llm_calls;
  const stale = report.stale === true;
  const attempt = report.last_attempt;
  const failedAttempt = attempt && attempt.status !== "succeeded" ? attempt : null;

  return (
    <header className={`space-y-3 ${className}`} data-testid="industry-report-header">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">
            <span className="font-mono text-sm text-slate-500 mr-2">{report.code}</span>
            {report.name}
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Edition v{report.version} · {report.period_key || na("no period key")} · as of{" "}
            <span data-testid="report-as-of">{fmtDateTime(report.as_of, "no as-of on this edition")}</span> · taxonomy{" "}
            {report.taxonomy_version}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {stale && (
            <Badge tone="warn" icon={Clock} testId="badge-stale">
              Stale — {report.stale_reason || "reason not recorded"}
            </Badge>
          )}
          {report.degraded.length > 0 && (
            <Badge tone="mixed" icon={AlertTriangle} testId="badge-degraded">
              Degraded ({report.degraded.length})
            </Badge>
          )}
        </div>
      </div>

      {failedAttempt && (
        <div className="card-tight border-warn-500/40 text-xs space-y-1" role="status" data-testid="last-attempt">
          <div className="text-slate-200">
            The most recent refresh ({failedAttempt.period_key || "period not recorded"}) {failedAttempt.status} —
            attempt {failedAttempt.attempts} of {failedAttempt.max_attempts}, {fmtDateTime(failedAttempt.at)}.
          </div>
          <div className="text-slate-400">
            {failedAttempt.error_type || "error type not recorded"}:{" "}
            {failedAttempt.error_message || "no message recorded"}
          </div>
          <div className="text-slate-500">
            The edition below is the last one that passed validation; it has not been replaced.
          </div>
        </div>
      )}

      {statsReason && (
        <div className="card-tight text-xs" role="status" data-testid="stats-unavailable">
          <span className="text-slate-200">No statistics row on this edition</span>{" "}
          <span className="text-slate-400">— {statsReason}.</span>{" "}
          <span className="text-slate-500">
            The observed numbers below are whatever the writer could record without one; the rest say so.
          </span>
        </div>
      )}

      {report.degraded.length > 0 && (
        <details className="card-tight text-xs" data-testid="degraded-list">
          <summary className="cursor-pointer text-slate-300">
            {report.degraded.length} degradation{report.degraded.length === 1 ? "" : "s"} on this edition
          </summary>
          <ul className="mt-1 space-y-0.5 text-slate-400">
            {report.degraded.map((d) => (
              <li key={d}>
                {degradationText(d)} <span className="font-mono text-[10px] text-slate-600">{d}</span>
              </li>
            ))}
          </ul>
          {/* The labels say WHAT degraded; `errors` is the detail behind
              them, and an edition that carries one and shows only the
              label is asking the reader to take the degradation on
              faith. */}
          {report.errors.length > 0 && (
            <div className="mt-2" data-testid="report-errors">
              <div className="text-slate-500 uppercase tracking-widest text-[10px]">
                {report.errors.length} error{report.errors.length === 1 ? "" : "s"} recorded on this run
              </div>
              <ul className="mt-1 space-y-0.5 text-slate-400">
                {report.errors.map((e, i) => (
                  <li key={i} className="font-mono text-[10px] break-words">
                    {errorText(e)}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </details>
      )}

      <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 text-xs" data-testid="report-coverage">
        <div className="card-tight">
          <dt className="text-slate-500 uppercase tracking-widest text-[10px]">Membership</dt>
          <dd className="text-slate-200">
            {isNum(nMembers) ? `${nMembers} classified constituents` : na("membership not recorded")}
          </dd>
        </div>
        <div className="card-tight">
          <dt className="text-slate-500 uppercase tracking-widest text-[10px]">Priced this period</dt>
          <dd className="text-slate-200" data-testid="coverage-priced">
            {isNum(nPriced) && isNum(nMembers)
              ? `${nPriced} of ${nMembers} (${fmtShare(coverage.coverage, "coverage not recorded")})`
              : na("price coverage not recorded")}
          </dd>
          {excluded.length > 0 && (
            <dd className="text-slate-500 mt-1" data-testid="coverage-excluded">
              Excluded:{" "}
              {excluded
                .map((e) => `${String(e.ticker ?? "?")} (${String(e.reason ?? "reason not recorded")})`)
                .join(", ")}
            </dd>
          )}
        </div>
        <div className="card-tight">
          <dt className="text-slate-500 uppercase tracking-widest text-[10px]">Weighting</dt>
          <dd className="text-slate-200" data-testid="method-weighting">
            {weighting.length > 0 ? weighting.join(" and ") : na(statsReason ?? "weighting not recorded on the statistics row")}
          </dd>
          <dd className="text-slate-500 mt-1">
            {isNum(breadth.sessions)
              ? `Breadth mean window: ${breadth.sessions} sessions${breadth.basis ? ` — ${String(breadth.basis)}` : ""}`
              : na(statsReason ?? "breadth window not recorded")}
          </dd>
        </div>
        <div className="card-tight">
          <dt className="text-slate-500 uppercase tracking-widest text-[10px]">Prices</dt>
          <dd className="text-slate-200">
            {coverage.prices_max_date
              ? `Latest close ${fmtDate(String(coverage.prices_max_date))}`
              : na("no close on file for this period")}
          </dd>
          <dd className="text-slate-500 mt-1">
            Generated {fmtDateTime(report.generated_at)} · narrative:{" "}
            {(report.generation as Record<string, unknown>)?.generation_mode
              ? humanize(String((report.generation as Record<string, unknown>).generation_mode))
              : na("generation mode not recorded")}
          </dd>
          {/* What the edition cost to write. Zero is a real number here —
              a deterministic edition made no calls — so it is printed
              with the call count beside it rather than left out, which
              would read as "not measured". */}
          <dd className="text-slate-500 mt-1" data-testid="llm-cost">
            LLM cost{" "}
            {isNum(report.llm_cost_usd)
              ? `$${report.llm_cost_usd.toFixed(4)}`
              : na("cost not recorded on this edition")}
            {isNum(llmCalls) ? ` over ${llmCalls} call${llmCalls === 1 ? "" : "s"}` : ""}
          </dd>
        </div>
      </dl>

      <div className="card-tight text-xs" data-testid="benchmark-definitions">
        <div className="text-slate-500 uppercase tracking-widest text-[10px] mb-1">Benchmarks</div>
        {bms.length === 0 ? (
          <p className="text-slate-400">{na("this edition recorded no benchmark definitions")}</p>
        ) : (
          <ul className="space-y-0.5 text-slate-300">
            {bms.map((b) => (
              <li key={b.id} data-testid={`benchmark-${b.id}`}>
                <span className="font-mono text-[11px] text-slate-400">{b.id}</span> — {b.definition || na("no definition recorded")}
                {b.available ? (
                  isNum(b.n) ? <span className="text-slate-500"> (n={b.n})</span> : null
                ) : (
                  <span className="text-warn-500"> — not available this period</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="text-[11px] leading-snug text-slate-500 space-y-1" data-testid="report-attribution">
        <p>
          {report.attribution} {report.mapping_caveat ? `Mappings are ${report.mapping_caveat}.` : ""}
        </p>
        {/* Printed only when it is a DIFFERENT claim. The report row's
            `attribution` falls back to the registry's own string, so on
            most editions these two are the same sentence and printing
            both just looks like a page that cannot count. */}
        {taxonomyAttribution && taxonomyAttribution.trim() !== report.attribution.trim() && (
          <p data-testid="taxonomy-attribution">{taxonomyAttribution}</p>
        )}
        {securityReferenceCaveat && (
          <p data-testid="security-reference-caveat">Research map symbol crosswalk: {securityReferenceCaveat}</p>
        )}
      </div>
      <p className="text-[11px] leading-snug text-slate-500" data-testid="report-disclaimer">
        {report.disclaimer}
      </p>
    </header>
  );
}
