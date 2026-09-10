import React, { useMemo } from "react";
import {
  SCORECARD_FAMILIES,
  type ScorecardCategory,
  type ScorecardDetail,
  type ScorecardDisagreement,
  type ScorecardFeature,
  type ScorecardProfiles,
  type ScorecardSummary,
} from "@/types/scorecard";
import ContributionBars, { fromContributions } from "./ContributionBars";
import { familyLabel, featureMissingReason, fmtCoverage, fmtPercentile, fmtRaw, fmtScore, fmtZ, humanize, isNum, na, scoreTone } from "./format";

/**
 * One ticker's scorecard: the headline (0–100 score, universe and sector
 * percentile, coverage), the eight family composites, the top ± three
 * contributors, and — when the full detail is passed — the feature table
 * with two explicitly labelled column groups: **Observed** (the reported
 * input, its fiscal period and availability date) and **Model read** (the
 * spec's z and contribution). The two never share a column so a reader
 * cannot mistake a model number for a reported one.
 *
 * Null-safe by construction: a family the run could not score, a null
 * overall (fewer than five families), or a feature without an input all
 * render as "n/a (reason)". Nothing is drawn as 0 or "neutral". Returns
 * null when there is no scorecard at all — the caller decides what to say.
 */
export interface ScorecardPanelProps {
  scorecard: ScorecardSummary | ScorecardDetail | null | undefined;
  title?: string;
  /** Hide the feature table even when features are present (memo card). */
  compact?: boolean;
  className?: string;
}

const OVERALL_REASON = "insufficient coverage: fewer than 5 families scored";
const PROFILE_THRESHOLD = 0.5;

function hasFeatures(s: ScorecardSummary | ScorecardDetail): s is ScorecardDetail {
  return Array.isArray((s as ScorecardDetail).features);
}

/** "reads as a compounder", "reads as an early inflection", "reads as a
 *  compounder with an early inflection", "reads as neither", or n/a. A
 *  model read of the sub-composites, not a judgement about the company. */
export function profileText(p: ScorecardProfiles | null | undefined): string {
  if (!p || (!isNum(p.compounder) && !isNum(p.inflection))) return na("profile not computed");
  const compounder = isNum(p.compounder) && p.compounder >= PROFILE_THRESHOLD;
  const inflection = isNum(p.inflection) && p.inflection >= PROFILE_THRESHOLD;
  if (compounder && inflection) return "reads as a compounder with an early inflection";
  if (compounder) return "reads as a compounder";
  if (inflection) return "reads as an early inflection";
  return "reads as neither a compounder nor an inflection";
}

function directionText(d: ScorecardDisagreement["direction"]): string {
  return d === "narrative_above_quant" ? "the memo is more positive than the quant read" : "the memo is more negative than the quant read";
}

function CategoryRow({ family, cat }: { family: string; cat: ScorecardCategory | undefined }) {
  const score = cat?.score ?? null;
  const reason = !cat ? "not scored" : cat.n_available === 0 ? "no applicable inputs" : `${cat.n_available} of ${cat.n_features} inputs`;
  const width = isNum(score) ? Math.max(0, Math.min(100, score)) : 0;
  return (
    <li className="text-xs" data-testid={`category-${family}`} data-missing={isNum(score) ? undefined : "true"}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-slate-200">{familyLabel(family)}</span>
        <span className="font-mono whitespace-nowrap">
          <span className={scoreTone(score)}>{fmtScore(score, reason)}</span>
          {cat && (
            <span className="text-slate-500">
              {" "}
              · z {fmtZ(cat.z, reason)} · {fmtPercentile(cat.percentile, reason)} pct · {cat.n_available}/{cat.n_features}
            </span>
          )}
        </span>
      </div>
      <div aria-hidden="true" className="h-1.5 mt-0.5 rounded bg-ink-700 overflow-hidden">
        {isNum(score) && <div className="h-full bg-accent-600" style={{ width: `${width}%` }} />}
      </div>
    </li>
  );
}

function FeatureTable({ features, versionKey, latestPeriod, availableAt }: { features: ScorecardFeature[]; versionKey: string; latestPeriod: string | null; availableAt: string | null }) {
  const observedNote = `as of ${latestPeriod ?? "n/a (period unknown)"}, available ${availableAt ?? "n/a (availability unknown)"}`;
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-xs" data-testid="feature-table">
        <caption className="text-left text-[11px] text-slate-400 mb-1">
          Observed inputs ({observedNote}) beside the {versionKey} model read. A missing input reads n/a with its reason; it is never scored as zero.
        </caption>
        <thead className="uppercase tracking-wider text-slate-400">
          <tr>
            <th scope="col" rowSpan={2} className="text-left px-2 py-1 align-bottom">
              Feature
            </th>
            <th scope="col" rowSpan={2} className="text-left px-2 py-1 align-bottom">
              Family
            </th>
            <th scope="colgroup" colSpan={2} className="text-left px-2 py-1 border-b border-ink-700 text-slate-200" data-testid="observed-header">
              Observed
            </th>
            <th scope="colgroup" colSpan={2} className="text-left px-2 py-1 border-b border-ink-700 text-accent-500" data-testid="model-read-header">
              Model read ({versionKey})
            </th>
          </tr>
          <tr>
            <th scope="col" className="text-right px-2 py-1">
              Value
            </th>
            <th scope="col" className="text-left px-2 py-1">
              Formula
            </th>
            <th scope="col" className="text-right px-2 py-1">
              z
            </th>
            <th scope="col" className="text-right px-2 py-1">
              Contribution
            </th>
          </tr>
        </thead>
        <tbody>
          {features.map((f) => {
            const missing = !isNum(f.raw);
            const reason = featureMissingReason(f);
            return (
              <tr key={f.name} className="border-t border-ink-700/60" data-testid={`feature-${f.name}`} data-applicable={f.applicable ? "true" : "false"}>
                <th scope="row" className="text-left px-2 py-1 font-normal text-slate-200 whitespace-nowrap">
                  {humanize(f.name)}
                  {f.sign === -1 && <span className="text-slate-500 text-[10px] ml-1">(lower is better)</span>}
                </th>
                <td className="px-2 py-1 text-slate-400 whitespace-nowrap">{familyLabel(f.family)}</td>
                <td className="px-2 py-1 text-right font-mono whitespace-nowrap text-slate-200" data-missing={missing ? "true" : undefined}>
                  {fmtRaw(f.raw, f.unit, reason)}
                </td>
                <td className="px-2 py-1 font-mono text-slate-500 whitespace-nowrap">{f.formula}</td>
                <td className="px-2 py-1 text-right font-mono whitespace-nowrap text-slate-300">{fmtZ(f.z, reason)}</td>
                <td className="px-2 py-1 text-right font-mono whitespace-nowrap text-slate-300">
                  {isNum(f.contribution) ? `${f.contribution > 0 ? "+" : ""}${f.contribution.toFixed(3)}` : na(reason)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function ScorecardPanel({ scorecard, title = "Fundamental Factor Scorecard", compact = false, className = "" }: ScorecardPanelProps) {
  const families = useMemo(() => {
    if (!scorecard) return [];
    // Canonical order first, then any family the backend added that the
    // client does not know yet — shown rather than dropped.
    const known = SCORECARD_FAMILIES as readonly string[];
    const extra = Object.keys(scorecard.categories ?? {}).filter((k) => !known.includes(k) && !k.startsWith("_"));
    return [...known, ...extra];
  }, [scorecard]);

  if (!scorecard) return null;

  const detail = hasFeatures(scorecard) ? scorecard : null;
  const overallMissing = !isNum(scorecard.overall_score);
  const disagreement = scorecard.disagreement ?? null;

  return (
    <div className={`card space-y-4 ${className}`} data-testid="scorecard-panel">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="section-title">
          {title}
          {detail?.ticker ? ` · ${detail.ticker}` : ""}
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
          <span className="badge-neutral" title="Scoring spec version">
            {scorecard.version_key}
          </span>
          <span>as of {scorecard.as_of}</span>
          {scorecard.is_month_end && <span className="text-slate-500">month-end</span>}
          {scorecard.stale && (
            <span className="badge-mixed" title="The latest run is older than 45 days" data-testid="stale-badge">
              stale
            </span>
          )}
          {detail?.sector && <span className="text-slate-500">{detail.sector}</span>}
        </div>
      </div>

      <dl className="grid grid-cols-2 md:grid-cols-4 gap-2" data-testid="scorecard-headline">
        <div className="card-tight !p-2">
          <dt className="text-[10px] uppercase tracking-widest text-slate-500">Overall score</dt>
          <dd className={`text-lg font-semibold font-mono ${scoreTone(scorecard.overall_score)}`} data-testid="overall-score">
            {fmtScore(scorecard.overall_score, OVERALL_REASON)}
          </dd>
          <dd className="text-[10px] text-slate-500">0–100 · 50 = universe median · z {fmtZ(scorecard.overall_z, OVERALL_REASON)}</dd>
        </div>
        <div className="card-tight !p-2">
          <dt className="text-[10px] uppercase tracking-widest text-slate-500">Universe percentile</dt>
          <dd className="text-lg font-semibold font-mono text-slate-200" data-testid="universe-percentile">
            {fmtPercentile(scorecard.universe_percentile, overallMissing ? "unranked" : "not ranked")}
          </dd>
        </div>
        <div className="card-tight !p-2">
          <dt className="text-[10px] uppercase tracking-widest text-slate-500">Sector percentile</dt>
          <dd className="text-lg font-semibold font-mono text-slate-200" data-testid="sector-percentile">
            {fmtPercentile(scorecard.sector_percentile, overallMissing ? "unranked" : "sector too small")}
          </dd>
        </div>
        <div className="card-tight !p-2">
          <dt className="text-[10px] uppercase tracking-widest text-slate-500">Coverage</dt>
          <dd className="text-lg font-semibold font-mono text-slate-200" data-testid="coverage">
            {fmtCoverage(scorecard.coverage)}
          </dd>
          <dd className="text-[10px] text-slate-500">of applicable inputs available</dd>
        </div>
      </dl>

      <p className="text-xs text-slate-300" data-testid="profile-line">
        <span className="text-accent-500 uppercase tracking-widest text-[10px] mr-2">Model read</span>
        {profileText(scorecard.profiles)}
        <span className="text-slate-500"> — from the quality/profitability/capital-allocation/earnings-quality and growth sub-composites; a scenario label, not a recommendation.</span>
      </p>

      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <div className="section-title mb-2">Family composites</div>
          <ul className="space-y-1.5" data-testid="category-list">
            {families.map((fam) => (
              <CategoryRow key={fam} family={fam} cat={scorecard.categories?.[fam]} />
            ))}
          </ul>
        </div>
        <div className="space-y-3">
          <ContributionBars title="Top positive contributors" items={fromContributions(scorecard.top_positive ?? [])} emptyText="No positive contributors (overall not scored)." data-testid="top-positive" />
          <ContributionBars title="Top negative contributors" items={fromContributions(scorecard.top_negative ?? [])} emptyText="No negative contributors (overall not scored)." data-testid="top-negative" />
        </div>
      </div>

      {disagreement && (
        <div className="card-tight border-warn-500/40 bg-warn-500/5 text-xs space-y-1" role="note" data-testid="disagreement">
          <div className="text-warn-500 font-medium">
            Memo / scorecard disagreement · {disagreement.severity} · {humanize(disagreement.dimension).toLowerCase()}
          </div>
          <p className="text-slate-200">
            {disagreement.note} ({directionText(disagreement.direction)}; gap {disagreement.gap > 0 ? "+" : ""}
            {disagreement.gap.toFixed(0)} points.)
          </p>
          {scorecard.reconciliation ? (
            <p className="text-slate-300">
              <span className="text-slate-400">PM reconciliation: </span>
              {scorecard.reconciliation}
            </p>
          ) : (
            <p className="text-slate-400">PM reconciliation: n/a (not yet written).</p>
          )}
        </div>
      )}

      {detail && !compact && <FeatureTable features={detail.features} versionKey={scorecard.version_key} latestPeriod={scorecard.latest_period} availableAt={scorecard.data_available_at} />}

      <p className="text-[11px] text-slate-500 leading-relaxed" data-testid="scorecard-footnote">
        <span className="text-slate-300">Observed</span> figures come from reported statements (latest period {scorecard.latest_period ?? "n/a (unknown)"}, available{" "}
        {scorecard.data_available_at ?? "n/a (unknown)"}). The <span className="text-accent-500">model read</span> is spec {scorecard.version_key} scored against the universe on{" "}
        {scorecard.as_of}: sector-neutral z-scores, equal family weights. It is a scenario output for research and education, not a recommendation.
      </p>
    </div>
  );
}
