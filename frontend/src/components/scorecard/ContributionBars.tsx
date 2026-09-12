import React, { useMemo } from "react";
import type { ScorecardContribution, ScorecardFeature } from "@/types/scorecard";
import { familyLabel, featureMissingReason, fmtZ, humanize, isNum } from "./format";

/**
 * Per-feature contributions to the overall z as horizontal bars. Every bar
 * is paired with its number in text, so colour and length are never the
 * only carrier; a feature whose contribution is null prints "n/a (reason)"
 * and draws no bar, because a missing input is not a zero contribution.
 * With `overallZ` the footer shows the listed contributions' sum next to
 * the overall z — they are equal when the list is complete, which is the
 * additivity invariant the backend guarantees (plan §4.2).
 */
export interface ContributionItem {
  feature: string;
  family: string;
  z: number | null;
  contribution: number | null;
  /** Why `contribution` is null; defaults to a generic reason. */
  reason?: string;
}

export interface ContributionBarsProps {
  items: ContributionItem[];
  title?: string;
  /** When given, the footer compares the sum of listed contributions to it. */
  overallZ?: number | null;
  emptyText?: string;
  className?: string;
  "data-testid"?: string;
}

/** Adapts the API's top-N lists (numbers only). */
export function fromContributions(rows: ScorecardContribution[]): ContributionItem[] {
  return rows.map((r) => ({ feature: r.feature, family: r.family, z: r.z, contribution: r.contribution }));
}

/** Adapts the full feature list, carrying the missing reason through. */
export function fromFeatures(rows: ScorecardFeature[]): ContributionItem[] {
  return rows.map((f) => ({ feature: f.name, family: f.family, z: f.z, contribution: f.contribution, reason: featureMissingReason(f) }));
}

function fmtContribution(v: number | null, reason: string): string {
  if (!isNum(v)) return `n/a (${reason})`;
  const s = v.toFixed(3);
  return v > 0 ? `+${s}` : s;
}

export default function ContributionBars({ items, title, overallZ, emptyText = "No contributions to show.", className = "", ...rest }: ContributionBarsProps) {
  const max = useMemo(() => items.reduce((m, it) => (isNum(it.contribution) ? Math.max(m, Math.abs(it.contribution)) : m), 0), [items]);
  const sum = useMemo(() => {
    const nums = items.map((it) => it.contribution).filter(isNum);
    return nums.length === 0 ? null : nums.reduce((a, b) => a + b, 0);
  }, [items]);
  const testId = rest["data-testid"] ?? "contribution-bars";

  return (
    <div className={className} data-testid={testId}>
      {title && <div className="section-title mb-2">{title}</div>}
      {items.length === 0 ? (
        <p className="text-xs text-slate-400" role="status">
          {emptyText}
        </p>
      ) : (
        <ul className="space-y-1.5">
          {items.map((it) => {
            const c = it.contribution;
            const has = isNum(c);
            const width = has && max > 0 ? Math.max(2, (Math.abs(c) / max) * 100) : 0;
            const positive = has && c > 0;
            return (
              <li key={`${it.family}:${it.feature}`} className="text-xs" data-testid={`contribution-${it.feature}`} data-missing={has ? undefined : "true"}>
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate">
                    <span className="text-slate-200">{humanize(it.feature)}</span>
                    <span className="text-slate-500"> · {familyLabel(it.family)}</span>
                  </span>
                  <span className="font-mono whitespace-nowrap text-slate-300">
                    <span className="sr-only">z </span>
                    <span title="Sector-neutral z-score">{fmtZ(it.z, it.reason ?? "not computed")}</span>
                    <span className="text-slate-500"> · </span>
                    <span className="sr-only">contribution </span>
                    <span title="Contribution to the overall z" className={has ? (positive ? "text-accent-500" : "text-danger-500") : "text-slate-500"}>
                      {fmtContribution(c, it.reason ?? "not computed")}
                    </span>
                  </span>
                </div>
                <div aria-hidden="true" className="relative h-1.5 mt-0.5 rounded bg-ink-700 overflow-hidden">
                  {has && (
                    <div
                      className={`absolute top-0 h-full ${positive ? "left-1/2 bg-accent-600" : "right-1/2 bg-danger-500"}`}
                      style={{ width: `${width / 2}%` }}
                      data-testid={`bar-${it.feature}`}
                    />
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
      {overallZ !== undefined && items.length > 0 && (
        <p className="mt-2 text-[11px] text-slate-400 font-mono" data-testid="contribution-sum">
          Σ listed contributions {sum === null ? "n/a (none computed)" : fmtZ(sum)} · overall z {fmtZ(overallZ, "insufficient coverage")}
        </p>
      )}
    </div>
  );
}
