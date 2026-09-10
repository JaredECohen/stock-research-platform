import React, { useMemo, useState } from "react";
import { SCORECARD_EXPORT_CONTRACT, SCORECARD_FAMILIES, SCORECARD_SCORE_SCALE_LONG, type ScorecardUniverse, type ScorecardUniverseRow } from "@/types/scorecard";
import { familyLabel, fmtCoverage, fmtPercentile, fmtScore, fmtZ, humanize, isNum, na, scoreTone } from "./format";

/**
 * The universe on one as-of date: one row per ticker with the 0–100 score,
 * universe and sector percentile, coverage and the eight family scores as
 * mini-bars. Headers sort (click or Enter/Space) and announce it through
 * `aria-sort`; an unscored value sorts last in either direction because a
 * gap is not a small number, and every gap prints as "n/a (reason)" in the
 * cell itself — a tooltip is not a reason a screen reader or a copy-paste
 * can see. Filters: sector and minimum coverage. The export link, when the
 * page supplies one, must carry `contract=v1`.
 */
export interface ScorecardUniverseTableProps {
  universe: ScorecardUniverse | null | undefined;
  /** Row selection (ticker view). */
  onSelect?: (ticker: string) => void;
  /** `/api/scorecard/export?...&contract=v1` for the displayed as_of. */
  exportHref?: string;
  /** Initial minimum coverage filter (0–1). */
  initialMinCoverage?: number;
  className?: string;
}

type FamilyKey = (typeof SCORECARD_FAMILIES)[number];
type SortKey = "rank" | "ticker" | "sector" | "overall_score" | "universe_percentile" | "sector_percentile" | "coverage" | FamilyKey;
type SortDir = "ascending" | "descending";

const FOCUS = "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 rounded-sm";
const NUMERIC_DEFAULT_DESC: SortKey[] = ["overall_score", "universe_percentile", "sector_percentile", "coverage", ...SCORECARD_FAMILIES];

function valueOf(row: ScorecardUniverseRow, key: SortKey): number | string | null {
  switch (key) {
    case "rank":
      return isNum(row.rank) ? row.rank : null;
    case "ticker":
      return row.ticker;
    case "sector":
      return row.sector ?? null;
    case "overall_score":
    case "universe_percentile":
    case "sector_percentile":
    case "coverage": {
      const v = row[key];
      return isNum(v) ? v : null;
    }
    default: {
      const v = row.category_score?.[key];
      return isNum(v) ? v : null;
    }
  }
}

export function compareRows(a: ScorecardUniverseRow, b: ScorecardUniverseRow, key: SortKey, dir: SortDir): number {
  const sign = dir === "ascending" ? 1 : -1;
  const va = valueOf(a, key);
  const vb = valueOf(b, key);
  // Missing values sort last regardless of direction.
  if (va === null && vb === null) return a.ticker < b.ticker ? -1 : 1;
  if (va === null) return 1;
  if (vb === null) return -1;
  if (typeof va === "string" || typeof vb === "string") {
    const sa = String(va);
    const sb = String(vb);
    return sa < sb ? -sign : sa > sb ? sign : 0;
  }
  return (va - vb) * sign;
}

// The universe row does not say *why* a family is unscored (the sector mask
// and the coverage floor both yield null), so the cell names the two
// possibilities rather than pretending to know; the detail view has the
// per-feature reason.
const FAMILY_MISSING = "not scored";
const FAMILY_MISSING_TITLE = "Family not scored for this name: masked for its sector, or fewer than half of its inputs were available. Open the ticker for the per-feature reason.";

function MiniBar({ value }: { value: number | null | undefined }) {
  const has = isNum(value);
  return (
    <span className="inline-flex items-center gap-1" title={has ? `${value.toFixed(0)} / 100` : FAMILY_MISSING_TITLE}>
      <span aria-hidden="true" className="inline-block w-10 h-1.5 rounded bg-ink-700 overflow-hidden align-middle">
        {has && <span className="block h-full bg-accent-600" style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />}
      </span>
      <span className={`font-mono text-[11px] ${has ? "text-slate-300" : "text-slate-500"}`} data-missing={has ? undefined : "true"}>
        {has ? value.toFixed(0) : na(FAMILY_MISSING)}
      </span>
    </span>
  );
}

export default function ScorecardUniverseTable({ universe, onSelect, exportHref, initialMinCoverage = 0, className = "" }: ScorecardUniverseTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortDir, setSortDir] = useState<SortDir>("ascending");
  const [sector, setSector] = useState<string>("");
  const [minCoverage, setMinCoverage] = useState<number>(initialMinCoverage);

  const rows = universe?.rows ?? [];
  const sectors = useMemo(() => Array.from(new Set(rows.map((r) => r.sector).filter((s): s is string => !!s))).sort(), [rows]);
  const filtered = useMemo(() => rows.filter((r) => (!sector || r.sector === sector) && (isNum(r.coverage) ? r.coverage : 0) >= minCoverage), [rows, sector, minCoverage]);
  const sorted = useMemo(() => [...filtered].sort((a, b) => compareRows(a, b, sortKey, sortDir)), [filtered, sortKey, sortDir]);

  function toggle(key: SortKey) {
    if (key === sortKey) setSortDir((d) => (d === "ascending" ? "descending" : "ascending"));
    else {
      setSortKey(key);
      // Scores read best-first; names and ranks read A→Z / 1→N.
      setSortDir(NUMERIC_DEFAULT_DESC.includes(key) ? "descending" : "ascending");
    }
  }

  // `key` is set here because the family headers are rendered from a map.
  const header = (key: SortKey, label: React.ReactNode, align: "left" | "right" = "right", title?: string) => (
    <th key={key} scope="col" aria-sort={sortKey === key ? sortDir : "none"} className={`${align === "left" ? "text-left" : "text-right"} px-2 py-1 whitespace-nowrap`} title={title}>
      <button type="button" onClick={() => toggle(key)} className={`inline-flex items-center gap-1 ${align === "right" ? "justify-end w-full" : ""} ${FOCUS}`}>
        <span>{label}</span>
        <span aria-hidden="true" className="text-slate-500">
          {sortKey === key ? (sortDir === "ascending" ? "▲" : "▼") : "↕"}
        </span>
      </button>
    </th>
  );

  if (!universe) {
    return (
      <div className={`card-tight text-xs text-slate-400 ${className}`} role="status" data-testid="universe-empty">
        {na("no scorecard run yet")}. The worker writes the first universe after the daily scorecard loop runs.
      </div>
    );
  }

  const exportOk = !!exportHref && exportHref.includes(`contract=${SCORECARD_EXPORT_CONTRACT}`);

  return (
    <div className={`space-y-3 ${className}`} data-testid="scorecard-universe">
      <div className="flex flex-wrap items-end justify-between gap-3 text-xs">
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1 text-slate-400">
            Sector
            <select className="input !py-1 text-xs" value={sector} onChange={(e) => setSector(e.target.value)} data-testid="sector-filter">
              <option value="">All sectors</option>
              {sectors.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-slate-400">
            Min coverage
            <select className="input !py-1 text-xs" value={String(minCoverage)} onChange={(e) => setMinCoverage(Number(e.target.value))} data-testid="coverage-filter">
              {[0, 0.5, 0.6, 0.8].map((c) => (
                <option key={c} value={String(c)}>
                  {c === 0 ? "Any" : `≥ ${Math.round(c * 100)}%`}
                </option>
              ))}
            </select>
          </label>
          <span className="text-slate-500 pb-1" data-testid="universe-count">
            {sorted.length} of {universe.universe_size} names · {universe.version_key} · as of {universe.as_of}
          </span>
        </div>
        {exportOk && (
          <a href={exportHref} className={`btn-ghost !py-1 !px-2 text-xs ${FOCUS}`} data-testid="export-link" title={`Frozen column order, contract ${SCORECARD_EXPORT_CONTRACT}`}>
            Export CSV (contract {SCORECARD_EXPORT_CONTRACT})
          </a>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full text-sm" data-testid="universe-table">
          <caption className="text-left text-xs text-slate-400 mb-2">
            Scorecard universe, {universe.version_key} as of {universe.as_of}: {universe.universe_size} names ranked by overall score. An unscored name shows n/a and sorts last.
          </caption>
          <thead className="text-xs uppercase tracking-wider text-slate-400">
            <tr>
              {header("rank", "Rank", "right")}
              {header("ticker", "Ticker", "left")}
              {header("sector", "Sector", "left")}
              {header("overall_score", "Score", "right", SCORECARD_SCORE_SCALE_LONG)}
              {header("universe_percentile", "Univ pct", "right", "Percentile rank across the universe")}
              {header("sector_percentile", "Sector pct", "right", "Percentile rank within sector")}
              {header("coverage", "Coverage", "right", "Share of applicable inputs available")}
              {SCORECARD_FAMILIES.map((f) => header(f, familyLabel(f), "left"))}
              <th scope="col" className="text-left px-2 py-1 whitespace-nowrap">
                Top +
              </th>
              <th scope="col" className="text-left px-2 py-1 whitespace-nowrap">
                Top −
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 && (
              <tr>
                <td colSpan={9 + SCORECARD_FAMILIES.length} className="px-2 py-3 text-xs text-slate-400" role="status">
                  No names match the filters.
                </td>
              </tr>
            )}
            {sorted.map((r) => {
              const unscored = !isNum(r.overall_score);
              const pos = r.top_positive?.[0];
              const neg = r.top_negative?.[0];
              // No overall means no rank and no contributor list; a scored
              // name with an empty list simply had none on that side.
              const noContributor = unscored ? "overall not scored" : "none listed";
              return (
                <tr key={r.ticker} className="border-t border-ink-700/60 table-row-hover" data-testid={`row-${r.ticker}`} data-ticker={r.ticker} data-unscored={unscored ? "true" : undefined}>
                  <td className="px-2 py-1 text-right font-mono text-slate-400 whitespace-nowrap" data-missing={isNum(r.rank) ? undefined : "true"}>
                    {isNum(r.rank) ? r.rank : na(unscored ? "unscored" : "not ranked")}
                  </td>
                  <th scope="row" className="px-2 py-1 text-left font-normal whitespace-nowrap">
                    {onSelect ? (
                      <button type="button" onClick={() => onSelect(r.ticker)} className={`font-mono text-accent-500 underline underline-offset-2 ${FOCUS}`}>
                        {r.ticker}
                      </button>
                    ) : (
                      <span className="font-mono text-slate-100">{r.ticker}</span>
                    )}
                    {r.company_name && <span className="ml-2 text-xs text-slate-500">{r.company_name}</span>}
                  </th>
                  <td className="px-2 py-1 text-xs text-slate-400 whitespace-nowrap">{r.sector ?? na("no sector")}</td>
                  <td className={`px-2 py-1 text-right font-mono ${scoreTone(r.overall_score)}`} data-missing={unscored ? "true" : undefined}>
                    {fmtScore(r.overall_score, "insufficient coverage")}
                  </td>
                  <td className="px-2 py-1 text-right font-mono text-slate-300">{fmtPercentile(r.universe_percentile, "unranked")}</td>
                  <td className="px-2 py-1 text-right font-mono text-slate-300">{fmtPercentile(r.sector_percentile, "unranked")}</td>
                  <td className="px-2 py-1 text-right font-mono text-slate-400">{fmtCoverage(r.coverage)}</td>
                  {SCORECARD_FAMILIES.map((f) => (
                    <td key={f} className="px-2 py-1 whitespace-nowrap" data-family={f}>
                      <MiniBar value={r.category_score?.[f]} />
                    </td>
                  ))}
                  <td className="px-2 py-1 text-xs whitespace-nowrap text-slate-400">
                    {pos ? (
                      <span title={`${familyLabel(pos.family)} · z ${fmtZ(pos.z)}`}>
                        {humanize(pos.feature)} <span className="text-accent-500 font-mono">{fmtZ(pos.z)}</span>
                      </span>
                    ) : (
                      <span data-missing="true">{na(noContributor)}</span>
                    )}
                  </td>
                  <td className="px-2 py-1 text-xs whitespace-nowrap text-slate-400">
                    {neg ? (
                      <span title={`${familyLabel(neg.family)} · z ${fmtZ(neg.z)}`}>
                        {humanize(neg.feature)} <span className="text-danger-500 font-mono">{fmtZ(neg.z)}</span>
                      </span>
                    ) : (
                      <span data-missing="true">{na(noContributor)}</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-slate-500">Ranks are a model read of reported fundamentals relative to today's constituent list — a research queue, not a buy or sell list.</p>
    </div>
  );
}
