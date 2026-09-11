import React, { useMemo, useState } from "react";
import type { IndustryCompanies, IndustryCompanyRow } from "@/types/industries";
import { fmtCap, fmtDate, fmtPct, fmtPctSigned, fmtPrice, isNum, na } from "./format";

/**
 * The group's classified membership — who is in it, how each name got
 * there, and which of them the statistics row could actually price.
 *
 * The table is the place where the two counts that are easiest to
 * conflate are kept apart: the caption states membership (`count`) and
 * price coverage (`n_priced`) as separate numbers over the whole
 * membership, and every unpriced row carries the server's reason instead
 * of a zero. A capped page says how many rows it dropped.
 *
 * Provenance is a column, not a footnote: a row assigned from the
 * research map and a row derived from the data provider's own industry
 * label are different kinds of evidence, and the badge's title is the
 * API's `source_label` verbatim — including the map author's caveat that
 * it is research, not a licensed issuer mapping.
 */

type SortKey = "ticker" | "company_name" | "sub_industry_name" | "source" | "market_cap" | "weight_mcw" | "last_close" | "ret_1m";
type SortDir = "ascending" | "descending";

const NUMERIC: readonly SortKey[] = ["market_cap", "weight_mcw", "last_close", "ret_1m"];

function ret1m(row: IndustryCompanyRow): number | null {
  const v = row.returns?.["1m"];
  return isNum(v) ? v : null;
}

function cell(row: IndustryCompanyRow, key: SortKey): string | number | null {
  switch (key) {
    case "ticker":
      return row.ticker;
    case "company_name":
      return row.company_name ?? "";
    case "sub_industry_name":
      return row.sub_industry_name ?? "";
    case "source":
      return row.classification.source ?? "";
    case "market_cap":
      return row.market_cap;
    case "weight_mcw":
      return row.weight_mcw;
    case "last_close":
      return row.last_close;
    case "ret_1m":
      return ret1m(row);
    default:
      return null;
  }
}

/** Missing values sort last in BOTH directions: a name with no price is
 *  not "the smallest", and letting it float to the top of an ascending
 *  sort would read as one. */
function compare(a: IndustryCompanyRow, b: IndustryCompanyRow, key: SortKey, dir: SortDir): number {
  const av = cell(a, key);
  const bv = cell(b, key);
  const aMissing = av === null || av === "";
  const bMissing = bv === null || bv === "";
  if (aMissing && bMissing) return a.ticker.localeCompare(b.ticker);
  if (aMissing) return 1;
  if (bMissing) return -1;
  const sign = dir === "ascending" ? 1 : -1;
  if (typeof av === "number" && typeof bv === "number") return (av - bv) * sign;
  return String(av).localeCompare(String(bv)) * sign;
}

/** Where a row's group assignment came from, in two words, with the
 *  API's own label as the title. */
export function sourceBadge(row: IndustryCompanyRow): { label: string; title: string; tone: string } {
  const source = row.classification.source;
  const title = row.classification.source_label || source || "source not recorded";
  if (source === "research_map") return { label: "Research map", title, tone: "border-accent-600/40 text-accent-500" };
  if (source === "provider_alias") return { label: "Provider-derived", title, tone: "border-ink-700 text-slate-300" };
  return { label: source || "unrecorded", title, tone: "border-ink-700 text-slate-400" };
}

export interface CompaniesTableProps {
  companies: IndustryCompanies;
  /** `payload.sections.companies.facts.n_priced` from the edition being
   *  read, when there is one. The edition counts its priced names at
   *  write time and this response counts them at read time; when the two
   *  disagree the page says so rather than showing the reader two
   *  numbers and letting them pick. */
  editionNPriced?: number | null;
  className?: string;
}

export default function CompaniesTable({ companies, editionNPriced = null, className = "" }: CompaniesTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("weight_mcw");
  const [sortDir, setSortDir] = useState<SortDir>("descending");
  const rows = companies.items ?? [];
  const sorted = useMemo(() => [...rows].sort((a, b) => compare(a, b, sortKey, sortDir)), [rows, sortKey, sortDir]);

  function toggle(key: SortKey) {
    if (key === sortKey) setSortDir((d) => (d === "ascending" ? "descending" : "ascending"));
    else {
      setSortKey(key);
      setSortDir(NUMERIC.includes(key) ? "descending" : "ascending");
    }
  }

  const header = (key: SortKey, label: string, align: "left" | "right" = "right", title?: string) => (
    <th
      key={key}
      scope="col"
      aria-sort={sortKey === key ? sortDir : "none"}
      className={`${align === "left" ? "text-left" : "text-right"} px-2 py-1 whitespace-nowrap`}
      title={title}
    >
      <button
        type="button"
        onClick={() => toggle(key)}
        className={`inline-flex items-center gap-1 ${align === "right" ? "justify-end w-full" : ""}`}
      >
        <span>{label}</span>
        <span aria-hidden="true" className="text-slate-500">
          {sortKey === key ? (sortDir === "ascending" ? "▲" : "▼") : "↕"}
        </span>
      </button>
    </th>
  );

  if (rows.length === 0) {
    return (
      <div className={`card-tight text-xs text-slate-400 ${className}`} role="status" data-testid="companies-empty">
        {na("no company is classified into this group in the current universe")}. Membership comes from{" "}
        {companies.membership_source || "the classification table"}.
      </div>
    );
  }

  return (
    <div className={`space-y-2 ${className}`} data-testid="industry-companies">
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm" data-testid="companies-table">
          <caption className="text-left text-xs text-slate-400 mb-2" data-testid="companies-caption">
            {companies.name} ({companies.code}) membership as of{" "}
            {companies.as_of ? fmtDate(companies.as_of) : na("no statistics as-of")}: {companies.count} classified
            constituents, {companies.n_priced} priced by the latest statistics row.{" "}
            {companies.truncated > 0
              ? `${companies.truncated} further member${companies.truncated === 1 ? "" : "s"} not shown (page limit ${companies.limit}).`
              : ""}{" "}
            An unpriced name shows its reason and sorts last.
          </caption>
          <thead className="text-xs uppercase tracking-wider text-slate-400">
            <tr>
              {header("ticker", "Ticker", "left")}
              {header("company_name", "Company", "left")}
              {header("sub_industry_name", "Sub-industry", "left", "8-digit GICS sub-industry from the classification row")}
              {header("source", "Source", "left", "Where the group assignment came from")}
              {header("market_cap", "Market cap")}
              {header("weight_mcw", "Weight (mcw)", "right", "Market-cap weight inside the group")}
              {header("last_close", "Last close")}
              {header("ret_1m", "1M return")}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => {
              const badge = sourceBadge(row);
              const r1m = ret1m(row);
              return (
                <tr key={row.ticker} className="border-t border-ink-800/70" data-testid={`company-row-${row.ticker}`}>
                  <th scope="row" className="px-2 py-1 text-left font-normal whitespace-nowrap">
                    <span className="font-mono">{row.ticker}</span>
                  </th>
                  <td className="px-2 py-1 text-slate-300">{row.company_name || na("name not on file")}</td>
                  <td className="px-2 py-1 text-slate-300" data-testid={`sub-industry-${row.ticker}`}>
                    {row.sub_industry_name ? (
                      <>
                        {row.sub_industry_name}{" "}
                        <span className="font-mono text-[10px] text-slate-500">{row.sub_industry_code}</span>
                      </>
                    ) : (
                      <span className="text-slate-500">{na("no sub-industry on this row")}</span>
                    )}
                  </td>
                  <td className="px-2 py-1">
                    <span className={`badge text-[10px] ${badge.tone}`} title={badge.title} data-testid={`source-badge-${row.ticker}`}>
                      {badge.label}
                    </span>
                  </td>
                  <td className="px-2 py-1 text-right tabular-nums">{fmtCap(row.market_cap)}</td>
                  <td className="px-2 py-1 text-right tabular-nums">
                    {/* Two decimals, not a whole percent: a group of any
                        real size has members at a few tenths of a
                        percent of its market cap, and rounding those to
                        "0%" prints a nonzero weight as nothing. */}
                    {fmtPct(row.weight_mcw, row.priced ? "no market cap on file" : (row.unpriced_reason ?? "not priced"))}
                  </td>
                  <td className="px-2 py-1 text-right tabular-nums">
                    {row.priced ? (
                      <>
                        {fmtPrice(row.last_close)}{" "}
                        <span className="text-[10px] text-slate-500">{fmtDate(row.last_date)}</span>
                      </>
                    ) : (
                      <span className="text-slate-500" data-testid={`unpriced-${row.ticker}`}>
                        {na(row.unpriced_reason ?? "not priced this period")}
                      </span>
                    )}
                  </td>
                  <td className="px-2 py-1 text-right tabular-nums">
                    {fmtPctSigned(r1m, row.return_reasons?.["1m"] ?? row.unpriced_reason ?? "not computed")}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {typeof editionNPriced === "number" && editionNPriced !== companies.n_priced && (
        <p className="card-tight text-[11px] text-warn-500" role="status" data-testid="coverage-disagreement">
          Coverage figures disagree: the edition's own companies facts count {editionNPriced} priced, this membership
          read counts {companies.n_priced} of {companies.count}. They are computed differently — a name counts as priced
          here only when the statistics row stored a close for it and did not exclude it, which is the count reconciled
          with the excluded list above. Both are shown rather than one quietly winning.
        </p>
      )}
      <p className="text-[11px] text-slate-500" data-testid="companies-basis">
        {companies.counts_basis} Membership source: {companies.membership_source}.{" "}
        {Object.entries(companies.membership_states ?? {})
          .map(([state, n]) => `${n} ${state}`)
          .join(", ")}
        .
      </p>
      <p className="text-[11px] text-slate-500" data-testid="companies-caveats">
        {companies.mapping_caveat ? `Group assignments are ${companies.mapping_caveat}. ` : ""}
        {companies.security_reference_caveat}
      </p>
    </div>
  );
}
