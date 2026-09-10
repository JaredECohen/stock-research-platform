import React from "react";
import { fmtCurrency, fmtMultiple, fmtPct } from "@/lib/format";
import type { CompsResult, CompsRow } from "@/types";

/**
 * Target, peers and the peer median. Every empty metric prints "—" (the
 * generic "field absent" spelling) rather than a zero, and the table
 * scrolls inside its own container so a phone never scrolls sideways.
 */
const COLUMNS: Array<{ key: keyof CompsRow; label: string; fmt: (v: number | null | undefined) => string }> = [
  { key: "market_cap", label: "Market cap", fmt: (v) => fmtCurrency(v, { compact: true }) },
  { key: "revenue_growth", label: "Revenue growth", fmt: (v) => fmtPct(v) },
  { key: "gross_margin", label: "Gross margin", fmt: (v) => fmtPct(v) },
  { key: "operating_margin", label: "Operating margin", fmt: (v) => fmtPct(v) },
  { key: "ev_ebitda", label: "EV / EBITDA", fmt: (v) => fmtMultiple(v) },
  { key: "pe", label: "P / E", fmt: (v) => fmtMultiple(v) },
  { key: "fcf_yield", label: "FCF yield", fmt: (v) => fmtPct(v) },
];

function Row({ row, kind }: { row: CompsRow; kind: "target" | "peer" | "median" }) {
  const cls =
    kind === "target"
      ? "bg-accent-600/10 text-slate-100 font-medium"
      : kind === "median"
        ? "border-t border-ink-700 text-slate-300 italic"
        : "text-slate-300";
  return (
    <tr className={cls} data-kind={kind}>
      <th scope="row" className="text-left px-3 py-2 whitespace-nowrap font-medium">
        <span className="font-mono">{row.ticker}</span>
        {kind !== "median" ? <span className="text-slate-400 font-normal ml-2">{row.company_name}</span> : null}
      </th>
      {COLUMNS.map((c) => (
        <td key={c.key} className="px-3 py-2 text-right font-mono whitespace-nowrap">
          {c.fmt(row[c.key] as number | null | undefined)}
        </td>
      ))}
    </tr>
  );
}

export default function SampleCompsTable({ comps, headingLevel = 2 }: { comps: CompsResult; headingLevel?: 2 | 3 }) {
  const H = `h${headingLevel}` as "h2" | "h3";
  return (
    <section aria-labelledby="sample-comps-heading">
      <H id="sample-comps-heading" className="text-lg font-semibold mb-3">Comparable companies</H>
      <div className="overflow-x-auto rounded-xl border border-ink-700">
        <table className="min-w-full text-sm">
          <caption className="sr-only">
            {comps.target.ticker} against its peer set; the last row is the peer median.
          </caption>
          <thead className="bg-ink-900/60 text-xs uppercase tracking-wider text-slate-400">
            <tr>
              <th scope="col" className="text-left px-3 py-2">Company</th>
              {COLUMNS.map((c) => (
                <th key={c.key} scope="col" className="text-right px-3 py-2 whitespace-nowrap">{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <Row row={comps.target} kind="target" />
            {comps.peers.map((p) => (
              <Row key={p.ticker} row={p} kind="peer" />
            ))}
            {comps.median ? <Row row={{ ...comps.median, ticker: "Median", company_name: "" }} kind="median" /> : null}
          </tbody>
        </table>
      </div>
      {comps.interpretation ? <p className="text-sm text-slate-300 mt-3 leading-relaxed">{comps.interpretation}</p> : null}
      {comps.history?.interpretation ? (
        <p className="text-xs text-slate-400 mt-2">
          Against its own history ({comps.history.lookback_label}): {comps.history.interpretation}
        </p>
      ) : null}
    </section>
  );
}
