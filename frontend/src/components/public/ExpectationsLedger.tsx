import React from "react";
import { fmtPrice, fmtUpside } from "@/lib/format";
import type { ExpectationsLedger as Ledger, GuidanceCellValue, LedgerCell, LedgerColumn, LedgerColumnKey } from "@/types/public";

/**
 * The four expectation columns from docs/research/README.md, rendered
 * from the sample payload's `expectations_ledger` (built server-side from
 * memo fields only). Two rules the research process insists on and this
 * component makes visible:
 *
 *   - observed data is styled apart from interpretation (a blue rail and
 *     an "Observed" tag vs. the accent rail and "Interpretation");
 *   - a blank leg says WHY it is blank — "not captured" or "n/a" plus the
 *     backend's reason — and is never shown as a zero.
 *
 * Reusable in the app later; it takes the payload and a heading level.
 */
export const COLUMN_META: Record<LedgerColumnKey, { title: string; hint: string }> = {
  reported_consensus: { title: "Reported consensus", hint: "What the street expects" },
  management_guidance: { title: "Management guidance", hint: "What the company said it will deliver" },
  price_implied: { title: "Price-implied", hint: "What the current price already assumes" },
  our_forecast: { title: "Our forecast", hint: "What the committee expects, and what would prove it wrong" },
};

const ORDER: LedgerColumnKey[] = ["reported_consensus", "management_guidance", "price_implied", "our_forecast"];

function isGuidance(v: unknown): v is GuidanceCellValue {
  return !!v && typeof v === "object" && !Array.isArray(v) && "direction" in (v as object);
}

function CellValue({ cell }: { cell: LedgerCell }) {
  const v = cell.value;
  if (typeof v === "number") {
    const isUpside = /upside/i.test(cell.label) || /upside/i.test(cell.source);
    return <span className="font-mono text-slate-100">{isUpside ? fmtUpside(v) : fmtPrice(v)}</span>;
  }
  if (Array.isArray(v)) {
    return (
      <ul className="list-disc pl-4 space-y-1">
        {v.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ul>
    );
  }
  if (isGuidance(v)) {
    const prior = v.prior || "—";
    const current = v.current || "—";
    return (
      <div>
        <div>
          <span className="font-mono text-slate-100">{prior}</span>
          <span className="text-slate-400 mx-1.5" aria-hidden>→</span>
          <span className="sr-only"> to </span>
          <span className="font-mono text-slate-100">{current}</span>
          <span className="ml-2 badge-neutral text-[10px] capitalize">{v.direction}</span>
        </div>
        {v.rationale ? <p className="text-slate-400 mt-1">{v.rationale}</p> : null}
      </div>
    );
  }
  if (typeof v === "string" && v.trim()) return <p>{v}</p>;
  return <span className="text-slate-400">n/a</span>;
}

function Cell({ cell }: { cell: LedgerCell }) {
  const observed = cell.basis === "observed";
  return (
    <div
      data-basis={cell.basis}
      className={`rounded-md border-l-2 pl-3 py-1.5 text-sm ${
        observed ? "border-sky-400 bg-sky-400/[0.06]" : "border-accent-500 bg-accent-600/[0.06]"
      }`}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <div className="text-xs font-medium text-slate-300">{cell.label}</div>
        <span
          className={`text-[10px] uppercase tracking-wider font-semibold ${observed ? "text-sky-300" : "text-accent-500"}`}
        >
          {observed ? "Observed" : "Interpretation"}
        </span>
      </div>
      <CellValue cell={cell} />
      {cell.as_of ? <div className="text-[11px] text-slate-400 mt-1">as of {cell.as_of.slice(0, 10)}</div> : null}
    </div>
  );
}

function Column({ id, column, headingLevel }: { id: LedgerColumnKey; column: LedgerColumn; headingLevel: 3 | 4 }) {
  const meta = COLUMN_META[id];
  const H = `h${headingLevel}` as "h3" | "h4";
  const headingId = `ledger-${id}`;
  return (
    <section aria-labelledby={headingId} className="card-tight flex flex-col gap-2" data-status={column.status}>
      <div>
        <H id={headingId} className="text-sm font-semibold">{meta.title}</H>
        <div className="text-[11px] text-slate-400">{meta.hint}</div>
      </div>
      {column.status === "available" && column.items.length > 0 ? (
        column.items.map((cell, i) => <Cell key={`${cell.source}-${i}`} cell={cell} />)
      ) : (
        <p className="text-sm text-slate-400 italic">
          {column.status === "not_captured" ? "Not captured" : "n/a"}
          {column.reason ? ` — ${column.reason}` : ""}
        </p>
      )}
    </section>
  );
}

interface Props {
  ledger: Ledger;
  /** The page's heading hierarchy: columns render one level below `headingLevel`. */
  headingLevel?: 2 | 3;
}

export default function ExpectationsLedger({ ledger, headingLevel = 2 }: Props) {
  const H = `h${headingLevel}` as "h2" | "h3";
  const cols = (ledger.columns && ledger.columns.length ? ledger.columns : ORDER).filter((c) => c in COLUMN_META);
  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-2 mb-3">
        <div>
          <H className="text-lg font-semibold">Expectations ledger</H>
          <p className="text-xs text-slate-400 mt-0.5 max-w-2xl">{ledger.note}</p>
        </div>
        <dl className="flex items-center gap-4 text-[11px] text-slate-300" aria-label="Legend">
          <div className="flex items-center gap-1.5">
            <dt className="h-3 w-1 rounded bg-sky-400" aria-hidden />
            <dd>Observed — quoted data</dd>
          </div>
          <div className="flex items-center gap-1.5">
            <dt className="h-3 w-1 rounded bg-accent-500" aria-hidden />
            <dd>Interpretation — the committee's view</dd>
          </div>
        </dl>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {cols.map((c) => (
          <Column key={c} id={c} column={ledger[c]} headingLevel={headingLevel === 2 ? 3 : 4} />
        ))}
      </div>
    </div>
  );
}
