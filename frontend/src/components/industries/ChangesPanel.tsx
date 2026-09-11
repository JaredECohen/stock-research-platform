import React from "react";
import type { IndustryChanges, IndustryFactDelta } from "@/types/industries";
import { fmtByPath, fmtDate, fmtPctSigned, humanize, isNum, na, unitFor } from "./format";

/**
 * What moved between two editions.
 *
 * The delta table is the honesty test of this whole feature: a fact
 * present on one side and missing on the other has **no** delta, and the
 * server says which side it was missing on. Subtracting a missing value
 * from a present one yields a number that looks like a move and is not
 * one, so a row without a `delta` prints its `reason` and nothing else.
 *
 * The two editions are named with their versions and as-ofs, and
 * `adjacent` says whether the comparison is against the edition this one
 * actually replaced or one the reader picked — different claims.
 */

function side(v: Record<string, unknown> | undefined, fallback: string): string {
  if (!v || typeof v.version !== "number") return na(fallback);
  return `v${v.version} · ${String(v.period_key || "no period")} · as of ${fmtDate(String(v.as_of ?? ""))}`;
}

/** A delta is a MOVE, so it is signed even for a quantity whose level is
 *  not (breadth going from 50% to 75% is "+25.00%"). Which quantities
 *  are percents at all is `format.unitFor`'s call — the same one the
 *  facts view makes, so the two cards on this tab cannot disagree about
 *  what `returns.1m.n` is. */
function isPercent(key: string): boolean {
  const unit = unitFor(key);
  return unit === "return" || unit === "share";
}

function value(key: string, v: unknown): string {
  if (v === null || v === undefined) return na("missing");
  if (isNum(v)) return fmtByPath(key, v);
  return String(v);
}

function DeltaRow({ name, row }: { name: string; row: IndustryFactDelta }) {
  return (
    <tr className="border-t border-ink-800/70" data-testid={`delta-${name}`}>
      <th scope="row" className="px-2 py-1 text-left font-normal">
        <span className="font-mono text-[11px] text-slate-400">{name}</span>
      </th>
      <td className="px-2 py-1 text-right tabular-nums text-slate-300">{value(name, row.from)}</td>
      <td className="px-2 py-1 text-right tabular-nums text-slate-300">{value(name, row.to)}</td>
      <td className="px-2 py-1 text-right tabular-nums">
        {isNum(row.delta) ? (
          <span className={row.delta > 0 ? "text-accent-500" : row.delta < 0 ? "text-danger-500" : "text-slate-300"}>
            {isPercent(name) ? fmtPctSigned(row.delta) : row.delta}
          </span>
        ) : (
          <span className="text-slate-500">{na(row.reason ?? "no reason recorded")}</span>
        )}
      </td>
    </tr>
  );
}

export default function ChangesPanel({ changes, className = "" }: { changes: IndustryChanges; className?: string }) {
  const deltas = Object.entries(changes.facts_delta ?? {});
  const moved = deltas.filter(([, d]) => isNum(d.delta) && d.delta !== 0).length;
  const unmeasurable = deltas.filter(([, d]) => !isNum(d.delta)).length;
  const constituents = (changes.constituents ?? {}) as Record<string, unknown>;
  const added = Array.isArray(constituents.added) ? (constituents.added as string[]) : [];
  const removed = Array.isArray(constituents.removed) ? (constituents.removed as string[]) : [];
  const analyst = (changes.analyst_view ?? {}) as Record<string, unknown>;

  return (
    <div className={`space-y-3 ${className}`} data-testid="industry-changes">
      <p className="text-xs text-slate-400" data-testid="changes-basis">
        {side(changes.from as Record<string, unknown>, "from-edition not recorded")} →{" "}
        {side(changes.to as Record<string, unknown>, "to-edition not recorded")} ·{" "}
        {changes.adjacent
          ? "the edition this one replaced"
          : "a comparison you chose; these editions are not adjacent"}
        . {moved} fact{moved === 1 ? "" : "s"} moved; {unmeasurable} could not be differenced.
      </p>

      <div className="overflow-x-auto">
        <table className="min-w-full text-xs" data-testid="changes-table">
          <caption className="text-left text-xs text-slate-400 mb-2">
            Observed facts on both editions. A fact missing on either side has no delta — the reason names the side.
          </caption>
          <thead className="uppercase tracking-wider text-slate-400">
            <tr>
              <th scope="col" className="text-left px-2 py-1">
                Fact
              </th>
              <th scope="col" className="text-right px-2 py-1">
                From
              </th>
              <th scope="col" className="text-right px-2 py-1">
                To
              </th>
              <th scope="col" className="text-right px-2 py-1">
                Change
              </th>
            </tr>
          </thead>
          <tbody>
            {deltas.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-2 py-3 text-slate-400" role="status">
                  {na("neither edition stored a comparable statistics row")}
                </td>
              </tr>
            ) : (
              deltas.map(([name, row]) => <DeltaRow key={name} name={name} row={row} />)
            )}
          </tbody>
        </table>
      </div>

      <div className="card-tight text-xs" data-testid="changes-constituents">
        <div className="text-slate-500 uppercase tracking-widest text-[10px] mb-1">Membership</div>
        <p className="text-slate-300">
          Added: {added.length > 0 ? added.join(", ") : na("none")} · Removed:{" "}
          {removed.length > 0 ? removed.join(", ") : na("none")}
          {isNum(constituents.n_from) && isNum(constituents.n_to)
            ? ` · ${constituents.n_from} → ${constituents.n_to} members`
            : ""}
        </p>
      </div>

      <div className="card-tight text-xs space-y-1" data-testid="changes-analyst-view">
        <div className="text-slate-500 uppercase tracking-widest text-[10px]">
          Analyst interpretation — what the edition itself said changed
        </div>
        <p className="text-slate-300">
          {typeof analyst.what_changed === "string" && analyst.what_changed
            ? analyst.what_changed
            : na("this edition wrote no what-changed line")}
        </p>
        {["from", "to"].map((k) => (
          <p key={k} className="text-slate-500">
            <span className="uppercase text-[10px] tracking-widest">{humanize(k)}: </span>
            {typeof analyst[k] === "string" && analyst[k] ? String(analyst[k]) : na("no analyst view recorded")}
          </p>
        ))}
      </div>
    </div>
  );
}
