import React from "react";
import { useConfig } from "@/auth/ConfigProvider";
import { COMPARISON_ROWS, matrixKnown, rowCells } from "./allowance";

/**
 * Free Explorer vs Pro, row by row, from the backend's matrix. When the
 * config fetch fell back (empty matrix) the table still lists what the
 * product does but shows no allowance — never a guessed number.
 */
export default function FeatureComparison({ compact = false, headingLevel = 2 }: { compact?: boolean; headingLevel?: 2 | 3 }) {
  const { config } = useConfig();
  const H = `h${headingLevel}` as "h2" | "h3";
  const known = matrixKnown(config.features);
  const rows = compact ? COMPARISON_ROWS.slice(0, 5) : COMPARISON_ROWS;

  return (
    <section aria-labelledby="compare-heading" className={compact ? "mt-16" : "mt-10"}>
      <H id="compare-heading" className="text-2xl font-semibold tracking-tight">
        {compact ? "Explore free, underwrite on Pro" : "What each plan includes"}
      </H>
      <p className="text-sm text-slate-400 mt-1 max-w-2xl">
        Free Explorer opens the committee's stored work. Pro is for underwriting: run research, DCF, comps and portfolios on your own
        tickers. All allowances are per UTC calendar month.
      </p>
      <div className="overflow-x-auto rounded-xl border border-ink-700 mt-4">
        <table className="min-w-full text-sm">
          <caption className="sr-only">Feature allowances by plan</caption>
          <thead className="bg-ink-900/60 text-xs uppercase tracking-wider text-slate-400">
            <tr>
              <th scope="col" className="text-left px-3 py-2">Feature</th>
              <th scope="col" className="text-left px-3 py-2 whitespace-nowrap">Free Explorer</th>
              <th scope="col" className="text-left px-3 py-2 text-accent-500">Pro</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const cells = known ? rowCells(config.features, r.feature) : null;
              return (
                <tr key={r.feature} className="border-t border-ink-700 align-top" data-feature={r.feature}>
                  <th scope="row" className="text-left px-3 py-2 font-medium text-slate-200">{r.label}</th>
                  <td className="px-3 py-2 text-slate-300">{cells ? cells.free : "—"}</td>
                  <td className="px-3 py-2 text-slate-100">{cells ? cells.pro : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {!known ? (
        <p className="text-xs text-slate-400 mt-2" role="status" data-testid="matrix-unknown">
          Allowances could not be loaded just now; the app's account page shows the current numbers.
        </p>
      ) : null}
    </section>
  );
}
