import React from "react";
import { fmtPct, fmtPrice, fmtUpside } from "@/lib/format";
import type { DCFResult, DCFScenario } from "@/types";

/**
 * Base / bull / bear side by side. Two conventions from the underwriting
 * process are stated on the card rather than assumed: the discount rate
 * and terminal growth are held constant across scenarios (only the
 * operating assumptions move), and the three columns are scenarios, not
 * a confidence interval. Prices and upsides the engine could not compute
 * are `null` and print as "n/a", never as $0.00.
 */
function avg(xs: number[]): number | null {
  const v = xs.filter((x) => typeof x === "number" && Number.isFinite(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

function Scenario({ s }: { s: DCFScenario }) {
  const growth = avg(s.assumptions?.revenue_growth || []);
  const margins = s.assumptions?.operating_margin || [];
  const exitMargin = margins.length ? margins[margins.length - 1] : null;
  const tone = s.name === "bull" ? "text-accent-500" : s.name === "bear" ? "text-danger-500" : "text-slate-100";
  return (
    <div className="card-tight" data-scenario={s.name}>
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-widest font-semibold text-slate-300">{s.label || s.name}</div>
        {s.tv_clamped ? <span className="badge-mixed text-[10px]">terminal value capped</span> : null}
      </div>
      <dl className="mt-2 space-y-1.5 text-sm">
        <div className="flex justify-between gap-2">
          <dt className="text-slate-400">Implied price</dt>
          <dd className={`font-mono ${tone}`}>{fmtPrice(s.implied_share_price)}</dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt className="text-slate-400">vs. price at model</dt>
          <dd className={`font-mono ${tone}`}>{fmtUpside(s.upside_pct)}</dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt className="text-slate-400">Avg. revenue growth</dt>
          <dd className="font-mono">{fmtPct(growth)}</dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt className="text-slate-400">Exit-year operating margin</dt>
          <dd className="font-mono">{fmtPct(exitMargin)}</dd>
        </div>
      </dl>
    </div>
  );
}

export default function SampleDCFCard({ dcf, headingLevel = 2 }: { dcf: DCFResult; headingLevel?: 2 | 3 }) {
  const H = `h${headingLevel}` as "h2" | "h3";
  const scenarios = [dcf.base, dcf.bull, dcf.bear].filter((s): s is DCFScenario => !!s);
  const waccs = scenarios.map((s) => s.assumptions?.wacc).filter((x): x is number => typeof x === "number");
  const tgs = scenarios.map((s) => s.assumptions?.terminal_growth).filter((x): x is number => typeof x === "number");
  const same = (xs: number[]) => xs.length > 0 && xs.every((x) => Math.abs(x - xs[0]) < 1e-9);
  const constant = same(waccs) && same(tgs);

  return (
    <section aria-labelledby="sample-dcf-heading">
      <div className="flex flex-wrap items-end justify-between gap-2 mb-3">
        <div>
          <H id="sample-dcf-heading" className="text-lg font-semibold">DCF scenarios</H>
          <p className="text-xs text-slate-400 mt-0.5">
            Price at model: <span className="font-mono text-slate-200">{fmtPrice(dcf.current_price)}</span>
          </p>
        </div>
        <span className="badge-neutral">Scenarios, not confidence intervals</span>
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        {scenarios.map((s) => (
          <Scenario key={s.name} s={s} />
        ))}
      </div>
      <p className="text-xs text-slate-400 mt-3">
        {constant ? (
          <>
            Discount rate <span className="font-mono text-slate-200">{fmtPct(waccs[0])}</span> and terminal growth{" "}
            <span className="font-mono text-slate-200">{fmtPct(tgs[0])}</span> held constant across scenarios; only the operating
            assumptions move.
          </>
        ) : (
          <>
            Discount rate and terminal growth vary by scenario:{" "}
            {scenarios.map((s, i) => (
              <span key={s.name}>
                {i > 0 ? "; " : ""}
                {s.label || s.name} {fmtPct(s.assumptions?.wacc)} / {fmtPct(s.assumptions?.terminal_growth)}
              </span>
            ))}
            .
          </>
        )}
      </p>
      {dcf.summary ? <p className="text-sm text-slate-300 mt-2 leading-relaxed">{dcf.summary}</p> : null}
      {dcf.guardrails && dcf.guardrails.length > 0 ? (
        <ul className="mt-2 space-y-1 text-xs" aria-label="Model sanity checks">
          {dcf.guardrails.map((g, i) => (
            <li key={i} className={g.severity === "error" ? "text-danger-500" : "text-warn-500"}>
              {g.severity === "error" ? "Check failed: " : "Caution: "}
              {g.message}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
