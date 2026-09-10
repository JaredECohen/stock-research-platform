import React, { useMemo } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useReducedMotion } from "@/components/public/hooks";
import type { DoubleLassoResult, FF6Factor, FF6RegressionResult, LassoVerdict, QuintileLSResult, ScorecardEvaluation as EvaluationRow, ScorecardEvaluationResponse } from "@/types/scorecard";
import { fmtReturn, fmtStat, humanize, isNum, na } from "./format";

/**
 * Does the scorecard predict anything? Three worker-computed evaluations,
 * each with its own card: the top-vs-bottom quintile long/short monthly
 * spread (stats, quintile table, skipped months), the FF5 + momentum
 * regression (alpha with t-stat, betas, "insufficient" when n < 24), and
 * the double-selection LASSO verdict (independent | subsumed |
 * insufficient_data). Every card renders the backend's `caveats` verbatim
 * — the unadjusted-price, current-constituent, current-sector and
 * restatement residues are part of the result, not a footnote the client
 * paraphrases. A kind the worker has not run yet says so rather than
 * disappearing.
 */
export interface ScorecardEvaluationProps {
  evaluation: ScorecardEvaluationResponse | null | undefined;
  /** Fixed chart size (tests, print). */
  width?: number;
  height?: number;
  className?: string;
}

const FOCUS = "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500";
const STROKE = "#52E0C4";
const GRID = "#243056";
const AXIS = "#94a3b8";
const FACTORS: FF6Factor[] = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"];

export const VERDICT_TEXT: Record<LassoVerdict, { label: string; tone: string; gloss: string }> = {
  independent: { label: "Independent information", tone: "badge-bull", gloss: "The overall z predicts next-month excess return after the standard characteristics are controlled for (cluster t ≥ 2)." },
  subsumed: { label: "Subsumed by known characteristics", tone: "badge-mixed", gloss: "After controls the coefficient on the overall z is not distinguishable from zero (|t| < 2)." },
  insufficient_data: { label: "Insufficient data", tone: "badge-neutral", gloss: "Fewer than 24 month-ends or 2,000 observations: no verdict is drawn." },
};

function verdictInfo(v: string): { label: string; tone: string; gloss: string } {
  return VERDICT_TEXT[v as LassoVerdict] ?? { label: humanize(v), tone: "badge-neutral", gloss: "Verdict not recognised by this client version." };
}

function Caveats({ caveats, kind }: { caveats: string[] | undefined; kind: string }) {
  const list = caveats ?? [];
  return (
    <div className="text-[11px]" data-testid={`caveats-${kind}`}>
      <div className="text-slate-400 uppercase tracking-widest text-[10px] mb-1">Caveats (verbatim from the evaluation)</div>
      {list.length === 0 ? (
        <p className="text-slate-500">{na("no caveats recorded")}</p>
      ) : (
        <ul className="list-disc pl-4 space-y-0.5 text-slate-300">
          {list.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Stat({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="card-tight !p-2" title={title}>
      <div className="text-[10px] uppercase tracking-widest text-slate-500">{label}</div>
      <div className="font-mono text-sm text-slate-200">{value}</div>
    </div>
  );
}

function Sample({ row }: { row: EvaluationRow }) {
  return (
    <p className="text-[11px] text-slate-500">
      Sample {row.sample_start ?? "n/a"} to {row.sample_end ?? "n/a"} · n = {row.n_obs.toLocaleString("en-US")} · computed {row.created_at}
    </p>
  );
}

function NotRun({ kind, title }: { kind: string; title: string }) {
  return (
    <section className="card space-y-2" data-testid={`eval-${kind}`} data-state="not-run">
      <h3 className="section-title">{title}</h3>
      <p className="text-xs text-slate-400" role="status">
        {na("not run yet")}. The worker computes this evaluation on the first scorecard run after each month-end.
      </p>
    </section>
  );
}

function QuintileCard({ row, result, width, height, animate }: { row: EvaluationRow; result: QuintileLSResult; width?: number; height: number; animate: boolean }) {
  const cumulative = useMemo(() => {
    let acc = 0;
    return result.months.map((m) => {
      acc += m.spread;
      return { as_of: m.as_of, cumulative: acc, spread: m.spread };
    });
  }, [result.months]);
  const insufficient = result.n_months < 24;
  const label = useMemo(() => {
    if (result.months.length === 0) return "Quintile long/short spread: no scored months.";
    const first = result.months[0].as_of;
    const last = result.months[result.months.length - 1].as_of;
    const end = cumulative[cumulative.length - 1].cumulative;
    return `Cumulative top-minus-bottom quintile spread, ${first} to ${last}, ${result.months.length} months, ending at ${fmtReturn(end)}; ${result.skipped_months.length} skipped month${result.skipped_months.length === 1 ? "" : "s"}.`;
  }, [result, cumulative]);

  const chart = (
    <LineChart data={cumulative} width={width} height={width ? height : undefined} margin={{ top: 12, right: 16, left: 4, bottom: 4 }}>
      <CartesianGrid stroke={GRID} vertical={false} />
      <XAxis dataKey="as_of" stroke={AXIS} tick={{ fontSize: 11 }} tickLine={false} minTickGap={24} />
      <YAxis stroke={AXIS} tick={{ fontSize: 11 }} tickLine={false} width={48} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
      <Tooltip
        contentStyle={{ background: "#0E1525", border: `1px solid ${GRID}`, borderRadius: 8, fontSize: 12 }}
        labelStyle={{ color: "#e2e8f0" }}
        itemStyle={{ color: "#cbd5e1" }}
        formatter={(value: unknown) => [fmtReturn(typeof value === "number" ? value : null), "cumulative spread"]}
      />
      <Line type="monotone" dataKey="cumulative" name="cumulative spread" stroke={STROKE} strokeWidth={2} dot={false} isAnimationActive={animate} />
    </LineChart>
  );

  return (
    <section className="card space-y-3" data-testid="eval-quintile_ls" data-state={insufficient ? "insufficient" : "ok"}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="section-title">Quintile long/short spread</h3>
        {insufficient && (
          <span className="badge-neutral" data-testid="quintile-insufficient">
            insufficient: {result.n_months} of 24 months
          </span>
        )}
      </div>
      <Sample row={row} />
      <p className="text-xs text-slate-400">Top 20% minus bottom 20% by overall z, equal weight, monthly rebalance; legs need at least 15 names or the month is skipped and listed below.</p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <Stat label="Mean monthly spread" value={fmtReturn(result.mean_spread, "no months")} />
        <Stat label="Stdev" value={fmtReturn(result.stdev, "no months")} />
        <Stat label="Sharpe (ann.)" value={fmtStat(result.sharpe_annualized, insufficient ? "insufficient months" : "not computed")} />
        <Stat label="t-stat" value={fmtStat(result.t_stat, insufficient ? "insufficient months" : "not computed")} />
        <Stat label="Hit rate" value={isNum(result.hit_rate) ? `${(result.hit_rate * 100).toFixed(0)}%` : na("no months")} />
        <Stat label="Max drawdown" value={fmtReturn(result.max_drawdown, insufficient ? "insufficient months" : "not computed")} />
        <Stat label="Months" value={String(result.n_months)} />
        <Stat label="Monotonic quintiles" value={result.monotonic === null ? na("not computed") : result.monotonic ? "yes" : "no"} title="Mean return rises from quintile 1 to 5" />
      </div>
      {result.months.length > 0 && (
        <figure className="min-w-0">
          <figcaption className="text-xs text-slate-300 mb-1">Cumulative spread (sum of monthly spreads)</figcaption>
          <div role="img" aria-label={label} tabIndex={0} className={`rounded-md ${FOCUS}`} style={{ height: width ? undefined : height }}>
            {width ? chart : <ResponsiveContainer width="100%" height="100%">{chart}</ResponsiveContainer>}
          </div>
        </figure>
      )}
      <div className="grid md:grid-cols-2 gap-3">
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs" data-testid="quintile-table">
            <caption className="text-left text-[11px] text-slate-400 mb-1">Mean monthly return by quintile (5 = highest overall z).</caption>
            <thead className="uppercase tracking-wider text-slate-400">
              <tr>
                <th scope="col" className="text-left px-2 py-1">
                  Quintile
                </th>
                <th scope="col" className="text-right px-2 py-1">
                  Mean return
                </th>
                <th scope="col" className="text-right px-2 py-1">
                  Avg names
                </th>
              </tr>
            </thead>
            <tbody>
              {result.quintile_table.length === 0 && (
                <tr>
                  <td colSpan={3} className="px-2 py-2 text-slate-500">
                    {na("insufficient months to bucket")}
                  </td>
                </tr>
              )}
              {result.quintile_table.map((q) => (
                <tr key={q.q} className="border-t border-ink-700/60">
                  <th scope="row" className="text-left px-2 py-1 font-normal text-slate-200">
                    Q{q.q}
                  </th>
                  <td className="px-2 py-1 text-right font-mono text-slate-300">{fmtReturn(q.mean_ret, "no names")}</td>
                  <td className="px-2 py-1 text-right font-mono text-slate-400">{q.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="text-xs" data-testid="skipped-months">
          <div className="text-slate-400 uppercase tracking-widest text-[10px] mb-1">Skipped months</div>
          {result.skipped_months.length === 0 ? (
            <p className="text-slate-500">None.</p>
          ) : (
            <ul className="space-y-0.5 text-slate-300">
              {result.skipped_months.map((s) => (
                <li key={s.as_of}>
                  <span className="font-mono">{s.as_of}</span> — {s.reason}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <Caveats caveats={result.caveats} kind="quintile_ls" />
    </section>
  );
}

function FF6Card({ row, result }: { row: EvaluationRow; result: FF6RegressionResult }) {
  const reason = result.insufficient ? `insufficient: ${result.n_months} of 24 months` : "not computed";
  return (
    <section className="card space-y-3" data-testid="eval-ff6_regression" data-state={result.insufficient ? "insufficient" : "ok"}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="section-title">Fama-French 5 + momentum regression</h3>
        {result.insufficient && (
          <span className="badge-neutral" data-testid="ff6-insufficient">
            insufficient: {result.n_months} of 24 months
          </span>
        )}
      </div>
      <Sample row={row} />
      <p className="text-xs text-slate-400">
        OLS of the {result.series === "spread" ? "quintile spread" : "long-leg excess return"} on MKT-RF, SMB, HML, RMW, CMA and MOM (Ken French monthly series); HC0 robust t-stats.
      </p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <Stat label="Alpha (monthly)" value={fmtReturn(result.alpha_monthly, reason)} />
        <Stat label="Alpha (annualised)" value={fmtReturn(result.alpha_annualized, reason)} />
        <Stat label="Alpha t-stat" value={fmtStat(result.alpha_t, reason)} />
        <Stat label="R²" value={fmtStat(result.r_squared, reason)} />
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-xs" data-testid="ff6-table">
          <caption className="text-left text-[11px] text-slate-400 mb-1">Factor loadings with t-stats.</caption>
          <thead className="uppercase tracking-wider text-slate-400">
            <tr>
              <th scope="col" className="text-left px-2 py-1">
                Factor
              </th>
              <th scope="col" className="text-right px-2 py-1">
                Beta
              </th>
              <th scope="col" className="text-right px-2 py-1">
                t
              </th>
            </tr>
          </thead>
          <tbody>
            {FACTORS.map((f) => (
              <tr key={f} className="border-t border-ink-700/60">
                <th scope="row" className="text-left px-2 py-1 font-normal font-mono text-slate-200">
                  {f}
                </th>
                <td className="px-2 py-1 text-right font-mono text-slate-300">{fmtStat(result.betas?.[f], reason)}</td>
                <td className="px-2 py-1 text-right font-mono text-slate-400">{fmtStat(result.beta_t?.[f], reason)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Caveats caveats={result.caveats} kind="ff6_regression" />
    </section>
  );
}

function LassoCard({ row, result }: { row: EvaluationRow; result: DoubleLassoResult }) {
  const info = verdictInfo(result.verdict);
  const insufficient = result.verdict === "insufficient_data";
  const reason = insufficient ? "insufficient data" : "not computed";
  return (
    <section className="card space-y-3" data-testid="eval-double_lasso" data-state={insufficient ? "insufficient" : "ok"} data-verdict={result.verdict}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="section-title">Double-selection LASSO</h3>
        <span className={info.tone} data-testid="lasso-verdict">
          {info.label}
        </span>
      </div>
      <Sample row={row} />
      <p className="text-xs text-slate-300">{result.interpretation || na("no interpretation recorded")}</p>
      <p className="text-[11px] text-slate-500">{info.gloss}</p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <Stat label="Coef on overall z" value={isNum(result.coef_d) ? result.coef_d.toFixed(4) : na(reason)} title="Post-selection OLS coefficient on the overall z (monthly excess return per 1 z)" />
        <Stat label="SE (month-clustered)" value={isNum(result.se_cluster_month) ? result.se_cluster_month.toFixed(4) : na(reason)} />
        <Stat label="t-stat" value={fmtStat(result.t_stat, reason)} />
        <Stat label="p-value" value={isNum(result.p_value) ? result.p_value.toFixed(3) : na(reason)} />
        <Stat label="Naive coef" value={isNum(result.naive_coef) ? result.naive_coef.toFixed(4) : na(reason)} title="y ~ d with no controls" />
        <Stat label="Full OLS coef" value={isNum(result.full_ols_coef) ? result.full_ols_coef.toFixed(4) : na(reason)} title="y ~ d + all controls" />
        <Stat label="Observations" value={result.n_obs.toLocaleString("en-US")} />
        <Stat label="Months · controls" value={`${result.n_months} · ${result.p_controls}`} />
      </div>
      <div className="grid md:grid-cols-2 gap-3 text-xs">
        <div data-testid="lasso-selected-y">
          <div className="text-slate-400 uppercase tracking-widest text-[10px] mb-1">Controls selected for return (S₁)</div>
          {result.selected_y.length === 0 ? <p className="text-slate-500">{na(insufficient ? "insufficient data" : "none selected")}</p> : <p className="font-mono text-slate-300">{result.selected_y.join(", ")}</p>}
        </div>
        <div data-testid="lasso-selected-d">
          <div className="text-slate-400 uppercase tracking-widest text-[10px] mb-1">Controls selected for the score (S₂)</div>
          {result.selected_d.length === 0 ? <p className="text-slate-500">{na(insufficient ? "insufficient data" : "none selected")}</p> : <p className="font-mono text-slate-300">{result.selected_d.join(", ")}</p>}
        </div>
      </div>
      <Caveats caveats={result.caveats} kind="double_lasso" />
    </section>
  );
}

export default function ScorecardEvaluation({ evaluation, width, height = 220, className = "" }: ScorecardEvaluationProps) {
  const reduced = useReducedMotion();
  const byKind = useMemo(() => {
    const m = new Map<string, EvaluationRow>();
    for (const e of evaluation?.evaluations ?? []) m.set(e.kind, e);
    return m;
  }, [evaluation]);

  if (!evaluation) {
    return (
      <div className={`card-tight text-xs text-slate-400 ${className}`} role="status" data-testid="evaluation-empty">
        {na("no evaluation yet")}. The worker evaluates the scorecard after each month-end; results appear here with their caveats.
      </div>
    );
  }

  const quintile = byKind.get("quintile_ls");
  const ff6 = byKind.get("ff6_regression");
  const lasso = byKind.get("double_lasso");

  return (
    <div className={`space-y-4 ${className}`} data-testid="scorecard-evaluation">
      <p className="text-xs text-slate-400">
        Evaluations of spec {evaluation.version_key} computed by the worker from month-end runs. They describe a scoring rule's past association with returns — model outputs for research and education, not a
        forecast or a recommendation.
      </p>
      {quintile && quintile.kind === "quintile_ls" ? <QuintileCard row={quintile} result={quintile.result} width={width} height={height} animate={!reduced} /> : <NotRun kind="quintile_ls" title="Quintile long/short spread" />}
      {ff6 && ff6.kind === "ff6_regression" ? <FF6Card row={ff6} result={ff6.result} /> : <NotRun kind="ff6_regression" title="Fama-French 5 + momentum regression" />}
      {lasso && lasso.kind === "double_lasso" ? <LassoCard row={lasso} result={lasso.result} /> : <NotRun kind="double_lasso" title="Double-selection LASSO" />}
    </div>
  );
}
