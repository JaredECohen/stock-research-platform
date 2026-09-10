import React from "react";
import { formatShortUtc } from "@/lib/entitlements";
import type { SampleScreenerRow as Row } from "@/types/public";

/**
 * Where the company sits in the stored screener ranking, with its factor
 * scores as labelled bars. Scores are 0–100 quant ranks against the
 * universe at the time of scoring — a ranking, not a forecast.
 */
const FACTORS: Array<{ key: keyof Row; label: string }> = [
  { key: "quality", label: "Quality" },
  { key: "growth", label: "Growth" },
  { key: "valuation", label: "Valuation" },
  { key: "earnings_momentum", label: "Earnings momentum" },
  { key: "risk", label: "Risk" },
  { key: "macro_fit", label: "Macro fit" },
];

function Bar({ label, value }: { label: string; value: unknown }) {
  const v = typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null;
  return (
    <div>
      <div className="flex justify-between text-xs">
        <span className="text-slate-300">{label}</span>
        <span className="font-mono text-slate-200">{v === null ? "n/a" : Math.round(v)}</span>
      </div>
      <div
        className="h-1.5 mt-1 rounded bg-ink-800 overflow-hidden"
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={v ?? undefined}
        aria-valuetext={v === null ? "not available" : `${Math.round(v)} of 100`}
      >
        {v !== null ? <div className="h-full bg-accent-500" style={{ width: `${v}%` }} /> : null}
      </div>
    </div>
  );
}

export default function SampleScreenerRow({ row, headingLevel = 2 }: { row: Row; headingLevel?: 2 | 3 }) {
  const H = `h${headingLevel}` as "h2" | "h3";
  const scored = formatShortUtc(row.scored_at ?? null);
  return (
    <section aria-labelledby="sample-screener-heading" className="card">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <H id="sample-screener-heading" className="text-lg font-semibold">Screener rank</H>
        <div className="text-sm text-slate-300">
          <span className="font-mono text-slate-100">#{row.rank}</span>
          {typeof row.universe_size === "number" && row.universe_size > 0 ? <span className="text-slate-400"> of {row.universe_size}</span> : null}
          {typeof row.pm_score === "number" ? (
            <span className="ml-3">
              Stock score <span className="font-mono text-slate-100">{Math.round(row.pm_score)}</span>
              <span className="text-slate-400"> / 100</span>
            </span>
          ) : null}
        </div>
      </div>
      <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2 mt-3">
        {FACTORS.map((f) => (
          <Bar key={f.key} label={f.label} value={row[f.key]} />
        ))}
      </div>
      <dl className="mt-4 space-y-2 text-sm">
        {row.one_line_thesis ? (
          <div>
            <dt className="text-xs uppercase tracking-wider text-slate-400">One-line thesis</dt>
            <dd className="text-slate-200">{row.one_line_thesis}</dd>
          </div>
        ) : null}
        {row.main_catalyst ? (
          <div>
            <dt className="text-xs uppercase tracking-wider text-slate-400">Main catalyst</dt>
            <dd className="text-slate-200">{row.main_catalyst}</dd>
          </div>
        ) : null}
        {row.main_risk ? (
          <div>
            <dt className="text-xs uppercase tracking-wider text-slate-400">Main risk</dt>
            <dd className="text-slate-200">{row.main_risk}</dd>
          </div>
        ) : null}
      </dl>
      <p className="text-[11px] text-slate-400 mt-3">
        Quant factor ranks against the universe{scored ? ` as scored on ${scored}` : ""}. A ranking, not a forecast.
      </p>
    </section>
  );
}
