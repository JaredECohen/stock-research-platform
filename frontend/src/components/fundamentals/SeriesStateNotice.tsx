import React from "react";
import { Link } from "react-router-dom";
import { seriesName } from "@/lib/fundamentals/a11y";
import { reasonText } from "@/lib/fundamentals/format";
import { seriesId, statusCounts } from "@/lib/fundamentals/transform";
import type { MetricSeries, MissingReason, UnavailableTickerWire } from "@/types";

/**
 * Everything about the displayed data that is not a number: companies
 * the server could not draw (with the remedy — for `not_backfilled` that
 * is a research run, which is the only path that loads history), series
 * whose stored data is stale, values that were estimated, and the
 * server's own warnings. Text, never colour alone; nothing here is
 * summarised away as "some data missing".
 */
export interface SeriesStateNoticeProps {
  unavailable: UnavailableTickerWire[];
  series: MetricSeries[];
  warnings?: string[];
  metricLabels?: Record<string, string>;
  className?: string;
}

export default function SeriesStateNotice({ unavailable, series, warnings = [], metricLabels, className = "" }: SeriesStateNoticeProps) {
  const stale = series.filter((s) => s.provenance?.stale);
  const estimated = series.map((s) => ({ s, n: statusCounts(s).estimated })).filter((x) => x.n > 0);
  if (unavailable.length === 0 && stale.length === 0 && estimated.length === 0 && warnings.length === 0) return null;

  return (
    <div className={`card-tight border-warn-500/40 bg-warn-500/5 text-sm space-y-2 ${className}`} role="status" data-testid="series-state">
      {unavailable.length > 0 && (
        <div>
          <div className="text-warn-500 font-medium">Not drawn</div>
          <ul className="mt-1 space-y-1 text-slate-300">
            {unavailable.map((u) => (
              <li key={u.ticker} data-testid={`unavailable-${u.ticker}`}>
                <span className="font-mono">{u.ticker}</span> — {reasonText(u.reason as MissingReason)}.{" "}
                {u.remedy ? <span className="text-slate-400">{u.remedy} </span> : null}
                {u.reason === "not_backfilled" && (
                  <Link to={`/app/research?ticker=${encodeURIComponent(u.ticker)}`} className="text-accent-500 underline underline-offset-2">
                    Run research on {u.ticker}
                  </Link>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
      {stale.length > 0 && (
        <ul className="space-y-1 text-slate-300">
          {stale.map((s) => (
            <li key={seriesId(s)} data-testid="stale-badge">
              <span className="badge badge-mixed text-[10px] mr-1.5">stale</span>
              {seriesName(s, metricLabels)}: {s.provenance.stale_reason ?? "stored data may be out of date"}. Drawn at reduced opacity.
            </li>
          ))}
        </ul>
      )}
      {estimated.length > 0 && (
        <ul className="space-y-1 text-slate-300">
          {estimated.map(({ s, n }) => (
            <li key={seriesId(s)} data-testid="estimated-badge">
              <span className="badge badge-neutral text-[10px] mr-1.5">estimated</span>
              {seriesName(s, metricLabels)}: {n} value{n === 1 ? "" : "s"} computed with a documented fallback (hollow marker on the chart, ≈ in the table).
            </li>
          ))}
        </ul>
      )}
      {warnings.length > 0 && (
        <ul className="space-y-1 text-slate-300" data-testid="series-warnings">
          {warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
