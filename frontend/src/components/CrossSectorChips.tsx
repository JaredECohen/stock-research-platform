import React from "react";
import { Link } from "react-router-dom";

/**
 * Cross-sector pull-through chips.
 *
 * Surfaces tickers in OTHER sectors that the sector analyst flagged as
 * thesis-relevant for the current name (Phase 6 cross-talk). Each chip is a
 * Link into the Research page for that ticker so users can pivot quickly.
 */
export default function CrossSectorChips({
  tickers,
  className = "",
}: {
  tickers: string[];
  className?: string;
}) {
  if (!tickers || tickers.length === 0) return null;
  return (
    <div className={`flex flex-wrap items-center gap-1.5 ${className}`}>
      <span className="text-[11px] uppercase tracking-wider text-slate-500 mr-1">
        Cross-sector pull-through:
      </span>
      {tickers.map((t) => (
        <Link
          key={t}
          to={`/research?ticker=${encodeURIComponent(t)}`}
          className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-mono font-medium border border-accent-600/40 bg-accent-700/10 text-accent-500 hover:bg-accent-700/25 transition-colors"
        >
          {t}
        </Link>
      ))}
    </div>
  );
}
