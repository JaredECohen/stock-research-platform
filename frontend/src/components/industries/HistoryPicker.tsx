import React from "react";
import type { IndustryHistory } from "@/types/industries";
import { fmtDate, fmtDateTime, na } from "./format";

/**
 * Prior editions of one group's report, newest first.
 *
 * Choosing one puts `?version=` in the URL — the edition is addressable,
 * which is what makes "the numbers moved between these two weeks" a
 * claim a reader can check rather than take.
 *
 * Two honesty rules ride on this list:
 *
 *   * an edition that was **not** published (`pending_review`,
 *     `superseded`) is still listed, labelled with its status, because a
 *     gap in the version numbers is more alarming than a labelled row.
 *     The one exception is an edition kept for audit only (a template —
 *     owner decision 1 never displays one): the server does not list it,
 *     and `withheld` counts it, so the gap it leaves is explained here;
 *   * the response's `truncated` count is printed. A capped history that
 *     shows 26 rows and says nothing looks like a group that has only
 *     ever had 26 editions.
 */

export interface HistoryPickerProps {
  history: IndustryHistory;
  /** The edition being viewed: an edition number, or "latest". */
  value: string;
  onSelect: (version: string) => void;
  className?: string;
}

export default function HistoryPicker({ history, value, onSelect, className = "" }: HistoryPickerProps) {
  const items = history.items ?? [];
  if (items.length === 0) {
    return (
      <div className={`card-tight text-xs text-slate-400 ${className}`} role="status" data-testid="history-empty">
        {na("no editions on file for this group yet")}
      </div>
    );
  }

  return (
    <div className={`space-y-2 min-w-0 ${className}`} data-testid="industry-history">
      <label className="flex flex-col gap-1 text-xs text-slate-400 min-w-0">
        Edition
        {/* Same reason as the group picker: an unconstrained select is as
            wide as its longest option, and these carry a period key, an
            as-of and a status. */}
        <select
          className="input !py-1 text-xs w-full min-w-0"
          data-testid="history-select"
          value={value}
          onChange={(e) => onSelect(e.target.value)}
        >
          <option value="latest">Latest published</option>
          {items.map((item) => (
            <option key={item.version} value={String(item.version)}>
              v{item.version} · {item.period_key || na("no period")} · as of {fmtDate(item.as_of)}
              {item.is_latest_good ? " · published" : ` · ${item.status}`}
              {item.degraded.length > 0 ? ` · degraded (${item.degraded.length})` : ""}
            </option>
          ))}
        </select>
      </label>
      <p className="text-[11px] text-slate-500" data-testid="history-count">
        {history.count} edition{history.count === 1 ? "" : "s"} listed
        {history.truncated > 0 ? `, ${history.truncated} older not shown (limit ${history.limit})` : ""}.
        {history.withheld > 0
          ? ` ${history.withheld} edition${history.withheld === 1 ? " was" : "s were"} kept for audit only ` +
            "(no validated analyst edition) and " +
            (history.withheld === 1 ? "is" : "are") +
            " not published, so version numbers skip."
          : ""}
        {history.last_attempt
          ? ` Last refresh attempt: ${history.last_attempt.status} at ${fmtDateTime(history.last_attempt.at)}.`
          : " No refresh attempt is recorded for this group."}
      </p>
    </div>
  );
}
