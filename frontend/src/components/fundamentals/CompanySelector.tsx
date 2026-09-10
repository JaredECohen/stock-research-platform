import React, { useId } from "react";
import { X } from "lucide-react";
import TickerPicker from "@/components/TickerPicker";
import type { CompanyOut } from "@/types";

/**
 * The companies on the chart, in selection order (which drives line
 * colour). Removable chips plus the shared `TickerPicker` to add one; at
 * the maximum the picker gives way to a sentence saying why. `max` is the
 * plan's ceiling once a series response has reported it, otherwise the
 * absolute request ceiling — the backend stays the gate either way.
 */
export interface CompanySelectorProps {
  tickers: string[];
  onChange: (tickers: string[]) => void;
  universe: CompanyOut[];
  universeLoading?: boolean;
  max: number;
  /** Where `max` comes from, for the sentence at the ceiling. */
  maxSource?: "plan" | "absolute";
  className?: string;
}

const FOCUS = "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500";

export default function CompanySelector({ tickers, onChange, universe, universeLoading = false, max, maxSource = "absolute", className = "" }: CompanySelectorProps) {
  const labelId = useId();
  const atMax = tickers.length >= max;

  const add = (t: string) => {
    const u = t.trim().toUpperCase();
    if (!u || tickers.includes(u) || atMax) return;
    onChange([...tickers, u]);
  };
  const remove = (t: string) => onChange(tickers.filter((x) => x !== t));

  return (
    <div role="group" aria-labelledby={labelId} className={className} data-testid="company-selector">
      <div id={labelId} className="section-title mb-2">
        Companies
      </div>
      {tickers.length > 0 && (
        <ul aria-label="Selected companies" className="flex flex-wrap gap-1.5 mb-2">
          {tickers.map((t) => (
            <li key={t} className="inline-flex items-center gap-1 rounded-md border border-ink-700 bg-ink-900/60 px-2 py-1 text-xs font-mono text-slate-200" data-testid={`chip-${t}`}>
              <span>{t}</span>
              <button
                type="button"
                aria-label={`Remove ${t}`}
                title={`Remove ${t}`}
                onClick={() => remove(t)}
                onKeyDown={(e) => {
                  // Chips are removable from the keyboard without hunting
                  // for the × target.
                  if (e.key === "Backspace" || e.key === "Delete") {
                    e.preventDefault();
                    remove(t);
                  }
                }}
                className={`rounded p-0.5 text-slate-400 hover:text-slate-100 ${FOCUS}`}
              >
                <X size={12} aria-hidden="true" />
              </button>
            </li>
          ))}
        </ul>
      )}
      {atMax ? (
        <p className="text-xs text-slate-400" role="note" data-testid="company-max">
          Up to {max} companies per chart{maxSource === "plan" ? " on this plan" : ""}. Remove one to add another.
        </p>
      ) : (
        // The label wraps the picker so its input is named without changing
        // TickerPicker's props; the visible legend already says "Companies".
        <label className="block">
          <span className="sr-only">Add a company</span>
          <TickerPicker key={tickers.length} value="" onChange={add} universe={universe} loading={universeLoading} placeholder="Add a company…" />
        </label>
      )}
    </div>
  );
}
