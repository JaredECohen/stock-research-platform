import React, { useEffect, useMemo, useRef, useState } from "react";
import type { CompanyOut } from "@/types";

interface Props {
  value: string;
  onChange: (ticker: string) => void;
  universe: CompanyOut[];
  loading?: boolean;
  placeholder?: string;
  className?: string;
  /** Allow committing tickers that aren't in `universe` (e.g. for the
   * Research page's lazy-introduction flow). Default true. */
  allowUnknown?: boolean;
}

/**
 * Typeahead ticker picker — single component, used by Research / Comps /
 * DCF Lab. Filters the dropdown by ticker OR company name as the user
 * types; click or Enter commits. Closes on outside click. Caps the
 * visible options to 60 with a "keep typing" affordance below.
 */
export default function TickerPicker({
  value,
  onChange,
  universe,
  loading = false,
  placeholder,
  className = "",
  allowUnknown = true,
}: Props) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);

  const tickerSet = useMemo(
    () => new Set(universe.map((c) => c.ticker.toUpperCase())),
    [universe],
  );

  const filtered = useMemo(() => {
    const q = draft.trim().toLowerCase();
    if (!q) return universe;
    return universe.filter(
      (c) =>
        c.ticker.toLowerCase().includes(q) ||
        c.company_name.toLowerCase().includes(q),
    );
  }, [universe, draft]);

  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  function commit(t: string) {
    const upper = t.trim().toUpperCase();
    if (!upper) return;
    if (!allowUnknown && !tickerSet.has(upper)) {
      // Snap back to current value if user typed garbage and we don't allow it.
      setDraft(value);
      setOpen(false);
      return;
    }
    onChange(upper);
    setDraft(upper);
    setOpen(false);
  }

  return (
    <div ref={ref} className={`relative ${className}`}>
      <input
        type="text"
        className="input w-full"
        value={draft}
        onChange={(e) => {
          setDraft(e.target.value.toUpperCase());
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            const target = filtered[0]?.ticker ?? draft;
            commit(target);
          } else if (e.key === "Escape") {
            setDraft(value);
            setOpen(false);
          }
        }}
        placeholder={
          loading
            ? "Loading tickers…"
            : universe.length === 0
            ? "No tickers available"
            : placeholder ?? "Type a ticker or company name…"
        }
        disabled={loading}
        spellCheck={false}
        autoComplete="off"
      />
      {open && filtered.length > 0 && (
        <ul
          className="absolute z-10 left-0 right-0 mt-1 max-h-72 overflow-y-auto rounded-md border border-ink-700 bg-ink-900 shadow-lg"
          role="listbox"
        >
          {filtered.slice(0, 60).map((c) => (
            <li
              key={c.ticker}
              role="option"
              aria-selected={c.ticker === value}
              onMouseDown={(e) => {
                e.preventDefault();
                commit(c.ticker);
              }}
              className={`px-3 py-1.5 cursor-pointer text-sm flex items-center gap-2 ${
                c.ticker === value
                  ? "bg-accent-600/15 text-accent-400"
                  : "text-slate-200 hover:bg-ink-800"
              }`}
            >
              <span className="font-mono w-14">{c.ticker}</span>
              <span className="text-slate-400 truncate">{c.company_name}</span>
              {c.sector && (
                <span className="ml-auto text-xs text-slate-500">{c.sector}</span>
              )}
            </li>
          ))}
          {filtered.length > 60 && (
            <li className="px-3 py-1 text-xs text-slate-500 border-t border-ink-800">
              {filtered.length - 60} more — keep typing to narrow
            </li>
          )}
        </ul>
      )}
    </div>
  );
}
