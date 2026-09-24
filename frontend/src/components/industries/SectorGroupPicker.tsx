import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { IndustryGroupNode, IndustryTaxonomy } from "@/types/industries";
import { na } from "./format";

/**
 * The industry-group picker: sectors → groups, from the registry.
 *
 * Two renderings of one list, because one control cannot be good at both:
 *
 *   * **Wide**: a single `role="listbox"` with `role="group"` per sector.
 *     The box itself takes focus and moves an `aria-activedescendant`
 *     with the arrow keys — so a screen reader announces the option under
 *     the cursor without the focus ring leaving the control, and Home/End
 *     reach the ends of a 25-group list without 24 keystrokes.
 *   * **Narrow**: the platform `<select>` with an `<optgroup>` per
 *     sector, which on a phone is a native wheel rather than a list the
 *     viewer has to scroll inside a scrolling page.
 *
 * Counts are whatever the response carries — the number of sectors,
 * groups and constituents comes from the registry on every call and is
 * never hardcoded here.
 *
 * Groups and sectors are shown by name only: `code` is the public slug the
 * URL and the option values use, and no taxonomy code is ever printed
 * (owner decision 2026-09-24 — our own labels, no licensed codes).
 */

const NARROW_QUERY = "(max-width: 767px)";

function useNarrowViewport(): boolean {
  const [narrow, setNarrow] = useState<boolean>(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
    return window.matchMedia(NARROW_QUERY).matches;
  });
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mql = window.matchMedia(NARROW_QUERY);
    const onChange = (e: MediaQueryListEvent) => setNarrow(e.matches);
    if (typeof mql.addEventListener === "function") {
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    }
    return undefined;
  }, []);
  return narrow;
}

export function optionId(code: string): string {
  return `industry-group-option-${code}`;
}

/** The edition pointer as one short line. `stale_by_age` is the only
 *  staleness this response can know (the report endpoint owns the other
 *  half), so the word "age" is in the label rather than a bare "stale". */
export function pointerSummary(group: IndustryGroupNode): string {
  const p = group.latest_report;
  if (!p) return na("no published edition yet");
  const bits = [`v${p.version}`, p.period_key || na("no period")];
  if (p.stale_by_age) bits.push(`stale by age${typeof p.age_days === "number" ? ` (${p.age_days}d)` : ""}`);
  if (p.degraded) bits.push(`degraded (${p.degraded_reasons.length})`);
  return bits.join(" · ");
}

/**
 * The server's sentence about a group this universe cannot cover, or null
 * when it can.
 *
 * A group below the floor on MEMBERSHIP will report `insufficient_sample`
 * every week for as long as its classified membership stays as it is —
 * which is a different thing from a group waiting on the weekly price
 * warm-up, and different again from a universe short of companies. The
 * picker is where a reader decides which group to open. The text is the
 * API's, including which remedy applies; the picker never composes one.
 */
export function notCoverableNote(group: IndustryGroupNode): string | null {
  const cov = group.universe_coverage;
  if (!cov || cov.coverable) return null;
  return cov.explanation;
}

export interface SectorGroupPickerProps {
  taxonomy: IndustryTaxonomy;
  /** The selected group code, or null on the index page. */
  value: string | null;
  onSelect: (code: string) => void;
  className?: string;
}

export default function SectorGroupPicker({ taxonomy, value, onSelect, className = "" }: SectorGroupPickerProps) {
  const narrow = useNarrowViewport();
  const flat = useMemo(
    () => taxonomy.sectors.flatMap((s) => s.industry_groups.map((g) => ({ sector: s, group: g }))),
    [taxonomy],
  );
  const codes = useMemo(() => flat.map((e) => e.group.code), [flat]);
  const [active, setActive] = useState<string | null>(value ?? codes[0] ?? null);
  const listRef = useRef<HTMLUListElement | null>(null);

  // The selection is the URL's, so it wins whenever it changes under us
  // (a deep link, the back button, a pick from the other rendering).
  useEffect(() => {
    if (value) setActive(value);
  }, [value]);

  const move = useCallback(
    (to: number) => {
      if (codes.length === 0) return;
      const next = codes[Math.max(0, Math.min(codes.length - 1, to))];
      setActive(next);
      // Attribute selector rather than `#id`: the code is data, and a
      // querySelector with an interpolated id is a parse error waiting
      // for the first taxonomy whose codes are not bare digits.
      const el = listRef.current?.querySelector<HTMLElement>(`[data-code="${next}"]`);
      el?.scrollIntoView?.({ block: "nearest" });
    },
    [codes],
  );

  const onKeyDown = (e: React.KeyboardEvent<HTMLUListElement>) => {
    const i = active ? codes.indexOf(active) : -1;
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        move(i + 1);
        break;
      case "ArrowUp":
        e.preventDefault();
        move(i <= 0 ? 0 : i - 1);
        break;
      case "Home":
        e.preventDefault();
        move(0);
        break;
      case "End":
        e.preventDefault();
        move(codes.length - 1);
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        if (active) onSelect(active);
        break;
      default:
        break;
    }
  };

  if (flat.length === 0) {
    return (
      <div className={`card-tight text-xs text-slate-400 ${className}`} role="status" data-testid="industry-picker-empty">
        {na("the active taxonomy has no industry groups")}
      </div>
    );
  }

  if (narrow) {
    return (
      <div className={`min-w-0 ${className}`} data-testid="industry-picker">
        <label className="flex flex-col gap-1 text-xs text-slate-400 min-w-0">
          Industry group
          {/* `w-full min-w-0` is load-bearing, not decoration: a bare
              `<select>` sizes itself to its WIDEST option, and these
              options carry the group name plus its edition pointer. On a
              375px phone that made the select 729px wide and gave the
              whole page a horizontal scrollbar. */}
          <select
            className="input text-sm w-full min-w-0"
            data-testid="industry-picker-select"
            value={value ?? ""}
            onChange={(e) => e.target.value && onSelect(e.target.value)}
          >
            <option value="">Choose an industry group…</option>
            {taxonomy.sectors.map((sector) => (
              <optgroup key={sector.code} label={sector.name}>
                {sector.industry_groups.map((g) => (
                  <option key={g.code} value={g.code}>
                    {g.name} — {g.constituent_count} in universe
                    {notCoverableNote(g) ? " · too few to report on" : ""} · {pointerSummary(g)}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>
      </div>
    );
  }

  return (
    <div className={className} data-testid="industry-picker">
      <ul
        ref={listRef}
        role="listbox"
        aria-label="Industry groups"
        tabIndex={0}
        aria-activedescendant={active ? optionId(active) : undefined}
        onKeyDown={onKeyDown}
        data-testid="industry-picker-listbox"
        className="max-h-[28rem] overflow-y-auto rounded-lg border border-ink-800 bg-ink-900/40 focus:outline-none focus:ring-2 focus:ring-accent-600/50"
      >
        {taxonomy.sectors.map((sector) => (
          <li key={sector.code} role="group" aria-label={sector.name}>
            <div className="px-3 py-1 text-[10px] uppercase tracking-widest text-slate-500 bg-ink-900/70 sticky top-0">
              {sector.name}
            </div>
            <ul role="presentation">
              {sector.industry_groups.map((g) => {
                const selected = value === g.code;
                const notCoverable = notCoverableNote(g);
                return (
                  <li
                    key={g.code}
                    id={optionId(g.code)}
                    role="option"
                    aria-selected={selected}
                    data-code={g.code}
                    data-active={active === g.code ? "true" : undefined}
                    onClick={() => onSelect(g.code)}
                    onMouseEnter={() => setActive(g.code)}
                    className={`px-3 py-2 text-sm cursor-pointer border-l-2 ${
                      selected
                        ? "border-accent-600 bg-accent-600/10 text-accent-500"
                        : active === g.code
                          ? "border-slate-500 bg-ink-800/60 text-slate-100"
                          : "border-transparent text-slate-300 hover:bg-ink-800"
                    }`}
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <span>{g.name}</span>
                      <span className="text-[11px] text-slate-500 whitespace-nowrap">
                        {g.constituent_count} in universe
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-500">{pointerSummary(g)}</div>
                    {/* Said here, not only on the report: this is where a
                        reader chooses what to open, and "4 in universe"
                        alone does not tell them the group can never be
                        reported on. "Classified" is load-bearing: the
                        shortfall is of constituents this taxonomy counts,
                        which is not the same as a shortfall of companies,
                        and the server's sentence that follows says which
                        remedy applies. */}
                    {notCoverable && (
                      <div className="text-[11px] text-warn-500" data-testid={`not-coverable-${g.code}`}>
                        Too few classified companies in this universe to report on — {notCoverable}
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          </li>
        ))}
      </ul>
      <p className="mt-1 text-[11px] text-slate-500">
        {taxonomy.node_counts.industry_group ?? flat.length} industry groups across{" "}
        {taxonomy.node_counts.sector ?? taxonomy.sectors.length} sectors, from the active taxonomy{" "}
        {taxonomy.taxonomy_version.key}. Arrow keys move, Enter opens.
      </p>
    </div>
  );
}
