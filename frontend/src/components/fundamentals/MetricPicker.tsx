import React, { useId, useMemo } from "react";
import type { MetricSpec, UnitType } from "@/types";

/**
 * Checkbox list of catalog metrics grouped by family. Selection order is
 * preserved (it drives dash pattern and axis side), each option carries
 * its unit and the formula in statement line-item terms (title for the
 * pointer, sr-only text for the screen reader), and unticked options are
 * disabled once the ceiling is reached with a sentence saying why. Native
 * inputs inside a fieldset: arrow keys, Space and labels come for free.
 */
export interface MetricPickerProps {
  catalog: MetricSpec[] | null;
  selected: string[];
  onChange: (metrics: string[]) => void;
  max: number;
  maxSource?: "plan" | "absolute";
  className?: string;
}

const UNIT_BADGE: Record<UnitType, string> = {
  currency: "currency",
  percent: "%",
  ratio: "ratio",
  multiple: "×",
  count: "count",
};

export function familyLabel(family: string): string {
  const s = family.replace(/_/g, " ");
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

export default function MetricPicker({ catalog, selected, onChange, max, maxSource = "absolute", className = "" }: MetricPickerProps) {
  const idBase = useId();
  const atMax = selected.length >= max;

  const groups = useMemo(() => {
    const byFamily = new Map<string, MetricSpec[]>();
    for (const spec of catalog ?? []) {
      const list = byFamily.get(spec.family) ?? [];
      list.push(spec);
      byFamily.set(spec.family, list);
    }
    return Array.from(byFamily.entries());
  }, [catalog]);

  const toggle = (id: string, checked: boolean) => {
    if (checked) {
      if (selected.includes(id) || atMax) return;
      onChange([...selected, id]);
    } else {
      onChange(selected.filter((m) => m !== id));
    }
  };

  return (
    <fieldset className={className} data-testid="metric-picker">
      <legend className="section-title mb-2">Metrics</legend>
      {!catalog && (
        <p className="text-xs text-slate-500" aria-busy="true">
          Loading the metric catalog…
        </p>
      )}
      {catalog && catalog.length === 0 && <p className="text-xs text-slate-500">The catalog is empty.</p>}
      <div className="space-y-3">
        {groups.map(([family, specs]) => (
          <div key={family}>
            <div className="text-[11px] uppercase tracking-wider text-slate-500 mb-1">{familyLabel(family)}</div>
            <ul className="space-y-1">
              {specs.map((spec) => {
                const checked = selected.includes(spec.id);
                const disabled = !checked && atMax;
                const descId = `${idBase}-${spec.id}`;
                return (
                  <li key={spec.id}>
                    <label
                      className={`flex items-start gap-2 text-sm rounded px-1 py-0.5 ${disabled ? "text-slate-500" : "text-slate-200 hover:bg-ink-800/60"}`}
                      title={spec.formula_text}
                    >
                      <input
                        type="checkbox"
                        className="mt-1 accent-accent-600"
                        checked={checked}
                        disabled={disabled}
                        onChange={(e) => toggle(spec.id, e.target.checked)}
                        aria-describedby={descId}
                        data-metric={spec.id}
                      />
                      <span className="flex-1 min-w-0">
                        <span>{spec.label}</span>{" "}
                        <span className="badge text-[10px] border-ink-700 text-slate-400 align-middle" aria-hidden="true">
                          {UNIT_BADGE[spec.unit_type] ?? spec.unit_type}
                        </span>
                        <span id={descId} className="sr-only">
                          {spec.unit_type}, {spec.kind}. {spec.formula_text}
                          {spec.sign_note ? ` ${spec.sign_note}` : ""}
                        </span>
                      </span>
                    </label>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>
      {atMax && (
        <p className="text-xs text-slate-400 mt-2" role="note" data-testid="metric-max">
          Up to {max} metrics per chart{maxSource === "plan" ? " on this plan" : ""}. Untick one to choose another.
        </p>
      )}
    </fieldset>
  );
}
