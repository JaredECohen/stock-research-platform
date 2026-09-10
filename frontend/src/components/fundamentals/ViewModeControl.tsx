import React, { useId } from "react";
import type { LayoutResult, SelectableMode } from "@/lib/fundamentals/layout";
import { VIEW_MODES, type ViewMode } from "@/types/fundamentals";

/**
 * auto / dual axis / small multiples / indexed / table as a native radio
 * group. The layout engine reports which modes the current series allow;
 * a forbidden one stays visible but disabled, with the engine's reason as
 * the tooltip and as sr-only text, so the reader learns *why* ("indexing a
 * percent series is misleading") instead of finding a dead control.
 */
export interface ViewModeControlProps {
  view: ViewMode;
  onChange: (view: ViewMode) => void;
  availability?: LayoutResult["availability"] | null;
  className?: string;
}

const LABELS: Record<ViewMode, string> = {
  auto: "Auto",
  "dual-axis": "Dual axis",
  "small-multiples": "Small multiples",
  indexed: "Indexed",
  table: "Table",
};

function isSelectable(mode: ViewMode): mode is SelectableMode {
  return mode !== "auto" && mode !== "table";
}

export default function ViewModeControl({ view, onChange, availability, className = "" }: ViewModeControlProps) {
  const name = useId();
  return (
    <fieldset className={className} data-testid="view-mode">
      <legend className="section-title mb-2">View</legend>
      <div className="flex flex-wrap gap-1" role="presentation">
        {VIEW_MODES.map((mode) => {
          const avail = isSelectable(mode) && availability ? availability[mode] : null;
          const disabled = !!avail && !avail.enabled;
          const reason = disabled ? avail?.reason ?? "not available for these series" : null;
          const checked = view === mode;
          return (
            <label
              key={mode}
              title={reason ?? undefined}
              className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs ${
                disabled
                  ? "border-ink-800 text-slate-600 cursor-not-allowed"
                  : checked
                    ? "border-accent-600/50 bg-accent-600/10 text-accent-500 cursor-pointer"
                    : "border-ink-700 text-slate-300 hover:bg-ink-800 cursor-pointer"
              }`}
            >
              <input
                type="radio"
                name={name}
                value={mode}
                checked={checked}
                disabled={disabled}
                onChange={() => onChange(mode)}
                className="accent-accent-600"
                data-view={mode}
              />
              {LABELS[mode]}
              {reason && <span className="sr-only"> (unavailable: {reason})</span>}
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
