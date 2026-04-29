import React from "react";
import type { MacroBroadcast } from "@/types";

/**
 * Macro regime banner.
 *
 * Renders the current macro regime label with this name's alignment
 * (favored / pressured / neutral) and the regime's favored / pressured
 * sector lists. Color follows alignment so users see at-a-glance whether
 * the macro tape is working for or against the thesis.
 */
const REGIME_LABEL: Record<string, string> = {
  sticky_inflation: "Sticky Inflation",
  late_cycle_slowdown: "Late-Cycle Slowdown",
  credit_stress: "Credit Stress",
  soft_landing: "Soft Landing",
  mixed: "Mixed Regime",
};

function alignmentClasses(alignment?: string): string {
  switch ((alignment || "").toLowerCase()) {
    case "favored":
      return "border-accent-600/40 bg-accent-700/10 text-accent-500";
    case "pressured":
      return "border-danger-500/40 bg-danger-500/10 text-danger-500";
    default:
      return "border-ink-700 bg-ink-800/60 text-slate-300";
  }
}

export default function MacroRegimeBanner({
  broadcast,
  alignment,
  sector,
}: {
  broadcast?: MacroBroadcast;
  alignment?: string;
  sector?: string;
}) {
  if (!broadcast || !broadcast.regime) return null;
  const label = REGIME_LABEL[broadcast.regime] || broadcast.regime;
  const cls = alignmentClasses(alignment);
  const align = (alignment || "neutral").toLowerCase();
  return (
    <div
      className={`card-tight ${cls} flex flex-col md:flex-row md:items-center md:justify-between gap-2`}
    >
      <div>
        <div className="text-[11px] uppercase tracking-widest opacity-80">Macro regime</div>
        <div className="text-sm font-semibold mt-0.5">{label}</div>
        {sector && (
          <div className="text-xs opacity-80 mt-0.5">
            {sector} alignment:{" "}
            <span className="font-medium capitalize">{align}</span>
          </div>
        )}
      </div>
      <div className="text-xs space-y-0.5 md:text-right">
        {broadcast.favored_sectors && broadcast.favored_sectors.length > 0 && (
          <div>
            <span className="opacity-70">Favored:</span>{" "}
            <span className="font-medium">{broadcast.favored_sectors.join(", ")}</span>
          </div>
        )}
        {broadcast.pressured_sectors && broadcast.pressured_sectors.length > 0 && (
          <div>
            <span className="opacity-70">Pressured:</span>{" "}
            <span className="font-medium">{broadcast.pressured_sectors.join(", ")}</span>
          </div>
        )}
      </div>
    </div>
  );
}
