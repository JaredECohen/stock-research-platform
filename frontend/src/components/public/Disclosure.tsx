import React from "react";
import { RESEARCH_ONLY } from "./ctas";

/**
 * Research-and-education-only wording. Sample pages pass the backend's
 * `disclosures` (which add the build time); everything else uses the
 * shared sentence. `role="note"` so it is announced as an aside, not
 * skipped as decoration.
 */
export default function Disclosure({ lines, compact = false }: { lines?: string[]; compact?: boolean }) {
  const items = lines && lines.length > 0 ? lines : [RESEARCH_ONLY];
  return (
    <aside role="note" aria-label="Disclosure" className={`${compact ? "" : "card"} text-xs text-slate-400 leading-relaxed`}>
      {items.length === 1 ? (
        <p>{items[0]}</p>
      ) : (
        <ul className="space-y-1">
          {items.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      )}
    </aside>
  );
}
