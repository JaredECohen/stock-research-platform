import React, { useId } from "react";
import { YEAR_CHOICES, type YearsChoice } from "@/hooks/useFundamentalsState";

/**
 * 5 years / 10 years / Max as a native radio group. "Max" omits `years`
 * from the request, so the plan's full allowance applies without it being
 * reported as a cap; the sentence under the group states what the last
 * response actually drew, so a Free reader on "Max" sees "5 years" as a
 * fact, not a nag. (The upgrade copy lives in EntitlementNotice and only
 * appears when the URL asked for more than the plan draws.)
 */
export interface TimeRangeControlProps {
  years: YearsChoice;
  onChange: (years: YearsChoice) => void;
  /** `limits.applied.max_years` from the last series response; undefined before one. */
  appliedYears?: number | null;
  capped?: boolean;
  className?: string;
}

function choiceLabel(c: YearsChoice): string {
  return c === null ? "Max" : `${c} years`;
}

export default function TimeRangeControl({ years, onChange, appliedYears, capped = false, className = "" }: TimeRangeControlProps) {
  const name = useId();
  const showApplied = appliedYears !== undefined;
  return (
    <fieldset className={className} data-testid="time-range">
      <legend className="section-title mb-2">Range</legend>
      <div className="flex flex-wrap gap-1" role="presentation">
        {YEAR_CHOICES.map((c) => {
          const checked = years === c;
          return (
            <label
              key={String(c)}
              className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs cursor-pointer ${
                checked ? "border-accent-600/50 bg-accent-600/10 text-accent-500" : "border-ink-700 text-slate-300 hover:bg-ink-800"
              }`}
            >
              <input type="radio" name={name} value={c === null ? "max" : String(c)} checked={checked} onChange={() => onChange(c)} className="accent-accent-600" />
              {choiceLabel(c)}
            </label>
          );
        })}
      </div>
      {showApplied && (
        <p className="text-xs text-slate-500 mt-1.5" data-testid="range-applied">
          {capped
            ? `Drawn ${appliedYears} years: the most this plan draws.`
            : appliedYears === null
              ? "Drawn: every fiscal year on record."
              : `Drawn: the last ${appliedYears} fiscal years${years === null ? " (the most this plan draws)" : ""}.`}
        </p>
      )}
    </fieldset>
  );
}
