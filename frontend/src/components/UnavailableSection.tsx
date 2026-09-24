import type { SectionAvailability } from "@/types";
import { UNAVAILABLE_TEXT, hiddenItemsNote, reasonText } from "@/lib/memoSections";

/**
 * W2a — the placeholder a memo renderer shows in place of a section the
 * presenter hid (a template filled it). It says the section is unavailable
 * in this version and why, in one line each; it never shows the hidden text.
 *
 * Variants:
 *  - `card`  — a standalone `card-tight` with the section title (MemoCard grid).
 *  - `inline` — no wrapper or title, for a slot inside a card that already
 *    carries the heading (the thesis box, the PM view, the Confidence card).
 *  - `paper` — the print-friendly form for the Full Investment Memo. The PDF
 *    popup has no Tailwind, only its own stylesheet, so this variant relies on
 *    plain elements and the `text-slate-*` classes that stylesheet maps.
 *
 * `detail` is an extra line under the reason — the PM's intake rationale for
 * a skipped analyst, so the reader sees why rather than just that.
 */
export default function UnavailableSection({
  title,
  availability,
  section,
  detail,
  reasonLine,
  variant = "card",
  className = "",
}: {
  title?: string;
  availability?: SectionAvailability;
  // The map key, surfaced as `data-section` so tests and QA can find the
  // placeholder for a given section.
  section?: string;
  detail?: string;
  // Overrides the reason sentence, for a slot with no map entry of its own.
  reasonLine?: string;
  variant?: "card" | "inline" | "paper";
  className?: string;
}) {
  const reason = reasonLine ?? reasonText(availability);
  const items = hiddenItemsNote(availability);
  const body = (
    <>
      <div
        className={
          variant === "paper"
            ? "text-sm italic text-slate-300 print:text-slate-700"
            : "text-sm italic text-slate-400"
        }
      >
        {UNAVAILABLE_TEXT}
      </div>
      {reason && (
        <div
          className={
            variant === "paper"
              ? "text-xs text-slate-400 print:text-slate-600 mt-0.5"
              : "text-xs text-slate-500 mt-0.5"
          }
          data-testid="unavailable-reason"
        >
          {reason}
        </div>
      )}
      {detail && (
        <div
          className={
            variant === "paper"
              ? "text-xs text-slate-400 print:text-slate-600 mt-0.5"
              : "text-xs text-slate-500 mt-0.5"
          }
          data-testid="unavailable-detail"
        >
          PM&apos;s reason: {detail}
        </div>
      )}
      {items && (
        <div className="text-[11px] text-slate-500 mt-0.5" data-testid="hidden-items-note">
          {items}
        </div>
      )}
    </>
  );

  if (variant === "card") {
    return (
      <div
        className={`card-tight border-ink-700 ${className}`}
        data-testid="unavailable-section"
        data-section={section}
      >
        {title && <div className="section-title mb-1">{title}</div>}
        {body}
      </div>
    );
  }
  return (
    <div className={className} data-testid="unavailable-section" data-section={section}>
      {title && (
        <div className="text-xs uppercase tracking-widest text-slate-400 print:text-slate-600 mb-1">
          {title}
        </div>
      )}
      {body}
    </div>
  );
}

/**
 * The one-line note for a section the presenter kept but marked degraded
 * (reduced inputs, template items removed, a rating whose PM input was a
 * template). Renders nothing for an available section.
 */
export function DegradedNote({
  availability,
  className = "",
}: {
  availability?: SectionAvailability;
  className?: string;
}) {
  if (availability?.status !== "degraded") return null;
  const parts = [reasonText(availability), hiddenItemsNote(availability)].filter(Boolean);
  if (parts.length === 0) return null;
  return (
    <div
      className={`text-[11px] text-warn-500 print:text-amber-800 ${className}`}
      data-testid="degraded-note"
    >
      {parts.join(" · ")}
    </div>
  );
}
