// Compact amber badge shown wherever a DCF's numbers are displayed when the
// engine floored the Gordon denominator (WACC − terminal growth ≤ 0.5%).
// The terminal value is then a cap, not a valuation, so the implied prices
// beside it must not be read as trustworthy. The explanation lives in the
// `title` tooltip so the badge stays small in dense DCF layouts.

export const TERMINAL_CLAMP_LABEL = "Terminal value clamped";

export const TERMINAL_CLAMP_TITLE =
  "WACC minus terminal growth was at or below the engine's 0.5% floor, so " +
  "the Gordon terminal value was capped at that floor. The terminal value " +
  "and implied prices are not trustworthy — lower terminal growth or raise WACC.";

export default function TerminalClampBadge({ className = "" }: { className?: string }) {
  return (
    <span
      role="status"
      title={TERMINAL_CLAMP_TITLE}
      className={`inline-flex items-center gap-1 rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-amber-400 print:text-amber-700 ${className}`}
    >
      <span aria-hidden="true">⚠</span>
      {TERMINAL_CLAMP_LABEL}
    </span>
  );
}
