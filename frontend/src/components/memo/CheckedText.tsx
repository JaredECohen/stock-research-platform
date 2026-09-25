import React from "react";
import type { NumberClaim } from "@/types";
import { CLAIM_TITLE, FLAGGED_STATUSES } from "@/lib/memoQuality";

/**
 * W2b 7(a) — memo prose with its checked figures marked in place.
 *
 * `claims` are the number check's stored claims for this one field
 * (`claimsFor(memo, field)`). Each names `[start, end)` in the field's text
 * and the exact `raw` slice. A flagged figure (not found in the data the
 * analysts were given, or found under a different metric) gets a dotted
 * underline and a tooltip; a declared PM assumption gets a lighter mark
 * saying so. Everything else renders as plain text.
 *
 * Defensive by contract: the offsets index the text AS STORED, and text can
 * move under them (a presenter placeholder, a news patch, a prefix written
 * after the check). A claim whose slice is not `raw`, that runs out of
 * range, or that overlaps one already placed is skipped — the figure then
 * renders unmarked, which is wrong in the safe direction. This never
 * throws, so a bad offset can never take the memo page down.
 */
export default function CheckedText({
  text,
  claims,
}: {
  text: string | null | undefined;
  claims?: readonly NumberClaim[] | null;
}) {
  const s = text ?? "";
  const spans = placeable(s, claims ?? []);
  if (spans.length === 0) return <>{s}</>;
  const parts: React.ReactNode[] = [];
  let at = 0;
  spans.forEach((c, i) => {
    if (c.start > at) parts.push(s.slice(at, c.start));
    const flagged = FLAGGED_STATUSES.has(c.status);
    parts.push(
      <span
        key={i}
        data-claim-status={c.status}
        title={CLAIM_TITLE[c.status]}
        className={
          flagged
            ? "underline decoration-dotted decoration-warn-500 underline-offset-2 cursor-help"
            : "underline decoration-dotted decoration-slate-500 underline-offset-2 cursor-help"
        }
      >
        {s.slice(c.start, c.end)}
      </span>,
    );
    at = c.end;
  });
  if (at < s.length) parts.push(s.slice(at));
  return <>{parts}</>;
}

/** The claims that can be marked, in text order: valid integer offsets in
 * range, a slice equal to `raw`, and no overlap with an earlier one. */
export function placeable(text: string, claims: readonly NumberClaim[]): NumberClaim[] {
  const ok = claims
    .filter(
      (c) =>
        c != null &&
        Number.isInteger(c.start) &&
        Number.isInteger(c.end) &&
        c.start >= 0 &&
        c.end > c.start &&
        c.end <= text.length &&
        typeof c.raw === "string" &&
        text.slice(c.start, c.end) === c.raw,
    )
    .sort((a, b) => a.start - b.start || a.end - b.end);
  const out: NumberClaim[] = [];
  let end = 0;
  for (const c of ok) {
    if (c.start < end) continue;
    out.push(c);
    end = c.end;
  }
  return out;
}
