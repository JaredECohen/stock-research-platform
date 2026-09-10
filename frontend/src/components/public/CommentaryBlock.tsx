import React from "react";
import { formatShortUtc } from "@/lib/entitlements";
import type { SampleCommentary } from "@/types/public";

/**
 * The one-paragraph framing the worker asked a model to write when the
 * sample was built. The provenance line is not optional: which model,
 * when, and that it is a model output — the reader should never mistake
 * it for a human analyst's note or a recommendation.
 */
export default function CommentaryBlock({ commentary, headingLevel = 2 }: { commentary: SampleCommentary; headingLevel?: 2 | 3 }) {
  const H = `h${headingLevel}` as "h2" | "h3";
  const when = formatShortUtc(commentary.generated_at || null);
  return (
    <section aria-labelledby="sample-commentary-heading" className="card border-accent-600/30">
      <H id="sample-commentary-heading" className="text-xs uppercase tracking-widest text-accent-500 font-semibold">
        Committee commentary
      </H>
      <p className="text-sm text-slate-200 leading-relaxed mt-2">{commentary.text}</p>
      <p className="text-[11px] text-slate-400 mt-3">
        Written by model <span className="font-mono">{commentary.model || "unknown"}</span>
        {when ? ` on ${when}` : ""}. A model output for context — not a recommendation.
      </p>
    </section>
  );
}
