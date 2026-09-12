import React from "react";
import type { FAQEntry } from "@/content/faq";
import { FOCUS_RING } from "./ctas";

/**
 * Native <details>/<summary> — keyboard-operable and announced without
 * any script, and deep-linkable by entry id.
 */
export default function FAQ({
  entries,
  headingLevel = 2,
  title = "Frequently asked questions",
}: {
  entries: FAQEntry[];
  headingLevel?: 1 | 2 | 3;
  title?: string;
}) {
  const H = `h${headingLevel}` as "h1" | "h2" | "h3";
  return (
    <section aria-labelledby="faq-heading" className="mt-16">
      <H id="faq-heading" className={headingLevel === 1 ? "text-3xl font-semibold tracking-tight" : "text-2xl font-semibold tracking-tight"}>
        {title}
      </H>
      <div className="mt-4 divide-y divide-ink-700 border-y border-ink-700">
        {entries.map((e) => (
          <details key={e.id} id={e.id} className="group py-3">
            <summary className={`cursor-pointer list-none flex items-center justify-between gap-3 text-base font-medium text-slate-100 rounded-sm ${FOCUS_RING}`}>
              <span>{e.question}</span>
              <span aria-hidden className="text-slate-400 group-open:rotate-45 motion-safe:transition-transform">+</span>
            </summary>
            <p className="text-sm text-slate-300 leading-relaxed mt-2 max-w-3xl">{e.answer}</p>
          </details>
        ))}
      </div>
    </section>
  );
}
