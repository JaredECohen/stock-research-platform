import React from "react";
import { FileSearch, Gavel, MessageSquare, PenLine } from "lucide-react";
import type { LucideIcon } from "lucide-react";

/**
 * Agents → committee → memo → your questions. Describes what the pipeline
 * does today (the roster in backend/app/agents), not what it might do.
 */
const STEPS: Array<{ icon: LucideIcon; title: string; body: string }> = [
  {
    icon: FileSearch,
    title: "Specialists gather the evidence",
    body:
      "Separate agents read the filings, the latest earnings call, reported fundamentals, prices and macro context, and each writes up what it found with its sources and its confidence.",
  },
  {
    icon: Gavel,
    title: "The committee argues",
    body:
      "A bull case and a bear case are built and tested against each other; a risk committee challenges the thesis and lists the risks it thinks are underweighted. Disagreements are recorded, not smoothed over.",
  },
  {
    icon: PenLine,
    title: "The portfolio manager writes the memo",
    body:
      "One rating, one mispricing thesis — consensus view, our view, the gap and what would falsify it — plus a valuation verdict from a DCF and comps. Missing evidence stays visibly missing.",
  },
  {
    icon: MessageSquare,
    title: "You interrogate it",
    body:
      "Ask the PM about any stored memo, adjust DCF assumptions in the lab, compare peers, screen the universe, and re-run the committee when the facts change.",
  },
];

export default function HowItWorks() {
  return (
    <section aria-labelledby="how-heading" className="mt-16">
      <h2 id="how-heading" className="text-2xl font-semibold tracking-tight">How it works</h2>
      <ol className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 mt-4">
        {STEPS.map((s, i) => (
          <li key={s.title} className="card flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-accent-600/15 border border-accent-600/30 flex items-center justify-center text-accent-500">
                <s.icon size={16} aria-hidden />
              </div>
              <span className="font-mono text-xs text-slate-400">0{i + 1}</span>
            </div>
            <h3 className="text-base font-semibold">{s.title}</h3>
            <p className="text-sm text-slate-300 leading-relaxed">{s.body}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}
