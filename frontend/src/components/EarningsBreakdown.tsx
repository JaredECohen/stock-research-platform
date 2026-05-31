// Renders the structured earnings-call payload (EarningsStructured)
// inside the Earnings Analyst card so users see the actual call breakdown
// (tone, guidance changes, tone signals with quoted evidence, Q&A themes
// with response-quality coloring, most defended/pressed segments, forward
// catalysts) — not just the headline + summary.

import type { EarningsStructured } from "@/types";

const TONE_BADGE: Record<EarningsStructured["overall_tone"], string> = {
  constructive: "bg-accent-500/15 text-accent-400 border-accent-500/30",
  measured: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  cautious: "bg-warn-500/15 text-warn-500 border-warn-500/30",
};

const TONE_SIGNAL_BADGE: Record<string, string> = {
  constructive: "bg-accent-500/15 text-accent-400 border-accent-500/30",
  measured: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  cautious: "bg-warn-500/15 text-warn-500 border-warn-500/30",
  defensive: "bg-warn-500/15 text-warn-500 border-warn-500/30",
  evasive: "bg-danger-500/15 text-danger-500 border-danger-500/30",
};

const DIRECTION_BADGE: Record<string, string> = {
  raised: "bg-accent-500/15 text-accent-400 border-accent-500/30",
  lowered: "bg-danger-500/15 text-danger-500 border-danger-500/30",
  reaffirmed: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  introduced: "bg-blue-500/15 text-blue-300 border-blue-500/30",
  withdrawn: "bg-danger-500/15 text-danger-500 border-danger-500/30",
  unclear: "bg-slate-700 text-slate-400 border-slate-600",
};

const QA_BADGE: Record<string, string> = {
  clear: "bg-accent-500/15 text-accent-400 border-accent-500/30",
  partial: "bg-warn-500/15 text-warn-500 border-warn-500/30",
  deflected: "bg-warn-500/15 text-warn-500 border-warn-500/30",
  evasive: "bg-danger-500/15 text-danger-500 border-danger-500/30",
};

const MATERIALITY_BADGE: Record<string, string> = {
  high: "bg-accent-500/15 text-accent-400 border-accent-500/30",
  medium: "bg-warn-500/15 text-warn-500 border-warn-500/30",
  low: "bg-slate-500/15 text-slate-300 border-slate-500/30",
};

function Pill({
  text,
  classes,
}: {
  text: string;
  classes: string;
}) {
  return (
    <span
      className={`text-[10px] uppercase tracking-widest px-1.5 py-0.5 rounded border ${classes}`}
    >
      {text}
    </span>
  );
}

export default function EarningsBreakdown({
  structured,
}: {
  structured: EarningsStructured;
}) {
  const {
    period,
    overall_tone,
    guidance_changes,
    tone_signals,
    qa_themes,
    most_defended_segment,
    most_pressed_segment,
    forward_catalysts,
  } = structured;

  const hasAnything =
    overall_tone ||
    period ||
    guidance_changes?.length ||
    tone_signals?.length ||
    qa_themes?.length ||
    most_defended_segment?.name ||
    most_pressed_segment?.name ||
    forward_catalysts?.length;
  if (!hasAnything) return null;

  return (
    <div className="mt-3 border-t border-ink-700 pt-3 space-y-3">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="section-title">Call breakdown</div>
        <div className="flex items-center gap-2">
          {period && (
            <span className="text-[10px] uppercase tracking-widest text-slate-500">
              {period}
            </span>
          )}
          {overall_tone && (
            <Pill text={overall_tone} classes={TONE_BADGE[overall_tone]} />
          )}
        </div>
      </div>

      {guidance_changes?.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">
            Guidance changes
          </div>
          <ul className="space-y-1.5">
            {guidance_changes.map((g, i) => (
              <li
                key={i}
                className="text-xs text-slate-300 rounded border border-ink-700 bg-ink-900/40 px-2 py-1.5"
              >
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <span className="font-medium text-slate-100">{g.metric}</span>
                  <Pill
                    text={g.direction}
                    classes={
                      DIRECTION_BADGE[g.direction] || DIRECTION_BADGE.unclear
                    }
                  />
                </div>
                {(g.prior || g.current) && (
                  <div className="text-[11px] text-slate-400 mt-0.5">
                    {g.prior && (
                      <span>
                        Prior:{" "}
                        <span className="text-slate-300 font-mono">
                          {g.prior}
                        </span>
                      </span>
                    )}
                    {g.prior && g.current && <span className="mx-1.5">→</span>}
                    {g.current && (
                      <span>
                        Current:{" "}
                        <span className="text-slate-200 font-mono">
                          {g.current}
                        </span>
                      </span>
                    )}
                  </div>
                )}
                {g.rationale && (
                  <div className="text-[11px] text-slate-400 mt-0.5">
                    {g.rationale}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {tone_signals?.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">
            Tone signals
          </div>
          <ul className="space-y-1.5">
            {tone_signals.map((t, i) => (
              <li
                key={i}
                className="text-xs text-slate-300 rounded border border-ink-700 bg-ink-900/40 px-2 py-1.5"
              >
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <span className="text-slate-200">
                    {t.speaker && (
                      <span className="font-medium">{t.speaker}</span>
                    )}
                    {t.speaker && t.segment && (
                      <span className="text-slate-500"> · </span>
                    )}
                    {t.segment && (
                      <span className="text-slate-400">{t.segment}</span>
                    )}
                  </span>
                  <Pill
                    text={t.classification}
                    classes={
                      TONE_SIGNAL_BADGE[t.classification] ||
                      TONE_SIGNAL_BADGE.measured
                    }
                  />
                </div>
                {t.evidence && (
                  <div className="text-[11px] text-slate-400 italic mt-0.5 border-l-2 border-ink-600 pl-2">
                    “{t.evidence}”
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {qa_themes?.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">
            Q&A themes
          </div>
          <ul className="space-y-1.5">
            {qa_themes.map((q, i) => (
              <li
                key={i}
                className="text-xs text-slate-300 rounded border border-ink-700 bg-ink-900/40 px-2 py-1.5"
              >
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <span className="text-slate-200">
                    <span className="font-medium">{q.theme}</span>
                    {q.analyst && (
                      <span className="text-slate-500"> · {q.analyst}</span>
                    )}
                  </span>
                  <Pill
                    text={q.response_quality}
                    classes={
                      QA_BADGE[q.response_quality] || QA_BADGE.partial
                    }
                  />
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {(most_defended_segment?.name || most_pressed_segment?.name) && (
        <div className="grid sm:grid-cols-2 gap-2">
          {most_defended_segment?.name && (
            <div className="rounded border border-accent-500/30 bg-accent-500/[0.04] px-2 py-1.5">
              <div className="text-[10px] uppercase tracking-widest text-accent-400 mb-0.5">
                Most defended
              </div>
              <div className="text-sm text-slate-100">
                {most_defended_segment.name}
              </div>
              {most_defended_segment.why && (
                <div className="text-[11px] text-slate-400 mt-0.5">
                  {most_defended_segment.why}
                </div>
              )}
            </div>
          )}
          {most_pressed_segment?.name && (
            <div className="rounded border border-warn-500/30 bg-warn-500/[0.04] px-2 py-1.5">
              <div className="text-[10px] uppercase tracking-widest text-warn-500 mb-0.5">
                Most pressed
              </div>
              <div className="text-sm text-slate-100">
                {most_pressed_segment.name}
              </div>
              {most_pressed_segment.why && (
                <div className="text-[11px] text-slate-400 mt-0.5">
                  {most_pressed_segment.why}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {forward_catalysts?.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">
            Forward catalysts
          </div>
          <ul className="space-y-1">
            {forward_catalysts.map((c, i) => (
              <li
                key={i}
                className="text-xs text-slate-300 flex items-center justify-between gap-2 flex-wrap"
              >
                <span>
                  <span className="text-slate-100">{c.event ?? "—"}</span>
                  {c.expected_quarter && (
                    <span className="text-slate-500">
                      {" "}
                      · {c.expected_quarter}
                    </span>
                  )}
                </span>
                {c.materiality && (
                  <Pill
                    text={c.materiality}
                    classes={
                      MATERIALITY_BADGE[c.materiality.toLowerCase()] ||
                      MATERIALITY_BADGE.medium
                    }
                  />
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
