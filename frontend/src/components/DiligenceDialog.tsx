// Wave 9 — renders the PM↔specialist deep-research dialog audit trail.
// Round 0 (the parallel fan-out) is suppressed since its findings are
// already visible in the main agent grid; this panel surfaces ONLY the
// PM's follow-up rounds — the dig-deeper questions and the specialist's
// new view in response.

import { useState } from "react";
import type { RoundFindings } from "../types";

const TARGET_LABEL: Record<string, string> = {
  sector: "Sector Analyst",
  earnings: "Earnings Analyst",
  filing: "Filing Analyst",
  valuation: "Valuation Analyst",
  comps: "Comps Analyst",
  macro: "Macro Analyst",
  risk: "Risk Analyst",
  technical: "Technical Analyst",
};

export default function DiligenceDialog({
  rounds,
}: {
  rounds: RoundFindings[];
}) {
  const followUps = rounds.filter((r) => r.round > 0);
  if (followUps.length === 0) return null;

  const totalQuestions = followUps.reduce(
    (n, r) => n + (r.pm_questions?.length ?? 0),
    0,
  );
  const finalRound = followUps[followUps.length - 1];
  const exitedByConsensus = finalRound?.early_exit ?? false;

  return (
    <div className="card-tight border-accent-600/30 bg-accent-600/[0.03]">
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="section-title">Diligence dialog</div>
        <div className="text-[10px] uppercase tracking-widest text-slate-500">
          {totalQuestions} follow-up{totalQuestions === 1 ? "" : "s"} ·{" "}
          {followUps.length} round{followUps.length === 1 ? "" : "s"} ·{" "}
          {exitedByConsensus ? "PM satisfied" : "round budget hit"}
        </div>
      </div>
      <p className="text-xs text-slate-400 leading-relaxed mb-3">
        After the parallel fan-out, the PM critiqued the specialists' findings
        and asked targeted follow-ups. Each question below was answered by the
        named specialist with a fresh LLM call — re-fired with the question
        prepended to its prompt. The latest-round answer is what flows into the
        final PM synthesis.
      </p>
      <div className="space-y-3">
        {followUps.map((r) => (
          <RoundBlock key={r.round} round={r} />
        ))}
      </div>
    </div>
  );
}

function RoundBlock({ round }: { round: RoundFindings }) {
  // Tell apart a real "no more questions" exit (good signal) from an
  // LLM failure (which used to render as a confusing "LLM call failed"
  // quote followed by the satisfied-PM tagline).
  const rationale = round.pm_rationale ?? "";
  const failedExit = /LLM call failed|critique unavailable/i.test(rationale);
  return (
    <div className="rounded border border-ink-700 bg-ink-900/40 p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="text-[11px] uppercase tracking-widest text-accent-400 font-semibold">
          Round {round.round}
          {round.early_exit && !failedExit && " · PM declared no further questions"}
          {round.early_exit && failedExit && " · dialog skipped"}
        </div>
        {rationale && !failedExit && (
          <div className="text-[11px] text-slate-500 italic max-w-[60%] text-right">
            "{rationale}"
          </div>
        )}
      </div>
      {round.pm_questions.length === 0 && round.early_exit && (
        failedExit ? (
          <div className="text-xs text-warn-500 leading-relaxed">
            PM critique was unavailable for this memo (the LLM call did not
            return). Round-0 findings shipped as-is — no follow-up dialog ran.
            This usually means the configured PM model / provider is offline
            or unkeyed in this environment.
          </div>
        ) : (
          <div className="text-xs text-slate-500">
            PM ended the dialog — round-N findings already triangulate.
          </div>
        )
      )}
      <div className="space-y-3">
        {round.pm_questions.map((q, i) => {
          const finding = round.findings?.[q.target_agent];
          return (
            <QuestionBlock
              key={i}
              targetAgent={q.target_agent}
              question={q.question}
              whyItMatters={q.why_it_matters}
              answer={finding?.summary}
              answerHeadline={finding?.headline}
              answerKeyPoints={finding?.key_points}
              answerConfidence={finding?.confidence}
            />
          );
        })}
      </div>
    </div>
  );
}

function QuestionBlock({
  targetAgent,
  question,
  whyItMatters,
  answer,
  answerHeadline,
  answerKeyPoints,
  answerConfidence,
}: {
  targetAgent: string;
  question: string;
  whyItMatters?: string;
  answer?: string;
  answerHeadline?: string;
  answerKeyPoints?: string[];
  answerConfidence?: number;
}) {
  const [open, setOpen] = useState(false);
  const label = TARGET_LABEL[targetAgent] ?? targetAgent;
  const hasAnswer = !!answer;

  return (
    <div className="rounded bg-ink-800/40 border border-ink-700/60">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full text-left px-3 py-2 flex items-start justify-between gap-2 hover:bg-ink-800/70 transition-colors"
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-slate-500">
            <span className="text-accent-400 font-semibold">PM →</span>
            <span>{label}</span>
          </div>
          <div className="text-sm text-slate-100 mt-1 leading-snug">
            {question}
          </div>
          {whyItMatters && (
            <div className="text-[11px] text-slate-500 mt-1 italic">
              Why it matters: {whyItMatters}
            </div>
          )}
        </div>
        <div className="text-[11px] text-slate-500 shrink-0 mt-1">
          {hasAnswer ? (open ? "hide" : "show answer") : "no answer"}
        </div>
      </button>
      {open && hasAnswer && (
        <div className="px-3 pb-3 border-t border-ink-700/60 pt-2 text-sm text-slate-300 space-y-2">
          {answerHeadline && (
            <div className="text-slate-100 font-medium">{answerHeadline}</div>
          )}
          <div className="leading-relaxed">{answer}</div>
          {answerKeyPoints && answerKeyPoints.length > 0 && (
            <ul className="list-disc pl-5 space-y-1 text-slate-300">
              {answerKeyPoints.slice(0, 5).map((p, i) => (
                <li key={i}>{p}</li>
              ))}
            </ul>
          )}
          {typeof answerConfidence === "number" && (
            <div className="text-[10px] uppercase tracking-widest text-slate-500">
              specialist confidence: {Math.round(answerConfidence * 100)}%
            </div>
          )}
        </div>
      )}
    </div>
  );
}
