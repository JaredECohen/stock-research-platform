import React from "react";
import type { NumberCheck, StockMemoOut } from "@/types";
import { isHidden } from "@/lib/memoSections";
import {
  capText,
  checksCoverHiddenText,
  CLAIM_TITLE,
  CRITIC_ASSESSMENT_TEXT,
  declaredAssumptionFor,
  fieldLabel,
  FLAGGED_STATUSES,
  orderedCaps,
  pmConfidenceIsTemplate,
  qualityOf,
  ratingCheckLines,
  sourceLabel,
  sourceLabels,
} from "@/lib/memoQuality";

/**
 * W2b — "Research checks": what the backend's quality stage found, in the
 * reader's words.
 *
 *  - Figures (7a): how many figures were checked against the data the
 *    analysts were given and how they came out, each flagged figure with
 *    where it appears, the PM's declared assumptions (labelled "PM
 *    assumption"), and the supporting points withheld from the memo behind
 *    a disclosure.
 *  - Rating check (7b): the check in sentences built from the record's
 *    fields (never the backend's note, which carries check names), the
 *    PM's reason, and whether a live critic reviewed it.
 *  - Confidence (7c): PM confidence before and after the caps, each cap as
 *    a sentence, the binding one first.
 *  - Sources: what the traced figures trace to, as labels.
 *
 * It follows the presenter's section map (W2a, owner decision 2): when the
 * memo's confidence is unavailable in this version, the raw -> final
 * numbers and the caps are not printed either — they would put back the
 * number the placeholder withholds. When the PM synthesis was template-
 * filled, the PM's "raw" confidence is the template's number, so only the
 * final (earned) confidence is printed. Flagged figures in hidden sections
 * were already dropped from the record by the presenter; the tallies and
 * cited sources are memo-wide, and say so when sections are hidden.
 *
 * Machine vocabulary (cap codes, field paths, source refs, industry
 * codes) never reaches the page: `lib/memoQuality` maps each to a label.
 *
 * Renders nothing for a memo without `quality` (every memo stored before
 * W2b), so those pages are unchanged.
 */
export default function QualityPanel({
  memo,
  variant = "card",
  className = "",
}: {
  memo: StockMemoOut;
  // `paper` is the print-friendly form for the Full Investment Memo; the
  // PDF popup maps only the `text-slate-*` classes, so it avoids others.
  variant?: "card" | "paper";
  className?: string;
}) {
  const q = qualityOf(memo);
  const nc = q?.number_check ?? null;
  const rec = q?.rating_reconciliation ?? null;
  const conf = q?.confidence ?? null;
  if (!q || (!nc && !rec && !conf)) return null;
  const paper = variant === "paper";
  const muted = paper ? "text-slate-400 print:text-slate-600" : "text-slate-400";
  const body = paper ? "text-slate-200 print:text-slate-800" : "text-slate-300";
  const head = paper
    ? "text-xs uppercase tracking-wider text-slate-400 print:text-slate-600 font-semibold"
    : "text-[10px] uppercase tracking-widest text-slate-500";
  const confidenceHidden = isHidden(memo, "confidence_score");
  const templateConfidence = pmConfidenceIsTemplate(memo);
  const wholeRun = checksCoverHiddenText(memo);

  return (
    <div
      className={`${paper ? "" : "card-tight"} text-sm space-y-3 ${className}`}
      data-testid="research-checks"
    >
      {!paper && <div className="section-title">Research checks</div>}

      {nc && (
        <div data-testid="research-checks-figures">
          <div className={head}>Figures</div>
          <FiguresBlock nc={nc} body={body} muted={muted} paper={paper} wholeRun={wholeRun} />
        </div>
      )}

      {rec && (
        <div data-testid="research-checks-rating">
          <div className={head}>Rating check</div>
          <RatingBlock memo={memo} body={body} muted={muted} />
        </div>
      )}

      {conf && (
        <div data-testid="research-checks-confidence">
          <div className={head}>Confidence</div>
          {confidenceHidden ? (
            <p className={`mt-1 ${muted}`}>Confidence is unavailable in this version.</p>
          ) : (
            <>
              <p className={`mt-1 ${body}`} data-testid="research-checks-confidence-line">
                {templateConfidence
                  ? `Confidence ${Math.round(conf.final)} after the research checks. The PM synthesis was template-filled, so it has no confidence of its own to show.`
                  : conf.final < conf.raw
                    ? `PM confidence ${Math.round(conf.raw)} → ${Math.round(conf.final)} after the research checks.`
                    : `PM confidence ${Math.round(conf.final)}; no research check lowered it.`}
              </p>
              {conf.caps.length > 0 && (
                <ul className={`mt-1 list-disc pl-5 space-y-0.5 text-xs ${body}`}>
                  {orderedCaps(conf).map((c, i) => (
                    <li key={`${c.code}-${i}`} data-binding={c.code === conf.binding ? "true" : undefined}>
                      <span className="font-mono">≤{Math.round(c.cap)}</span> — {capText(c)}
                      {c.code === conf.binding && conf.final < conf.raw ? " (binding)" : ""}
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
      )}

      {nc && nc.checked && nc.sources_cited.length > 0 && (
        <p className={`text-xs ${muted}`} data-testid="research-checks-sources">
          {wholeRun ? "Figures across the whole memo, including sections not shown, trace to:" : "Figures trace to:"}{" "}
          {sourceLabels(nc.sources_cited).join(" · ")}.
        </p>
      )}
    </div>
  );
}

function FiguresBlock({
  nc,
  body,
  muted,
  paper,
  wholeRun,
}: {
  nc: NumberCheck;
  body: string;
  muted: string;
  // The PDF prints the withheld points as a list: a toggle means nothing
  // on paper, and the popup loads none of the app's CSS to hide one.
  paper: boolean;
  // The tallies cover sections the presenter hid (see the header).
  wholeRun: boolean;
}) {
  const [showWithheld, setShowWithheld] = React.useState(false);
  if (!nc.checked) {
    return (
      <p className={`mt-1 ${muted}`} data-testid="research-checks-unchecked">
        Figures were not source-checked for this version.
      </p>
    );
  }
  const n = (k: string) => Number(nc.counts?.[k] ?? 0) || 0;
  const flagged = nc.claims.filter((c) => FLAGGED_STATUSES.has(c.status));
  const assumptions = nc.claims.filter((c) => c.status === "assumption");
  const notFound = n("untraceable");
  const parts = [
    `${n("claims_total")} checked`,
    `${n("traced")} traced to source data`,
    `${n("weak")} matched by value only`,
    `${notFound} not found in source data`,
  ];
  if (n("mis_anchored") > 0) parts.push(`${n("mis_anchored")} matched a different metric`);
  if (n("assumption") > 0) parts.push(`${n("assumption")} PM assumption${n("assumption") === 1 ? "" : "s"}`);
  const withheld = nc.withheld ?? [];
  return (
    <div className="mt-1 space-y-1.5">
      <p className={body} data-testid="research-checks-counts">
        {parts.join(" · ")}
      </p>
      {wholeRun && (
        <p className={`text-xs ${muted}`} data-testid="research-checks-scope">
          These counts cover the whole memo as generated, including sections not shown in this
          version and the withheld points.
        </p>
      )}
      {n("weak") > 0 && (
        <p className={`text-xs ${muted}`}>
          A value-only match is a round figure found in the data without the metric it names; it
          is counted, not flagged.
        </p>
      )}
      {flagged.length > 0 && (
        <ul className={`list-disc pl-5 space-y-0.5 text-xs ${body}`} data-testid="research-checks-flagged">
          {flagged.map((c, i) => (
            <li key={i}>
              <span className="font-mono">{c.raw}</span> in {fieldLabel(c.field)} —{" "}
              {c.status === "mis_anchored" ? "matches a different metric" : "not found in the data the analysts were given"}
            </li>
          ))}
        </ul>
      )}
      {paper && (flagged.length > 0 || assumptions.length > 0) && (
        <p className={`text-xs ${muted}`}>These figures are underlined with dots where they appear in the memo.</p>
      )}
      {assumptions.length > 0 && (
        <ul className={`list-disc pl-5 space-y-0.5 text-xs ${body}`} data-testid="research-checks-assumptions">
          {assumptions.map((c, i) => {
            const declared = declaredAssumptionFor(nc, c);
            const horizon = typeof declared?.horizon === "string" ? declared.horizon : "";
            const basis = typeof declared?.basis_ref === "string" ? sourceLabel(declared.basis_ref) : "";
            return (
              <li key={i} title={CLAIM_TITLE.assumption}>
                <span className="font-mono">{c.raw}</span> in {fieldLabel(c.field)} — PM assumption
                {horizon ? `, ${horizon}` : ""}
                {basis ? `, based on ${basis.toLowerCase()}` : ""}
              </li>
            );
          })}
        </ul>
      )}
      {withheld.length > 0 && (
        <div data-testid="research-checks-withheld">
          {paper ? (
            <p className={`text-xs ${body}`}>
              {withheld.length} withheld point{withheld.length === 1 ? "" : "s"}:
            </p>
          ) : (
            <button
              type="button"
              className="text-xs text-accent-500 hover:text-accent-400 print:hidden"
              aria-expanded={showWithheld}
              onClick={() => setShowWithheld((v) => !v)}
            >
              {showWithheld ? "Hide" : "Show"} {withheld.length} withheld point{withheld.length === 1 ? "" : "s"}
            </button>
          )}
          <p className={`text-xs ${muted}`}>
            Supporting points whose figures were not found in the source data are removed from the
            memo and kept here verbatim.
          </p>
          {(paper || showWithheld) && (
            <ul className={`mt-1 list-disc pl-5 space-y-0.5 text-xs ${body}`}>
              {withheld.map((w, i) => (
                <li key={i}>
                  <span className={muted}>{fieldLabel(w.field)}:</span> {w.text}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {nc.lists_not_withheld.length > 0 && (
        <p className={`text-xs ${muted}`}>
          Kept with flags, because most of the list was flagged:{" "}
          {Array.from(new Set(nc.lists_not_withheld.map(fieldLabel))).join(", ")}.
        </p>
      )}
      {nc.unchecked_fields.length > 0 && (
        <p className={`text-xs ${muted}`} data-testid="research-checks-unchecked-fields">
          Changed by a news update after the check, so not checked:{" "}
          {Array.from(new Set(nc.unchecked_fields.map(fieldLabel))).join(", ")}.
        </p>
      )}
    </div>
  );
}

function RatingBlock({ memo, body, muted }: { memo: StockMemoOut; body: string; muted: string }) {
  const rec = qualityOf(memo)?.rating_reconciliation;
  if (!rec) return null;
  const [line, ...after] = ratingCheckLines(memo);
  return (
    <div className="mt-1 space-y-0.5" data-testid="research-checks-rating-text">
      {line && <p className={body}>{line}</p>}
      {rec.divergence && rec.reason.trim() && (
        <p className={`text-xs ${body}`}>PM&apos;s reason: &ldquo;{rec.reason.trim()}&rdquo;</p>
      )}
      {rec.divergence && rec.reason.trim() && (
        <p className={`text-xs ${muted}`}>{CRITIC_ASSESSMENT_TEXT[rec.critic_assessment] ?? ""}</p>
      )}
      {after.map((l, i) => (
        <p key={i} className={`text-xs ${muted}`}>
          {l}
        </p>
      ))}
    </div>
  );
}
