import React, { useMemo, useRef } from "react";
import type { IndustryAnchor, IndustryClaim, IndustryInterpretation, IndustryReport } from "@/types/industries";
import { INDUSTRY_FACTS_ONLY_SECTIONS, INDUSTRY_SECTIONS, INDUSTRY_SECTION_LABELS } from "@/types/industries";
import FactsView from "./FactsView";
import { fmtByPath, fmtPctSigned, humanize, isNum, na, unitFor } from "./format";

/**
 * The edition's sections, one tab each, with the two layers separated by
 * a heading that names them: **Observed data** (server-computed `facts`)
 * and **Analyst interpretation** (the model's or the template's reading
 * of those facts). That separation is the whole research contract of
 * this feature, and it is a structural heading rather than a style so a
 * screen reader hits it too.
 *
 * The tab list is `payload.section_order` when the edition carries one,
 * so a section added by a later writer shows up without a deploy here;
 * `INDUSTRY_SECTIONS` is only the fallback ordering. A section the order
 * names but the payload lacks is rendered as a stated absence rather
 * than dropped.
 *
 * Only analyst-written editions reach this component (owner decision 1).
 * A section a template filled inside one is listed in
 * `display.hidden_sections` and its interpretation is replaced by the
 * server's `hidden_reason`; its observed facts still render.
 *
 * The outlook's forward numbers are registered forecast assumptions
 * (owner decision 1): a table labelled "Analyst assumptions — not
 * observed data" prints each beside the observation it is anchored to,
 * and every such claim is badged "Analyst assumption", never as a fact.
 *
 * Keyboard: the tablist is a roving tabindex — Left/Right move and
 * select, Home/End reach the ends. Only the selected tab is tabbable, so
 * Tab from the page moves past the strip rather than through thirteen
 * stops.
 */

const FACTS_HEADING = "Observed data";
const INTERPRETATION_HEADING = "Analyst interpretation";
/** The label every registered forward number carries (owner decision 1):
 *  it is the analyst's assumption, printed beside the observation it
 *  departs from, and never presented as data. */
const ASSUMPTIONS_HEADING = "Analyst assumptions — not observed data";
const ASSUMPTION_BADGE = "Analyst assumption";

function sectionLabel(name: string): string {
  return (INDUSTRY_SECTION_LABELS as Record<string, string>)[name] ?? humanize(name);
}

/** The outlook's anchors catalogue (`facts.anchors`), or none. Read
 *  defensively: an edition written before the catalogue existed has none,
 *  and neither does any other section. */
function anchorsOf(facts: Record<string, unknown> | undefined): IndustryAnchor[] {
  const raw = facts?.anchors;
  return Array.isArray(raw)
    ? raw.filter((a): a is IndustryAnchor => !!a && typeof (a as IndustryAnchor).path === "string")
    : [];
}

function claimBadge(type: string): string {
  return type === "forecast_assumption" ? ASSUMPTION_BADGE : humanize(type);
}

/** An anchor's observed value in its family's unit. A multiple prints as
 *  "26.5x"; a rate takes the unit its path claims, and a rate path the
 *  formatter does not recognise is still a rate, never a bare decimal. */
function fmtAnchor(anchor: IndustryAnchor): string {
  if (!isNum(anchor.value)) return na("not on file");
  if (anchor.family === "multiple") return `${anchor.value}x`;
  return unitFor(anchor.path) === "plain" ? fmtPctSigned(anchor.value) : fmtByPath(anchor.path, anchor.value);
}

/** The outlook's registered forecast assumptions, one row each: the
 *  assumed value and horizon, the observed fact it is measured against
 *  (read from the edition's own anchors catalogue — the page does not
 *  resolve fact paths itself), what would break it, and which scenarios
 *  rest on it. */
function AssumptionsTable({
  assumptions,
  anchors,
  scenarios,
}: {
  assumptions: IndustryClaim[];
  anchors: IndustryAnchor[];
  scenarios: IndustryInterpretation["scenarios"];
}) {
  const byPath = new Map(anchors.map((a) => [a.path, a]));
  const usedBy = (id: string) =>
    Object.entries(scenarios ?? {})
      .filter(([, s]) => (s.assumption_ids ?? []).includes(id))
      .map(([name]) => humanize(name));
  return (
    <div className="space-y-1 text-xs" data-testid="assumptions-table">
      <div className="text-slate-500 uppercase tracking-widest text-[10px]">{ASSUMPTIONS_HEADING}</div>
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead className="text-slate-500">
            <tr>
              <th scope="col" className="pr-2 font-normal">Assumption</th>
              <th scope="col" className="pr-2 font-normal">Horizon</th>
              <th scope="col" className="pr-2 font-normal">Measured against (observed)</th>
              <th scope="col" className="pr-2 font-normal">Falsifier</th>
              <th scope="col" className="font-normal">Scenarios</th>
            </tr>
          </thead>
          <tbody>
            {assumptions.map((a, i) => {
              const anchor = a.anchor ? byPath.get(a.anchor) : undefined;
              const used = a.id ? usedBy(a.id) : [];
              return (
                <tr key={a.id || i} className="align-top border-t border-ink-800" data-testid={`assumption-${a.id ?? i}`}>
                  <td className="pr-2 text-slate-200">
                    <span className="text-slate-500 mr-1">{a.id}</span>
                    {a.value}
                  </td>
                  <td className="pr-2 text-slate-300">{a.horizon}</td>
                  <td className="pr-2 text-slate-300">
                    {anchor ? (
                      <>
                        {humanize(anchor.path)}: {fmtAnchor(anchor)}
                      </>
                    ) : (
                      na("anchor not in this edition's catalogue")
                    )}
                  </td>
                  <td className="pr-2 text-slate-400">{a.falsifier}</td>
                  <td className="text-slate-400">{used.length > 0 ? used.join(", ") : na("no scenario lists it")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** The analyst layer: prose, the eight-stage spine when the section is
 *  ordered by it, the registered forecast assumptions (outlook), scenarios
 *  (labelled as scenarios), and each claim with the evidence it rests on
 *  and what would break it. */
function Interpretation({
  interp,
  mode,
  anchors = [],
}: {
  interp: IndustryInterpretation;
  mode: string;
  anchors?: IndustryAnchor[];
}) {
  const assumptions = (interp.claims ?? []).filter((c) => c.type === "forecast_assumption");
  return (
    <div className="text-sm space-y-3" data-testid="interpretation-view">
      <p className="text-slate-200 whitespace-pre-line">{interp.text}</p>

      {Array.isArray(interp.stages) && interp.stages.length > 0 && (
        <ol className="space-y-1 text-xs" data-testid="interpretation-stages">
          {interp.stages.map((s, i) => (
            <li key={s.id || i}>
              <span className="text-slate-500 mr-1">{i + 1}. {humanize(s.id)}:</span>
              <span className="text-slate-300">{s.text}</span>
            </li>
          ))}
        </ol>
      )}

      {assumptions.length > 0 && (
        <AssumptionsTable assumptions={assumptions} anchors={anchors} scenarios={interp.scenarios} />
      )}

      {interp.scenarios && Object.keys(interp.scenarios).length > 0 && (
        <div className="space-y-1 text-xs" data-testid="interpretation-scenarios">
          <div className="text-slate-500 uppercase tracking-widest text-[10px]">
            Scenarios — not forecasts, not recommendations
          </div>
          <ul className="space-y-1">
            {Object.entries(interp.scenarios).map(([name, s]) => (
              <li key={name}>
                <span className="text-slate-400">{humanize(name)}:</span>{" "}
                <span className="text-slate-300">{s.text}</span>
                {s.assumption_ids && s.assumption_ids.length > 0 && (
                  <span className="text-slate-500" data-testid={`scenario-uses-${name}`}>
                    {" "}
                    (uses {s.assumption_ids.join(", ")})
                  </span>
                )}
                {s.falsifiers && s.falsifiers.length > 0 && (
                  <div className="text-slate-500">
                    Falsifier{s.falsifiers.length === 1 ? "" : "s"}: {s.falsifiers.join("; ")}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {Array.isArray(interp.claims) && interp.claims.length > 0 && (
        <details className="text-xs" data-testid="interpretation-claims">
          <summary className="cursor-pointer text-slate-400">
            {interp.claims.length} claim{interp.claims.length === 1 ? "" : "s"}, with basis and falsifier
          </summary>
          <ul className="mt-1 space-y-1">
            {interp.claims.map((c, i) => (
              <li key={i} className="border-l border-ink-800 pl-2">
                <span className="badge text-[10px] border-ink-700 text-slate-400 mr-1">{claimBadge(c.type)}</span>
                <span className="text-slate-300">{c.text}</span>
                <div className="text-slate-500">
                  Basis: {c.basis.length > 0 ? c.basis.join(", ") : na("no basis recorded")}
                </div>
                <div className="text-slate-500">
                  Falsifier:{" "}
                  {c.falsifier || (
                    <span>
                      {na(
                        c.type === "observed_fact"
                          ? "an observed fact carries no falsifier"
                          : "none recorded on this claim",
                      )}
                    </span>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </details>
      )}

      <p className="text-[11px] text-slate-500" data-testid="interpretation-provenance">
        Written by: {mode === "llm" ? "the industry analyst model" : `deterministic template (${mode || "mode not recorded"})`}.
      </p>
    </div>
  );
}

export interface ReportTabsProps {
  report: IndustryReport;
  /** The selected section; the page keeps it in the URL. */
  section: string;
  onSelect: (section: string) => void;
  /** Extra content for a section, rendered under the observed-data
   *  heading — the companies table, the changes panel. */
  extras?: Record<string, React.ReactNode>;
  className?: string;
}

export default function ReportTabs({ report, section, onSelect, extras = {}, className = "" }: ReportTabsProps) {
  const order = useMemo(() => {
    const fromPayload = report.payload?.section_order;
    return Array.isArray(fromPayload) && fromPayload.length > 0 ? fromPayload : [...INDUSTRY_SECTIONS];
  }, [report]);
  const current = order.includes(section) ? section : order[0];
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  const go = (to: number) => {
    const i = Math.max(0, Math.min(order.length - 1, to));
    const next = order[i];
    onSelect(next);
    tabRefs.current[next]?.focus();
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const i = order.indexOf(current);
    switch (e.key) {
      case "ArrowRight":
        e.preventDefault();
        go(i + 1 >= order.length ? 0 : i + 1);
        break;
      case "ArrowLeft":
        e.preventDefault();
        go(i - 1 < 0 ? order.length - 1 : i - 1);
        break;
      case "Home":
        e.preventDefault();
        go(0);
        break;
      case "End":
        e.preventDefault();
        go(order.length - 1);
        break;
      default:
        break;
    }
  };

  const body = report.payload?.sections?.[current];
  const narrativeMode = report.payload?.narrative_by_section?.[current] ?? report.payload?.analyst_narrative ?? "";
  const factsOnly = INDUSTRY_FACTS_ONLY_SECTIONS.includes(current);
  // A section a template filled inside an analyst edition (owner decision
  // 1). The server has already nulled its interpretation; the page says why
  // in the server's words instead of printing "no interpretation", which
  // would read as a section the analyst skipped.
  const hidden = (report.display?.hidden_sections ?? []).includes(current);

  return (
    <div className={`space-y-3 ${className}`} data-testid="industry-report-tabs">
      <div
        role="tablist"
        aria-label="Report sections"
        onKeyDown={onKeyDown}
        className="card-tight flex flex-wrap gap-1"
      >
        {order.map((name) => (
          <button
            key={name}
            type="button"
            role="tab"
            id={`industry-tab-${name}`}
            aria-selected={name === current}
            aria-controls={`industry-panel-${name}`}
            tabIndex={name === current ? 0 : -1}
            ref={(el) => {
              tabRefs.current[name] = el;
            }}
            onClick={() => onSelect(name)}
            data-testid={`tab-${name}`}
            className={`px-2 py-1 rounded text-xs ${
              name === current ? "bg-accent-600/15 text-accent-500 border border-accent-600/30" : "text-slate-300 border border-transparent hover:bg-ink-800"
            }`}
          >
            {sectionLabel(name)}
          </button>
        ))}
      </div>

      <section
        role="tabpanel"
        id={`industry-panel-${current}`}
        aria-labelledby={`industry-tab-${current}`}
        tabIndex={0}
        className="space-y-4"
        data-testid={`panel-${current}`}
      >
        <h2 className="section-title">{sectionLabel(current)}</h2>

        {!body ? (
          <p className="card-tight text-xs text-slate-400" role="status" data-testid="section-missing">
            {na(`this edition carries no "${sectionLabel(current)}" section`)}
          </p>
        ) : (
          <>
            <div className="card space-y-2">
              <h3 className="text-xs uppercase tracking-widest text-slate-500" data-testid="heading-facts">
                {FACTS_HEADING}
              </h3>
              {extras[current]}
              <FactsView facts={body.facts ?? {}} />
            </div>

            <div className="card space-y-2">
              <h3 className="text-xs uppercase tracking-widest text-slate-500" data-testid="heading-interpretation">
                {INTERPRETATION_HEADING}
              </h3>
              {hidden ? (
                <p className="text-xs text-slate-400" role="status" data-testid="interpretation-hidden">
                  {report.display?.hidden_reason || "Analyst interpretation unavailable in this version."}
                </p>
              ) : body.interpretation ? (
                <Interpretation interp={body.interpretation} mode={narrativeMode} anchors={anchorsOf(body.facts)} />
              ) : (
                <p className="text-xs text-slate-400" data-testid="interpretation-absent">
                  {na(
                    factsOnly
                      ? "this section is observed data only — the analyst does not write it"
                      : "no analyst interpretation on this edition",
                  )}
                </p>
              )}
            </div>
          </>
        )}
      </section>
    </div>
  );
}

export { ASSUMPTION_BADGE, ASSUMPTIONS_HEADING, FACTS_HEADING, INTERPRETATION_HEADING };
