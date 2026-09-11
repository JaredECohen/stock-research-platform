import React from "react";
import { fmtDateTime, fmtPctSigned, humanize, isNum, na } from "./format";

/**
 * A generic, honest renderer for a report section's `facts`.
 *
 * The sections are heterogeneous by design — the server computes what
 * each one can observe, and a group with no filings on file has a
 * different `themes` shape from one with twenty. Rather than thirteen
 * bespoke layouts that silently drop a key the backend adds later, this
 * walks the object and renders whatever is there, with three rules:
 *
 *   * **`{value: null, reason}` is the API's way of saying "missing, and
 *     here is why".** It renders as "n/a (reason)" — never as a blank
 *     row, never as 0, and never dropped for being null.
 *   * **A null without a reason is still reported**, as
 *     "n/a (no reason recorded)". Swallowing it would hide a backend that
 *     forgot to say why.
 *   * **Nothing is truncated silently.** Long lists print a counted
 *     "+N more" line rather than stopping.
 *
 * Keys the reader does not need repeated on every section (the
 * attribution and caveat strings, which the header already carries
 * verbatim) are dropped by name and listed in `OMITTED_KEYS`, so the
 * omission is a decision on the page rather than an accident.
 */

/** Rendered elsewhere on the page, verbatim; skipped here to avoid
 *  printing the same rights statement thirteen times. */
export const OMITTED_KEYS: readonly string[] = ["attribution", "mapping_caveat", "disclaimer"];

const LIST_CAP = 12;

const ISO_DATETIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/;

/**
 * Is a number at this PATH a rate the analytics layer emitted as a
 * fraction?
 *
 * The path, not the leaf key: `returns.1m.median` and
 * `benchmark_relative.universe_ew.1m.value` are returns, but their leaf
 * keys are "median" and "value". Deciding on the leaf alone printed
 * "Equal weight: +1.08%" directly above "Median: 0.009186" — the same
 * quantity, two units, in adjacent rows.
 *
 * Sample sizes travel inside those same blocks, so they are excluded by
 * name: `n`, `n_mcw`, `benchmark_n` are counts however deep they sit.
 */
const RATE_ANCESTORS = /(^|\.)(returns?|ret|benchmark_relative|breadth|dispersion|margins?|growth|coverage|weight|yield|pct|share)(\.|$)/i;
const COUNT_LEAF = /^(n|n_[a-z_]+|[a-z_]*_n|count|sessions|order|id|code|version|year|limit|budget|min_sample|attempts|max_attempts)$/i;

export function isRate(path: string): boolean {
  const leaf = path.split(".").pop() ?? path;
  if (COUNT_LEAF.test(leaf)) return false;
  return RATE_ANCESTORS.test(path);
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** `{value: null, reason}` / `{value: x, ...}` — the API's missing-value
 *  cell. Returns the rendered string, or null when this is not one. */
function missingCell(v: Record<string, unknown>): string | null {
  if (!("reason" in v)) return null;
  const hasValue = "value" in v;
  if (hasValue && v.value === null) return na(String(v.reason ?? "no reason recorded"));
  if (!hasValue && Object.keys(v).length === 1) return na(String(v.reason ?? "no reason recorded"));
  return null;
}

function Scalar({ path, value }: { path: string; value: unknown }) {
  if (value === null || value === undefined) {
    return <span className="text-slate-500">{na("no reason recorded")}</span>;
  }
  if (typeof value === "boolean") return <span>{value ? "yes" : "no"}</span>;
  if (isNum(value)) {
    return <span className="tabular-nums">{isRate(path) && Math.abs(value) <= 10 ? fmtPctSigned(value) : String(value)}</span>;
  }
  if (typeof value === "string" && ISO_DATETIME.test(value)) {
    // The same rendering the header gives an as-of, for the same reason:
    // a naive timestamp is UTC and must not be restamped by the viewer's
    // offset just because it appears deeper in the payload.
    return <span>{fmtDateTime(value)}</span>;
  }
  return <span>{String(value)}</span>;
}

function Node({ name, path, value, depth }: { name: string; path: string; value: unknown; depth: number }) {
  if (isPlainObject(value)) {
    const missing = missingCell(value);
    if (missing !== null) {
      return (
        <div className="flex flex-wrap gap-x-2">
          <span className="text-slate-500">{humanize(name)}:</span>
          <span className="text-slate-400">{missing}</span>
        </div>
      );
    }
    const entries = Object.entries(value).filter(([k]) => !OMITTED_KEYS.includes(k));
    if (entries.length === 0) {
      return (
        <div className="flex flex-wrap gap-x-2">
          <span className="text-slate-500">{humanize(name)}:</span>
          <span className="text-slate-400">{na("empty on this edition")}</span>
        </div>
      );
    }
    return (
      <div className={depth > 0 ? "mt-1" : ""}>
        <div className="text-slate-500">{humanize(name)}</div>
        <div className="pl-3 border-l border-ink-800 space-y-0.5">
          {entries.map(([k, v]) => (
            <Node key={k} name={k} path={`${path}.${k}`} value={v} depth={depth + 1} />
          ))}
        </div>
      </div>
    );
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return (
        <div className="flex flex-wrap gap-x-2">
          <span className="text-slate-500">{humanize(name)}:</span>
          <span className="text-slate-400">{na("none on this edition")}</span>
        </div>
      );
    }
    const shown = value.slice(0, LIST_CAP);
    return (
      <div className={depth > 0 ? "mt-1" : ""}>
        <div className="text-slate-500">
          {humanize(name)} <span className="text-slate-600">({value.length})</span>
        </div>
        <ul className="pl-3 border-l border-ink-800 space-y-0.5">
          {shown.map((item, i) => (
            <li key={i}>
              {isPlainObject(item) || Array.isArray(item) ? (
                <Node name={`#${i + 1}`} path={path} value={item} depth={depth + 1} />
              ) : (
                <Scalar path={path} value={item} />
              )}
            </li>
          ))}
          {value.length > shown.length && (
            <li className="text-slate-500">
              +{value.length - shown.length} more not shown
            </li>
          )}
        </ul>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap gap-x-2">
      <span className="text-slate-500">{humanize(name)}:</span>
      <span className="text-slate-200">
        <Scalar path={path} value={value} />
      </span>
    </div>
  );
}

export default function FactsView({ facts, className = "" }: { facts: Record<string, unknown>; className?: string }) {
  const entries = Object.entries(facts ?? {}).filter(([k]) => !OMITTED_KEYS.includes(k));
  if (entries.length === 0) {
    return (
      <p className={`text-xs text-slate-400 ${className}`} data-testid="facts-empty">
        {na("this section recorded no observed data")}
      </p>
    );
  }
  return (
    <div className={`text-xs space-y-1 ${className}`} data-testid="facts-view">
      {entries.map(([k, v]) => (
        <Node key={k} name={k} path={k} value={v} depth={0} />
      ))}
    </div>
  );
}
