import React from "react";
import { fmtByPath, fmtDateTime, humanize, isNum, na, unitFor } from "./format";

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
 * Is a number at this PATH a percent?
 *
 * The unit rules live in `format.unitFor` because two renderers share
 * them — this one and the changes table — and a quantity that printed as
 * "+5.82%" on one card and "0.058221" on the next was the bug that moved
 * them there. Kept as a named export because it is the thing worth
 * asserting on directly in a test.
 */
export function isRate(path: string): boolean {
  const unit = unitFor(path);
  return unit === "return" || unit === "share";
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
    // No magnitude guard: a number whose path says "percent" prints as a
    // percent whatever its size. The old `<= 10` bound let a
    // misclassified count print raw beside a correctly classified rate,
    // which hides the misclassification instead of showing it.
    return <span className="tabular-nums">{fmtByPath(path, value)}</span>;
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
