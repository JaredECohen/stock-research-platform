import React, { useState } from "react";
import type { NewsAlert, NewsSeverity } from "@/types";

/**
 * Pending news alerts side panel.
 *
 * Severity-color-coded list of NewsAlerts the news agent has pushed into the
 * hot cache for this ticker. Collapsible (defaults to expanded when there's
 * at least one `material` or `breaking` alert).
 */
function severityClasses(s: NewsSeverity): string {
  switch (s) {
    case "breaking":
      return "border-danger-500/50 bg-danger-500/10 text-danger-500";
    case "material":
      return "border-warn-500/50 bg-warn-500/10 text-warn-500";
    default:
      return "border-ink-700 bg-ink-800/60 text-slate-300";
  }
}

function fmtDate(s?: string | null): string {
  if (!s) return "";
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return s.slice(0, 10);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export default function NewsAlertPanel({ alerts }: { alerts?: NewsAlert[] }) {
  const items = alerts || [];
  const hasMaterial = items.some((a) => a.severity === "material" || a.severity === "breaking");
  const [open, setOpen] = useState(hasMaterial);

  if (items.length === 0) {
    return (
      <div className="card-tight">
        <div className="section-title mb-1">Hot News</div>
        <div className="text-xs text-slate-500">No pending alerts.</div>
      </div>
    );
  }

  return (
    <div className="card-tight">
      <button
        type="button"
        className="w-full flex items-center justify-between text-left"
        onClick={() => setOpen(!open)}
      >
        <span className="section-title">Hot News ({items.length})</span>
        <span className="text-xs text-slate-500">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <ul className="mt-2 space-y-2">
          {items.map((a, i) => (
            <li
              key={i}
              className={`rounded-md border px-2 py-1.5 ${severityClasses(a.severity)}`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-[10px] uppercase tracking-widest opacity-80">
                  {a.severity}
                </span>
                {a.published_at && (
                  <span className="text-[10px] opacity-70">{fmtDate(a.published_at)}</span>
                )}
              </div>
              <div className="text-sm font-medium mt-0.5 leading-snug">
                {a.url ? (
                  <a
                    href={a.url}
                    target="_blank"
                    rel="noreferrer"
                    className="hover:underline"
                  >
                    {a.title}
                  </a>
                ) : (
                  a.title
                )}
              </div>
              {a.summary && (
                <div className="text-xs opacity-80 mt-0.5 line-clamp-3">{a.summary}</div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
