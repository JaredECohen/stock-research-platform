import React, { useEffect, useState } from "react";
import { api } from "@/api/client";

type Change = {
  field: string;
  from: unknown;
  to: unknown;
  rationale: string;
};

type Version = {
  version: number;
  parent_version: number | null;
  trigger: string;
  generated_at: string;
  assumption_changes: Change[];
  has_result: boolean;
};

const TRIGGER_BADGE: Record<string, string> = {
  initial: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  earnings_update: "bg-accent-500/15 text-accent-500 border-accent-500/30",
  memo_rebuild: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  force_refresh: "bg-slate-500/15 text-slate-300 border-slate-500/30",
};

function fmtVal(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (Math.abs(v) < 1) return `${(v * 100).toFixed(1)}%`;
    return v.toFixed(2);
  }
  return String(v);
}

function fmtDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function DCFVersionHistory({ ticker }: { ticker: string }) {
  const [rows, setRows] = useState<Version[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setRows(null);
    setError(null);
    api
      .dcfVersionHistory(ticker)
      .then((r) => {
        if (!cancelled) setRows(r.versions);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  if (error) {
    return (
      <div className="text-xs text-danger-500">
        DCF history unavailable: {error}
      </div>
    );
  }
  if (rows === null) {
    return <div className="text-xs text-slate-500">Loading DCF history…</div>;
  }
  if (rows.length === 0) {
    return (
      <div className="text-xs text-slate-500">
        No DCF versions yet for {ticker} — run an analysis first.
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <div className="section-title">DCF assumption drift</div>
      <ol className="border-l border-ink-700 pl-4 space-y-3">
        {rows.map((v) => {
          const trig = TRIGGER_BADGE[v.trigger] || TRIGGER_BADGE.memo_rebuild;
          return (
            <li key={v.version} className="relative">
              <div className="absolute -left-[21px] top-1 h-2 w-2 rounded-full bg-accent-500" />
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-medium text-slate-200">
                  v{v.version}
                </span>
                <span
                  className={`text-[10px] uppercase tracking-widest px-2 py-0.5 rounded border ${trig}`}
                >
                  {v.trigger.replace("_", " ")}
                </span>
                {v.parent_version !== null && (
                  <span className="text-[10px] text-slate-500">
                    ← from v{v.parent_version}
                  </span>
                )}
                <span className="text-[10px] text-slate-500">
                  {fmtDate(v.generated_at)}
                </span>
              </div>
              {v.assumption_changes.length > 0 ? (
                <ul className="mt-1 text-xs text-slate-300 space-y-0.5">
                  {v.assumption_changes.map((c, i) => (
                    <li key={i}>
                      <span className="text-slate-400">{c.field}</span>:{" "}
                      <span className="text-slate-200">{fmtVal(c.from)}</span> →{" "}
                      <span className="text-accent-500">{fmtVal(c.to)}</span>
                      {c.rationale && (
                        <span className="text-slate-500"> — {c.rationale}</span>
                      )}
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="mt-1 text-xs text-slate-500">
                  No assumption changes.
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
