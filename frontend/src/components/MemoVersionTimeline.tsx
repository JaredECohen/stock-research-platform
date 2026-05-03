import React, { useEffect, useState } from "react";
import { api } from "@/api/client";

type RevisionLogEntry = {
  version: number;
  trigger: string;
  at?: string;
  parent_version?: number | null;
  fields_patched?: string[];
  rationales?: Record<string, string>;
  delta_summary?: string;
  critic_skipped?: boolean;
  alert?: { title?: string; severity?: string; source?: string };
};

type Row = {
  version: number;
  trigger: string;
  parent_version: number | null;
  generated_at: string;
  revision_log: RevisionLogEntry[];
  rating_label: string | null;
  confidence_score: number | null;
};

const TRIGGER_BADGE: Record<string, string> = {
  first_run: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  full_reanalysis: "bg-accent-500/15 text-accent-500 border-accent-500/30",
  incremental_patch: "bg-warn-500/15 text-warn-500 border-warn-500/30",
  force_refresh: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  scheduled: "bg-slate-500/15 text-slate-300 border-slate-500/30",
};

function fmtDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function MemoVersionTimeline({ ticker }: { ticker: string }) {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setRows(null);
    setError(null);
    api
      .memoHistory(ticker)
      .then((r) => {
        if (!cancelled) setRows(r);
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
        Memo history unavailable: {error}
      </div>
    );
  }
  if (rows === null) {
    return <div className="text-xs text-slate-500">Loading memo history…</div>;
  }
  if (rows.length === 0) {
    return (
      <div className="text-xs text-slate-500">
        No memo history yet for {ticker}.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="section-title">Memo version timeline</div>
      <ol className="border-l border-ink-700 pl-4 space-y-3">
        {rows.map((r) => {
          const log = r.revision_log?.[0];
          const trig = TRIGGER_BADGE[r.trigger] || TRIGGER_BADGE.scheduled;
          return (
            <li key={r.version} className="relative">
              <div className="absolute -left-[21px] top-1 h-2 w-2 rounded-full bg-accent-500" />
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-medium text-slate-200">
                  v{r.version}
                </span>
                <span
                  className={`text-[10px] uppercase tracking-widest px-2 py-0.5 rounded border ${trig}`}
                >
                  {r.trigger.replace("_", " ")}
                </span>
                {r.parent_version !== null && (
                  <span className="text-[10px] text-slate-500">
                    ← from v{r.parent_version}
                  </span>
                )}
                <span className="text-[10px] text-slate-500">
                  {fmtDate(r.generated_at)}
                </span>
                {r.rating_label && (
                  <span className="text-[10px] text-slate-300">
                    {r.rating_label}
                    {r.confidence_score !== null
                      ? ` · ${Math.round(r.confidence_score)}`
                      : ""}
                  </span>
                )}
              </div>
              {log?.delta_summary && (
                <div className="mt-1 text-xs text-slate-300">
                  <span className="text-slate-400">Delta:</span> {log.delta_summary}
                </div>
              )}
              {log?.fields_patched && log.fields_patched.length > 0 && (
                <div className="mt-1 text-xs text-slate-400">
                  Patched: {log.fields_patched.join(", ")}
                </div>
              )}
              {log?.alert?.title && (
                <div className="mt-1 text-xs text-slate-400">
                  Alert: <span className="text-slate-200">{log.alert.title}</span>
                  {log.alert.severity ? ` (${log.alert.severity})` : ""}
                </div>
              )}
              {log?.critic_skipped && (
                <div className="mt-1 text-[10px] text-warn-500">
                  Critic skipped (incremental patch policy).
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
