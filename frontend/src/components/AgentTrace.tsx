import React from "react";
import type { AgentTrace as AgentTraceT } from "@/types";
import { CheckCircle2, Loader2, Circle } from "lucide-react";

export function AgentTrace({ trace }: { trace: AgentTraceT[] }) {
  if (!trace?.length) return null;
  return (
    <div className="card-tight">
      <div className="section-title mb-2">Agent Trace</div>
      <ul className="space-y-1.5 text-sm">
        {trace.map((t, i) => (
          <li key={`${t.agent}-${i}`} className="flex items-start gap-2">
            {t.status === "done" ? (
              <CheckCircle2 size={14} className="text-accent-500 mt-1 shrink-0" />
            ) : t.status === "running" ? (
              <Loader2 size={14} className="text-warn-500 mt-1 shrink-0 animate-spin" />
            ) : (
              <Circle size={14} className="text-slate-500 mt-1 shrink-0" />
            )}
            <div>
              <div className="font-medium text-slate-100">{t.agent}</div>
              <div className="text-slate-400 text-xs">{t.detail}</div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
