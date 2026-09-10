import React from "react";
import { Clock } from "lucide-react";
import { formatExactUtc } from "@/lib/entitlements";
import { FOCUS_RING } from "./ctas";

/**
 * When the sample was built and what is missing from it. Samples are
 * rebuilt weekly from stored research; saying so — with the exact time
 * and the backend's degraded notes — is what keeps a stale page honest.
 */
export default function DataFreshness({ builtAt, degraded = [] }: { builtAt: string | null; degraded?: string[] }) {
  const when = formatExactUtc(builtAt);
  return (
    <div className="text-xs text-slate-400 flex flex-col gap-1" data-testid="data-freshness">
      <div className="flex items-center gap-1.5">
        <Clock size={12} aria-hidden />
        {when ? (
          <span>
            Built from stored research on <time dateTime={builtAt || undefined}>{when}</time>; not updated in real time.
          </span>
        ) : (
          <span>This sample has not been built yet.</span>
        )}
      </div>
      {degraded.length > 0 ? (
        <details>
          <summary className={`cursor-pointer text-slate-300 rounded-sm inline-block ${FOCUS_RING}`}>
            What is missing from this sample ({degraded.length})
          </summary>
          <ul className="list-disc pl-5 mt-1 space-y-0.5">
            {degraded.map((d, i) => (
              <li key={i}>{d}</li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}
