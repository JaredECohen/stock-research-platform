import React from "react";
import { Link } from "react-router-dom";
import { formatShortUtc } from "@/lib/entitlements";
import Disclosure from "@/components/public/Disclosure";
import PublicShell from "@/components/public/PublicShell";
import { SampleUnavailable } from "@/components/public/SampleShowcase";
import { FOCUS_RING, SAMPLES_PATH } from "@/components/public/ctas";
import { useSampleList } from "@/components/public/useSamples";

const KIND_LABELS: Record<string, string> = {
  memo: "memo",
  dcf: "DCF",
  comps: "comps",
  fundamentals: "fundamentals",
  prices: "prices",
  screener_row: "screener rank",
  commentary: "commentary",
};

/** /samples — the curated list, built or not. */
export default function Samples() {
  const list = useSampleList();
  return (
    <PublicShell title="Sample research" description="Read-only research memos, DCF scenarios, comps and fundamentals for three companies, from stored research runs.">
      <div className="pt-12">
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">Sample research</h1>
        <p className="text-slate-300 mt-2 max-w-2xl">
          Stored research for a small, fixed set of companies, rebuilt on a schedule. Read-only: nothing here runs a model or calls a data
          provider.
        </p>
      </div>

      <div className="mt-8">
        {list.status === "loading" ? (
          <div className="card text-sm text-slate-400" role="status">Loading samples…</div>
        ) : list.status === "error" ? (
          <SampleUnavailable />
        ) : list.samples.length === 0 ? (
          <div className="card text-sm text-slate-400" role="status" data-testid="samples-empty">
            No public samples are configured on this deployment yet.
          </div>
        ) : (
          <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {list.samples.map((s) => {
              const built = formatShortUtc(s.built_at);
              return (
                <li key={s.ticker} className="card flex flex-col">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="font-mono text-lg font-semibold">{s.ticker}</span>
                    {s.sector ? <span className="text-xs text-slate-400">{s.sector}</span> : null}
                  </div>
                  <div className="text-sm text-slate-200 mt-0.5">{s.company_name || "Company name not stored"}</div>
                  <div className="text-xs text-slate-400 mt-2">{built ? `Built ${built}` : "Not built yet"}</div>
                  {s.kinds.length > 0 ? (
                    <ul className="flex flex-wrap gap-1 mt-2" aria-label={`Sections available for ${s.ticker}`}>
                      {s.kinds.map((k) => (
                        <li key={k} className="badge-neutral text-[10px]">{KIND_LABELS[k] || k}</li>
                      ))}
                    </ul>
                  ) : null}
                  <Link
                    to={`${SAMPLES_PATH}/${s.ticker}`}
                    className={`btn-ghost text-sm mt-4 self-start motion-safe:transition-colors ${FOCUS_RING}`}
                    aria-label={`Open the ${s.ticker} sample`}
                  >
                    Open sample
                  </Link>
                </li>
              );
            })}
          </ul>
        )}
      </div>
      <div className="mt-12">
        <Disclosure />
      </div>
    </PublicShell>
  );
}
