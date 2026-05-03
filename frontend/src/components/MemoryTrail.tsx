import React, { useEffect, useState } from "react";
import { api } from "@/api/client";

type Source = {
  source_kind: string;
  source_id: string;
  facts: Record<string, string[]>;
};

type Entry = {
  date: string;
  trigger: string;
  body: string;
  structured_facts: { sources?: Source[]; extractor_version?: number } | null;
};

type Memory = {
  ticker: string;
  path: string;
  entry_count: number;
  historical_context: string;
  entries: Entry[];
};

const FACT_LABEL: Record<string, string> = {
  guidance_changes: "Guidance",
  capex_commentary: "Capex",
  m_and_a: "M&A",
  leadership_changes: "Leadership",
  segment_signals: "Segments",
};

export default function MemoryTrail({ ticker }: { ticker: string }) {
  const [data, setData] = useState<Memory | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    api
      .stockMemory(ticker)
      .then((r) => {
        if (!cancelled) setData(r);
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
        Memory unavailable: {error}
      </div>
    );
  }
  if (data === null) {
    return <div className="text-xs text-slate-500">Loading memory…</div>;
  }
  if (data.entries.length === 0) {
    return (
      <div className="text-xs text-slate-500">
        No memory entries for {ticker} yet — earnings + filing deltas
        accumulate here over time.
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <div className="section-title">
        Long-term memory · {data.entry_count} entries
      </div>
      <div className="text-[10px] uppercase tracking-widest text-slate-600">
        {data.path}
      </div>
      <ol className="space-y-2">
        {data.entries.map((e, i) => (
          <li key={i} className="border-t border-ink-700 pt-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-slate-200">
                {e.date}
              </span>
              <span className="text-[10px] uppercase tracking-widest text-slate-500">
                {e.trigger}
              </span>
            </div>
            <pre className="text-xs text-slate-300 whitespace-pre-wrap font-sans mt-1 leading-relaxed">
              {e.body}
            </pre>
            {e.structured_facts?.sources?.length ? (
              <div className="mt-2 space-y-1">
                {e.structured_facts.sources.map((s, j) => {
                  const factEntries = Object.entries(s.facts || {}).filter(
                    ([, v]) => v && v.length > 0,
                  );
                  if (factEntries.length === 0) return null;
                  return (
                    <div key={j} className="text-[11px]">
                      <div className="text-slate-500">
                        {s.source_kind}: {s.source_id}
                      </div>
                      <ul className="mt-0.5 list-disc pl-4 text-slate-300 space-y-0.5">
                        {factEntries.map(([cat, items]) =>
                          items.slice(0, 3).map((it, k) => (
                            <li key={`${cat}-${k}`}>
                              <span className="text-slate-400">
                                {FACT_LABEL[cat] || cat}:
                              </span>{" "}
                              {it}
                            </li>
                          )),
                        )}
                      </ul>
                    </div>
                  );
                })}
              </div>
            ) : null}
          </li>
        ))}
      </ol>
    </div>
  );
}
