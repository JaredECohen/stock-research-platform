import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { track } from "@/lib/analytics";
import { formatShortUtc } from "@/lib/entitlements";
import type { SampleSummary } from "@/types/public";
import CommentaryBlock from "./CommentaryBlock";
import DataFreshness from "./DataFreshness";
import Disclosure from "./Disclosure";
import ExpectationsLedger from "./ExpectationsLedger";
import SafeSection from "./SafeSection";
import SampleCompsTable from "./SampleCompsTable";
import SampleDCFCard from "./SampleDCFCard";
import SampleFundamentalsChart from "./SampleFundamentalsChart";
import SampleMemoSummary from "./SampleMemoSummary";
import { BTN_GHOST, FOCUS_RING, PRICING_PATH, SAMPLES_PATH } from "./ctas";
import { useSample, useSampleList } from "./useSamples";

/**
 * "See the product" on the landing page: the three curated companies from
 * `/api/public/samples`, one at a time, rendered from stored research.
 * Read-only by construction — the public routes never generate — and
 * every failure mode has a rendering: list unreachable, nothing
 * configured, listed but not built, one section malformed.
 */
function pickDefault(samples: SampleSummary[]): string | null {
  const built = samples.find((s) => s.built_at);
  return (built || samples[0])?.ticker ?? null;
}

export function SampleUnavailable() {
  return (
    <div className="card" role="status" data-testid="samples-unavailable">
      <div className="text-base font-semibold">Sample research is temporarily unavailable</div>
      <p className="text-sm text-slate-400 mt-1">
        The public samples could not be loaded right now. The app, the methodology page and pricing are unaffected.
      </p>
      <div className="flex flex-wrap gap-2 mt-3">
        <Link to="/methodology" className={`${BTN_GHOST} text-sm`}>Read the methodology</Link>
        <Link to={PRICING_PATH} className={`${BTN_GHOST} text-sm`}>See pricing</Link>
      </div>
    </div>
  );
}

export default function SampleShowcase() {
  const list = useSampleList();
  const [selected, setSelected] = useState<string | null>(null);
  const ticker = selected ?? pickDefault(list.samples);
  const detail = useSample(list.status === "ok" ? ticker : null);

  useEffect(() => {
    if (detail.status === "ok" && detail.ticker) track("sample_view", { ticker: detail.ticker, page: "landing" });
  }, [detail.status, detail.ticker]);

  const summary = useMemo(() => list.samples.find((s) => s.ticker === ticker) || null, [list.samples, ticker]);

  return (
    <section aria-labelledby="samples-heading" className="mt-16">
      <div className="flex flex-wrap items-end justify-between gap-3 mb-4">
        <div>
          <h2 id="samples-heading" className="text-2xl font-semibold tracking-tight">See the research, not a demo reel</h2>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">
            Three companies, read straight from stored research runs. Nothing on this page calls a model or a data provider.
          </p>
        </div>
        <Link to={SAMPLES_PATH} className={`${BTN_GHOST} text-sm`}>View sample research</Link>
      </div>

      {list.status === "loading" ? (
        <div className="card text-sm text-slate-400" role="status">Loading sample research…</div>
      ) : list.status === "error" ? (
        <SampleUnavailable />
      ) : list.samples.length === 0 ? (
        <div className="card text-sm text-slate-400" role="status" data-testid="samples-empty">
          No public samples are configured on this deployment yet.
        </div>
      ) : (
        <>
          <div role="group" aria-label="Sample company" className="flex flex-wrap gap-2 mb-4">
            {list.samples.map((s) => {
              const on = s.ticker === ticker;
              return (
                <button
                  key={s.ticker}
                  type="button"
                  aria-pressed={on}
                  onClick={() => setSelected(s.ticker)}
                  className={`px-3 py-1.5 rounded-lg border text-sm motion-safe:transition-colors ${FOCUS_RING} ${
                    on ? "border-accent-600/50 bg-accent-600/15 text-accent-500" : "border-ink-700 text-slate-300 hover:bg-ink-800"
                  }`}
                >
                  <span className="font-mono font-semibold">{s.ticker}</span>
                  {s.company_name ? <span className="ml-2 text-slate-400">{s.company_name}</span> : null}
                </button>
              );
            })}
          </div>

          <div aria-busy={detail.status === "loading"} aria-live="polite" className="space-y-6">
            {detail.status === "loading" || detail.status === "idle" ? (
              <div className="card text-sm text-slate-400" role="status">Loading {ticker}…</div>
            ) : detail.status !== "ok" || !detail.sample ? (
              <SampleUnavailable />
            ) : (
              <>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <DataFreshness builtAt={detail.sample.built_at} degraded={detail.sample.degraded} />
                  <Link to={`${SAMPLES_PATH}/${detail.sample.ticker}`} className={`text-sm ${FOCUS_RING} text-accent-500 hover:text-accent-600 underline underline-offset-2 rounded-sm`}>
                    Open the full {detail.sample.ticker} sample
                  </Link>
                </div>

                {!detail.sample.built_at ? (
                  <div className="card text-sm text-slate-400" role="status" data-testid="sample-unbuilt">
                    {summary?.company_name || detail.sample.ticker} is listed as a public sample but has not been built yet
                    {summary?.built_at ? ` (last build ${formatShortUtc(summary.built_at)})` : ""}.
                  </div>
                ) : null}

                {detail.sample.memo ? (
                  <SafeSection label="The memo summary">
                    <SampleMemoSummary memo={detail.sample.memo} headingLevel={3} />
                  </SafeSection>
                ) : null}

                <SafeSection label="The expectations ledger">
                  <ExpectationsLedger ledger={detail.sample.expectations_ledger} headingLevel={3} />
                </SafeSection>

                {detail.sample.dcf ? (
                  <SafeSection label="The DCF scenarios">
                    <SampleDCFCard dcf={detail.sample.dcf} headingLevel={3} />
                  </SafeSection>
                ) : null}

                {detail.sample.comps ? (
                  <SafeSection label="The comparable-company table">
                    <SampleCompsTable comps={detail.sample.comps} headingLevel={3} />
                  </SafeSection>
                ) : null}

                {detail.sample.fundamentals ? (
                  <SafeSection label="The fundamentals chart">
                    <SampleFundamentalsChart fundamentals={detail.sample.fundamentals} headingLevel={3} />
                  </SafeSection>
                ) : null}

                {detail.sample.commentary ? (
                  <SafeSection label="The commentary">
                    <CommentaryBlock commentary={detail.sample.commentary} headingLevel={3} />
                  </SafeSection>
                ) : null}

                <Disclosure lines={detail.sample.disclosures} compact />
              </>
            )}
          </div>
        </>
      )}
    </section>
  );
}
