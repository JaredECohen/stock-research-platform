import React, { useEffect } from "react";
import { Link, useParams } from "react-router-dom";
import { normalizeTicker } from "@/api/publicClient";
import { track } from "@/lib/analytics";
import MemoCard from "@/components/MemoCard";
import CommentaryBlock from "@/components/public/CommentaryBlock";
import DataFreshness from "@/components/public/DataFreshness";
import Disclosure from "@/components/public/Disclosure";
import ExpectationsLedger from "@/components/public/ExpectationsLedger";
import PublicShell from "@/components/public/PublicShell";
import SafeSection from "@/components/public/SafeSection";
import SampleCompsTable from "@/components/public/SampleCompsTable";
import SampleDCFCard from "@/components/public/SampleDCFCard";
import SampleFundamentalsChart from "@/components/public/SampleFundamentalsChart";
import SamplePriceChart from "@/components/public/SamplePriceChart";
import { SampleUnavailable } from "@/components/public/SampleShowcase";
import SampleScreenerRow from "@/components/public/SampleScreenerRow";
import { BTN_GHOST, BTN_PRIMARY, FOCUS_RING, SAMPLES_PATH, SIGN_UP_PATH } from "@/components/public/ctas";
import { useVisitor } from "@/components/public/hooks";
import { useSample } from "@/components/public/useSamples";

/**
 * /samples/:ticker — everything the sample carries, including the full
 * read-only committee card. The route param is normalised before any
 * request; an unlisted ticker is a friendly 404 with links to the
 * tickers that are public.
 */
export default function SampleDetail() {
  const { ticker: raw } = useParams();
  const ticker = normalizeTicker(raw);
  const detail = useSample(ticker);
  const { signedIn, loading } = useVisitor();

  useEffect(() => {
    if (detail.status === "ok" && detail.ticker) track("sample_view", { ticker: detail.ticker, page: "sample" });
  }, [detail.status, detail.ticker]);

  const title = detail.sample?.company_name ? `${detail.sample.company_name} (${detail.sample.ticker}) sample` : `${ticker || "Sample"} sample`;

  return (
    <PublicShell title={title} description={`Stored research memo, expectations ledger, DCF scenarios, comps and fundamentals for ${ticker || "a sample company"}.`}>
      <div className="pt-12">
        <Link to={SAMPLES_PATH} className={`text-sm text-slate-300 hover:text-slate-100 rounded-sm ${FOCUS_RING}`}>← All samples</Link>
      </div>

      {!ticker || detail.status === "not_found" ? (
        <div className="card mt-4 max-w-xl" role="status" data-testid="sample-not-found">
          <h1 className="text-2xl font-semibold">Not a public sample</h1>
          <p className="text-sm text-slate-400 mt-2">
            Only a small, fixed set of companies is published here. Research on any other company happens inside the app, against your
            allowance.
          </p>
          {detail.sampleTickers.length > 0 ? (
            <ul className="flex flex-wrap gap-2 mt-3" aria-label="Public samples">
              {detail.sampleTickers.map((t) => (
                <li key={t}>
                  <Link to={`${SAMPLES_PATH}/${t}`} className={`${BTN_GHOST} text-sm font-mono`}>{t}</Link>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : detail.status === "loading" || detail.status === "idle" ? (
        <div className="mt-4">
          <h1 className="text-2xl font-semibold font-mono">{ticker}</h1>
          <div className="card text-sm text-slate-400 mt-4" role="status">Loading the {ticker} sample…</div>
        </div>
      ) : detail.status !== "ok" || !detail.sample ? (
        <div className="mt-4">
          <h1 className="text-2xl font-semibold font-mono">{ticker}</h1>
          <div className="mt-4">
            <SampleUnavailable />
          </div>
        </div>
      ) : (
        <div className="mt-4 space-y-10">
          <header>
            <div className="text-xs uppercase tracking-widest text-accent-500 font-semibold">Sample research</div>
            <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight mt-1">
              {detail.sample.company_name || detail.sample.ticker}{" "}
              <span className="font-mono text-slate-400 text-xl">{detail.sample.ticker}</span>
            </h1>
            {detail.sample.sector ? <div className="text-sm text-slate-400 mt-1">{detail.sample.sector}</div> : null}
            <div className="mt-3">
              <DataFreshness builtAt={detail.sample.built_at} degraded={detail.sample.degraded} />
            </div>
            {!detail.sample.built_at ? (
              <div className="card text-sm text-slate-400 mt-4" role="status" data-testid="sample-unbuilt">
                This company is listed as a public sample but has not been built yet.
              </div>
            ) : null}
          </header>

          <SafeSection label="The expectations ledger">
            <ExpectationsLedger ledger={detail.sample.expectations_ledger} />
          </SafeSection>

          {detail.sample.commentary ? (
            <SafeSection label="The commentary">
              <CommentaryBlock commentary={detail.sample.commentary} />
            </SafeSection>
          ) : null}

          {detail.sample.dcf ? (
            <SafeSection label="The DCF scenarios">
              <SampleDCFCard dcf={detail.sample.dcf} />
            </SafeSection>
          ) : null}

          {detail.sample.comps ? (
            <SafeSection label="The comparable-company table">
              <SampleCompsTable comps={detail.sample.comps} />
            </SafeSection>
          ) : null}

          {detail.sample.fundamentals ? (
            <SafeSection label="The fundamentals chart">
              <SampleFundamentalsChart fundamentals={detail.sample.fundamentals} />
            </SafeSection>
          ) : null}

          {detail.sample.prices ? (
            <SafeSection label="The price history">
              <SamplePriceChart prices={detail.sample.prices} />
            </SafeSection>
          ) : null}

          {detail.sample.screener_row ? (
            <SafeSection label="The screener rank">
              <SampleScreenerRow row={detail.sample.screener_row} />
            </SafeSection>
          ) : null}

          {detail.sample.memo ? (
            <section aria-labelledby="full-memo-heading">
              <h2 id="full-memo-heading" className="text-lg font-semibold mb-3">The full committee memo</h2>
              <SafeSection label="The committee memo">
                <MemoCard memo={detail.sample.memo} />
              </SafeSection>
            </section>
          ) : null}

          <div className="card flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-base font-semibold">Research a company of your own</div>
              <p className="text-sm text-slate-400 mt-1">Sign up to open stored memos and put the committee on your tickers.</p>
            </div>
            {loading ? null : signedIn ? (
              <Link to="/app" className={BTN_PRIMARY}>Continue to app</Link>
            ) : (
              <div className="flex flex-wrap gap-2">
                <Link to={SIGN_UP_PATH} className={BTN_PRIMARY}>Sign up free</Link>
                <Link to="/pricing" className={BTN_GHOST}>See pricing</Link>
              </div>
            )}
          </div>

          <Disclosure lines={detail.sample.disclosures} />
        </div>
      )}
    </PublicShell>
  );
}
