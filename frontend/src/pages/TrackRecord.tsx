import React, { useEffect, useState } from "react";
import { api, isApiError } from "@/api/client";
import RateLimitNotice from "@/components/RateLimitNotice";
import UpgradePrompt from "@/components/UpgradePrompt";
import type { EntitlementRefusal, RateLimitRefusal } from "@/types";
import {
  COUNTED_REASON_LABELS,
  EXCLUSION_REASON_LABELS,
  RATING_SOURCE_LABELS,
  type TrackRecordOut,
} from "@/types/trackRecord";

// W6 / FIX-007 (owner decision 2026-09-24): the record stays visible but is
// PROVISIONAL until coverage clears the server's thresholds. It shows
// coverage and SPY-relative alpha beside the absolute hit rate, and it says
// which outcomes it does not count and that they were kept, not deleted.

const HORIZONS = [30, 90, 180, 365];

function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

function fmtSignedPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined) return "—";
  const s = (v * 100).toFixed(digits);
  return v > 0 ? `+${s}%` : `${s}%`;
}

/** "Bullish 54 · Neutral 11": largest first, name as the tie-break. */
function countLine(counts: Record<string, number>, label: (k: string) => string = (k) => k): string {
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([k, n]) => `${label(k)} ${n}`)
    .join(" · ");
}

export default function TrackRecord() {
  const [horizon, setHorizon] = useState(90);
  const [ticker, setTicker] = useState("");
  const [sector, setSector] = useState("");
  const [data, setData] = useState<TrackRecordOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  // 402 / 429 from the backend render a specific prompt instead of the
  // generic error line; the track record is a Pro feature under the wall.
  const [refusal, setRefusal] = useState<EntitlementRefusal | null>(null);
  const [rateLimit, setRateLimit] = useState<RateLimitRefusal | null>(null);

  const handleFailure = (e: unknown) => {
    if (isApiError(e) && e.entitlement) {
      setRefusal(e.entitlement);
    } else if (isApiError(e) && e.rateLimit) {
      setRateLimit(e.rateLimit);
    } else {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const refresh = React.useCallback(() => {
    setLoading(true);
    setError(null);
    setRefusal(null);
    api
      .trackRecord({
        horizon_days: horizon,
        ticker: ticker.trim() || undefined,
        sector: sector.trim() || undefined,
      })
      .then(setData)
      .catch(handleFailure)
      .finally(() => setLoading(false));
  }, [horizon, ticker, sector]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Goes through the api client (not a bare fetch) so the bearer token and
  // structured-error handling apply — the endpoint is Pro-gated and
  // globally rate-limited (1 per 10 min) under the login wall.
  const triggerEvaluator = async () => {
    setEvaluating(true);
    setError(null);
    setRateLimit(null);
    try {
      await api.evaluateOutcomes();
      refresh();
    } catch (e) {
      handleFailure(e);
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <div className="text-2xl font-semibold tracking-tight">Track record</div>
        <div className="text-sm text-slate-400 mt-1">
          Realized forward returns for every eligible memo. Hit rate uses absolute return;
          alpha is measured against SPY over the same sessions.
        </div>
      </div>

      <div className="card-tight">
        <div className="flex flex-wrap items-end gap-3">
          <div className="space-y-1">
            <label className="text-xs uppercase tracking-widest text-slate-500" htmlFor="tr-horizon">
              Horizon (days)
            </label>
            <select
              id="tr-horizon"
              className="bg-ink-800 border border-ink-700 rounded-md px-2 py-1.5 text-sm text-slate-100"
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
            >
              {HORIZONS.map((h) => (
                <option key={h} value={h}>{h}d</option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-xs uppercase tracking-widest text-slate-500">
              Ticker (optional)
            </label>
            <input
              className="bg-ink-800 border border-ink-700 rounded-md px-2 py-1.5 text-sm text-slate-100 w-32"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="NVDA"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs uppercase tracking-widest text-slate-500">
              Sector (optional)
            </label>
            <input
              className="bg-ink-800 border border-ink-700 rounded-md px-2 py-1.5 text-sm text-slate-100 w-44"
              value={sector}
              onChange={(e) => setSector(e.target.value)}
              placeholder="Technology"
            />
          </div>
          <button
            type="button"
            onClick={refresh}
            disabled={loading}
            className="px-3 py-1.5 text-sm rounded-md bg-accent-600/30 border border-accent-600/40 text-accent-100 hover:bg-accent-600/45 disabled:opacity-50"
          >
            {loading ? "Loading…" : "Refresh"}
          </button>
          <button
            type="button"
            onClick={triggerEvaluator}
            disabled={evaluating}
            className="px-3 py-1.5 text-sm rounded-md bg-ink-800 border border-ink-700 text-slate-200 hover:bg-ink-700 disabled:opacity-50"
            title="Run the daily outcome scorer now (production runs it via APScheduler)."
          >
            {evaluating ? "Evaluating…" : "Score now"}
          </button>
        </div>
      </div>

      {refusal && <UpgradePrompt refusal={refusal} onDismiss={() => setRefusal(null)} />}
      {rateLimit && (
        <RateLimitNotice
          refusal={rateLimit}
          onRetry={() => void triggerEvaluator()}
          onDismiss={() => setRateLimit(null)}
          preservedNote="The scorer runs at most once every 10 minutes for everyone; nothing to re-enter."
        />
      )}

      {error && (
        <div className="card-tight border-danger-500/40 bg-danger-500/5 text-danger-500 text-sm">
          {error}
        </div>
      )}

      {data && data.total > 0 && <Record data={data} />}

      {data && <CountedNote data={data} />}

      {data && <ExclusionNote data={data} />}

      {data && data.total === 0 && (
        <div className="card-tight border-warn-500/40 bg-warn-500/5 text-warn-500 text-sm">
          No outcomes for this filter yet. Click <strong>Score now</strong> to
          evaluate any memos whose forward window has come of age.
        </div>
      )}
    </div>
  );
}

function Record({ data }: { data: TrackRecordOut }) {
  const row = data.coverage.horizons.find((h) => h.horizon_days === data.horizon_days);
  const companies = row?.companies ?? data.provisional.companies;
  const universePct = fmtPct(row?.universe_pct ?? null, 0);
  const sources = data.rating_mix_by_source;
  const sourceTotals: Record<string, number> = {};
  for (const [source, mix] of Object.entries(sources)) {
    sourceTotals[source] = Object.values(mix ?? {}).reduce((a, b) => a + b, 0);
  }
  const llmShare = data.total ? (sourceTotals.llm_pm ?? 0) / data.total : 0;
  return (
    <>
      {data.provisional.is_provisional && (
        <div role="status" className="card-tight border-warn-500/40 bg-warn-500/5 text-warn-500 text-sm">
          <strong>Provisional:</strong> {data.provisional.companies} companies ({universePct} of the{" "}
          {data.coverage.universe_companies}-company universe) and {data.provisional.directional} directional
          calls evaluated at {data.horizon_days} days. This record stays provisional until at least{" "}
          {data.provisional.min_companies} companies and {data.provisional.min_directional} directional calls
          are evaluated. Several memos on one company are not independent calls.
        </div>
      )}

      <div className="grid md:grid-cols-4 gap-4">
        <Stat
          label="Memos evaluated"
          value={String(data.total)}
          sub={`${companies} companies · ${universePct} of universe`}
        />
        <Stat
          label="Thesis hit rate"
          value={fmtPct(data.thesis_hit_rate, 0)}
          sub={`absolute return · always-Bullish would score ${fmtPct(data.base_rate.always_bullish_hit_rate, 0)}`}
        />
        <Stat
          label={`Beat ${data.benchmark}`}
          value={fmtPct(data.alpha.beat_benchmark_rate, 0)}
          sub={`median alpha ${fmtSignedPct(data.alpha.directional_median)} · per-company ${fmtSignedPct(data.alpha.company_weighted_median)}`}
        />
        <Stat
          label={`Avg alpha vs ${data.benchmark}`}
          value={fmtSignedPct(data.avg_alpha)}
          sub={`avg return ${fmtSignedPct(data.avg_forward_return)}`}
        />
      </div>

      <div className="card-tight text-sm text-slate-300 space-y-1">
        <div>Ratings: {countLine(data.rating_mix)}</div>
        <div>Rated by: {countLine(sourceTotals, (k) => RATING_SOURCE_LABELS[k] ?? k)}</div>
        {llmShare < 0.5 && (
          <div className="text-slate-400">
            Most of these calls were rated by the deterministic keyword PM or a news patch, not the LLM
            committee, so this record mostly measures that process.
          </div>
        )}
      </div>

      <div className="card-tight overflow-x-auto">
        <table className="w-full text-sm" aria-label="Coverage by horizon">
          <thead>
            <tr className="text-left text-xs uppercase tracking-widest text-slate-500">
              <th className="py-1 pr-4">Horizon</th>
              <th className="py-1 pr-4">Memos</th>
              <th className="py-1 pr-4">Companies</th>
              <th className="py-1 pr-4">Directional calls</th>
              <th className="py-1">% of universe</th>
            </tr>
          </thead>
          <tbody>
            {data.coverage.horizons.map((h) => (
              <tr key={h.horizon_days} className="border-t border-ink-700">
                <td className="py-1 pr-4">{h.horizon_days}d</td>
                <td className="py-1 pr-4">{h.memos}</td>
                <td className="py-1 pr-4">{h.companies}</td>
                <td className="py-1 pr-4">{h.directional}</td>
                <td className="py-1">{fmtPct(h.universe_pct, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {data.coverage.late_evaluation_candidates > 0 && (
          <div className="text-xs text-slate-400 mt-2">
            {data.coverage.late_evaluation_candidates} of the {data.horizon_days}-day outcomes were evaluated
            well after their horizon (late-evaluation candidates); their baselines have not been verified.
          </div>
        )}
      </div>
    </>
  );
}

function CountedNote({ data }: { data: TrackRecordOut }) {
  const disclosed = Object.entries(data.eligibility.eligible_by_reason).filter(
    ([reason, n]) => n > 0 && reason in COUNTED_REASON_LABELS,
  );
  if (disclosed.length === 0) return null;
  return (
    <div className="card-tight text-xs text-slate-400 space-y-0.5" aria-label="Outcomes counted with a caveat">
      <div>Counted, with a caveat (of {data.total} outcomes above):</div>
      <ul className="list-disc pl-5">
        {disclosed.map(([reason, n]) => (
          <li key={reason}>
            {n} outcomes on {COUNTED_REASON_LABELS[reason]}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ExclusionNote({ data }: { data: TrackRecordOut }) {
  const { excluded, unclassified, excluded_by_reason } = data.eligibility;
  if (excluded + unclassified === 0) return null;
  return (
    <div className="card-tight text-xs text-slate-400 space-y-0.5" aria-label="Outcomes not counted">
      <div>Not counted (kept, not deleted):</div>
      <ul className="list-disc pl-5">
        {Object.entries(excluded_by_reason).map(([reason, n]) => (
          <li key={reason}>
            {n} outcomes on {EXCLUSION_REASON_LABELS[reason] ?? reason}
          </li>
        ))}
        {unclassified > 0 && <li>{unclassified} awaiting classification</li>}
      </ul>
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="card-tight">
      <div className="text-xs uppercase tracking-widest text-slate-500">
        {label}
      </div>
      <div className="text-2xl font-semibold mt-1">{value}</div>
      {sub && <div className="text-xs text-slate-400 mt-0.5">{sub}</div>}
    </div>
  );
}
