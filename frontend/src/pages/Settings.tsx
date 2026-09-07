import React, { useEffect, useState } from "react";
import { api } from "@/api/client";
import type { LLMStatus, ProvidersStatusResponse } from "@/types";
import { parseTimestamp, providerLabel } from "@/components/ProviderHealthBanner";

// Naive-UTC backend stamps would otherwise display as-is and invite a
// local-time misreading; show the normalised instant, raw only if unparseable.
function formatTimestamp(value: string | number): string {
  const t = parseTimestamp(value);
  return t === null ? String(value) : new Date(t).toISOString().replace(".000Z", "Z");
}

function yesNo(v: boolean | undefined): { text: string; cls: string } {
  if (v === undefined) return { text: "—", cls: "text-slate-500" };
  return v ? { text: "yes", cls: "text-accent-500" } : { text: "no", cls: "text-slate-500" };
}

/**
 * Read-only LLM health: per-provider configured/breaker state, failover
 * history, role → model routing and degradation reasons. Every section is
 * optional because the deployed backend may predate the fields; the card
 * then says so rather than rendering blanks that look like a bug.
 */
function ProviderHealthCard({ llm, mode, llmConfigured }: { llm?: LLMStatus; mode: string; llmConfigured: boolean }) {
  const breakers = llm?.breakers || {};
  const configured: Record<string, boolean | undefined> = {
    openai: llm?.openai_configured,
    anthropic: llm?.anthropic_configured,
    gemini: llm?.gemini_configured,
  };
  // Union of the providers we know how to describe and whatever the backend
  // reports breakers for, so a new provider still shows up without a UI change.
  const rows = Array.from(
    new Set([
      ...Object.keys(configured).filter((k) => configured[k] !== undefined),
      ...Object.keys(breakers),
    ]),
  );
  const anyOpen = Object.values(breakers).some((b) => b?.is_open);
  const degraded = llm?.degraded === true || anyOpen || (mode === "live" && !llmConfigured);
  const reasons = llm?.degradation_reasons || [];
  const fo = llm?.failover;
  const roleModels = Object.entries(llm?.role_models || {});
  const reported = llm?.degraded !== undefined || llm?.breakers !== undefined;

  return (
    <div className={`card ${degraded ? "border-warn-500/30" : ""}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="section-title">Provider health</div>
        <span className={degraded ? "badge-mixed" : "badge-bull"}>{degraded ? "degraded" : "healthy"}</span>
      </div>
      {!reported && (
        <div className="text-xs text-slate-500 mb-2">
          This backend does not report breaker or degradation state yet; status below is limited to key configuration.
        </div>
      )}
      {reasons.length > 0 && (
        <ul className="text-sm text-warn-500 list-disc pl-4 mb-3 space-y-0.5">
          {reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      )}

      {rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-xs text-slate-500">
              <tr>
                <th className="text-left py-2">Provider</th>
                <th className="text-left">Configured</th>
                <th className="text-left">Healthy</th>
                <th className="text-right">Failures</th>
                <th className="text-left pl-3">Breaker open</th>
                <th className="text-right">Since last failure</th>
                <th className="text-right">Cooldown</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((key) => {
                const b = breakers[key];
                const conf = yesNo(configured[key]);
                const healthy = yesNo(b ? !b.is_open : undefined);
                return (
                  <tr key={key} className="border-t border-ink-800">
                    <td className="py-2 font-mono">{providerLabel(key)}</td>
                    <td className={conf.cls}>{conf.text}</td>
                    <td className={healthy.cls}>{healthy.text}</td>
                    <td className="text-right font-mono">{b ? b.failure_count : "—"}</td>
                    <td className={`pl-3 ${b?.is_open ? "text-warn-500" : "text-slate-500"}`}>{b ? String(b.is_open) : "—"}</td>
                    <td className="text-right font-mono">
                      {b && b.seconds_since_last_failure !== null ? `${Math.round(b.seconds_since_last_failure)}s` : "—"}
                    </td>
                    <td className="text-right font-mono">{b ? `${b.cooldown_seconds}s` : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {fo && (
        <div className="text-sm mt-3">
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Failover</div>
          <div className="text-slate-300">
            {fo.enabled ? "Enabled" : "Disabled"} · {fo.count} event{fo.count === 1 ? "" : "s"}
            {fo.last_to && (
              <>
                {" "}· last: <span className="font-mono">{fo.last_from ? providerLabel(fo.last_from) : "?"}</span> →{" "}
                <span className="font-mono">{providerLabel(fo.last_to)}</span>
                {fo.last_at != null && <span className="text-slate-500"> at {formatTimestamp(fo.last_at)}</span>}
                {fo.last_reason && <span className="text-slate-500"> ({fo.last_reason})</span>}
              </>
            )}
          </div>
        </div>
      )}

      {roleModels.length > 0 && (
        <div className="text-sm mt-3">
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Role models</div>
          <div className="grid sm:grid-cols-2 gap-x-6 gap-y-1">
            {roleModels.map(([role, model]) => (
              <div key={role} className="flex justify-between border-b border-ink-800 py-1">
                <span className="text-slate-300">{role}</span>
                <span className="font-mono text-slate-200">{model}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {reported && (
        <div className="text-[11px] text-slate-500 mt-3">
          Breakers are per process: this reflects the web service that answered the request, not the worker.
        </div>
      )}
    </div>
  );
}

export default function Settings() {
  const [status, setStatus] = useState<ProvidersStatusResponse | null>(null);

  useEffect(() => {
    api.providersStatus().then(setStatus);
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Settings & Status</h1>
        <p className="text-slate-400 text-sm mt-1">Mode, provider configuration, and feature flags.</p>
      </div>

      {status && (
        <>
          <div className="card">
            <div className="section-title mb-2">Mode</div>
            <div className="text-sm">
              Current mode: <span className={`font-medium ${status.mode === "demo" ? "text-warn-500" : "text-accent-500"}`}>{status.mode}</span>
            </div>
            <div className="text-sm mt-1">
              LLM configured: <span className={status.llm_configured ? "text-accent-500" : "text-slate-400"}>{String(status.llm_configured)}</span>
            </div>
            {status.llm && (
              <div className="text-sm mt-1 space-y-0.5">
                <div>
                  Active provider:{" "}
                  <span className={status.llm.active_provider === "none" ? "text-slate-400" : "text-accent-500"}>
                    {status.llm.active_provider}
                  </span>
                  <span className="text-slate-500"> · choice: {status.llm.provider_choice}</span>
                </div>
                <div className="text-xs text-slate-400">
                  OpenAI: <span className={status.llm.openai_configured ? "text-accent-500" : "text-slate-500"}>
                    {status.llm.openai_configured ? "configured" : "not set"}
                  </span>{" "}
                  · strong <span className="font-mono">{status.llm.openai_strong_model ?? "—"}</span>
                  {" / "}
                  cheap <span className="font-mono">{status.llm.openai_cheap_model ?? "—"}</span>
                </div>
                <div className="text-xs text-slate-400">
                  Anthropic: <span className={status.llm.anthropic_configured ? "text-accent-500" : "text-slate-500"}>
                    {status.llm.anthropic_configured ? "configured" : "not set"}
                  </span>{" "}
                  · strong <span className="font-mono">{status.llm.anthropic_strong_model ?? "—"}</span>
                  {" / "}
                  cheap <span className="font-mono">{status.llm.anthropic_cheap_model ?? "—"}</span>
                </div>
              </div>
            )}
          </div>

          <ProviderHealthCard llm={status.llm} mode={status.mode} llmConfigured={status.llm_configured} />

          <div className="card">
            <div className="section-title mb-2">Feature flags</div>
            <div className="grid sm:grid-cols-2 gap-2 text-sm">
              {Object.entries(status.feature_flags).map(([k, v]) => (
                <div key={k} className="flex justify-between border-b border-ink-800 py-1.5">
                  <span className="text-slate-300">{k}</span>
                  <span className={v ? "text-accent-500" : "text-slate-500"}>{String(v)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="section-title mb-2">Providers</div>
            <table className="w-full text-sm">
              <thead className="text-xs text-slate-500">
                <tr>
                  <th className="text-left py-2">Name</th>
                  <th className="text-left">Configured</th>
                  <th className="text-left">Healthy</th>
                  <th className="text-left">Capabilities</th>
                  <th className="text-left">Notes</th>
                </tr>
              </thead>
              <tbody>
                {Object.values(status.providers).map((p) => (
                  <tr key={p.name} className="border-t border-ink-800">
                    <td className="py-2 font-mono">{p.name}</td>
                    <td className={p.configured ? "text-accent-500" : "text-slate-500"}>{String(p.configured)}</td>
                    <td className={p.healthy ? "text-accent-500" : "text-slate-500"}>{String(p.healthy)}</td>
                    <td className="text-slate-300">{p.capabilities.join(", ")}</td>
                    <td className="text-slate-400">{p.notes || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {status.missing_api_keys.length > 0 && (
            <div className="card-tight border-warn-500/30 bg-warn-500/5">
              <div className="section-title mb-1 text-warn-500">Missing API keys</div>
              <div className="text-sm text-slate-300">
                Set the following environment variables to enable live providers: <span className="font-mono">{status.missing_api_keys.join(", ")}</span>.
                <br />
                Until then, MarketMosaic falls back to a coherent demo dataset for ~28 large-cap stocks.
              </div>
            </div>
          )}

          <div className="text-[11px] text-slate-500">
            MarketMosaic is for investment research and education only. It does not provide personalized financial,
            investment, legal, or tax advice.
          </div>
        </>
      )}
    </div>
  );
}
