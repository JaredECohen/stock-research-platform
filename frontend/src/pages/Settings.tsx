import React, { useEffect, useState } from "react";
import { api } from "@/api/client";
import type { ProvidersStatusResponse } from "@/types";

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
                  · strong <span className="font-mono">{status.llm.openai_strong_model}</span>
                  {" / "}
                  cheap <span className="font-mono">{status.llm.openai_cheap_model}</span>
                </div>
                <div className="text-xs text-slate-400">
                  Anthropic: <span className={status.llm.anthropic_configured ? "text-accent-500" : "text-slate-500"}>
                    {status.llm.anthropic_configured ? "configured" : "not set"}
                  </span>{" "}
                  · strong <span className="font-mono">{status.llm.anthropic_strong_model}</span>
                  {" / "}
                  cheap <span className="font-mono">{status.llm.anthropic_cheap_model}</span>
                </div>
              </div>
            )}
          </div>

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
