import React, { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Send, Sparkles } from "lucide-react";
import { api } from "@/api/client";
import { Markdown } from "@/components/Markdown";
import { AgentTrace } from "@/components/AgentTrace";
import MemoCard from "@/components/MemoCard";
import type { ChatResponse } from "@/types";

const STARTERS = [
  "Analyze NVDA as a long-term investment.",
  "Compare MSFT and GOOGL from a portfolio manager's perspective.",
  "Find 5 high-quality stocks that could benefit from falling rates.",
  "Build a 10-stock portfolio for a soft landing with continued AI infrastructure spending.",
  "What sectors benefit if inflation stays sticky?",
  "Run a DCF for MSFT using base-case assumptions.",
  "Show me reasonable valuation growth stocks.",
];

interface Turn {
  user: string;
  response?: ChatResponse;
  loading?: boolean;
  error?: string;
}

export default function Chat() {
  const [params] = useSearchParams();
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);

  useEffect(() => {
    const q = params.get("q");
    if (q && turns.length === 0) {
      void send(q);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const send = async (msg: string) => {
    if (!msg.trim()) return;
    setTurns((prev) => [...prev, { user: msg, loading: true }]);
    setInput("");
    try {
      const r = await api.chat(msg);
      setTurns((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = { user: msg, response: r, loading: false };
        return copy;
      });
    } catch (e) {
      setTurns((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = { user: msg, loading: false, error: String(e) };
        return copy;
      });
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Ask the PM</h1>
        <p className="text-slate-400 text-sm mt-1">Ask the orchestrator anything: stocks, sectors, scenarios, portfolios.</p>
      </div>

      {turns.length === 0 && (
        <div className="card">
          <div className="section-title mb-2">Try one of these prompts</div>
          <div className="grid md:grid-cols-2 gap-2">
            {STARTERS.map((s) => (
              <button key={s} className="card-tight text-left hover:border-accent-600 transition-colors" onClick={() => void send(s)}>
                <Sparkles size={14} className="text-accent-500 inline mr-2" />
                <span className="text-sm">{s}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-6">
        {turns.map((t, i) => (
          <div key={i} className="space-y-3">
            <div className="card-tight bg-ink-900/60">
              <div className="section-title mb-1">You</div>
              <div className="text-sm">{t.user}</div>
            </div>
            {t.loading && (
              <div className="card-tight">
                <div className="section-title mb-1">PM Orchestrator</div>
                <div className="text-sm text-slate-400">Routing through agents…</div>
              </div>
            )}
            {t.error && (
              <div className="card-tight border-danger-500/40 text-sm text-danger-500">{t.error}</div>
            )}
            {t.response && (
              <div className="grid lg:grid-cols-[1fr_280px] gap-4">
                <div className="space-y-4">
                  <div className="card">
                    <div className="section-title mb-2">PM Response</div>
                    <Markdown text={t.response.answer} />
                  </div>
                  {t.response.memo && <MemoCard memo={t.response.memo} />}
                  {t.response.portfolio && (
                    <div className="card-tight">
                      <div className="section-title mb-1">Portfolio Holdings</div>
                      <table className="w-full text-sm">
                        <thead className="text-xs text-slate-500">
                          <tr>
                            <th className="text-left">Ticker</th>
                            <th className="text-left">Sector</th>
                            <th className="text-right">Weight</th>
                          </tr>
                        </thead>
                        <tbody>
                          {t.response.portfolio.holdings.map((h) => (
                            <tr key={h.ticker} className="border-t border-ink-800">
                              <td className="font-mono py-1">{h.ticker}</td>
                              <td className="text-slate-300">{h.sector}</td>
                              <td className="text-right font-mono">{(h.weight * 100).toFixed(1)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
                <AgentTrace trace={t.response.agent_trace} />
              </div>
            )}
          </div>
        ))}
      </div>

      <form
        className="sticky bottom-0 pt-3 bg-ink-950/80 backdrop-blur"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <div className="flex gap-2">
          <input
            className="input flex-1"
            placeholder="Ask the PM…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button type="submit" className="btn-primary">
            <Send size={14} /> Send
          </button>
        </div>
        <div className="text-[11px] text-slate-500 mt-2">
          Research and education only. Not personalized financial advice.
        </div>
      </form>
    </div>
  );
}
