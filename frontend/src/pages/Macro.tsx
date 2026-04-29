import React, { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "@/api/client";
import type { MacroScenarioResult } from "@/types";

const SCENARIOS = [
  { key: "soft landing", label: "Soft Landing" },
  { key: "recession", label: "Recession" },
  { key: "sticky inflation", label: "Sticky Inflation" },
  { key: "falling rates", label: "Falling Rates" },
  { key: "ai capex boom", label: "AI Capex Boom" },
];

interface MacroPoint { date: string; value: number | null }
interface MacroSeries { series_id: string; name: string; units: string; points: MacroPoint[] }

export default function Macro() {
  const [scenario, setScenario] = useState("soft landing");
  const [result, setResult] = useState<MacroScenarioResult | null>(null);
  const [series, setSeries] = useState<MacroSeries[]>([]);

  useEffect(() => {
    api.macroAnalyze(scenario).then(setResult);
  }, [scenario]);

  useEffect(() => {
    api.macroSeries().then((s) => setSeries(s as MacroSeries[]));
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Macro</h1>
        <p className="text-slate-400 text-sm mt-1">Scenario analysis and macro snapshot from FRED-compatible series.</p>
      </div>

      <div className="card-tight flex gap-2 flex-wrap">
        {SCENARIOS.map((s) => (
          <button
            key={s.key}
            onClick={() => setScenario(s.key)}
            className={`badge ${scenario === s.key ? "border-accent-600 text-accent-500 bg-accent-600/15" : "border-ink-700 text-slate-300"}`}
          >
            {s.label}
          </button>
        ))}
      </div>

      {result && (
        <div className="grid lg:grid-cols-2 gap-4">
          <div className="card lg:col-span-2">
            <div className="section-title mb-1">Scenario: {result.scenario}</div>
            <p className="text-sm text-slate-200">{result.narrative}</p>
          </div>
          <div className="card">
            <div className="section-title mb-2">Sector impacts</div>
            <ul className="text-sm text-slate-300 space-y-1">
              {Object.entries(result.sector_impacts).map(([k, v]) => (
                <li key={k}>
                  <span className="font-medium text-slate-100">{k}:</span> <span className="text-slate-400">{v}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="card">
            <div className="section-title mb-2">Favored / Pressured</div>
            <div className="text-sm">
              <div className="mb-2">
                <span className="text-accent-500 font-medium">Favored: </span>
                {result.favored_sectors.join(", ")}
              </div>
              <div className="mb-3">
                <span className="text-danger-500 font-medium">Pressured: </span>
                {result.pressured_sectors.join(", ")}
              </div>
              <div className="section-title mb-1">Suggested research views</div>
              <ul className="list-disc pl-5 text-slate-300 space-y-0.5">
                {result.suggested_research_views.map((v, i) => <li key={i}>{v}</li>)}
              </ul>
              <div className="section-title mt-3 mb-1">Risks</div>
              <ul className="list-disc pl-5 text-slate-300 space-y-0.5">
                {result.risks.map((v, i) => <li key={i}>{v}</li>)}
              </ul>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="section-title mb-2">Macro snapshot</div>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
          {series.map((s) => (
            <div key={s.series_id} className="card-tight">
              <div className="text-xs text-slate-400">
                {s.name} <span className="text-slate-600">({s.series_id})</span>
              </div>
              <div className="text-base font-mono text-slate-100">
                {s.points[s.points.length - 1]?.value ?? "—"} <span className="text-xs text-slate-500">{s.units}</span>
              </div>
              <div style={{ height: 60 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={s.points} margin={{ top: 5, bottom: 0, left: 0, right: 0 }}>
                    <Line type="monotone" dataKey="value" stroke="#52E0C4" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
