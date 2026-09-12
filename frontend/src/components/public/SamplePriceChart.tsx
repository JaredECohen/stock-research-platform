import React from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmtPrice } from "@/lib/format";
import type { PricePoint } from "@/types/public";
import { useReducedMotion } from "./hooks";

/**
 * Closing prices from the stored provider cache — about a year of
 * sessions. The chart is decorative for assistive tech; the sentence
 * under it carries the same information (first and last close, span).
 */
export default function SamplePriceChart({ prices, headingLevel = 2 }: { prices: PricePoint[]; headingLevel?: 2 | 3 }) {
  const H = `h${headingLevel}` as "h2" | "h3";
  const reduced = useReducedMotion();
  const pts = prices.filter((p) => typeof p.close === "number" && Number.isFinite(p.close));
  if (pts.length === 0) return null;
  const first = pts[0];
  const last = pts[pts.length - 1];
  return (
    <section aria-labelledby="sample-prices-heading">
      <H id="sample-prices-heading" className="text-lg font-semibold mb-2">Price history</H>
      <div className="h-44 card-tight" aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={pts} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}>
            <XAxis dataKey="date" stroke="#94a3b8" tick={{ fontSize: 11 }} minTickGap={40} tickFormatter={(d: string) => d.slice(0, 7)} />
            <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} domain={["auto", "auto"]} tickFormatter={(v: number) => fmtPrice(v)} width={72} />
            <Tooltip
              formatter={(v) => (typeof v === "number" ? fmtPrice(v) : "n/a")}
              contentStyle={{ background: "#0E1525", border: "1px solid #243056", borderRadius: 8, fontSize: 12 }}
            />
            <Line type="monotone" dataKey="close" stroke="#52E0C4" dot={false} strokeWidth={1.5} isAnimationActive={!reduced} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-slate-400 mt-2">
        {pts.length} daily closes, {first.date} to {last.date}: from <span className="font-mono text-slate-200">{fmtPrice(first.close)}</span> to{" "}
        <span className="font-mono text-slate-200">{fmtPrice(last.close)}</span>. Stored at build time; not a live quote.
      </p>
    </section>
  );
}
