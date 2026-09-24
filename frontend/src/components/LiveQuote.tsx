// W5b — the live-quote chip on Research and DCF Lab.
//
// Owner decision (2026-09-24): market prices are pulled live and cached ~15
// minutes. The backend owns that policy (`quote_service`); this chip shows
// what it answered and, above all, HOW OLD it is, because a price without a
// time is how a months-old memo number used to read as "Current":
//
//   live, market open   $123.45  +2.45 (+2.02%) · 10:44 AM ET · may be delayed up to 15 min
//                       (the day too, "Tue Sep 22, 4:00 PM ET", when the provider's
//                        time is from an earlier session)
//   live, market closed $88.10 · At close, Wed Sep 23, 4:00 PM ET
//                       ("At close" only when the provider's own time is at or
//                        after the session close; otherwise "last quote …")
//   stale (amber)       $64.20 · last quote 10:24 AM ET, live quote unavailable
//   eod_close           $45.67 · last close, Tue Sep 22 (no live quote)
//   unavailable         nothing at all: a missing price never grows an error box
//
// Times are always America/New_York and say "ET", never the viewer's zone.
// Refetches: every 5 minutes while the market is open and the tab is
// visible, when the tab becomes visible again, and at the next open when
// the market is closed. The server's 15-minute cache means a poll costs a
// DB read, not a provider call. Not rendered on the public sample pages:
// anonymous showcase traffic must not draw on the quote quota.
import React, { useEffect, useRef, useState } from "react";
import { api } from "@/api/client";
import { logEvent } from "@/lib/logger";
import { fmtPrice, fmtUpside } from "@/lib/format";
import type { MarketState, QuoteOut } from "@/types/quotes";

export const POLL_MS = 5 * 60 * 1000;
/** Refetch this long after the next open, so the first poll sees an open market. */
export const NEXT_OPEN_SLACK_MS = 30 * 1000;
/** A tab that comes back within this window reuses what it has. */
const VISIBLE_REFETCH_MIN_MS = 60 * 1000;
/** setTimeout's ceiling (~24.8 days); a longer wait is simply not scheduled. */
const MAX_TIMEOUT_MS = 2 ** 31 - 1;

const ET = "America/New_York";
const timeFmt = new Intl.DateTimeFormat("en-US", { timeZone: ET, hour: "numeric", minute: "2-digit" });
const dayFmt = new Intl.DateTimeFormat("en-US", { timeZone: ET, weekday: "short", month: "short", day: "numeric" });
const shortDateFmt = new Intl.DateTimeFormat("en-US", { timeZone: ET, month: "short", day: "numeric" });
const etDateKey = new Intl.DateTimeFormat("en-CA", { timeZone: ET, year: "numeric", month: "2-digit", day: "2-digit" });

/** Backend timestamps on the memo are naive UTC ("2026-09-03T14:00:00");
 *  `Date` would read those as local time. Quote times always carry "Z". */
export function parseUtc(value: string): Date {
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(value);
  return new Date(hasZone || !value.includes("T") ? value : `${value}Z`);
}

export function fmtEtTime(iso: string): string {
  return `${timeFmt.format(parseUtc(iso))} ET`;
}

export function fmtEtDay(iso: string): string {
  // "Wed, Sep 23" -> "Wed Sep 23"
  return dayFmt.format(parseUtc(iso)).replace(",", "");
}

/** "2026-09-22": the New York calendar date of a UTC instant. Memo and model
 *  save times are naive UTC, so `iso.slice(0, 10)` is the UTC date, a day
 *  ahead for anything saved after 8 PM ET; this is the date the chip shows. */
export function fmtEtDate(iso: string): string {
  return etDateKey.format(parseUtc(iso));
}

function signed(v: number, digits = 2): string {
  return `${v > 0 ? "+" : ""}${v.toFixed(digits)}`;
}

type Loaded = { quote: QuoteOut; market: MarketState };

export default function LiveQuote({
  ticker,
  priceAtMemo,
  memoAt,
  onQuote,
  className = "",
}: {
  ticker: string;
  /** The memo's frozen quote; renders "Since memo: …" drift when set. */
  priceAtMemo?: number | null;
  memoAt?: string | null;
  /** Called with each answer (null when unavailable or on error). */
  onQuote?: (quote: QuoteOut | null) => void;
  className?: string;
}) {
  const symbol = ticker.trim().toUpperCase();
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [tick, setTick] = useState(0);
  const lastFetch = useRef(0);
  const onQuoteRef = useRef(onQuote);
  onQuoteRef.current = onQuote;

  useEffect(() => {
    if (!symbol) return;
    let cancelled = false;
    lastFetch.current = Date.now();
    // Promise.resolve first: a synchronous throw from the client becomes a
    // rejection handled below, never an exception out of the effect.
    Promise.resolve()
      .then(() => api.getQuotes([symbol]))
      .then((res) => {
        if (cancelled) return;
        const quote = res.quotes.find((q) => q.ticker === symbol) ?? null;
        setLoaded(quote ? { quote, market: res.market } : null);
        onQuoteRef.current?.(quote && quote.source !== "unavailable" ? quote : null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setLoaded(null);
        onQuoteRef.current?.(null);
        logEvent({
          kind: "live_quote_error",
          payload: { ticker: symbol, message: String((err as Error)?.message ?? err).slice(0, 200) },
        });
      });
    return () => {
      cancelled = true;
    };
  }, [symbol, tick]);

  // Coming back to the tab refetches (throttled), whatever the market state.
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === "visible" && Date.now() - lastFetch.current >= VISIBLE_REFETCH_MIN_MS) {
        setTick((t) => t + 1);
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, []);

  const isOpen = loaded?.market.is_open ?? null;
  const nextOpen = loaded?.market.next_open ?? null;
  useEffect(() => {
    if (isOpen === null) return;
    if (isOpen) {
      const id = window.setInterval(() => {
        if (document.visibilityState === "visible") setTick((t) => t + 1);
      }, POLL_MS);
      return () => window.clearInterval(id);
    }
    if (!nextOpen) return;
    const delay = Date.parse(nextOpen) - Date.now() + NEXT_OPEN_SLACK_MS;
    if (delay <= 0 || delay > MAX_TIMEOUT_MS) return;
    const id = window.setTimeout(() => setTick((t) => t + 1), delay);
    return () => window.clearTimeout(id);
  }, [isOpen, nextOpen]);

  if (!loaded || loaded.quote.ticker !== symbol) return null;
  const { quote, market } = loaded;
  if (quote.source === "unavailable" || quote.price == null) return null;

  let detail: React.ReactNode;
  let tone = "text-slate-400";
  if (quote.source === "live" && market.is_open) {
    const up = quote.change ?? 0;
    // Just after the open a delayed feed can still answer with the prior
    // session's print (and its change): a bare "4:00 PM ET" would read as
    // today, so an as_of outside today's session carries its day.
    const today = quote.as_of != null && fmtEtDate(quote.as_of) === market.session_date;
    detail = (
      <>
        {quote.change != null && (
          <span className={up > 0 ? "text-accent-500" : up < 0 ? "text-danger-500" : "text-slate-400"}>
            {signed(quote.change)}
            {quote.change_pct != null && ` (${signed(quote.change_pct)}%)`}
          </span>
        )}
        {quote.as_of && (
          <span> · {today ? fmtEtTime(quote.as_of) : `${fmtEtDay(quote.as_of)}, ${fmtEtTime(quote.as_of)}`}</span>
        )}
        {quote.delayed && <span> · may be delayed up to 15 min</span>}
      </>
    );
  } else if (quote.source === "live") {
    const atClose = quote.price_time != null && parseUtc(quote.price_time) >= parseUtc(market.session_close);
    detail = atClose ? (
      <span>
        · At close, {fmtEtDay(market.session_close)}, {fmtEtTime(market.session_close)}
      </span>
    ) : (
      <span>
        · last quote {quote.as_of ? `${fmtEtDay(quote.as_of)}, ${fmtEtTime(quote.as_of)}` : "time unknown"}
        {quote.delayed && " · may be delayed up to 15 min"}
      </span>
    );
  } else if (quote.source === "stale") {
    tone = "text-warn-500";
    const sameDay = quote.as_of != null && fmtEtDate(quote.as_of) === market.session_date;
    const when = quote.as_of ? (sameDay ? fmtEtTime(quote.as_of) : `${fmtEtDay(quote.as_of)}, ${fmtEtTime(quote.as_of)}`) : "time unknown";
    detail = <span>· last quote {when}, live quote unavailable</span>;
  } else {
    detail = <span>· last close{quote.as_of ? `, ${fmtEtDay(quote.as_of)}` : ""} (no live quote)</span>;
  }

  const drift =
    priceAtMemo != null && priceAtMemo > 0 ? (quote.price - priceAtMemo) / priceAtMemo : null;

  return (
    <div
      className={`text-sm ${className}`}
      data-testid="live-quote"
      data-source={quote.source}
      title={quote.provider ? `Source: ${quote.provider}` : undefined}
    >
      <div className={`flex flex-wrap items-baseline gap-x-2 ${tone}`}>
        <span className="font-mono text-base text-slate-100">{fmtPrice(quote.price)}</span>{" "}
        {detail}
      </div>
      {drift != null && (
        <div className="text-xs text-slate-500 mt-0.5">
          Since memo: {fmtUpside(drift)} (memo price {fmtPrice(priceAtMemo)}
          {memoAt ? `, ${shortDateFmt.format(parseUtc(memoAt))}` : ""})
        </div>
      )}
    </div>
  );
}
