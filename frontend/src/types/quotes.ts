// W5b live quotes — TypeScript mirror of `GET /api/quotes`
// (backend `app/schemas/quotes.py`, produced by `services/quote_service.py`).
// Mirrored by hand; the captured `test/fixtures/quotes.wire.json` is the
// route's own output and `backend/app/tests/test_quotes_ui_fixture_contract.py`
// fails when either side drifts, so re-capture rather than editing it.
//
// Every datetime is an ISO-8601 UTC string ending in "Z". The UI formats
// times in America/New_York explicitly, never in the viewer's zone.

/** Where the price came from, in the order the service falls back:
 *  a fresh provider quote, the last cached quote after a provider miss,
 *  the last stored daily close, or nothing at all. */
export type QuoteSource = "live" | "stale" | "eod_close" | "unavailable";

export interface QuoteOut {
  ticker: string;
  price: number | null;
  previous_close: number | null;
  change: number | null;
  /** Percent units as providers send them (1.2 means 1.2%). */
  change_pct: number | null;
  /** The provider's own trade/quote time, when it sent a plausible one. */
  price_time: string | null;
  /** When MarketMosaic fetched it (the shared cache row). */
  fetched_at: string | null;
  /** price_time, else fetched_at; for `eod_close`, that session's close. */
  as_of: string | null;
  source: QuoteSource;
  provider: string | null;
  /** Provider quotes may be delayed; a stored close is not. */
  delayed: boolean;
  /** provider_miss | refresh_deferred | unknown_ticker | no_stored_close */
  reason: string | null;
}

export interface MarketState {
  is_open: boolean;
  /** "open" | "pre_open" | "after_close" | "weekend" | "holiday:<name>" */
  reason: string;
  /** The most recent session that has opened (today while open). */
  session_date: string;
  session_open: string;
  session_close: string;
  early_close: boolean;
  next_open: string;
}

export interface QuotesOut {
  /** Request order, deduped, upper-cased. */
  quotes: QuoteOut[];
  market: MarketState;
  ttl_seconds_in_session: number;
}
