// W5b live-quote fixtures.
//
// `quotes.wire.json` is NOT hand-written: it is what the real
// `GET /api/quotes` route served through TestClient
// (`backend/app/scripts/capture_quotes_ui_fixture.py`), with only the clock
// and the provider seam pinned (its `meta.pinned` says exactly what).
// `backend/app/tests/test_quotes_ui_fixture_contract.py` re-runs the capture
// and requires the committed file to equal it, so a renamed key or a
// changed label rule fails a test instead of drifting. Re-capture; never
// edit it by hand.
//
// One body per chip state, all on Wednesday 2026-09-23:
//   liveOpen     10:45 ET, a fresh provider quote timed 10:44:30 ET
//   liveAtClose  17:00 ET, a provider quote timed at the 16:00 ET close
//   stale        10:45 ET, a 20-minute-old cached quote after a provider miss
//   eodClose     10:45 ET, no quote: the stored close of Tue Sep 22
//   unavailable  a symbol that is not a company (never sent to a provider)
import type { QuotesOut } from "@/types/quotes";
import wire from "./quotes.wire.json";

export const liveOpen = wire.live_open as QuotesOut;
export const liveAtClose = wire.live_at_close as QuotesOut;
export const staleQuote = wire.stale as QuotesOut;
export const eodClose = wire.eod_close as QuotesOut;
export const unavailable = wire.unavailable as QuotesOut;
