"""`GET /api/quotes` — live quotes with an honest label (W5b).

Imported from `app.schemas.quotes` directly, like `portfolio_exposure`: the
package's `__all__` (frozen by `test_schema_package_reexports`) is the memo
contract's surface, and this route is not part of it. `StockMemoOut` and
`DCFResult` are unchanged.

Every datetime here is aware UTC, so the browser never reads one as local
time; the UI formats it in America/New_York explicitly.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel


class QuoteOut(BaseModel):
    ticker: str
    price: float | None = None
    previous_close: float | None = None
    change: float | None = None
    # Percent units as providers send them (1.2 == 1.2%).
    change_pct: float | None = None
    # The provider's own trade/quote time, when it sent a plausible one.
    price_time: datetime | None = None
    # When MarketMosaic fetched it (the shared cache row).
    fetched_at: datetime | None = None
    # price_time, else fetched_at; for `eod_close`, that session's close.
    as_of: datetime | None = None
    source: Literal["live", "stale", "eod_close", "unavailable"]
    provider: str | None = None
    # Provider quotes may be delayed (plan-dependent); a stored close is not.
    delayed: bool = True
    # provider_miss | refresh_deferred | unknown_ticker | no_stored_close
    reason: str | None = None


class MarketStateOut(BaseModel):
    is_open: bool
    # "open" | "pre_open" | "after_close" | "weekend" | "holiday:<name>"
    reason: str
    # The most recent session that has opened (today while open).
    session_date: date
    session_open: datetime
    session_close: datetime
    early_close: bool
    next_open: datetime


class QuotesOut(BaseModel):
    quotes: list[QuoteOut]                # request order, deduped
    market: MarketStateOut
    ttl_seconds_in_session: int = 900
