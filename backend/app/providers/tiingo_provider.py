"""Tiingo provider — prices and news."""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

import httpx

from ..config import settings
from .base import ProviderStatus, log_safely
from .price_history import history_start, normalize_history

log = logging.getLogger(__name__)
BASE = "https://api.tiingo.com"
TIMEOUT = 10.0


class TiingoProvider:
    name: str = "tiingo"
    price_history_provenance = {
        "provider": "tiingo", "endpoint": "/tiingo/daily/{ticker}/prices",
        "close_basis": "raw_as_traded", "adjusted_close_basis": "split_and_dividend_adjusted",
    }

    def __init__(self) -> None:
        self.api_key = settings.tiingo_api_key

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=bool(self.api_key),
            healthy=bool(self.api_key),
            notes="" if self.api_key else "Set TIINGO_API_KEY to enable.",
            capabilities=["prices", "quote", "news"],
        )

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Token {self.api_key}"}

    def get_quote(self, ticker: str) -> dict[str, Any] | None:
        """`/iex/{ticker}` — IEX real-time last trade during market hours."""
        if not self.api_key:
            return None
        try:
            with httpx.Client(timeout=TIMEOUT, headers=self._headers()) as client:
                r = client.get(f"{BASE}/iex/{ticker.upper()}")
                if r.status_code != 200:
                    return None
                rows = r.json()
        except Exception as exc:  # pragma: no cover
            log_safely(log, f"Tiingo quote failed for {ticker}", exc)
            return None
        if not isinstance(rows, list) or not rows:
            return None
        item = rows[0]
        price = item.get("last") or item.get("tngoLast")
        prev_close = item.get("prevClose")
        return dict(
            ticker=item.get("ticker"),
            price=price,
            previous_close=prev_close,
            change=(price - prev_close) if (price is not None and prev_close is not None) else None,
            change_pct=((price - prev_close) / prev_close * 100.0) if (price is not None and prev_close) else None,
            day_low=item.get("low"),
            day_high=item.get("high"),
            volume=item.get("volume"),
            timestamp=item.get("timestamp"),
        )

    def get_price_history(self, ticker: str, days: int = 252) -> list[dict[str, Any]] | None:
        if not self.api_key:
            return None
        # Without startDate this endpoint returns only the latest close.
        # Match Polygon's calendar-day slack and retain the complete response.
        end = date.today()
        start = history_start(end, days)
        if start is None:
            return None
        try:
            with httpx.Client(timeout=TIMEOUT, headers=self._headers()) as client:
                r = client.get(f"{BASE}/tiingo/daily/{ticker.upper()}/prices", params={
                    "resampleFreq": "daily",
                    "startDate": start.isoformat(),
                    "endDate": end.isoformat(),
                })
                if r.status_code != 200:
                    log.warning("Tiingo price history ticker=%s status=%s", ticker, r.status_code)
                    return None
                rows = r.json()
            if not isinstance(rows, list):
                log.warning("Tiingo price history invalid payload ticker=%s", ticker)
                return None
            mapped = [
                dict(
                    date=row.get("date"), open=row.get("open"), high=row.get("high"),
                    low=row.get("low"), close=row.get("close"),
                    adjusted_close=row.get("adjClose"), volume=row.get("volume"),
                ) if isinstance(row, dict) else row
                for row in rows
            ]
            return normalize_history(mapped, provider=self.name, ticker=ticker, start=start, end=end, log=log)
        except Exception as exc:  # pragma: no cover
            log_safely(log, f"Tiingo fetch failed for {ticker}", exc)
            return None

    def get_news(self, ticker: str) -> list[dict[str, Any]] | None:
        return None

    # Stubs
    def get_company_profile(self, ticker: str): return None
    def get_financial_statements(self, ticker: str): return None
    def get_ratios(self, ticker: str): return None
    def get_key_metrics(self, ticker: str): return None
    def get_earnings(self, ticker: str): return None
    def get_earnings_transcripts(self, ticker: str): return None
    def get_filings(self, ticker: str): return None
    def get_estimates(self, ticker: str): return None
    def get_macro_series(self, series_id: str): return None
    def list_tickers(self) -> list[str]: return []
