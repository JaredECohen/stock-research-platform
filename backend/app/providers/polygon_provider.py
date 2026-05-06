"""Polygon.io provider — prices and news."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .base import ProviderStatus

log = logging.getLogger(__name__)
BASE = "https://api.polygon.io"
TIMEOUT = 10.0


class PolygonProvider:
    name: str = "polygon"

    def __init__(self) -> None:
        self.api_key = settings.polygon_api_key

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=bool(self.api_key),
            healthy=bool(self.api_key),
            notes="" if self.api_key else "Set POLYGON_API_KEY to enable.",
            capabilities=["prices", "quote", "news"],
        )

    def _get(self, path: str, **params: Any) -> Optional[Any]:
        if not self.api_key:
            return None
        try:
            params["apiKey"] = self.api_key
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.get(f"{BASE}{path}", params=params)
                if r.status_code != 200:
                    return None
                return r.json()
        except Exception as exc:  # pragma: no cover
            log.warning("Polygon fetch failed: %s", exc)
            return None

    def get_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Snapshot endpoint — last trade + previous-day close.

        Free tier limits to 5 calls/min; paid tiers are real-time.
        """
        data = self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}")
        if not data or "ticker" not in data:
            return None
        snap = data["ticker"]
        last_trade = (snap.get("lastTrade") or {})
        prev_day = (snap.get("prevDay") or {})
        day = (snap.get("day") or {})
        price = last_trade.get("p") or day.get("c")
        prev_close = prev_day.get("c")
        return dict(
            ticker=snap.get("ticker"),
            price=price,
            previous_close=prev_close,
            change=(price - prev_close) if (price is not None and prev_close) else None,
            change_pct=snap.get("todaysChangePerc"),
            day_low=day.get("l"),
            day_high=day.get("h"),
            volume=day.get("v"),
            timestamp=last_trade.get("t"),
        )

    def get_price_history(self, ticker: str, days: int = 252) -> Optional[List[Dict[str, Any]]]:
        end = date.today()
        start = end - timedelta(days=int(days * 1.6))
        data = self._get(f"/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start.isoformat()}/{end.isoformat()}")
        if not data or "results" not in data:
            return None
        return [
            dict(
                date=date.fromtimestamp(r["t"] / 1000).isoformat(),
                open=r.get("o"),
                high=r.get("h"),
                low=r.get("l"),
                close=r.get("c"),
                adjusted_close=r.get("c"),
                volume=r.get("v"),
            )
            for r in data["results"]
        ][-days:]

    def get_news(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        data = self._get("/v2/reference/news", ticker=ticker.upper(), limit=20)
        if not data or "results" not in data:
            return None
        return [
            dict(
                title=n.get("title"),
                source=(n.get("publisher") or {}).get("name"),
                published_at=n.get("published_utc"),
                url=n.get("article_url"),
                summary=n.get("description"),
                tickers=n.get("tickers", [ticker]),
                sentiment="neutral",
                relevance_score=0.6,
            )
            for n in data["results"]
        ]

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
    def list_tickers(self) -> List[str]: return []
