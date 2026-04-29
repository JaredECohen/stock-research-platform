"""Alpha Vantage provider — earnings transcripts and news/sentiment."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .base import ProviderStatus

log = logging.getLogger(__name__)
BASE_URL = "https://www.alphavantage.co/query"
TIMEOUT = 10.0


class AlphaVantageProvider:
    name: str = "alpha_vantage"

    def __init__(self) -> None:
        self.api_key = settings.alpha_vantage_api_key

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=bool(self.api_key),
            healthy=bool(self.api_key),
            notes="" if self.api_key else "Set ALPHA_VANTAGE_API_KEY to enable.",
            capabilities=["transcripts", "news", "fundamentals_fallback"],
        )

    def _get(self, **params: Any) -> Optional[Any]:
        if not self.api_key:
            return None
        try:
            params["apikey"] = self.api_key
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.get(BASE_URL, params=params)
                if r.status_code != 200:
                    log.warning("AlphaVantage %s -> %s", params, r.status_code)
                    return None
                return r.json()
        except Exception as exc:  # pragma: no cover
            log.warning("AlphaVantage request failed: %s", exc)
            return None

    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]:
        data = self._get(function="OVERVIEW", symbol=ticker)
        if not data or "Symbol" not in data:
            return None
        return dict(
            ticker=data.get("Symbol"),
            company_name=data.get("Name"),
            exchange=data.get("Exchange") or "",
            sector=data.get("Sector") or "",
            industry=data.get("Industry") or "",
            country=data.get("Country") or "US",
            currency=data.get("Currency") or "USD",
            market_cap=float(data.get("MarketCapitalization") or 0) or None,
            business_description=data.get("Description") or "",
            beta=float(data.get("Beta") or 0) or None,
            shares_outstanding=float(data.get("SharesOutstanding") or 0) or None,
            last_price=None,
        )

    def get_price_history(self, ticker: str, days: int = 252) -> Optional[List[Dict[str, Any]]]:
        return None

    def get_financial_statements(self, ticker: str) -> Optional[Dict[str, Any]]:
        return None

    def get_ratios(self, ticker: str) -> Optional[Dict[str, Any]]:
        return None

    def get_key_metrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        return None

    def get_earnings(self, ticker: str) -> Optional[Dict[str, Any]]:
        return None

    def get_earnings_transcripts(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        data = self._get(function="EARNINGS_CALL_TRANSCRIPT", symbol=ticker)
        if not data or "transcript" not in data:
            return None
        return [
            dict(
                ticker=ticker,
                period=data.get("quarter"),
                speakers=[item.get("speaker") for item in data.get("transcript", [])],
                prepared_remarks=" ".join(item.get("content", "") for item in data.get("transcript", []) if item.get("type") == "presentation"),
                qa=" ".join(item.get("content", "") for item in data.get("transcript", []) if item.get("type") == "qa"),
            )
        ]

    def get_filings(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        return None

    def get_news(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        data = self._get(function="NEWS_SENTIMENT", tickers=ticker, limit=20)
        if not data or "feed" not in data:
            return None
        return [
            dict(
                title=n.get("title"),
                source=n.get("source"),
                published_at=n.get("time_published"),
                url=n.get("url"),
                summary=n.get("summary"),
                tickers=[ticker],
                topics=[t.get("topic") for t in n.get("topics", [])],
                sentiment=n.get("overall_sentiment_label"),
                relevance_score=float(n.get("relevance_score") or 0.5),
            )
            for n in data.get("feed", [])
        ]

    def get_estimates(self, ticker: str) -> Optional[Dict[str, Any]]:
        return None

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        return None

    def list_tickers(self) -> List[str]:
        return []
