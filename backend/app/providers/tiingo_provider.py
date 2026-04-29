"""Tiingo provider — prices and news."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .base import ProviderStatus

log = logging.getLogger(__name__)
BASE = "https://api.tiingo.com"
TIMEOUT = 10.0


class TiingoProvider:
    name: str = "tiingo"

    def __init__(self) -> None:
        self.api_key = settings.tiingo_api_key

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=bool(self.api_key),
            healthy=bool(self.api_key),
            notes="" if self.api_key else "Set TIINGO_API_KEY to enable.",
            capabilities=["prices", "news"],
        )

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Token {self.api_key}"}

    def get_price_history(self, ticker: str, days: int = 252) -> Optional[List[Dict[str, Any]]]:
        if not self.api_key:
            return None
        try:
            with httpx.Client(timeout=TIMEOUT, headers=self._headers()) as client:
                r = client.get(f"{BASE}/tiingo/daily/{ticker}/prices", params={"resampleFreq": "daily"})
                if r.status_code != 200:
                    return None
                rows = r.json()
            return [
                dict(
                    date=row.get("date", "")[:10],
                    open=row.get("open"),
                    high=row.get("high"),
                    low=row.get("low"),
                    close=row.get("close"),
                    adjusted_close=row.get("adjClose"),
                    volume=row.get("volume"),
                )
                for row in rows
            ][-days:]
        except Exception as exc:  # pragma: no cover
            log.warning("Tiingo fetch failed: %s", exc)
            return None

    def get_news(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
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
    def list_tickers(self) -> List[str]: return []
