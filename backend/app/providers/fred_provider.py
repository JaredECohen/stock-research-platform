"""FRED macro provider."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .base import ProviderStatus

log = logging.getLogger(__name__)
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
TIMEOUT = 10.0

# Curated macro catalog. Editable here without touching consumers.
# Each entry: (series_id, display_name, units). Points are fetched
# on demand via `get_macro_series(series_id)`.
_MACRO_CATALOG: List[Dict[str, str]] = [
    {"series_id": "FEDFUNDS", "name": "Federal Funds Rate", "units": "%"},
    {"series_id": "DGS2", "name": "2-Year Treasury", "units": "%"},
    {"series_id": "DGS10", "name": "10-Year Treasury", "units": "%"},
    {"series_id": "DGS30", "name": "30-Year Treasury", "units": "%"},
    {"series_id": "CPIAUCSL", "name": "Headline CPI YoY", "units": "%"},
    {"series_id": "CORESTICKM159SFRBATL", "name": "Sticky Core CPI YoY", "units": "%"},
    {"series_id": "PCEPI", "name": "PCE Inflation YoY", "units": "%"},
    {"series_id": "UNRATE", "name": "Unemployment Rate", "units": "%"},
    {"series_id": "PAYEMS", "name": "Nonfarm Payrolls", "units": "thousand"},
    {"series_id": "GDPC1", "name": "Real GDP YoY", "units": "%"},
    {"series_id": "RSAFS", "name": "Retail Sales YoY", "units": "%"},
    {"series_id": "BAMLH0A0HYM2", "name": "High-Yield Spread", "units": "%"},
    {"series_id": "DCOILWTICO", "name": "Oil Price (WTI)", "units": "$/bbl"},
]


class FREDProvider:
    name: str = "fred"

    def __init__(self) -> None:
        self.api_key = settings.fred_api_key

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=bool(self.api_key),
            healthy=bool(self.api_key),
            notes="" if self.api_key else "Set FRED_API_KEY to enable.",
            capabilities=["macro"],
        )

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        try:
            params = dict(series_id=series_id, api_key=self.api_key, file_type="json", limit=24, sort_order="desc")
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.get(BASE_URL, params=params)
                if r.status_code != 200:
                    return None
                data = r.json()
            obs = list(reversed(data.get("observations", [])))
            return dict(
                series_id=series_id,
                name=series_id,
                units="",
                points=[
                    dict(date=o.get("date"), value=float(o["value"]) if o.get("value") not in (".", "", None) else None)
                    for o in obs
                ],
            )
        except Exception as exc:  # pragma: no cover
            log.warning("FRED fetch failed: %s", exc)
            return None

    def list_macro_series(self) -> List[Dict[str, Any]]:
        """Return the curated catalog metadata. Points fetched per series."""
        return [dict(c, points=[]) for c in _MACRO_CATALOG]

    # Stubs for other BaseProvider methods
    def get_company_profile(self, ticker: str): return None
    def get_price_history(self, ticker: str, days: int = 252): return None
    def get_financial_statements(self, ticker: str): return None
    def get_ratios(self, ticker: str): return None
    def get_key_metrics(self, ticker: str): return None
    def get_earnings(self, ticker: str): return None
    def get_earnings_transcripts(self, ticker: str): return None
    def get_filings(self, ticker: str): return None
    def get_news(self, ticker: str): return None
    def get_estimates(self, ticker: str): return None
    def list_tickers(self) -> List[str]: return []
