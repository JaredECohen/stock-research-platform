"""Financial Modeling Prep provider.

Implements the BaseProvider methods backed by the FMP REST API. Every method
catches network/HTTP errors and returns None so the data service can fall
back to demo data without breaking the request.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .base import ProviderStatus

log = logging.getLogger(__name__)
BASE_URL = "https://financialmodelingprep.com/api/v3"
TIMEOUT = 10.0


class FMPProvider:
    name: str = "fmp"

    def __init__(self) -> None:
        self.api_key = settings.fmp_api_key

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=bool(self.api_key),
            healthy=bool(self.api_key),
            notes="" if self.api_key else "Set FMP_API_KEY to enable.",
            capabilities=["profile", "prices", "financials", "ratios", "estimates"],
        )

    def _get(self, path: str, **params: Any) -> Optional[Any]:
        if not self.api_key:
            return None
        try:
            params["apikey"] = self.api_key
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.get(f"{BASE_URL}{path}", params=params)
                if r.status_code != 200:
                    log.warning("FMP %s -> %s", path, r.status_code)
                    return None
                return r.json()
        except Exception as exc:  # pragma: no cover - network paths
            log.warning("FMP request failed: %s", exc)
            return None

    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]:
        data = self._get(f"/profile/{ticker.upper()}")
        if not data:
            return None
        item = data[0] if isinstance(data, list) and data else None
        if not item:
            return None
        return dict(
            ticker=item.get("symbol"),
            company_name=item.get("companyName"),
            exchange=item.get("exchangeShortName") or "",
            sector=item.get("sector") or "",
            industry=item.get("industry") or "",
            sub_industry=item.get("industry"),
            country=item.get("country") or "US",
            currency=item.get("currency") or "USD",
            market_cap=item.get("mktCap"),
            cik=item.get("cik"),
            business_description=item.get("description") or "",
            fiscal_year_end=None,
            is_active=True,
            is_etf=item.get("isEtf", False),
            beta=item.get("beta"),
            shares_outstanding=item.get("sharesOutstanding"),
            last_price=item.get("price"),
        )

    def get_price_history(self, ticker: str, days: int = 252) -> Optional[List[Dict[str, Any]]]:
        data = self._get(f"/historical-price-full/{ticker.upper()}", serietype="line", timeseries=days)
        if not data:
            return None
        rows = data.get("historical", [])[::-1]
        return [
            dict(
                date=r.get("date"),
                open=r.get("open"),
                high=r.get("high"),
                low=r.get("low"),
                close=r.get("close"),
                adjusted_close=r.get("adjClose") or r.get("close"),
                volume=r.get("volume"),
            )
            for r in rows
        ]

    def get_financial_statements(self, ticker: str) -> Optional[Dict[str, Any]]:
        income = self._get(f"/income-statement/{ticker.upper()}", limit=4) or []
        balance = self._get(f"/balance-sheet-statement/{ticker.upper()}", limit=4) or []
        cash = self._get(f"/cash-flow-statement/{ticker.upper()}", limit=4) or []
        if not income:
            return None
        return dict(income=income, balance=balance, cash=cash)

    def get_ratios(self, ticker: str) -> Optional[Dict[str, Any]]:
        data = self._get(f"/ratios/{ticker.upper()}", limit=1)
        if not data:
            return None
        return data[0] if isinstance(data, list) and data else None

    def get_key_metrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        data = self._get(f"/key-metrics/{ticker.upper()}", limit=1)
        if not data:
            return None
        return data[0] if isinstance(data, list) and data else None

    def get_earnings(self, ticker: str) -> Optional[Dict[str, Any]]:
        return None

    def get_earnings_transcripts(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        return None

    def get_filings(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        return None

    def get_news(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        return None

    def get_estimates(self, ticker: str) -> Optional[Dict[str, Any]]:
        return None

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        return None

    def list_tickers(self) -> List[str]:
        return []
