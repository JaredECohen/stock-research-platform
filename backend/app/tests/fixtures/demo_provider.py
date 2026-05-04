"""DemoProvider — always-on data backend that satisfies the BaseProvider contract.

It builds the dataset on first access (in-memory) and exposes the same
endpoints as live providers, so any consumer that goes through the service
layer is provider-agnostic.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from ..data.demo_dataset import build_dataset
from .base import ProviderStatus


@lru_cache(maxsize=1)
def _dataset() -> Dict[str, Dict]:
    return build_dataset()


class DemoProvider:
    name: str = "demo"

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=True,
            healthy=True,
            notes="Always-on demo dataset (illustrative; not real-time).",
            capabilities=[
                "profile", "prices", "financials", "ratios", "earnings",
                "transcripts", "filings", "news", "estimates", "macro",
            ],
        )

    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]:
        d = _dataset().get(ticker.upper())
        return d["profile"] if d else None

    def get_price_history(self, ticker: str, days: int = 252) -> Optional[List[Dict[str, Any]]]:
        d = _dataset().get(ticker.upper())
        return d["prices"][-days:] if d else None

    def get_financial_statements(self, ticker: str) -> Optional[Dict[str, Any]]:
        d = _dataset().get(ticker.upper())
        if not d:
            return None
        return dict(
            income=d["income_statements"],
            balance=d["balance_sheets"],
            cash=d["cash_flows"],
        )

    def get_ratios(self, ticker: str) -> Optional[Dict[str, Any]]:
        d = _dataset().get(ticker.upper())
        return d["ratios"] if d else None

    def get_key_metrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        d = _dataset().get(ticker.upper())
        if not d:
            return None
        return dict(profile=d["profile"], ratios=d["ratios"], earnings=d["earnings"])

    def get_earnings(self, ticker: str) -> Optional[Dict[str, Any]]:
        d = _dataset().get(ticker.upper())
        return d["earnings"] if d else None

    def get_earnings_transcripts(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        d = _dataset().get(ticker.upper())
        return d["transcripts"] if d else None

    def get_filings(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        d = _dataset().get(ticker.upper())
        return d["filings"] if d else None

    def get_news(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        d = _dataset().get(ticker.upper())
        return d["news"] if d else None

    def get_estimates(self, ticker: str) -> Optional[Dict[str, Any]]:
        d = _dataset().get(ticker.upper())
        return d["estimates"] if d else None

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        macro = _dataset().get("_macro", {})
        for s in macro.get("series", []):
            if s["series_id"].lower() == series_id.lower():
                return s
        # also allow lookup by name keywords
        for s in macro.get("series", []):
            if series_id.lower() in s["name"].lower():
                return s
        return None

    def list_tickers(self) -> List[str]:
        return [t for t in _dataset().keys() if not t.startswith("_")]

    def list_macro_series(self) -> List[Dict[str, Any]]:
        return _dataset().get("_macro", {}).get("series", [])
