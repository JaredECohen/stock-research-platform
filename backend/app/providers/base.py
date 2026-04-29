"""Provider interface: contract every market data backend must satisfy.

Every method should return either a populated dict/list or `None` if the
provider doesn't support that endpoint. The data service is responsible for
falling back to the demo provider when a method returns None or raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ProviderStatus:
    name: str
    configured: bool
    healthy: bool = True
    notes: str = ""
    capabilities: List[str] = field(default_factory=list)


class BaseProvider(Protocol):
    name: str

    def status(self) -> ProviderStatus: ...

    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]: ...

    def get_price_history(self, ticker: str, days: int = 252) -> Optional[List[Dict[str, Any]]]: ...

    def get_financial_statements(self, ticker: str) -> Optional[Dict[str, Any]]: ...

    def get_ratios(self, ticker: str) -> Optional[Dict[str, Any]]: ...

    def get_key_metrics(self, ticker: str) -> Optional[Dict[str, Any]]: ...

    def get_earnings(self, ticker: str) -> Optional[Dict[str, Any]]: ...

    def get_earnings_transcripts(self, ticker: str) -> Optional[List[Dict[str, Any]]]: ...

    def get_filings(self, ticker: str) -> Optional[List[Dict[str, Any]]]: ...

    def get_news(self, ticker: str) -> Optional[List[Dict[str, Any]]]: ...

    def get_estimates(self, ticker: str) -> Optional[Dict[str, Any]]: ...

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]: ...

    def list_tickers(self) -> List[str]: ...
