"""Provider interface: contract every market data backend must satisfy.

Every method should return either a populated dict/list or `None` if the
provider doesn't support that endpoint. The data service is responsible for
falling back to the demo provider when a method returns None or raises.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


def log_safely(
    log: logging.Logger, msg: str, exc: BaseException, *, level: int = logging.WARNING,
) -> None:
    """Providers' entry to `agents.log_safety.log_safely` (type at `level`,
    redacted detail at DEBUG).

    Imported lazily because `app.agents.__init__` pulls in the orchestrator,
    which reaches `data_service`, which imports every provider — a
    module-level import here would cycle whenever a provider module is
    the first thing imported (as the provider unit tests do).
    """
    from ..agents.log_safety import log_safely as _impl
    _impl(log, msg, exc, level=level)


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
