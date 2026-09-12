"""Provider interface: contract every market data backend must satisfy.

Every method should return either a populated dict/list or `None` if the
provider doesn't support that endpoint. The data service is responsible for
falling back to the demo provider when a method returns None or raises.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol


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
    capabilities: list[str] = field(default_factory=list)


class BaseProvider(Protocol):
    name: str

    def status(self) -> ProviderStatus: ...

    def get_company_profile(self, ticker: str) -> dict[str, Any] | None: ...

    def get_price_history(self, ticker: str, days: int = 252) -> list[dict[str, Any]] | None: ...

    def get_financial_statements(self, ticker: str) -> dict[str, Any] | None: ...

    def get_ratios(self, ticker: str) -> dict[str, Any] | None: ...

    def get_key_metrics(self, ticker: str) -> dict[str, Any] | None: ...

    def get_earnings(self, ticker: str) -> dict[str, Any] | None: ...

    def get_earnings_transcripts(self, ticker: str) -> list[dict[str, Any]] | None: ...

    def get_filings(self, ticker: str) -> list[dict[str, Any]] | None: ...

    def get_news(self, ticker: str) -> list[dict[str, Any]] | None: ...

    def get_estimates(self, ticker: str) -> dict[str, Any] | None: ...

    def get_macro_series(self, series_id: str) -> dict[str, Any] | None: ...

    def list_tickers(self) -> list[str]: ...
