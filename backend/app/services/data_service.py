"""Provider-aware data service.

A single facade in front of the providers. It tries the configured live
providers first (when ENABLE_LIVE_DATA=true) and gracefully falls back to
DemoProvider for any method that returns None or raises. All callers go
through this service so they never need to know whether data came from a
live API or local fixtures.

Wave 1C: an `as_of_date` ContextVar lets `run_stock_memo` mark the entire
call tree as a backtest for a specific historical date. Provider methods
that respect the context filter their results to data observable on or
before that date; providers that don't yet support date filtering simply
ignore it (no-op, with the cache key still segregated so live and
backtest data don't collide).
"""
from __future__ import annotations

import contextvars
import logging
from datetime import date as _date
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

from ..config import settings
from ..providers.alpha_vantage_provider import AlphaVantageProvider
from ..providers.base import ProviderStatus
from ..providers.demo_provider import DemoProvider
from ..providers.fmp_provider import FMPProvider
from ..providers.fred_provider import FREDProvider
from ..providers.polygon_provider import PolygonProvider
from ..providers.sec_edgar_provider import SECEdgarProvider
from ..providers.tiingo_provider import TiingoProvider

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wave 1C — As-of-date context
# ---------------------------------------------------------------------------
_AS_OF_CONTEXT: contextvars.ContextVar[Optional[_date]] = contextvars.ContextVar(
    "as_of_date", default=None,
)


class as_of_context:
    """Context manager that pins the data layer to a historical date.

    All cache reads and writes inside the with-block use a per-date cache
    namespace so live and backtest data don't collide. Memory writes are
    skipped (a backtest run shouldn't pollute the long-term memory file).
    Provider methods that support date filtering should consult
    `current_as_of_date()` and clip their results accordingly.
    """

    def __init__(self, as_of: Optional[_date]) -> None:
        self._as_of = as_of
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "as_of_context":
        self._token = _AS_OF_CONTEXT.set(self._as_of)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _AS_OF_CONTEXT.reset(self._token)


def current_as_of_date() -> Optional[_date]:
    """Read the active as_of date, if any. Returns None for live mode."""
    return _AS_OF_CONTEXT.get()


class DataService:
    """Facade over all providers with demo fallback."""

    def __init__(self) -> None:
        self.demo = DemoProvider()
        self.fmp = FMPProvider()
        self.alpha = AlphaVantageProvider()
        self.fred = FREDProvider()
        self.polygon = PolygonProvider()
        self.tiingo = TiingoProvider()
        self.sec = SECEdgarProvider()

    # ------------------------------------------------------------------
    # Provider selection
    # ------------------------------------------------------------------

    def _live_chain(self, capability: str) -> List[Any]:
        if settings.use_demo_data_only:
            return []
        chains: Dict[str, List[Any]] = {
            "profile": [self.fmp, self.alpha],
            "prices": [self.fmp, self.tiingo, self.polygon],
            "financials": [self.fmp],
            "ratios": [self.fmp],
            "key_metrics": [self.fmp],
            "earnings": [self.fmp],
            "transcripts": [self.alpha],
            "filings": [self.sec],
            "news": [self.alpha, self.polygon],
            "estimates": [self.fmp],
            "macro": [self.fred],
        }
        return chains.get(capability, [])

    def _try_chain(self, capability: str, fn_name: str, *args, **kwargs) -> Optional[Any]:
        for provider in self._live_chain(capability):
            try:
                fn: Callable = getattr(provider, fn_name, None)
                if not fn:
                    continue
                result = fn(*args, **kwargs)
                if result:
                    return result
            except Exception as exc:  # pragma: no cover
                log.warning("Provider %s.%s failed: %s", provider.name, fn_name, exc)
        return None

    # ------------------------------------------------------------------
    # Provider status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, ProviderStatus]:
        return {
            p.name: p.status() for p in (
                self.demo, self.fmp, self.alpha, self.fred, self.polygon, self.tiingo, self.sec
            )
        }

    def mode(self) -> str:
        return "demo" if settings.use_demo_data_only or not settings.enable_live_data else "live"

    # ------------------------------------------------------------------
    # Endpoints (with fallback)
    # ------------------------------------------------------------------

    def list_tickers(self) -> List[str]:
        return self.demo.list_tickers()

    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]:
        live = self._try_chain("profile", "get_company_profile", ticker)
        return live or self.demo.get_company_profile(ticker)

    def get_price_history(self, ticker: str, days: int = 252) -> Optional[List[Dict[str, Any]]]:
        live = self._try_chain("prices", "get_price_history", ticker, days)
        return live or self.demo.get_price_history(ticker, days)

    def get_financial_statements(self, ticker: str) -> Optional[Dict[str, Any]]:
        live = self._try_chain("financials", "get_financial_statements", ticker)
        return live or self.demo.get_financial_statements(ticker)

    def get_ratios(self, ticker: str) -> Optional[Dict[str, Any]]:
        live = self._try_chain("ratios", "get_ratios", ticker)
        return live or self.demo.get_ratios(ticker)

    def get_key_metrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        live = self._try_chain("key_metrics", "get_key_metrics", ticker)
        return live or self.demo.get_key_metrics(ticker)

    def get_earnings(self, ticker: str) -> Optional[Dict[str, Any]]:
        live = self._try_chain("earnings", "get_earnings", ticker)
        return live or self.demo.get_earnings(ticker)

    def get_earnings_transcripts(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        live = self._try_chain("transcripts", "get_earnings_transcripts", ticker)
        return live or self.demo.get_earnings_transcripts(ticker)

    def get_filings(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        cik = (self.demo.get_company_profile(ticker) or {}).get("cik")
        live = self._try_chain("filings", "get_filings", ticker, cik=cik) if cik else None
        return live or self.demo.get_filings(ticker)

    def get_news(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        live = self._try_chain("news", "get_news", ticker)
        return live or self.demo.get_news(ticker)

    def get_estimates(self, ticker: str) -> Optional[Dict[str, Any]]:
        live = self._try_chain("estimates", "get_estimates", ticker)
        return live or self.demo.get_estimates(ticker)

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        live = self._try_chain("macro", "get_macro_series", series_id)
        return live or self.demo.get_macro_series(series_id)

    def list_macro_series(self) -> List[Dict[str, Any]]:
        return self.demo.list_macro_series()


@lru_cache(maxsize=1)
def get_data_service() -> DataService:
    return DataService()
