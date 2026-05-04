"""Provider-aware data service.

A single facade in front of the live providers (FMP, Alpha Vantage,
FRED, SEC EDGAR, Polygon, Tiingo). All callers go through this service
so they never need to know which provider answered.

Wave 9b — runtime is live-only. `DemoProvider` has been moved to
`tests/fixtures/` and is wired in only by `conftest.py` for unit tests.
Production never serves synthetic data: when no provider can satisfy a
call, methods return `None` / `[]` and callers handle the empty state
explicitly.

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

from ..providers.alpha_vantage_provider import AlphaVantageProvider
from ..providers.base import ProviderStatus
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


# ---------------------------------------------------------------------------
# Wave 8B — provider-agnostic as_of clipping
# ---------------------------------------------------------------------------
# When `current_as_of_date()` is set, every list-shaped historical payload
# returned by a provider is filtered to drop rows whose date field exceeds
# the cutoff. Provider interfaces stay unchanged — the clip is applied at
# the data_service facade. Live mode is a pure no-op (the if-guard short-
# circuits before any list iteration).

def _coerce_iso_date(value: Any) -> Optional[_date]:
    """Best-effort parse of a date-ish value into a `date`. Returns None on
    unparseable input. Accepts `date` / `datetime` / ISO string / `2024Q4`
    style period labels (treated as quarter end).
    """
    from datetime import datetime as _dt
    if value is None:
        return None
    if isinstance(value, _date) and not isinstance(value, _dt):
        return value
    if isinstance(value, _dt):
        return value.date()
    s = str(value).strip()
    if not s:
        return None
    # Plain ISO date.
    try:
        return _date.fromisoformat(s[:10])
    except (TypeError, ValueError):
        pass
    # Period label `2024Q4` → quarter-end date for ordering.
    import re as _re
    m = _re.match(r"^(\d{4})Q([1-4])$", s.upper())
    if m:
        year = int(m.group(1))
        q_end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[int(m.group(2))]
        return _date(year, q_end[0], q_end[1])
    # Annual `FY2024` or `2024` → year-end.
    m = _re.match(r"^(?:FY)?(\d{4})$", s.upper())
    if m:
        return _date(int(m.group(1)), 12, 31)
    return None


def _clip_dated_rows(
    rows: Optional[List[Dict[str, Any]]], primary_key: str,
    *, fallback_key: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Drop rows from `rows` whose `primary_key` (or `fallback_key`) date
    exceeds `current_as_of_date()`. Returns `rows` unchanged when no
    as_of is active or `rows` is None.

    Rows whose dates can't be parsed at all pass through — better to err
    toward "show it" than silently drop content the agent might need.
    Once we have richer date-handling at the provider layer, this can
    tighten to "drop unparseable", but that's a follow-up.
    """
    as_of = current_as_of_date()
    if as_of is None or not rows:
        return rows
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            out.append(r)
            continue
        d = _coerce_iso_date(r.get(primary_key))
        if d is None and fallback_key:
            d = _coerce_iso_date(r.get(fallback_key))
        if d is None or d <= as_of:
            out.append(r)
    return out


def _clip_statements(
    statements: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Clip the income/balance/cash period rows inside a statements dict.

    When `current_as_of_date()` is set, drop any row whose `period_end` /
    `period` is past the cutoff.
    """
    as_of = current_as_of_date()
    if as_of is None or not isinstance(statements, dict):
        return statements
    out = dict(statements)
    for key in ("income", "balance", "cash"):
        rows = out.get(key) or []
        out[key] = _clip_dated_rows(
            rows, "period_end", fallback_key="period",
        ) or []
    return out


def _ratios_from_clipped_statements(
    statements: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Compute ratios from clipped statements so the historical view's
    ratios reflect data observable at `current_as_of_date()` rather than
    today's snapshot."""
    if not isinstance(statements, dict):
        return None
    income = (statements.get("income") or [])
    balance = (statements.get("balance") or [])
    cash = (statements.get("cash") or [])
    if not income:
        return None
    latest = income[-1]
    bal = balance[-1] if balance else {}
    cf = cash[-1] if cash else {}
    from ..finance import ratios as R
    return {
        "PE": None,  # market-cap-dependent; backtest market cap not wired here
        "EV_Revenue": None,
        "EV_EBITDA": None,
        "PFCF": None,
        "FCF_yield": None,
        "ROIC": R.roic(latest, bal),
        "gross_margin": R.gross_margin(latest),
        "operating_margin": R.operating_margin(latest),
        "ebitda_margin": R.ebitda_margin(latest, cf),
        "fcf_margin": R.fcf_margin(cf, latest),
        "net_margin": R.net_margin(latest),
    }


class DataService:
    """Facade over the live provider chain.

    Wave 9b: no demo fallback at runtime. When all configured providers
    miss for a capability, the method returns `None` / `[]` and callers
    decide how to handle the empty state.

    Tests inject a `DemoProvider` via `register_test_provider` to drive
    deterministic responses without hitting any network.
    """

    def __init__(self) -> None:
        self.fmp = FMPProvider()
        self.alpha = AlphaVantageProvider()
        self.fred = FREDProvider()
        self.polygon = PolygonProvider()
        self.tiingo = TiingoProvider()
        self.sec = SECEdgarProvider()
        # Optional test override (wired by `tests/conftest.py`).
        self._test_provider: Optional[Any] = None

    # ------------------------------------------------------------------
    # Provider selection
    # ------------------------------------------------------------------

    def register_test_provider(self, provider: Optional[Any]) -> None:
        """Inject a fixture provider for tests. Pass None to clear.

        When set, the fixture sits at the **head** of every capability
        chain so it answers first; the live chain still runs as a
        fallback for tests that exercise both paths.
        """
        self._test_provider = provider

    def _live_chain(self, capability: str) -> List[Any]:
        chains: Dict[str, List[Any]] = {
            "profile": [self.fmp, self.alpha],
            "prices": [self.fmp, self.tiingo, self.polygon],
            # Wave 9b — Alpha Vantage as a financials fallback. FMP's
            # Starter tier returns 403 on most fundamentals endpoints;
            # AV Premium covers the same vocabulary.
            "financials": [self.fmp, self.alpha],
            "ratios": [self.fmp],
            "key_metrics": [self.fmp],
            "earnings": [self.fmp, self.alpha],
            "transcripts": [self.alpha],
            "filings": [self.sec],
            "news": [self.alpha, self.polygon],
            "estimates": [self.fmp],
            "macro": [self.fred],
        }
        chain = chains.get(capability, [])
        if self._test_provider is not None:
            return [self._test_provider, *chain]
        return chain

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
                self.fmp, self.alpha, self.fred, self.polygon, self.tiingo, self.sec
            )
        }

    def mode(self) -> str:
        return "live"

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def list_tickers(self) -> List[str]:
        """Return every ticker the platform has ever touched.

        Reads the `companies` table directly — covers both the curated
        S&P 100 (`auto_analysis`) and any ticker the user has researched
        on demand (`analyzed_on_demand`). Empty on cold start before the
        seeder runs.
        """
        from ..database import SessionLocal
        from ..models import Company
        with SessionLocal() as db:
            return [t for (t,) in db.query(Company.ticker).all()]

    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self._try_chain("profile", "get_company_profile", ticker)

    def get_price_history(self, ticker: str, days: int = 252) -> Optional[List[Dict[str, Any]]]:
        rows = self._try_chain("prices", "get_price_history", ticker, days)
        return _clip_dated_rows(rows, "date")

    def get_financial_statements(self, ticker: str) -> Optional[Dict[str, Any]]:
        statements = self._try_chain("financials", "get_financial_statements", ticker)
        return _clip_statements(statements)

    def get_ratios(self, ticker: str) -> Optional[Dict[str, Any]]:
        # Ratios are derived from latest statements; if a clip drops the
        # latest period, the ratio is no longer "as of" the historical
        # date. Trigger a recompute from the clipped statements when an
        # as-of context is active. No-op in live mode.
        if current_as_of_date() is not None:
            return _ratios_from_clipped_statements(
                self.get_financial_statements(ticker),
            )
        return self._try_chain("ratios", "get_ratios", ticker)

    def get_key_metrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self._try_chain("key_metrics", "get_key_metrics", ticker)

    def get_earnings(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self._try_chain("earnings", "get_earnings", ticker)

    def get_earnings_transcripts(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        rows = self._try_chain("transcripts", "get_earnings_transcripts", ticker)
        return _clip_dated_rows(rows, "date", fallback_key="period")

    def get_filings(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        cik = self._lookup_cik(ticker)
        if not cik:
            return None
        rows = self._try_chain("filings", "get_filings", ticker, cik=cik)
        return _clip_dated_rows(rows, "filing_date", fallback_key="period_end")

    def _lookup_cik(self, ticker: str) -> Optional[str]:
        """Resolve a ticker's CIK.

        Order:
          1. `companies.cik` column — populated by FMP profile or a
             previous SEC lookup.
          2. Live FMP profile fetch (FMP returns CIK with the profile).
          3. SEC's public ticker→CIK map — last-resort fallback used
             when the chain is on AV-only profiles (which don't include
             CIK). Backfilled into `companies` so subsequent calls skip
             the network.
        """
        from ..database import SessionLocal
        from ..models import Company
        ticker_up = ticker.upper()
        with SessionLocal() as db:
            row = db.get(Company, ticker_up)
            if row and row.cik:
                return row.cik
        profile = self.get_company_profile(ticker) or {}
        cik = profile.get("cik")
        if cik:
            return cik
        # SEC fallback. Persist back to `companies` so the next call is free.
        cik = self.sec.lookup_cik(ticker_up)
        if cik:
            try:
                with SessionLocal() as db:
                    row = db.get(Company, ticker_up)
                    if row is not None:
                        row.cik = cik
                        db.commit()
            except Exception:  # pragma: no cover
                log.debug("CIK persist failed for %s", ticker_up)
        return cik

    def get_news(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        rows = self._try_chain("news", "get_news", ticker)
        return _clip_dated_rows(rows, "published_at")

    def get_estimates(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self._try_chain("estimates", "get_estimates", ticker)

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        return self._try_chain("macro", "get_macro_series", series_id)

    def list_macro_series(self) -> List[Dict[str, Any]]:
        return self.fred.list_macro_series()


@lru_cache(maxsize=1)
def get_data_service() -> DataService:
    return DataService()
