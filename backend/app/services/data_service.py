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
from collections.abc import Callable, Collection
from datetime import date as _date
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

from ..config import settings
from ..providers.alpha_vantage_provider import AlphaVantageProvider
from ..providers.base import ProviderStatus
from ..providers.bls_provider import BLSProvider
from ..providers.census_provider import CensusProvider
from ..providers.eia_provider import EIAProvider
from ..providers.fmp_provider import FMPProvider
from ..providers.fred_provider import FREDProvider
from ..providers.gdelt_provider import GDELTProvider
from ..providers.ken_french_provider import KenFrenchProvider
from ..providers.polygon_provider import PolygonProvider
from ..providers.sec_edgar_provider import SECEdgarProvider
from ..providers.tiingo_provider import TiingoProvider
from .ticker_symbols import symbol_variants

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wave 1C — As-of-date context
# ---------------------------------------------------------------------------
_AS_OF_CONTEXT: contextvars.ContextVar[_date | None] = contextvars.ContextVar(
    "as_of_date", default=None,
)


# ---------------------------------------------------------------------------
# Which universe tiers the scheduled pullers are allowed to touch
# ---------------------------------------------------------------------------
# The automatic pollers make one provider call per ticker per pass — EDGAR
# every 30 minutes, transcripts daily — so their standing cost is linear in
# the size of the set they iterate. `auto_analysis` is the curated watch list,
# and it is the only tier that earns that recurring spend.
#
# `analyzed_on_demand` and `data_only` are excluded from the *automatic* pull,
# and it is worth being precise about what that does not mean: every company
# in the table, at any tier, stays fully available to manual, user-initiated
# research — search, on-demand memo generation, peer cohorts, chat ticker
# resolution. Those callers ask for `list_tickers()` with no `tiers` argument
# and see the whole table, which is why narrowing this must never be done at
# the query's default. What the exclusion prevents is a ticker joining the
# recurring poll forever merely because somebody once looked it up: each
# on-demand search inserts an `analyzed_on_demand` row, so an unfiltered
# poller's universe only ever grows.
AUTO_PULL_TIERS: tuple[str, ...] = ("auto_analysis",)


class as_of_context:
    """Context manager that pins the data layer to a historical date.

    All cache reads and writes inside the with-block use a per-date cache
    namespace so live and backtest data don't collide. Memory writes are
    skipped (a backtest run shouldn't pollute the long-term memory file).
    Provider methods that support date filtering should consult
    `current_as_of_date()` and clip their results accordingly.
    """

    def __init__(self, as_of: _date | None) -> None:
        self._as_of = as_of
        self._token: contextvars.Token | None = None

    def __enter__(self) -> as_of_context:
        self._token = _AS_OF_CONTEXT.set(self._as_of)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _AS_OF_CONTEXT.reset(self._token)


def current_as_of_date() -> _date | None:
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

def _coerce_iso_date(value: Any) -> _date | None:
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
    rows: list[dict[str, Any]] | None, primary_key: str,
    *, fallback_key: str | None = None,
) -> list[dict[str, Any]] | None:
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
    out: list[dict[str, Any]] = []
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
    statements: dict[str, Any] | None,
) -> dict[str, Any] | None:
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
    statements: dict[str, Any] | None,
) -> dict[str, Any] | None:
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
        # Sector-overlay providers (Phase: smart sector analyst).
        # Each works without an API key against the public endpoints.
        self.eia = EIAProvider()
        self.bls = BLSProvider()
        self.census = CensusProvider()
        # Academic factor returns provider (Fama-French + momentum).
        # No key required — pulls from the Tuck data library.
        self.ken_french = KenFrenchProvider()
        # GDELT — broad international news coverage, no key.
        self.gdelt = GDELTProvider()
        # Optional test override (wired by `tests/conftest.py`).
        self._test_provider: Any | None = None

    # ------------------------------------------------------------------
    # Provider selection
    # ------------------------------------------------------------------

    def register_test_provider(self, provider: Any | None) -> None:
        """Inject a fixture provider for tests. Pass None to clear.

        When set, the fixture sits at the **head** of every capability
        chain so it answers first; the live chain still runs as a
        fallback for tests that exercise both paths.
        """
        self._test_provider = provider

    def _live_chain(self, capability: str) -> list[Any]:
        chains: dict[str, list[Any]] = {
            "profile": [self.fmp, self.alpha],
            "prices": [self.fmp, self.tiingo, self.polygon, self.alpha],
            "quote": [self.fmp, self.tiingo, self.polygon],
            # Wave 9b — Alpha Vantage as a financials fallback. FMP's
            # Starter tier returns 403 on most fundamentals endpoints;
            # AV Premium covers the same vocabulary.
            "financials": [self.fmp, self.alpha],
            "ratios": [self.fmp],
            "key_metrics": [self.fmp],
            "earnings": [self.fmp, self.alpha],
            "transcripts": [self.alpha],
            "filings": [self.sec],
            # Same provider and same method as `filings`, different cost
            # shape: `get_filings_index` calls it with `fetch_text=False`,
            # so this capability is one submissions.json read instead of
            # up to ten document-body fetches. It is a separate chain
            # entry because it is a separate cache row with its own TTL.
            "filings_index": [self.sec],
            "news": [self.alpha, self.polygon, self.gdelt],
            "estimates": [self.fmp],
            "macro": [self.fred, self.eia, self.bls, self.census, self.ken_french],
            "energy": [self.eia],
            "inflation": [self.bls, self.fred],
            "labor": [self.bls, self.fred],
            "retail": [self.census, self.fred],
            "construction": [self.census, self.fred],
            "factor_returns": [self.ken_french],
        }
        chain = chains.get(capability, [])
        if settings.use_demo_data_only:
            # USE_DEMO_DATA=true with ENABLE_LIVE_DATA=false (every test run)
            # means "no provider calls": a developer .env carrying provider
            # keys must not turn the suite into paid traffic. The injected
            # test provider is the whole chain then.
            return [self._test_provider] if self._test_provider is not None else []
        if self._test_provider is not None:
            return [self._test_provider, *chain]
        return chain

    def _try_chain(self, capability: str, fn_name: str, *args, **kwargs) -> Any | None:
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

    def _try_chain_symbol(
        self, capability: str, fn_name: str, ticker: str, *args, **kwargs,
    ) -> Any | None:
        """`_try_chain`, retried across the spellings of a share-class ticker.

        There is no agreed spelling for a dual-class symbol: our universe
        file says `BRK.B`, the GICS map says `BRK-B`, EDGAR insists on
        `BRK-B`, and other feeds use `BRK/B`. Asking the chain for one
        spelling and treating a miss as "no such company" is what left
        Berkshire with no `companies` row at all — and with it, one of the
        ten `auto_update_memo` pins naming a ticker that did not exist.

        Costs nothing for an ordinary ticker: `symbol_variants` returns a
        single element when there is no separator to swap, so this is one
        pass over the chain, exactly as before. A separator-bearing symbol
        pays for the extra spellings only when the first one misses. The
        caller's spelling stays canonical for the cache key and the
        database — this changes what we *ask* for, never what we store.
        """
        variants = symbol_variants(ticker)
        for symbol in variants:
            result = self._try_chain(capability, fn_name, symbol, *args, **kwargs)
            if result:
                if symbol != variants[0]:
                    log.info(
                        "%s resolved %s under the provider spelling %s",
                        capability, variants[0], symbol,
                    )
                return result
        return None

    # ------------------------------------------------------------------
    # Provider status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, ProviderStatus]:
        return {
            p.name: p.status() for p in (
                self.fmp, self.alpha, self.fred, self.polygon, self.tiingo, self.sec,
                self.eia, self.bls, self.census, self.ken_french, self.gdelt,
            )
        }

    def mode(self) -> str:
        return "live"

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def list_tickers(self, *, tiers: Collection[str] | None = None) -> list[str]:
        """Return tickers from the `companies` table, ordered by ticker.

        With no `tiers` — the default, and what every existing caller
        uses — this is every ticker the platform has ever touched: the
        curated universe from `data/sp500.json` (S&P 500 + extensions,
        tagged `auto_analysis`), any ticker the user has researched on
        demand (`analyzed_on_demand`), and the `data_only` long tail.
        That unfiltered set is what manual research resolves against, so
        it must stay unfiltered: narrow it and an on-demand name stops
        being findable.

        Pass `tiers=AUTO_PULL_TIERS` for the curated tier alone — the
        scheduled pollers' universe. See that constant for why the other
        two tiers are kept out of the automatic pull.

        Ordered because the alternative is not "insertion order", it is
        "whatever the engine finds convenient" — rowid order on SQLite,
        genuinely arbitrary on Postgres. Any caller that slices or
        compares the result is non-deterministic without this, which is
        how `[:10]` in the news/social loops came to mean nothing in
        particular. Empty on cold start before the seeder runs.
        """
        from ..database import SessionLocal
        from ..models import Company
        with SessionLocal() as db:
            q = db.query(Company.ticker)
            if tiers is not None:
                q = q.filter(Company.universe_tier.in_(tuple(tiers)))
            return [t for (t,) in q.order_by(Company.ticker).all()]

    # ------------------------------------------------------------------
    # Read-through cache (Wave 9b Phase 2b)
    # ------------------------------------------------------------------
    # Each `get_X` method delegates to `_cached(capability, key, fn)`
    # which checks `provider_cache` first, falls through to the live
    # chain on miss / expiry, persists the response, and serves stale
    # rows when the provider also misses. Tests bypass the cache so
    # fixtures stay deterministic.

    def _cached(
        self, capability: str, key: str,
        fetcher: Callable[[], Any],
        *, force_refresh: bool = False,
        ttl_override: int | None = None,
    ) -> Any | None:
        # Skip cache when a test fixture is registered or an as-of
        # context is active — both want deterministic, point-in-time
        # responses, not yesterday's snapshot.
        if self._test_provider is not None or current_as_of_date() is not None:
            return fetcher()
        from . import provider_cache
        return provider_cache.cached_call(
            capability, key, fetcher,
            ttl_seconds=ttl_override,
            force_refresh=force_refresh,
        )

    def get_company_profile(
        self, ticker: str, *, force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        profile = self._cached(
            "profile", ticker.upper(),
            lambda: self._try_chain_symbol("profile", "get_company_profile", ticker),
            force_refresh=force_refresh,
        )
        # The provider spelling is a lookup detail. Memo composition and
        # persistence take their identity from this field, not the request.
        # Normalize after the cache read too, so old alias-bearing payloads
        # cannot keep filing BRK.B research under BRK-B for the profile TTL.
        return {**profile, "ticker": ticker.strip().upper()} if profile else None

    def get_price_history(
        self, ticker: str, days: int = 252, *, force_refresh: bool = False,
    ) -> list[dict[str, Any]] | None:
        # Fixture and historical-context behavior stays deterministic. Live
        # reads use durable per-date rows; expiry never removes old closes.
        if self._test_provider is None and current_as_of_date() is None and not settings.use_demo_data_only:
            from .price_history_service import fetch_and_store_prices, read_prices
            stored = read_prices(ticker, days=days)
            fresh = bool(stored and datetime.fromisoformat(stored[-1]["fetched_at"]) >= datetime.utcnow() - timedelta(hours=24)
                         and _date.fromisoformat(stored[-1]["date"]) >= _date.today() - timedelta(days=5))
            if len(stored) >= days and fresh and not force_refresh:
                return stored
            def refresh_prices():
                fetch_and_store_prices(ticker, days, service=self, verify_calendar=False)
                return read_prices(ticker, days=days) or None
            # Retain cache throttling for partial responses (e.g. a recent
            # IPO) and provider outages. Explicit history backfills bypass
            # these legacy response windows and validate their date range.
            rows = self._cached("prices", f"{ticker.upper()}:{days}", refresh_prices, force_refresh=force_refresh)
            return _clip_dated_rows(rows, "date")
        rows = self._cached(
            "prices", f"{ticker.upper()}:{days}",
            lambda: self._try_chain_symbol("prices", "get_price_history", ticker, days),
            force_refresh=force_refresh,
        )
        return _clip_dated_rows(rows, "date")

    def get_quote(
        self, ticker: str, *, force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Intraday last-trade quote with a 60s TTL.

        Bypassed during as-of backtests — historical runs read closes
        from `get_price_history` clipped to the as-of date, never live
        quotes. Live mode: provider chain returns a normalized dict
        with `price`, `previous_close`, `change_pct`, `timestamp`.
        """
        if current_as_of_date() is not None:
            return None
        return self._cached(
            "quote", ticker.upper(),
            lambda: self._try_chain_symbol("quote", "get_quote", ticker),
            force_refresh=force_refresh,
        )

    def get_financial_statements(
        self, ticker: str, *, force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        # Financials roll over only when a new 10-Q/K lands. Long TTL
        # is fine; force_refresh covers manual recomputes.
        statements = self._cached(
            "financials", ticker.upper(),
            lambda: self._try_chain_symbol("financials", "get_financial_statements", ticker),
            force_refresh=force_refresh,
            ttl_override=86400 * 7,
        )
        if not statements and self._test_provider is None and not settings.use_demo_data_only:
            from .fundamental_history_service import read_stored_financials
            # Normal valuation consumers expect annual statement rows. The
            # durable store also has quarters, but those must not be mixed
            # into an annual DCF or ratio series.
            statements = read_stored_financials(ticker, cadence="annual")
        return _clip_statements(statements)

    def get_ratios(
        self, ticker: str, *, force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        # Ratios are derived from latest statements; if a clip drops the
        # latest period, the ratio is no longer "as of" the historical
        # date. Trigger a recompute from the clipped statements when an
        # as-of context is active. No-op in live mode.
        if current_as_of_date() is not None:
            return _ratios_from_clipped_statements(
                self.get_financial_statements(ticker),
            )
        return self._cached(
            "ratios", ticker.upper(),
            lambda: self._try_chain_symbol("ratios", "get_ratios", ticker),
            force_refresh=force_refresh,
        )

    def get_key_metrics(
        self, ticker: str, *, force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        return self._cached(
            "key_metrics", ticker.upper(),
            lambda: self._try_chain_symbol("key_metrics", "get_key_metrics", ticker),
            force_refresh=force_refresh,
        )

    def get_earnings(
        self, ticker: str, *, force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        return self._cached(
            "earnings", ticker.upper(),
            lambda: self._try_chain_symbol("earnings", "get_earnings", ticker),
            force_refresh=force_refresh,
        )

    def get_earnings_transcripts(
        self, ticker: str, *, force_refresh: bool = False,
        prefer_cached: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Transcripts for `ticker`. Four AlphaVantage requests on a miss
        (the provider iterates four quarters).

        `prefer_cached=True` accepts a cached row at any age — see
        `get_filings` for why a universe-wide loop asks for that.
        """
        from . import provider_cache
        rows = self._cached(
            "transcripts", ticker.upper(),
            lambda: self._try_chain_symbol("transcripts", "get_earnings_transcripts", ticker),
            force_refresh=force_refresh,
            ttl_override=provider_cache.NEVER_EXPIRES if prefer_cached else None,
        )
        return _clip_dated_rows(rows, "date", fallback_key="period")

    def get_filings(
        self, ticker: str, *, force_refresh: bool = False,
        prefer_cached: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Filings WITH document bodies. The expensive read: up to ten
        multi-megabyte fetches per ticker, paced against SEC's ~10 req/s.

        `prefer_cached=True` serves a cached row at whatever age it is and
        only reaches the provider when there is no row at all. It exists for
        the one caller that sweeps the whole curated universe on a schedule
        (`history_service.backfill_ticker` under `history_backfill`), where
        the seven-day TTL rolling over would otherwise mean ~1,700 document
        downloads in a single nightly job.

        That is safe rather than a reintroduction of the never-expire bug,
        because nothing but a new accession changes a filed document, and
        `edgar_poller` responds to a new accession by calling
        `invalidate_filings_text` for that one ticker. Freshness is driven by
        the event, which is capped; the TTL stays the backstop for every
        other reader, whose cost is bounded by user activity rather than by
        the size of the universe.
        """
        from . import provider_cache
        cik = self._lookup_cik(ticker)
        if not cik:
            return None
        rows = self._cached(
            "filings", ticker.upper(),
            lambda: self._try_chain_symbol("filings", "get_filings", ticker, cik=cik),
            force_refresh=force_refresh,
            ttl_override=provider_cache.NEVER_EXPIRES if prefer_cached else None,
        )
        return _clip_dated_rows(rows, "filing_date", fallback_key="period_end")

    def get_filings_index(
        self, ticker: str, *, force_refresh: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Filing metadata WITHOUT document bodies — the cheap read.

        `get_filings` fetches the full text of every form it returns (up
        to ten per ticker, a few MB each, paced against SEC's ~10 req/s
        limit). Change detection does not need any of that: an accession
        number that was not in the last response is a new filing. This
        method asks the same provider with `fetch_text=False`, so a poll
        across the curated universe costs one JSON read per ticker rather
        than ~1,700 document downloads per pass.

        Cached under its own capability (`filings_index`, 15-minute TTL)
        so the short poll cadence cannot drag the expensive `filings`
        bodies along with it.
        """
        cik = self._lookup_cik(ticker)
        if not cik:
            return None
        rows = self._cached(
            "filings_index", ticker.upper(),
            lambda: self._try_chain_symbol(
                "filings_index", "get_filings", ticker, cik=cik, fetch_text=False,
            ),
            force_refresh=force_refresh,
        )
        return _clip_dated_rows(rows, "filing_date", fallback_key="period_end")

    def invalidate_filings_text(self, ticker: str) -> int:
        """Drop the cached filing *bodies* for one ticker. Returns rows deleted.

        Called by `edgar_poller` when the cheap index shows an accession
        it has not seen. Without it, better detection would just mean
        firing events about filings whose text is still the seven-day-old
        cached response — the poller would notice the new 10-Q and every
        downstream reader would keep reading the previous one.

        The key convention (`capability="filings"`, key = upper-case
        ticker) lives here, next to the code that writes those rows,
        rather than being restated in the poller.
        """
        from . import provider_cache
        return provider_cache.invalidate("filings", ticker.upper())

    def reads_from_cache(self, capability: str, key: str) -> bool:
        """True when a `prefer_cached` read of `(capability, key)` will be
        answered without consulting a provider.

        Lets a loop that fans out over the whole universe ask "does this
        ticker cost me a live provider call?" *before* it spends one, which
        is how `history_backfill` bounds its nightly cold reads.

        Reports True in the two modes where `_cached` bypasses the cache
        entirely — a registered test fixture, an active as-of context —
        because neither reaches a provider either. The question being asked
        is about cost, not about which row answered.
        """
        if self._test_provider is not None or current_as_of_date() is not None:
            return True
        from . import provider_cache
        return provider_cache.get(capability, key) is not None

    def _lookup_cik(self, ticker: str) -> str | None:
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
        # Silent under demo-only mode like every other provider read.
        cik = None if settings.use_demo_data_only else self.sec.lookup_cik(ticker_up)
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

    def get_news(
        self, ticker: str, *, force_refresh: bool = False,
    ) -> list[dict[str, Any]] | None:
        rows = self._cached(
            "news", ticker.upper(),
            lambda: self._try_chain_symbol("news", "get_news", ticker),
            force_refresh=force_refresh,
        )
        return _clip_dated_rows(rows, "published_at")

    def get_estimates(
        self, ticker: str, *, force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        return self._cached(
            "estimates", ticker.upper(),
            lambda: self._try_chain_symbol("estimates", "get_estimates", ticker),
            force_refresh=force_refresh,
        )

    def get_macro_series(
        self, series_id: str, *, force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        return self._cached(
            "macro", series_id,
            lambda: self._try_chain("macro", "get_macro_series", series_id),
            force_refresh=force_refresh,
        )

    def list_macro_series(self) -> list[dict[str, Any]]:
        """Catalog metadata across every macro-shaped provider.

        Returns the union of FRED + EIA + BLS + Census catalog rows,
        deduplicated by series_id. Used by `/api/data-catalog` and by
        any caller that wants to browse what's available before fetching.
        """
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for provider in (self.fred, self.eia, self.bls, self.census, self.ken_french):
            try:
                rows = provider.list_macro_series() or []
            except Exception:
                rows = []
            for row in rows:
                sid = row.get("series_id")
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                out.append(row)
        return out


@lru_cache(maxsize=1)
def get_data_service() -> DataService:
    return DataService()


def curated_poll_universe() -> tuple[list[str], int]:
    """Return `(tickers_to_poll, excluded)` for the scheduled pollers.

    `tickers_to_poll` is the `AUTO_PULL_TIERS` slice of the universe;
    `excluded` is how many companies the tier filter held back. Both
    pollers need the same pair, so it lives here rather than being
    copied into each — two copies of a constraint drift, and this one is
    the thing the user asked for.

    The count exists to be reported. A cap nobody can see is a cap that
    gets rediscovered as a bug: `/api/admin/cron-health` should say the
    poller skipped N companies, not quietly poll fewer than an operator
    expects. Clamped at zero because a row inserted between the two
    queries would otherwise show up as a negative skip count.
    """
    ds = get_data_service()
    selected = ds.list_tickers(tiers=AUTO_PULL_TIERS)
    total = len(ds.list_tickers())
    return selected, max(total - len(selected), 0)
