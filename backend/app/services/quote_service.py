"""Live quotes, cached ~15 minutes while the market is open (W5b).

Owner decision (2026-09-24): "market prices should be pulled live and
cached for ~15 minutes so it stays current." This module is that policy for
intraday quotes only. EOD closes in `daily_prices` stay the source for every
analytic (outcomes, scorecard, industry stats); no quote is ever written
there, and none overwrites a close.

Policy (`quote_expires_at`), applied to the ET date of the fetch:

1. fetched in `[open, close + CLOSE_SETTLE)` of a trading day: fresh for 15
   minutes. The settle window is the delayed feed still printing the close.
2. fetched before the open: fresh until the open (the pre-market quote is
   the prior close; nothing newer exists until 09:30).
3. fetched at or after `close + CLOSE_SETTLE`: the post-close print, fresh
   until the next open. One fetch carries the close overnight.
4. fetched on a weekend / holiday: fresh until the next open.

When a row has expired and the provider misses, the answer is labelled
rather than silently old, in this order:

* `stale`: the cached row, when it is no older than
  `provider_cache.max_stale_seconds("quote")` (3600 s; the env override
  still applies) or, with the market closed, when it came from the most
  recent session (after the close nothing newer exists, so a 15:55 quote is
  still the right answer at 18:00). Every stale serve and refusal goes to
  the shared `cache_cost_logs` ledger, like `cached_call`'s. With the
  market closed, an intraday row loses to a stored close of its own
  session or later: the close is newer information than any intraday
  print, so a 10:00 quote is never shown at 20:00 over that day's close.
* `eod_close`: the last close from the durable store, read DB-only. This
  fallback never calls a provider.
* `unavailable`: neither exists (or the ticker is unknown).

A batch of N tickers costs one known-ticker query, one cache read and one
cache write, whatever N is. Unknown tickers never reach a provider, so an
anonymous caller cannot spend the FMP quota on arbitrary strings. Provider
work per call is bounded: one FMP `/batch-quote` call for the plain symbols,
then at most `MAX_PROVIDER_FETCHES_PER_CALL` per-symbol calls inside a
`PER_REQUEST_BUDGET_SECONDS` wall-clock budget counted from before the
batch call; the rest are served stale or EOD with
`reason="refresh_deferred"`. The budget is soft: it stops new per-symbol
calls from starting, and a call already in flight runs to its providers'
own timeouts.

Internal price consumers (`DataService.get_quote` -> `get_current_price`,
which feeds `price_at_memo` and the DCF `current_price`) take a quote only
when `usable_as_current` says so: live, or stale by no more than the
15-minute policy. No quote older than ~15 minutes is ever used unlabelled
as the current price; the labelled chip is the only reader of older rows.

Inside a memo run (`safe_runner.in_memo_run()`, contract C6) any cached
quote older than 60 s is refetched even when the calendar calls it fresh,
which is exactly the old 60 s TTL: `price_at_memo` and the memo's DCF
`current_price` keep their meaning.
"""
from __future__ import annotations

import logging
import re
import time as _time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, TypedDict

from ..finance import market_calendar as cal
from . import provider_cache
from .ticker_symbols import is_multi_class

log = logging.getLogger(__name__)

# Owner: ~15 minutes during the regular session.
QUOTE_TTL_SECONDS = 900
# Close + this = the post-close print: the 15-minute feed delay plus the
# closing auction. Until then a quote is still refreshed every 15 minutes.
CLOSE_SETTLE_SECONDS = 1200
# The old quote TTL, kept as a floor inside memo runs (see module doc).
MEMO_QUOTE_MAX_AGE_SECONDS = 60
MAX_TICKERS_PER_REQUEST = 50
# Per-symbol provider calls one `get_quotes` may make (the batch call is
# one more). A 50-symbol cold list must not become a 50-call burst against
# a 300/min plan.
MAX_PROVIDER_FETCHES_PER_CALL = 10
# Wall-clock budget for provider work in one call, the batch included: a
# slow provider must not hold a page request for 10 x 10 s timeouts. Soft:
# checked before each per-symbol call starts.
PER_REQUEST_BUDGET_SECONDS = 20.0

CAPABILITY = "quote"
ENTITLEMENT_CAPABILITY = "entitlement"
BATCH_ENTITLEMENT_KEY = "fmp:/batch-quote"
# Statuses that mean "this plan does not include the endpoint". 401 is a
# bad key, which per-symbol calls would share, so it is not remembered.
BATCH_REFUSED_STATUSES = frozenset({402, 403})

# A provider time further ahead than this, or before 2000, is not shown.
_FUTURE_TOLERANCE = timedelta(minutes=5)
_EARLIEST = datetime(2000, 1, 1, tzinfo=UTC)
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-/]{0,14}$")

Source = Literal["live", "stale", "eod_close", "unavailable"]


class Quote(TypedDict):
    ticker: str
    price: float | None
    previous_close: float | None
    change: float | None
    change_pct: float | None
    day_low: float | None
    day_high: float | None
    volume: float | None
    price_time: str | None   # the provider's own trade/quote time, ISO-8601 UTC
    fetched_at: str | None   # when WE fetched it (the cache row), ISO-8601 UTC
    as_of: str | None        # price_time, else fetched_at; eod_close: the session close
    source: Source
    provider: str | None     # "fmp" | "tiingo" | "polygon" | "daily_prices"
    expires_at: str | None   # live rows only
    delayed: bool            # provider quotes: "may be delayed up to 15 min"
    reason: str | None       # provider_miss | refresh_deferred | unknown_ticker | no_stored_close


def _now() -> datetime:
    """Naive UTC; monkeypatched by tests (same idiom as `provider_cache._now`)."""
    return datetime.utcnow()


def _monotonic() -> float:
    return _time.monotonic()


def _aware(naive_utc: datetime) -> datetime:
    return naive_utc.replace(tzinfo=UTC)


def _naive(aware_utc: datetime) -> datetime:
    return aware_utc.astimezone(UTC).replace(tzinfo=None)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = _aware(value)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # NaN is not a price


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def quote_expires_at(fetched_at_utc: datetime) -> datetime:
    """When a quote fetched at `fetched_at_utc` stops being fresh (naive UTC in and out).

    The boundary instant is stale: a row fetched at 10:00:00 is stale at
    exactly 10:15:00.
    """
    day = cal.et_date(fetched_at_utc)
    bounds = cal.session_bounds(day)
    fetched = _aware(fetched_at_utc)
    if bounds is not None:
        open_at, close_at = bounds
        if fetched < open_at:
            return _naive(open_at)
        if fetched < close_at + timedelta(seconds=CLOSE_SETTLE_SECONDS):
            return fetched_at_utc + timedelta(seconds=QUOTE_TTL_SECONDS)
    return _naive(cal.next_open(fetched_at_utc))


def effective_floor(max_age_seconds: int | None = None) -> int | None:
    """The maximum row age this read accepts on top of the calendar policy.

    An explicit value wins (0 = always refetch). Otherwise 60 s inside a
    memo run, and no floor anywhere else.
    """
    if max_age_seconds is not None:
        return max_age_seconds
    from ..agents.safe_runner import in_memo_run
    return MEMO_QUOTE_MAX_AGE_SECONDS if in_memo_run() else None


def _usable(payload: Any) -> bool:
    return isinstance(payload, dict) and _float(payload.get("price")) is not None


def _is_fresh(fetched_at: datetime, now: datetime, floor: int | None) -> bool:
    if floor is not None:
        if floor <= 0 or (now - fetched_at).total_seconds() > floor:
            return False
    return now < quote_expires_at(fetched_at)


def _servable_stale(fetched_at: datetime, now: datetime, state: cal.MarketState) -> bool:
    age = (now - fetched_at).total_seconds()
    if age <= provider_cache.max_stale_seconds(CAPABILITY):
        return True
    # After the close nothing newer than the last session's prints exists,
    # so that session's row stays the right answer until the next open
    # (unless its close is already stored: `_superseding_close`).
    return not state.is_open and _aware(fetched_at) >= state.session_open_utc


def _parse_iso(value: str) -> datetime:
    """Naive UTC from one of this module's ISO-8601 `...Z` strings."""
    return _naive(datetime.fromisoformat(value.replace("Z", "+00:00")))


def usable_as_current(quote: Quote, now: datetime | None = None) -> bool:
    """May an internal caller use `quote["price"]` as the current price, unlabelled?

    `live` rows, yes. A `stale` row only while it is no older than the
    15-minute policy, which happens only under a tighter floor (the 60 s
    memo floor or a forced refresh) that expired it early. Anything older,
    and every stored close, is left to the caller's own close fallback:
    `get_current_price` feeds `price_at_memo` and the DCF `current_price`,
    which carry no source label, so an hours-old intraday quote must never
    reach them (owner default: no quote older than ~15 min is ever served
    as current). The labelled `/api/quotes` chip still shows older rows.
    """
    if quote["source"] == "live":
        return True
    if quote["source"] != "stale" or not quote["fetched_at"]:
        return False
    age = ((now or _now()) - _parse_iso(quote["fetched_at"])).total_seconds()
    return age <= QUOTE_TTL_SECONDS


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_tickers(tickers: Sequence[str]) -> list[str]:
    """Strip, upper-case, drop blanks, dedupe (order kept), validate, cap.

    Raises `ValueError` on a malformed symbol, an empty list, or more than
    `MAX_TICKERS_PER_REQUEST`; the route turns that into a 422.
    """
    out: list[str] = []
    for raw in tickers:
        ticker = str(raw or "").strip().upper()
        if not ticker:
            continue
        if not _TICKER_RE.fullmatch(ticker):
            raise ValueError(f"malformed ticker {ticker[:20]!r}")
        if ticker not in out:
            out.append(ticker)
    if not out:
        raise ValueError("no tickers given")
    if len(out) > MAX_TICKERS_PER_REQUEST:
        raise ValueError(f"at most {MAX_TICKERS_PER_REQUEST} tickers per request, got {len(out)}")
    return out


def _provider_timestamp(value: Any, *, now: datetime) -> datetime | None:
    """A provider's quote time as aware UTC, or None when it cannot be justified.

    Numbers are epoch seconds, milliseconds, microseconds or nanoseconds by
    magnitude (FMP sends seconds, Polygon nanoseconds); strings are ISO-8601
    (Tiingo), naive ones read as UTC. A time more than 5 minutes in the
    future or before 2000 is dropped, and the label falls back to our fetch
    time: the chip never shows a time we cannot justify.
    """
    if value is None or isinstance(value, bool):
        return None
    parsed: datetime | None = None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return _provider_timestamp(int(text), now=now)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            log.info("ignoring unparseable provider quote time %r", text[:40])
            return None
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        seconds = float(value)
        if seconds >= 1e17:
            seconds /= 1e9
        elif seconds >= 1e14:
            seconds /= 1e6
        elif seconds >= 1e11:
            seconds /= 1e3
        try:
            parsed = datetime.fromtimestamp(seconds, tz=UTC)
        except (OverflowError, OSError, ValueError):
            log.info("ignoring out-of-range provider quote time %r", value)
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    if parsed > _aware(now) + _FUTURE_TOLERANCE or parsed < _EARLIEST:
        log.info("ignoring implausible provider quote time %s (now %s)", _iso(parsed), _iso(now))
        return None
    return parsed


def _quote(
    ticker: str, payload: dict[str, Any], fetched_at: datetime, *,
    source: Source, now: datetime, reason: str | None = None,
) -> Quote:
    price_time = _provider_timestamp(payload.get("timestamp"), now=now)
    as_of = price_time or _aware(fetched_at)
    return Quote(
        ticker=ticker,
        price=_float(payload.get("price")),
        previous_close=_float(payload.get("previous_close")),
        change=_float(payload.get("change")),
        change_pct=_float(payload.get("change_pct")),
        day_low=_float(payload.get("day_low")),
        day_high=_float(payload.get("day_high")),
        volume=_float(payload.get("volume")),
        price_time=_iso(price_time),
        fetched_at=_iso(fetched_at),
        as_of=_iso(as_of),
        source=source,
        provider=(str(payload["provider"]) if payload.get("provider") else None),
        expires_at=_iso(quote_expires_at(fetched_at)) if source == "live" else None,
        delayed=True,
        reason=reason,
    )


def _unavailable(ticker: str, reason: str) -> Quote:
    return Quote(
        ticker=ticker, price=None, previous_close=None, change=None, change_pct=None,
        day_low=None, day_high=None, volume=None, price_time=None, fetched_at=None,
        as_of=None, source="unavailable", provider=None, expires_at=None,
        delayed=False, reason=reason,
    )


def _stored_close(ticker: str) -> dict[str, Any] | None:
    """The newest usable close in `daily_prices`. DB-only: never a provider."""
    from .price_history_service import read_prices

    for row in reversed(list(read_prices(ticker, days=10) or [])):
        close = _float(row.get("close"))
        if close is None:
            close = _float(row.get("adjusted_close"))
        if close is not None and close > 0:
            return {**row, "close": close}
    return None


def _eod_quote(ticker: str, row: dict[str, Any], reason: str | None) -> Quote:
    day = datetime.fromisoformat(str(row["date"])[:10]).date()
    bounds = cal.session_bounds(day)
    if bounds is not None:
        close_at = bounds[1]
    else:  # a stored date the calendar does not call a session: label it 16:00 ET
        close_at = datetime.combine(day, cal.REGULAR_CLOSE, tzinfo=cal._et()).astimezone(UTC)
    quote = _unavailable(ticker, reason or "provider_miss")
    quote["price"] = row["close"]
    quote["as_of"] = _iso(close_at)
    quote["source"] = "eod_close"
    quote["provider"] = "daily_prices"
    quote["reason"] = reason
    return quote


# ---------------------------------------------------------------------------
# Provider seams (tests replace these; nothing below them is mocked)
# ---------------------------------------------------------------------------

def _batch_enabled(ds: Any) -> bool:
    """FMP batch only when FMP is really in the live quote chain.

    Under `USE_DEMO_DATA=true` / `ENABLE_LIVE_DATA=false` the chain is empty
    (or the test provider alone), so a developer `.env` with a live FMP key
    still makes no FMP call; with a test provider registered the fixture is
    the provider.
    """
    return (
        getattr(ds, "_test_provider", None) is None
        and ds.fmp in ds._live_chain(CAPABILITY)
        and bool(getattr(ds.fmp, "api_key", None))
    )


def _fetch_batch(symbols: list[str], ds: Any) -> tuple[int | None, dict[str, dict[str, Any]]]:
    try:
        return ds.fmp.get_quotes(symbols)
    except Exception as exc:  # pragma: no cover — provider already returns, never raises
        log.warning("FMP batch quote failed: %s", type(exc).__name__)
        return None, {}


def _fetch_one(ticker: str, ds: Any) -> tuple[dict[str, Any] | None, str | None]:
    return ds._try_chain_symbol_with_provider(CAPABILITY, "get_quote", ticker)


def _batch_refused() -> bool:
    return provider_cache.get(
        ENTITLEMENT_CAPABILITY, BATCH_ENTITLEMENT_KEY,
        ttl_seconds=provider_cache.ttl_seconds_for(ENTITLEMENT_CAPABILITY),
    ) is not None


def _fetch(to_fetch: list[str], ds: Any) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Provider payloads for `to_fetch` (each tagged with its provider), and
    the symbols deferred by the per-call bounds."""
    fetched: dict[str, dict[str, Any]] = {}
    # The budget clock starts before the batch: a batch call that hung for
    # its full timeout has already spent the request's patience.
    started = _monotonic()
    # Share-class symbols stay out of the batch: FMP spells BRK.B as BRK-B,
    # and its 402s in the 2026-09-21 window were all the dotted spelling. A
    # refusal of a batch holding one could be the symbol, not the plan, and
    # would lock every caller out of the batch for a day. They go through
    # the per-symbol chain, which tries every spelling.
    plain = [t for t in to_fetch if not is_multi_class(t)]
    if plain and _batch_enabled(ds) and not _batch_refused():
        status, got = _fetch_batch(plain, ds)
        if status in BATCH_REFUSED_STATUSES:
            provider_cache.put(ENTITLEMENT_CAPABILITY, BATCH_ENTITLEMENT_KEY, {"status": status})
            log.warning(
                "FMP /batch-quote refused with %s; per-symbol quotes for %ds",
                status, provider_cache.ttl_seconds_for(ENTITLEMENT_CAPABILITY),
            )
        for ticker in plain:
            item = got.get(ticker)
            if _usable(item):
                fetched[ticker] = {**item, "provider": "fmp"}  # type: ignore[dict-item]

    deferred: set[str] = set()
    calls = 0
    for ticker in to_fetch:
        if ticker in fetched:
            continue
        if calls >= MAX_PROVIDER_FETCHES_PER_CALL or _monotonic() - started >= PER_REQUEST_BUDGET_SECONDS:
            deferred.add(ticker)
            continue
        calls += 1
        payload, provider = _fetch_one(ticker, ds)
        if _usable(payload):
            fetched[ticker] = {**payload, "provider": provider}  # type: ignore[dict-item]
    if deferred:
        log.info("quote refresh deferred for %d symbol(s) past the per-call bound", len(deferred))
    return fetched, deferred


def _known_tickers(keys: list[str]) -> set[str]:
    from sqlalchemy import select

    from ..database import SessionLocal
    from ..models import Company

    with SessionLocal() as db:
        return set(db.execute(select(Company.ticker).where(Company.ticker.in_(keys))).scalars())


def _superseding_close(
    ticker: str, fetched_at: datetime, state: cal.MarketState,
) -> dict[str, Any] | None:
    """The stored close that is newer information than an intraday row, if any.

    Only with the market closed (no close of today can exist while it is
    open), and only for a row fetched before its own session's close had
    settled: a post-close print already IS the close. The row loses to a
    stored close dated on or after its ET session date. DB-only.
    """
    if state.is_open:
        return None
    day = cal.et_date(fetched_at)
    bounds = cal.session_bounds(day)
    if bounds is None:  # fetched on a weekend/holiday: it carries the last close already
        return None
    if _aware(fetched_at) >= bounds[1] + timedelta(seconds=CLOSE_SETTLE_SECONDS):
        return None
    close = _stored_close(ticker)
    if close is None or str(close["date"])[:10] < day.isoformat():
        return None
    return close


def _resolve_miss(
    ticker: str, row: tuple[Any, datetime] | None, *, now: datetime,
    state: cal.MarketState, reason: str, resolve_eod: bool,
) -> Quote:
    if row is not None and _usable(row[0]):
        payload, fetched_at = row
        age = int((now - fetched_at).total_seconds())
        if _servable_stale(fetched_at, now, state):
            # Internal callers (resolve_eod=False) never take a row this old
            # (`usable_as_current`) and have their own close fallback, so
            # the durable read is spent only on the labelled path.
            close = _superseding_close(ticker, fetched_at, state) if resolve_eod else None
            if close is not None:
                log.info(
                    "quote provider miss; stored close %s supersedes the intraday row ticker=%s",
                    str(close["date"])[:10], ticker,
                )
                return _eod_quote(ticker, close, reason)
            log.warning(
                "quote provider miss, serving stale row ticker=%s age_seconds=%d reason=%s",
                ticker, age, reason,
            )
            provider_cache.record_stale(provider_cache.STALE_SERVED_KIND, CAPABILITY, ticker, age)
            return _quote(ticker, payload, fetched_at, source="stale", now=now, reason=reason)
        provider_cache.record_stale(provider_cache.STALE_REFUSED_KIND, CAPABILITY, ticker, age)
    if not resolve_eod:
        return _unavailable(ticker, reason)
    close = _stored_close(ticker)
    if close is None:
        return _unavailable(ticker, "no_stored_close")
    return _eod_quote(ticker, close, reason)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_quotes(
    tickers: Sequence[str], *,
    max_age_seconds: int | None = None,
    resolve_eod: bool = True,
    service: Any | None = None,
) -> dict[str, Quote]:
    """`{TICKER: Quote}` for every requested ticker, in request order.

    `max_age_seconds` is a floor on top of the calendar policy (0 = always
    refetch); inside a memo run it defaults to 60 s. `resolve_eod=False`
    skips the durable-close fallback, for internal callers that have their
    own (`get_current_price`). Returns `{}` under an as-of backtest, which
    never reads live quotes. Raises `ValueError` on malformed input and
    `CalendarUnavailable` when the exchange time zone cannot load, before
    any provider is called.
    """
    keys = normalize_tickers(tickers)
    from .data_service import current_as_of_date, get_data_service

    if current_as_of_date() is not None:
        return {}
    ds = service or get_data_service()
    now = _now()
    state = cal.market_state(now)
    floor = effective_floor(max_age_seconds)

    out: dict[str, Quote] = {}
    known = _known_tickers(keys)
    candidates = [k for k in keys if k in known]
    for key in keys:
        if key not in known:
            out[key] = _unavailable(key, "unknown_ticker")

    # A registered test provider wants deterministic, uncached answers, as
    # in `DataService._cached`.
    use_cache = getattr(ds, "_test_provider", None) is None
    rows = provider_cache.read_rows(CAPABILITY, candidates) if (use_cache and candidates) else {}
    to_fetch: list[str] = []
    for key in candidates:
        row = rows.get(key)
        if row is not None and _usable(row[0]) and _is_fresh(row[1], now, floor):
            out[key] = _quote(key, row[0], row[1], source="live", now=now)
        else:
            to_fetch.append(key)

    if to_fetch:
        fetched, deferred = _fetch(to_fetch, ds)
        if use_cache and fetched:
            provider_cache.put_many(CAPABILITY, fetched)
        for key in to_fetch:
            if key in fetched:
                out[key] = _quote(key, fetched[key], now, source="live", now=now)
            else:
                out[key] = _resolve_miss(
                    key, rows.get(key), now=now, state=state,
                    reason="refresh_deferred" if key in deferred else "provider_miss",
                    resolve_eod=resolve_eod,
                )
    return {key: out[key] for key in keys}


def market_state_out(now: datetime | None = None) -> dict[str, Any]:
    """The `MarketStateOut` body for `now` (default: the service clock)."""
    state = cal.market_state(now or _now())
    return {
        "is_open": state.is_open,
        "reason": state.reason,
        "session_date": state.session_date,
        "session_open": state.session_open_utc,
        "session_close": state.session_close_utc,
        "early_close": state.early_close,
        "next_open": state.next_open_utc,
    }
