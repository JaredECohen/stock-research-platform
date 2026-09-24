"""W5b — the calendar-aware live-quote policy (`services/quote_service.py`).

The clock is frozen in both `quote_service._now` and `provider_cache._now`
(the same idiom as `test_provider_cache.FakeClock`), so row ages are exact
and nothing asserts wall-clock time. The session DemoProvider is taken off
for the cache tests (`cache_ds`), because a registered test provider makes
`get_quotes` skip the cache on purpose. Providers are the `_fetch_batch` /
`_fetch_one` seams, stubbed; nothing here opens a socket. Tickers are unique
per test so rows from an earlier run cannot leak into assertions.
"""
from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import event, select

from app.agents.safe_runner import DegradationLog
from app.cache import CacheCostLog
from app.config import settings
from app.database import SessionLocal, engine
from app.finance import market_calendar as cal
from app.models import Company, ProviderCache
from app.providers.fmp_provider import FMPProvider
from app.services import price_history_service, provider_cache, quote_service
from app.services.data_service import as_of_context, get_data_service
from app.services.market_data_service import get_close_series, get_current_price

ET = ZoneInfo("America/New_York")


def et(*args: int) -> datetime:
    """Naive UTC for a wall-clock time in New York."""
    return datetime(*args, tzinfo=ET).astimezone(UTC).replace(tzinfo=None)


def z(value: datetime) -> str:
    return value.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


# Tuesday 2026-09-22, 10:45 ET: the market is open.
OPEN_NOW = et(2026, 9, 22, 10, 45)


@pytest.fixture
def clock(monkeypatch) -> Clock:
    fake = Clock(OPEN_NOW)
    monkeypatch.setattr(quote_service, "_now", fake)
    monkeypatch.setattr(provider_cache, "_now", fake)
    return fake


@pytest.fixture
def cache_ds():
    """`DataService` with the session DemoProvider off, so reads go through
    the cache (`test_provider_cache_never_expires` uses the same pattern)."""
    ds = get_data_service()
    previous = ds._test_provider
    ds.register_test_provider(None)
    try:
        yield ds
    finally:
        ds.register_test_provider(previous)


class Providers:
    """Stubbed provider seams with call records."""

    def __init__(self) -> None:
        self.batch_calls: list[list[str]] = []
        self.one_calls: list[str] = []
        self.batch_status: int | None = 200
        self.prices: dict[str, float] = {}
        self.batch_enabled = True

    def payload(self, ticker: str) -> dict | None:
        price = self.prices.get(ticker)
        if price is None:
            return None
        return {"ticker": ticker, "price": price, "previous_close": price - 1, "change": 1.0,
                "change_pct": 1.0, "timestamp": int(OPEN_NOW.replace(tzinfo=UTC).timestamp()) - 60}

    def fetch_batch(self, symbols, _ds):
        self.batch_calls.append(list(symbols))
        if self.batch_status != 200:
            return self.batch_status, {}
        return 200, {s: p for s in symbols if (p := self.payload(s)) is not None}

    def fetch_one(self, ticker, _ds):
        self.one_calls.append(ticker)
        p = self.payload(ticker)
        return (p, "tiingo") if p else (None, None)


@pytest.fixture
def providers(monkeypatch) -> Providers:
    stub = Providers()
    monkeypatch.setattr(quote_service, "_fetch_batch", stub.fetch_batch)
    monkeypatch.setattr(quote_service, "_fetch_one", stub.fetch_one)
    monkeypatch.setattr(quote_service, "_batch_enabled", lambda _ds: stub.batch_enabled)
    # The entitlement memo is one global row; each test starts without it.
    provider_cache.invalidate(quote_service.ENTITLEMENT_CAPABILITY, quote_service.BATCH_ENTITLEMENT_KEY)
    yield stub
    provider_cache.invalidate(quote_service.ENTITLEMENT_CAPABILITY, quote_service.BATCH_ENTITLEMENT_KEY)


def tickers(n: int, prefix: str = "QS") -> list[str]:
    """`n` unique known tickers (companies rows)."""
    stamp = time.perf_counter_ns() % 10**7
    out = [f"{prefix}{stamp}{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(n)]
    with SessionLocal() as db:
        for t in out:
            db.merge(Company(ticker=t, company_name=f"{t} Inc", sector="Test", industry="Test",
                             universe_tier="data_only"))
        db.commit()
    return out


def seed_row(clock: Clock, ticker: str, fetched_at: datetime, price: float = 50.0, **extra) -> None:
    saved = clock.now
    clock.now = fetched_at
    provider_cache.put("quote", ticker, {"ticker": ticker, "price": price, "provider": "fmp", **extra})
    clock.now = saved


def ledger(key: str) -> list[str]:
    with SessionLocal() as db:
        rows = db.execute(
            select(CacheCostLog.kind, CacheCostLog.note).where(CacheCostLog.subject == provider_cache.STALE_LOG_SUBJECT)
        ).all()
    return [kind for kind, note in rows if json.loads(note or "{}").get("key") == key]


# ---------------------------------------------------------------------------
# Expiry policy
# ---------------------------------------------------------------------------

def test_expiry_in_session_is_15_minutes():
    fetched = et(2026, 9, 22, 10, 0)
    assert quote_service.quote_expires_at(fetched) == et(2026, 9, 22, 10, 15)
    assert quote_service._is_fresh(fetched, et(2026, 9, 22, 10, 14, 59), None)
    assert not quote_service._is_fresh(fetched, et(2026, 9, 22, 10, 15), None)  # boundary is stale


def test_expiry_just_before_close_still_refreshes():
    assert quote_service.quote_expires_at(et(2026, 9, 22, 15, 55)) == et(2026, 9, 22, 16, 10)
    assert quote_service.quote_expires_at(et(2026, 9, 22, 16, 10)) == et(2026, 9, 22, 16, 25)


def test_post_close_print_lasts_until_next_open():
    # Friday 2026-09-25.
    assert quote_service.quote_expires_at(et(2026, 9, 25, 16, 20)) == et(2026, 9, 28, 9, 30)
    assert quote_service.quote_expires_at(et(2026, 9, 25, 16, 19, 59)) == et(2026, 9, 25, 16, 34, 59)


def test_half_day_settle_uses_13_00():
    assert quote_service.quote_expires_at(et(2026, 11, 27, 13, 25)) == et(2026, 11, 30, 9, 30)
    assert quote_service.quote_expires_at(et(2026, 11, 27, 13, 10)) == et(2026, 11, 27, 13, 25)


def test_pre_open_expires_at_open():
    assert quote_service.quote_expires_at(et(2026, 9, 22, 8, 0)) == et(2026, 9, 22, 9, 30)


def test_weekend_and_holiday_expire_at_next_open():
    assert quote_service.quote_expires_at(et(2026, 9, 26, 12, 0)) == et(2026, 9, 28, 9, 30)
    assert quote_service.quote_expires_at(et(2026, 9, 7, 12, 0)) == et(2026, 9, 8, 9, 30)


def test_dst_boundary_post_close():
    # Fri 2026-10-30 16:30 EDT -> Mon 2026-11-02 09:30 EST = 14:30 UTC.
    assert quote_service.quote_expires_at(et(2026, 10, 30, 16, 30)) == datetime(2026, 11, 2, 14, 30)


# ---------------------------------------------------------------------------
# Batch, SQL shape, bounds
# ---------------------------------------------------------------------------

def test_batch_cold_list_uses_one_provider_call(clock, cache_ds, providers):
    names = tickers(3)
    providers.prices = {t: 10.0 + i for i, t in enumerate(names)}
    got = quote_service.get_quotes(names)
    assert providers.batch_calls == [names] and providers.one_calls == []
    assert [q["source"] for q in got.values()] == ["live"] * 3
    assert got[names[0]]["provider"] == "fmp" and got[names[0]]["price"] == 10.0
    assert set(provider_cache.read_rows("quote", names)) == set(names)
    clock.now = et(2026, 9, 22, 10, 50)
    again = quote_service.get_quotes(names)
    assert len(providers.batch_calls) == 1 and providers.one_calls == []
    assert again[names[2]]["fetched_at"] == z(OPEN_NOW)


def test_list_read_is_constant_sql(clock, cache_ds, providers):
    names = tickers(20)
    provider_cache.put_many("quote", {t: {"ticker": t, "price": 5.0} for t in names})
    selects: list[str] = []

    def count(_conn, _cursor, statement, *_a):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    event.listen(engine, "before_cursor_execute", count)
    try:
        got = quote_service.get_quotes(names)
    finally:
        event.remove(engine, "before_cursor_execute", count)
    assert all(q["source"] == "live" for q in got.values())
    assert len(selects) == 2, selects  # known-ticker gate + one cache read, independent of N
    assert providers.batch_calls == [] and providers.one_calls == []


def test_batch_402_falls_back_and_is_remembered(clock, cache_ds, providers):
    names = tickers(2)
    providers.prices = {t: 20.0 for t in names}
    providers.batch_status = 402
    got = quote_service.get_quotes(names)
    assert providers.batch_calls == [names]
    assert providers.one_calls == names
    assert got[names[0]]["source"] == "live" and got[names[0]]["provider"] == "tiingo"
    remembered = provider_cache.get("entitlement", quote_service.BATCH_ENTITLEMENT_KEY)
    assert remembered == {"status": 402}
    # Next request (another process would read the same row): no batch attempt.
    more = tickers(2)
    providers.prices.update({t: 21.0 for t in more})
    quote_service.get_quotes(more)
    assert len(providers.batch_calls) == 1
    assert providers.one_calls == names + more


def test_per_symbol_fallback_is_bounded(clock, cache_ds, providers):
    names = tickers(15)
    providers.batch_enabled = False
    providers.prices = {t: 30.0 for t in names}
    for t in names[10:]:
        seed_row(clock, t, et(2026, 9, 22, 10, 20), price=29.0)  # 25 min old: expired, still servable
    got = quote_service.get_quotes(names)
    assert providers.one_calls == names[:10]
    assert [got[t]["source"] for t in names[:10]] == ["live"] * 10
    assert [(got[t]["source"], got[t]["reason"]) for t in names[10:]] == [("stale", "refresh_deferred")] * 5


def test_per_request_budget_defers_remaining(clock, cache_ds, providers, monkeypatch):
    names = tickers(4)
    providers.batch_enabled = False
    providers.prices = {t: 40.0 for t in names}
    ticks = iter([0.0, 0.0, 12.0, 25.0, 25.0, 25.0, 25.0])
    monkeypatch.setattr(quote_service, "_monotonic", lambda: next(ticks))
    got = quote_service.get_quotes(names)
    # Started at 0; calls at t=0 and t=12 fit, t=25 is past the 20 s budget.
    assert providers.one_calls == names[:2]
    assert [got[t]["reason"] for t in names[2:]] == ["no_stored_close"] * 2
    assert all(got[t]["source"] == "unavailable" for t in names[2:])


def test_unknown_ticker_never_reaches_provider(clock, cache_ds, providers):
    providers.prices = {"ZZZZQ": 1.0}
    got = quote_service.get_quotes(["zzzzq"])
    assert got["ZZZZQ"]["source"] == "unavailable" and got["ZZZZQ"]["reason"] == "unknown_ticker"
    assert providers.batch_calls == [] and providers.one_calls == []


def test_multiclass_symbols_skip_batch_and_402_writes_no_entitlement(clock, cache_ds, providers):
    stamp = time.perf_counter_ns() % 10**5
    dotted = f"QB{stamp}.B"
    with SessionLocal() as db:
        db.merge(Company(ticker=dotted, company_name="Share class", sector="Test", industry="Test",
                         universe_tier="data_only"))
        db.commit()
    plain = tickers(1)[0]
    providers.prices = {plain: 12.0}   # FMP refuses the dotted spelling (its 402s were all BRK.B)
    got = quote_service.get_quotes([plain, dotted])
    assert providers.batch_calls == [[plain]]          # the share class never rides in the batch
    assert providers.one_calls == [dotted]             # it goes symbol by symbol, every spelling
    assert got[dotted]["source"] == "unavailable"
    assert provider_cache.get("entitlement", quote_service.BATCH_ENTITLEMENT_KEY) is None
    # A share class alone makes no batch call at all.
    quote_service.get_quotes([dotted])
    assert providers.batch_calls == [[plain]]
    assert provider_cache.get("entitlement", quote_service.BATCH_ENTITLEMENT_KEY) is None


def test_demo_mode_never_calls_fmp_batch(clock, cache_ds, monkeypatch):
    """A developer `.env` with a live FMP key must not turn a demo-mode run
    into FMP traffic: under USE_DEMO_DATA / no live data the chain is empty."""
    assert settings.use_demo_data_only
    monkeypatch.setattr(cache_ds.fmp, "api_key", "fake-key-not-real")
    calls: list = []

    def refuse(self, path, **params):
        calls.append(path)
        raise AssertionError(f"FMP {path} called in demo mode")

    monkeypatch.setattr(FMPProvider, "_get_status", refuse)
    name = tickers(1)[0]
    got = quote_service.get_quotes([name])
    assert calls == []
    assert got[name]["source"] == "unavailable"
    assert quote_service._batch_enabled(cache_ds) is False


# ---------------------------------------------------------------------------
# Labelled misses
# ---------------------------------------------------------------------------

def test_in_session_provider_miss_serves_stale_with_ledger(clock, cache_ds, providers):
    name = tickers(1)[0]
    ts = int(et(2026, 9, 22, 10, 24).replace(tzinfo=UTC).timestamp())
    seed_row(clock, name, et(2026, 9, 22, 10, 25), price=77.0, timestamp=ts)  # 20 min old
    got = quote_service.get_quotes([name])[name]
    assert got["source"] == "stale" and got["price"] == 77.0 and got["reason"] == "provider_miss"
    assert got["as_of"] == z(et(2026, 9, 22, 10, 24)) == got["price_time"]
    assert got["fetched_at"] == z(et(2026, 9, 22, 10, 25))
    assert ledger(name) == [provider_cache.STALE_SERVED_KIND]


def test_after_close_session_row_is_served_stale_not_refused(clock, cache_ds, providers):
    name = tickers(1)[0]
    seed_row(clock, name, et(2026, 9, 22, 15, 10), price=81.0)
    clock.now = et(2026, 9, 22, 18, 0)   # 2h50m later: past the 3600 s cap, but the market is closed
    got = quote_service.get_quotes([name])[name]
    assert got["source"] == "stale" and got["price"] == 81.0


def _store_close(ticker: str, day: str, close: float) -> None:
    price_history_service.persist_prices(ticker, [{"date": day, "close": close, "volume": 1000}], source="fmp")


def test_old_row_falls_to_durable_close_without_provider_prices(clock, cache_ds, providers, monkeypatch):
    name = tickers(1)[0]
    seed_row(clock, name, et(2026, 9, 22, 8, 40), price=60.0)   # 2h05m old, in session
    _store_close(name, "2026-09-21", 58.5)

    def refuse(*_a, **_k):
        raise AssertionError("the EOD fallback must never call a provider")

    monkeypatch.setattr(price_history_service, "fetch_and_store_prices", refuse)
    got = quote_service.get_quotes([name])[name]
    assert got["source"] == "eod_close" and got["provider"] == "daily_prices"
    assert got["price"] == 58.5 and got["as_of"] == z(et(2026, 9, 21, 16, 0))
    assert got["delayed"] is False
    assert ledger(name) == [provider_cache.STALE_REFUSED_KIND]


def test_no_row_no_close_is_unavailable(clock, cache_ds, providers):
    name = tickers(1)[0]
    got = quote_service.get_quotes([name])[name]
    assert got["source"] == "unavailable" and got["reason"] == "no_stored_close" and got["price"] is None


def test_old_payload_rows_still_parse(clock, cache_ds, providers):
    name = tickers(1)[0]
    clock.now = OPEN_NOW
    fetched = et(2026, 9, 22, 10, 40)
    saved = clock.now
    clock.now = fetched
    provider_cache.put("quote", name, {"price": 1.0})   # a pre-W5b row: no provider, no time
    clock.now = saved
    got = quote_service.get_quotes([name])[name]
    assert got["source"] == "live" and got["price"] == 1.0
    assert got["price_time"] is None and got["as_of"] == z(fetched) and got["provider"] is None


# ---------------------------------------------------------------------------
# Memo floor and DataService contract
# ---------------------------------------------------------------------------

def test_memo_freshness_floor_refetches(clock, cache_ds, providers):
    name = tickers(1)[0]
    seed_row(clock, name, et(2026, 9, 22, 10, 43), price=90.0)   # 2 min old, fresh by calendar
    providers.prices = {name: 91.0}
    assert quote_service.get_quotes([name])[name]["price"] == 90.0
    assert providers.batch_calls == []
    with DegradationLog().activate():
        assert quote_service.effective_floor() == quote_service.MEMO_QUOTE_MAX_AGE_SECONDS
        assert quote_service.get_quotes([name])[name]["price"] == 91.0
    assert providers.batch_calls == [[name]]


def test_floor_resets_after_context():
    assert quote_service.effective_floor() is None
    with pytest.raises(RuntimeError):
        with DegradationLog().activate():
            assert quote_service.effective_floor() == 60
            raise RuntimeError("memo failed")
    assert quote_service.effective_floor() is None
    assert quote_service.effective_floor(0) == 0


def test_as_of_context_returns_none(clock, cache_ds, providers):
    name = tickers(1)[0]
    providers.prices = {name: 5.0}
    with as_of_context(date(2025, 1, 2)):
        assert cache_ds.get_quote(name) is None
        assert quote_service.get_quotes([name]) == {}
    assert providers.batch_calls == [] and providers.one_calls == []


def test_get_quote_eod_is_not_a_quote(clock, cache_ds, providers):
    name = tickers(1)[0]
    _store_close(name, "2026-09-21", 44.0)
    assert quote_service.get_quotes([name])[name]["source"] == "eod_close"
    assert cache_ds.get_quote(name) is None
    providers.prices = {name: 45.0}
    live = cache_ds.get_quote(name)
    assert live is not None and live["price"] == 45.0 and live["source"] == "live"


def test_get_quote_malformed_ticker_is_none(cache_ds):
    assert cache_ds.get_quote("A;B") is None


def test_calendar_failure_degrades_to_close(monkeypatch):
    """No tzdata: `get_current_price` answers the last close, never raises
    into `default_dcf_assumptions`."""
    def unavailable(*_a, **_k):
        raise cal.CalendarUnavailable("no tzdata")

    monkeypatch.setattr(cal, "market_state", unavailable)
    assert get_data_service().get_quote("NVDA") is None
    closes = get_close_series("NVDA", days=5)
    assert closes and get_current_price("NVDA") == closes[-1]


def test_provider_timestamp_normalization():
    now = datetime(2026, 9, 22, 15, 0)
    instant = datetime(2026, 9, 22, 14, 44, 30, tzinfo=UTC)
    seconds = int(instant.timestamp())
    for value in (seconds, seconds * 1000, seconds * 10**9, "2026-09-22T14:44:30Z",
                  "2026-09-22T10:44:30-04:00", str(seconds)):
        assert quote_service._provider_timestamp(value, now=now) == instant, value
    assert quote_service._provider_timestamp(seconds + 3600, now=now) is None   # an hour in the future
    assert quote_service._provider_timestamp(0, now=now) is None
    assert quote_service._provider_timestamp("garbage", now=now) is None
    assert quote_service._provider_timestamp(True, now=now) is None
    assert quote_service._provider_timestamp(None, now=now) is None


def test_normalize_tickers():
    assert quote_service.normalize_tickers([" aapl", "MSFT", "aapl", "", "brk.b"]) == ["AAPL", "MSFT", "BRK.B"]
    for bad in (["A;B"], [], [""], [f"T{i}" for i in range(51)], ["-AB"]):
        with pytest.raises(ValueError):
            quote_service.normalize_tickers(bad)


def test_quote_rows_are_only_quote_rows(clock, cache_ds, providers):
    """Quotes never touch `daily_prices`: an intraday price is not a close."""
    from app.models import DailyPrice

    name = tickers(1)[0]
    providers.prices = {name: 12.5}
    quote_service.get_quotes([name])
    with SessionLocal() as db:
        assert db.execute(select(DailyPrice).where(DailyPrice.ticker == name)).first() is None
        assert db.execute(select(ProviderCache).where(ProviderCache.key == name)).scalar_one().capability == "quote"


# ---------------------------------------------------------------------------
# Review fixes: stored close beats an old intraday row; internal callers
# never take an old stale quote as the current price
# ---------------------------------------------------------------------------

@pytest.fixture
def durable_closes(cache_ds, monkeypatch):
    """`get_price_history` as production's live mode answers it: the durable
    `daily_prices` rows (demo mode would read the absent fixture chain).
    DB-only, so `get_current_price`'s close fallback is the real one."""
    monkeypatch.setattr(
        cache_ds, "get_price_history",
        lambda ticker, days=252, **_k: price_history_service.read_prices(ticker, days=days) or None,
    )
    return cache_ds


def test_after_close_stored_close_supersedes_intraday_row(clock, durable_closes, providers):
    """A 10:00 quote must not be shown at 20:00 over that day's stored close,
    and must never become `price_at_memo` / the DCF `current_price`."""
    name = tickers(1)[0]
    seed_row(clock, name, et(2026, 9, 22, 10, 0), price=50.0)
    _store_close(name, "2026-09-22", 55.0)
    clock.now = et(2026, 9, 22, 20, 0)
    got = quote_service.get_quotes([name])[name]
    assert (got["source"], got["price"]) == ("eod_close", 55.0)
    assert got["as_of"] == z(et(2026, 9, 22, 16, 0))
    assert durable_closes.get_quote(name) is None
    assert get_current_price(name) == 55.0
    # Next morning, pre-open: still the close, not Tuesday's 10:00 print.
    clock.now = et(2026, 9, 23, 8, 0)
    assert quote_service.get_quotes([name])[name]["source"] == "eod_close"
    assert get_current_price(name) == 55.0


def test_after_close_row_with_only_an_older_close_stays_stale(clock, cache_ds, providers):
    """The session branch still serves the last session's row when the stored
    close is older than it (today's close not ingested yet), labelled."""
    name = tickers(1)[0]
    seed_row(clock, name, et(2026, 9, 22, 15, 10), price=81.0)
    _store_close(name, "2026-09-21", 79.0)
    clock.now = et(2026, 9, 22, 18, 0)
    got = quote_service.get_quotes([name])[name]
    assert (got["source"], got["price"]) == ("stale", 81.0)


def test_internal_callers_never_take_an_old_stale_quote(clock, durable_closes, providers):
    """Owner default: no quote older than ~15 min is ever served as current.
    The chip shows the old row labelled `stale`; `get_quote` (and so
    `get_current_price`) falls back to the stored close instead."""
    name = tickers(1)[0]
    _store_close(name, "2026-09-21", 48.0)
    # In session: a 20-minute-old row and a provider miss.
    seed_row(clock, name, et(2026, 9, 22, 10, 25), price=50.0)
    assert quote_service.get_quotes([name])[name]["source"] == "stale"
    assert durable_closes.get_quote(name) is None
    assert get_current_price(name) == 48.0
    # After the close, same-session row 10 hours old, no newer close stored.
    clock.now = et(2026, 9, 22, 20, 25)
    assert quote_service.get_quotes([name])[name]["source"] == "stale"
    assert get_current_price(name) == 48.0


def test_memo_floor_stale_within_15_minutes_is_still_current(clock, cache_ds, providers):
    """Under the 60 s memo floor a 5-minute-old row is expired early; on a
    miss it is `stale` but within the 15-minute policy, so it is used."""
    name = tickers(1)[0]
    seed_row(clock, name, et(2026, 9, 22, 10, 40), price=70.0)
    with DegradationLog().activate():
        quote = cache_ds.get_quote(name)
        assert quote is not None and quote["source"] == "stale" and quote["price"] == 70.0
        assert get_current_price(name) == 70.0


def test_usable_as_current_boundaries():
    live = quote_service._quote("X", {"price": 1.0}, OPEN_NOW, source="live", now=OPEN_NOW)
    assert quote_service.usable_as_current(live, OPEN_NOW)
    stale = quote_service._quote("X", {"price": 1.0}, et(2026, 9, 22, 10, 30), source="stale", now=OPEN_NOW)
    assert quote_service.usable_as_current(stale, OPEN_NOW)                      # exactly 15 min
    assert not quote_service.usable_as_current(stale, et(2026, 9, 22, 10, 45, 1))
    eod = dict(stale, source="eod_close")
    assert not quote_service.usable_as_current(eod, OPEN_NOW)  # type: ignore[arg-type]


def test_internal_get_quote_never_reads_the_durable_close(clock, cache_ds, providers, monkeypatch):
    """`resolve_eod=False`: `get_current_price` has its own close fallback, so
    `get_quote` must not spend a `daily_prices` read on every miss."""
    name = tickers(1)[0]
    reads: list[str] = []
    real = quote_service._stored_close
    monkeypatch.setattr(quote_service, "_stored_close", lambda t: reads.append(t) or real(t))
    assert cache_ds.get_quote(name) is None
    seed_row(clock, name, et(2026, 9, 22, 10, 0), price=50.0)
    clock.now = et(2026, 9, 22, 20, 0)
    assert cache_ds.get_quote(name) is None
    assert reads == []
    quote_service.get_quotes([name])          # the labelled path does read it
    assert reads == [name]


# ---------------------------------------------------------------------------
# Review fixes: the batch refusal memo, the budget, the one cache write
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", [401, 429, 500, None])
def test_non_entitlement_batch_failures_are_not_remembered(clock, cache_ds, providers, status):
    """Only 402/403 mean "not in this plan". A bad key, a rate limit, a 5xx or
    a timeout must not lock both processes out of the batch for a day."""
    names = tickers(2)
    providers.prices = {t: 20.0 for t in names}
    providers.batch_status = status
    got = quote_service.get_quotes(names)
    assert providers.one_calls == names and all(got[t]["source"] == "live" for t in names)
    assert provider_cache.get("entitlement", quote_service.BATCH_ENTITLEMENT_KEY) is None
    more = tickers(1)
    providers.prices[more[0]] = 21.0
    quote_service.get_quotes(more)
    assert providers.batch_calls == [names, more]   # the next request tries the batch again


def test_batch_refusal_memo_expires_after_a_day(clock, cache_ds, providers):
    names = tickers(1)
    providers.prices = {names[0]: 20.0}
    providers.batch_status = 402
    quote_service.get_quotes(names)
    assert providers.batch_calls == [names]
    providers.batch_status = 200
    clock.now = et(2026, 9, 23, 10, 44)
    later = tickers(1)
    providers.prices[later[0]] = 22.0
    quote_service.get_quotes(later)
    assert providers.batch_calls == [names]         # 23 h 59 min: still held
    clock.now = et(2026, 9, 23, 10, 46)
    last = tickers(1)
    providers.prices[last[0]] = 23.0
    quote_service.get_quotes(last)
    assert providers.batch_calls == [names, last]   # expired: the batch is tried again


def test_multiclass_alone_in_a_refusing_window_writes_no_entitlement(clock, cache_ds, providers):
    """Even with FMP answering 402, a share-class request never makes a batch
    call, so it can never record the plan-level refusal."""
    stamp = time.perf_counter_ns() % 10**5
    dotted = f"QC{stamp}.B"
    with SessionLocal() as db:
        db.merge(Company(ticker=dotted, company_name="Share class", sector="Test", industry="Test",
                         universe_tier="data_only"))
        db.commit()
    providers.batch_status = 402
    quote_service.get_quotes([dotted])
    assert providers.batch_calls == [] and providers.one_calls == [dotted]
    assert provider_cache.get("entitlement", quote_service.BATCH_ENTITLEMENT_KEY) is None


def test_budget_counts_the_batch_call(clock, cache_ds, providers, monkeypatch):
    """A batch that hung for 25 s leaves no budget for per-symbol calls."""
    names = tickers(3)
    providers.prices = {t: 40.0 for t in names}
    elapsed = [0.0]
    monkeypatch.setattr(quote_service, "_monotonic", lambda: elapsed[0])

    def hung_batch(symbols, _ds):
        providers.batch_calls.append(list(symbols))
        elapsed[0] += 25.0                          # the batch timed out
        return None, {}

    monkeypatch.setattr(quote_service, "_fetch_batch", hung_batch)
    got = quote_service.get_quotes(names)
    assert providers.batch_calls == [names] and providers.one_calls == []
    assert [(got[t]["source"], got[t]["reason"]) for t in names] == [("unavailable", "no_stored_close")] * 3
    assert provider_cache.get("entitlement", quote_service.BATCH_ENTITLEMENT_KEY) is None


def test_cold_list_is_one_cache_commit_whatever_n(clock, cache_ds, providers):
    """20 cold tickers: one batch call and ONE cache transaction, not a
    session and commit per key (the module doc's "one cache write")."""
    names = tickers(20)
    providers.prices = {t: 9.0 for t in names}
    commits: list[int] = []

    def on_commit(_conn):
        commits.append(1)

    event.listen(engine, "commit", on_commit)
    try:
        got = quote_service.get_quotes(names)
    finally:
        event.remove(engine, "commit", on_commit)
    assert all(got[t]["source"] == "live" for t in names)
    assert providers.batch_calls == [names]
    assert len(commits) == 1, commits
    assert set(provider_cache.read_rows("quote", names)) == set(names)


# ---------------------------------------------------------------------------
# Chat company-lites: one quote read for all of them, each price labelled
# ---------------------------------------------------------------------------

def test_chat_lites_prefetch_quotes_in_one_call_and_label_them(clock, cache_ds, providers, monkeypatch):
    from app.agents import llm, orchestrator, pm_context
    from app.services import memo_store

    live_t, stale_t, eod_t, seed_t = tickers(4, prefix="QL")
    with SessionLocal() as db:
        db.get(Company, seed_t).last_price = 12.0
        db.commit()
    seed_row(clock, live_t, et(2026, 9, 22, 10, 40), price=100.0)   # fresh
    seed_row(clock, stale_t, et(2026, 9, 22, 10, 25), price=200.0)  # 20 min old, provider misses
    _store_close(eod_t, "2026-09-21", 300.0)

    calls: list[list[str]] = []
    real = quote_service.get_quotes

    def spy(tks, **kwargs):
        calls.append(list(tks))
        return real(tks, **kwargs)

    monkeypatch.setattr(quote_service, "get_quotes", spy)
    monkeypatch.setattr(settings, "use_agents_sdk", False)
    monkeypatch.setattr(orchestrator, "_extract_tickers", lambda _text: [live_t, stale_t, eod_t, seed_t])
    monkeypatch.setattr(memo_store, "latest_memo", lambda _t: None)
    monkeypatch.setattr(pm_context, "build_pm_context", lambda **_k: "")
    prompts: list[str] = []
    monkeypatch.setattr(llm, "chat_text", lambda prompt, **_k: prompts.append(prompt) or "answer")

    assert orchestrator.Orchestrator()._answer_with_memo_context("compare them", []) is not None
    assert calls == [[live_t, stale_t, eod_t, seed_t]]
    lites = {row["ticker"]: row for row in json.loads(
        prompts[0].split("Company snapshots (use when no memo is available):\n", 1)[1].split("\n\nConversation", 1)[0]
    )}
    assert (lites[live_t]["last_price"], lites[live_t]["last_price_source"]) == (100.0, "live")
    assert lites[live_t]["last_price_as_of"] == z(et(2026, 9, 22, 10, 40))
    assert (lites[stale_t]["last_price"], lites[stale_t]["last_price_source"]) == (200.0, "stale")
    assert lites[stale_t]["last_price_as_of"] == z(et(2026, 9, 22, 10, 25))
    assert (lites[eod_t]["last_price"], lites[eod_t]["last_price_source"]) == (300.0, "eod_close")
    assert lites[eod_t]["last_price_as_of"] == z(et(2026, 9, 21, 16, 0))
    assert (lites[seed_t]["last_price"], lites[seed_t]["last_price_source"]) == (12.0, "profile_seed")
    assert lites[seed_t]["last_price_as_of"] is None
