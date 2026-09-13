"""Offline regressions for persisted placeholder and source-selection defects."""
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import DailyPrice
from app.services import price_history_service as prices


@pytest.fixture
def database(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(prices, "SessionLocal", factory)
    yield factory
    engine.dispose()


def tape(start, end, *, close=100, volume=1000):
    rows = []
    while start <= end:
        if start.weekday() < 5:
            rows.append({"date": start.isoformat(), "open": close, "high": close,
                         "low": close, "close": close, "volume": volume})
        start += timedelta(days=1)
    return rows


def test_dense_older_series_beats_newer_long_series_with_internal_gap(database):
    rows = tape(date(2024, 1, 1), date(2025, 12, 31))
    prices.persist_prices("SPY", rows, source="fmp")
    prices.persist_prices("ABC", rows[:300], source="fmp")
    prices.persist_prices("ABC", rows[:300] + rows[400:], source="alpha_vantage")
    result = prices.read_prices("ABC", days=252)
    assert {r["source"] for r in result} == {"fmp"}
    assert result.selection["internal_continuity_verified"] is True
    alpha = next(g for g in result.selection["candidate_sources"] if g["source"] == "alpha_vantage")
    assert alpha["internal_missing_benchmark_sessions"] == [r["date"] for r in rows[300:400]]


def test_legitimate_short_ipo_beats_long_broken_provider_history(database):
    rows = tape(date(2024, 1, 1), date(2025, 12, 31))
    prices.persist_prices("SPY", rows, source="fmp")
    prices.persist_prices("IPO", rows[-30:], source="tiingo")
    prices.persist_prices("IPO", rows[:300] + rows[-30:], source="fmp")
    result = prices.read_prices("IPO", days=252)
    assert len(result) == 30
    assert {r["source"] for r in result} == {"tiingo"}
    assert result.selection["internal_continuity_verified"] is True


def test_complete_fallback_beats_partial_primary_without_splicing(database):
    rows = tape(date(2024, 1, 1), date(2025, 12, 31))
    prices.persist_prices("SPY", rows, source="fmp")
    prices.persist_prices("ABC", rows[-5:], source="fmp")
    prices.persist_prices("ABC", rows, source="tiingo")
    result = prices.read_prices("ABC", days=252)
    assert len(result) == 252
    assert {r["source"] for r in result} == {"tiingo"}


def test_absent_calendar_keeps_explicitly_unverified_fallback(database):
    prices.persist_prices("ABC", tape(date(2025, 1, 1), date(2025, 1, 31)), source="fmp")
    prices.persist_prices("ABC", tape(date(2025, 1, 1), date(2025, 2, 28)), source="tiingo")
    result = prices.read_prices("ABC")
    assert {r["source"] for r in result} == {"tiingo"}
    assert result.selection["internal_continuity_verified"] is None
    assert not any(c["selection_calendar_available"] for c in result.selection["candidate_sources"])


def test_two_calendar_endpoints_do_not_prove_continuity(database):
    rows = tape(date(2025, 1, 1), date(2025, 12, 31))
    prices.persist_prices("ABC", rows, source="fmp")
    prices.persist_prices("SPY", [rows[0], rows[-1]], source="fmp")
    assert prices.read_prices("ABC").selection["internal_continuity_verified"] is None


def test_confirmed_placeholder_tail_remains_raw_but_never_becomes_a_trade(database):
    before = tape(date(2025, 1, 1), date(2025, 7, 16), close=374.3)
    after = tape(date(2025, 7, 17), date(2026, 9, 11), close=374.3, volume=0)
    prices.persist_prices("SPY", tape(date(2025, 1, 1), date(2026, 9, 11)), source="fmp")
    prices.persist_prices("ANOMALY", before + after, source="alpha_vantage")
    result = prices.read_prices("ANOMALY", days=120)
    assert result[-1]["date"] == "2025-07-16"
    candidate = result.selection["candidate_sources"][0]
    assert candidate["excluded_zero_volume_flat_dates"] == [r["date"] for r in after]
    report = prices.price_coverage("ANOMALY", date(2025, 1, 1))
    assert not report["coverage_complete"]
    assert not report["sources"][0]["current"]
    assert report["sources"][0]["excluded_zero_volume_flat_dates"] == [r["date"] for r in after]
    with database() as db:
        assert db.scalar(select(func.count()).select_from(DailyPrice).where(DailyPrice.ticker == "ANOMALY")) == len(before + after)
    # Even an empty bounded read carries every excluded identity to the admin API.
    from app.api.routes_admin import stored_market_prices
    response = stored_market_prices("ANOMALY", start=date(2026, 1, 1), days=None)
    assert response["rows"] == []
    assert response["selection"]["candidate_sources"][0]["excluded_zero_volume_flat_dates"]


@pytest.mark.parametrize("overrides", [{"volume": None}, {"open": None}, {"high": None}, {"low": None}, {"high": 101}, {"volume": 1}])
def test_unknown_volume_or_incomplete_ohlc_remains_eligible(database, overrides):
    row = {"date": "2025-01-02", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 0, **overrides}
    prices.persist_prices("ABC", [row], source="fmp")
    assert len(prices.read_prices("ABC")) == 1


def test_fresh_gapped_series_does_not_take_data_service_fast_path(database, monkeypatch):
    from app.config import settings
    from app.services.data_service import DataService
    rows = tape(date.today() - timedelta(days=450), date.today() - timedelta(days=1))
    prices.persist_prices("SPY", rows, source="fmp")
    prices.persist_prices("ABC", rows[:-50] + rows[-49:], source="fmp")
    monkeypatch.setattr(settings, "use_demo_data", False)
    monkeypatch.setattr(settings, "enable_live_data", True)
    ds = DataService()
    attempted = []
    monkeypatch.setattr(prices, "fetch_and_store_prices", lambda *a, **kw: attempted.append(a) or {})
    monkeypatch.setattr(ds, "_cached", lambda capability, key, fetcher, **kw: fetcher())
    ds.get_price_history("ABC", days=120)
    assert attempted, "last date and fetch timestamp alone cannot prove a healthy tape"


def test_old_cached_placeholder_payload_cannot_escape_durable_read_filter(database, monkeypatch):
    from app.config import settings
    from app.services.data_service import DataService
    before = tape(date(2024, 1, 1), date(2025, 7, 16))
    fake = tape(date.today() - timedelta(days=10), date.today() - timedelta(days=1), volume=0)
    prices.persist_prices("ABC", before + fake, source="alpha_vantage")
    monkeypatch.setattr(settings, "use_demo_data", False)
    monkeypatch.setattr(settings, "enable_live_data", True)
    ds = DataService()
    monkeypatch.setattr(ds, "_cached", lambda *a, **kw: fake)
    monkeypatch.setattr(prices, "fetch_and_store_prices", lambda *a, **kw: pytest.fail("no provider call on cache hit"))
    result = ds.get_price_history("ABC", days=120)
    assert result[-1]["date"] == "2025-07-16"
    assert not any(r["volume"] == 0 for r in result)


def test_old_gap_outside_requested_suffix_does_not_demote_current_series(database):
    rows = tape(date(2024, 1, 1), date(2025, 12, 31))
    prices.persist_prices("SPY", rows, source="fmp")
    prices.persist_prices("ABC", rows[:300], source="fmp")
    prices.persist_prices("ABC", rows[:100] + rows[150:], source="tiingo")
    result = prices.read_prices("ABC", days=30)
    assert {r["source"] for r in result} == {"tiingo"}
    assert result[-1]["date"] == rows[-1]["date"]
    assert result.selection["internal_continuity_verified"] is True
    selected = next(g for g in result.selection["candidate_sources"] if g["source"] == "tiingo")
    assert selected["continuity_start"] == rows[-30]["date"]
    full = next(g for g in prices.price_coverage("ABC", date(2024, 1, 1))["sources"] if g["source"] == "tiingo")
    assert full["internal_missing_benchmark_sessions"] == [r["date"] for r in rows[100:150]]


def test_normal_refresh_accepts_complete_requested_suffix_despite_older_gap(database):
    from types import SimpleNamespace
    rows = tape(date.today() - timedelta(days=700), date.today() - timedelta(days=1))
    prices.persist_prices("SPY", rows, source="fmp")
    broken_old = rows[:100] + rows[150:]
    provider = SimpleNamespace(name="fmp", get_price_history=lambda *args: broken_old)
    fallback = SimpleNamespace(name="tiingo", get_price_history=lambda *args: pytest.fail("complete requested suffix already answered"))
    result = prices.fetch_and_store_prices("ABC", 30, service=SimpleNamespace(_live_chain=lambda _: [provider, fallback]), verify_calendar=False)
    assert result["refresh_complete"] is True
    # Broader backfill coverage remains distinct from the successful 30-bar read.
    assert result["coverage"]["sources"][0]["internal_continuity_verified"] is False


@pytest.mark.parametrize("verify_calendar", [True, False])
def test_placeholder_only_refresh_cannot_freshen_complete_stale_real_tape(database, verify_calendar):
    from datetime import datetime
    from types import SimpleNamespace

    from sqlalchemy import update
    # End on the last known completed session. A later raw calendar-date
    # placeholder must neither move the usable boundary nor its fetched_at.
    end = prices._last_weekday(date.today())
    start = end - timedelta(days=200)
    rows = tape(start, end)
    prices.persist_prices("SPY", rows, source="fmp")
    prices.persist_prices("ABC", rows, source="fmp")
    stale = datetime.utcnow() - timedelta(days=3)
    with database() as db:
        db.execute(update(DailyPrice).where(DailyPrice.ticker == "ABC").values(fetched_at=stale))
        db.commit()
    placeholder = {"date": date.today().isoformat(), "open": 100, "high": 100, "low": 100, "close": 100, "volume": 0}
    provider = SimpleNamespace(name="fmp", get_price_history=lambda *a: [placeholder])
    result = prices.fetch_and_store_prices("ABC", 120, service=SimpleNamespace(_live_chain=lambda _: [provider]), verify_calendar=verify_calendar)
    attempt = result["attempts"][0]
    assert attempt["rows_upserted"] == 1 and attempt["usable_rows_upserted"] == 0
    assert attempt["excluded_zero_volume_flat_dates"] == [placeholder["date"]]
    assert not result["refresh_complete"]
    source = result["coverage"]["sources"][0]
    assert source["coverage_complete"] and source["current"]
    assert not source["fresh"]
    assert source["freshness_price_date"] == end.isoformat()
    assert source["last_fetched_at"] == stale.isoformat()
    assert source["last_raw_fetched_at"] > source["last_fetched_at"]
    selected = prices.read_prices("ABC", days=120)
    assert selected[-1]["date"] == end.isoformat()
    assert selected.selection["candidate_sources"][0]["last_fetched_at"] == stale.isoformat()


def test_fresh_old_real_bar_cannot_certify_stale_latest_usable_bar(database):
    from datetime import datetime
    from types import SimpleNamespace

    from sqlalchemy import update
    rows = tape(date.today() - timedelta(days=200), prices._last_weekday(date.today()))
    prices.persist_prices("SPY", rows, source="fmp")
    prices.persist_prices("ABC", rows, source="fmp")
    stale = datetime.utcnow() - timedelta(days=3)
    with database() as db:
        db.execute(update(DailyPrice).where(DailyPrice.ticker == "ABC").values(fetched_at=stale))
        db.commit()
    provider = SimpleNamespace(name="fmp", get_price_history=lambda *a: rows[:1])
    result = prices.fetch_and_store_prices("ABC", 120, service=SimpleNamespace(_live_chain=lambda _: [provider]), verify_calendar=False)
    assert result["attempts"][0]["usable_rows_upserted"] == 1
    assert not result["refresh_complete"]
    source = result["coverage"]["sources"][0]
    assert source["last_fetched_at"] == stale.isoformat()
    assert source["freshness_price_date"] == rows[-1]["date"]


def test_only_placeholders_have_no_usable_freshness(database):
    rows = tape(date(2025, 1, 1), date(2025, 1, 31), volume=0)
    report = prices.persist_prices("ABC", rows, source="fmp")
    assert report["rows_upserted"] == len(rows)
    assert report["usable_rows_upserted"] == 0
    source = prices.price_coverage("ABC", date(2025, 1, 1))["sources"][0]
    assert source["last_fetched_at"] is None and source["freshness_price_date"] is None
    assert source["last_raw_fetched_at"] is not None and source["fresh"] is False
