from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Company, DailyPrice, MarketDataSync, MemoOutcome, MemoSnapshot
from app.services import market_data_backfill as backfill
from app.services import price_history_service as prices


@pytest.fixture
def database(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(prices, "SessionLocal", factory)
    monkeypatch.setattr(backfill, "SessionLocal", factory)
    yield factory, engine
    engine.dispose()


def tape(start, end, close=100):
    result = []
    while start <= end:
        if start.weekday() < 5:
            result.append({"date": start.isoformat(), "close": close, "adjusted_close": close - 1, "volume": 1000})
        start += timedelta(days=1)
    return result


def test_store_is_idempotent_and_preserves_both_adjustment_values(database):
    rows = [{"date": "2025-01-02", "close": 100, "adjusted_close": 90, "volume": 200}]
    for _ in range(2):
        prices.persist_prices("ABC", rows, source="tiingo", provenance={"close_basis": "unadjusted"})
    with database[0]() as db:
        stored = db.execute(select(DailyPrice)).scalars().all()
        assert len(stored) == 1
        assert (stored[0].close, stored[0].adjusted_close, stored[0].close_basis) == (100, 90, "unadjusted")


@pytest.mark.parametrize("bad", [None, True, False, 0, -1, float("nan"), float("inf"), "broken"])
def test_bad_refresh_cannot_destroy_good_close(database, bad):
    prices.persist_prices("ABC", [{"date": "2025-01-02", "close": 100, "adjusted_close": 90}], source="fmp")
    result = prices.persist_prices("ABC", [{"date": "2025-01-02", "close": bad}], source="fmp")
    assert result["rows_upserted"] == 0
    assert result["rejected"][0]["date"] == "2025-01-02"
    assert prices.read_prices("ABC")[0]["close"] == 100


def test_missing_adjusted_value_does_not_erase_known_value(database):
    prices.persist_prices("ABC", [{"date": "2025-01-02", "close": 100, "adjusted_close": 90}], source="tiingo")
    prices.persist_prices("ABC", [{"date": "2025-01-02", "close": 101}], source="tiingo")
    assert prices.read_prices("ABC")[0]["adjusted_close"] == 90


def test_provider_histories_are_never_spliced(database):
    prices.persist_prices("ABC", [{"date": "2025-01-02", "close": 10}], source="fmp")
    prices.persist_prices("ABC", [{"date": "2025-01-03", "close": 1000}], source="tiingo")
    rows = prices.read_prices("ABC")
    assert len(rows) == 1
    assert len({row["source"] for row in rows}) == 1


def test_stored_api_uses_the_same_bar_count_selection_without_fetch(database):
    from app.api.routes_admin import stored_market_prices
    prices.persist_prices("ABC", tape(date(2025, 1, 1), date(2025, 1, 31)), source="fmp")
    response = stored_market_prices("ABC", days=3)
    assert response["count"] == 3
    assert response["rows"] == prices.read_prices("ABC", days=3)
    assert response["read_only"] is True


def test_duplicates_invalid_dates_are_fully_reported(database):
    result = prices.persist_prices("ABC", [{"date": "2025-01-02", "close": 1}, {"date": "2025-01-02", "close": 1}, {"date": "not-a-date", "close": 1}], source="fmp")
    assert result["duplicate_dates"] == ["2025-01-02"]
    assert result["rejected"] == [{"position": 2, "date": "not-a-date", "reason": "invalid_date_or_close"}]
    assert prices.read_prices("ABC")[0]["close"] == 1


def test_conflicting_duplicate_cannot_replace_existing_close(database):
    prices.persist_prices("ABC", [{"date": "2025-01-02", "close": 99}], source="fmp")
    report = prices.persist_prices("ABC", [{"date": "2025-01-02", "close": 100}, {"date": "2025-01-02", "close": 101}], source="fmp")
    assert report["rows_upserted"] == 0
    assert report["rejected"][0]["reason"] == "conflicting_duplicate"
    assert prices.read_prices("ABC")[0]["close"] == 99


def test_adjustment_bases_are_not_spliced_even_with_same_provider(database):
    prices.persist_prices("ABC", [{"date": "2025-01-02", "close": 100}], source="fmp", provenance={"close_basis": "raw"})
    prices.persist_prices("ABC", [{"date": "2025-01-03", "close": 10}], source="fmp", provenance={"close_basis": "split_adjusted"})
    rows = prices.read_prices("ABC")
    assert len(rows) == 1


def test_stale_complete_store_cannot_mask_failed_refresh(database):
    from sqlalchemy import update
    start = prices.minimum_start()
    rows = tape(start, date.today() - timedelta(days=1))
    prices.persist_prices("SPY", rows, source="fmp")
    prices.persist_prices("ABC", rows, source="tiingo")
    with database[0]() as db:
        db.execute(update(DailyPrice).where(DailyPrice.ticker == "ABC").values(fetched_at=datetime.utcnow() - timedelta(days=3)))
        db.commit()
    calls = []
    def failing(*args):
        calls.append("failed")
        return None
    def healthy(*args):
        calls.append("healthy")
        return rows
    providers = [SimpleNamespace(name="fmp", get_price_history=failing), SimpleNamespace(name="tiingo", get_price_history=healthy)]
    result = prices.fetch_and_store_prices("ABC", (date.today() - start).days, service=SimpleNamespace(_live_chain=lambda _: providers))
    assert calls == ["failed", "healthy"]
    assert result["refresh_complete"] is True


def test_failed_refresh_with_stale_good_rows_remains_incomplete(database):
    start = prices.minimum_start()
    rows = tape(start, date.today() - timedelta(days=1))
    prices.persist_prices("SPY", rows, source="fmp")
    prices.persist_prices("ABC", rows, source="fmp")
    provider = SimpleNamespace(name="fmp", get_price_history=lambda *args: None)
    result = prices.fetch_and_store_prices("ABC", (date.today() - start).days, service=SimpleNamespace(_live_chain=lambda _: [provider]))
    assert result["coverage"]["coverage_complete"] is True
    assert result["refresh_complete"] is False


def test_full_bounds_do_not_hide_interior_price_gaps(database):
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=730)
    rows = tape(start, end)
    prices.persist_prices("SPY", rows, source="fmp")
    missing = rows[100]["date"]
    prices.persist_prices("ABC", rows[:100] + rows[101:], source="fmp")
    report = prices.price_coverage("ABC", start)
    assert report["date_bounds_covered"] is True
    assert report["coverage_complete"] is False
    assert report["sources"][0]["missing_benchmark_sessions"] == [missing]


def test_two_endpoint_bars_are_not_two_years_of_coverage(database):
    end = date.today() - timedelta(days=3)
    start = end - timedelta(days=730)
    prices.persist_prices("SPY", [{"date": start.isoformat(), "close": 100}, {"date": end.isoformat(), "close": 100}], source="fmp")
    assert prices.price_coverage("SPY", start)["coverage_complete"] is False


def test_backfill_tries_next_provider_for_partial_history(database):
    start = prices.minimum_start()
    rows = tape(start, date.today() - timedelta(days=1))
    prices.persist_prices("SPY", rows, source="fmp")
    calls = []
    class Provider:
        price_history_provenance = {"close_basis": "unadjusted"}
        def __init__(self, name, values): self.name, self.values = name, values
        def get_price_history(self, ticker, days):
            calls.append((self.name, ticker, days))
            return self.values
    ds = SimpleNamespace(_live_chain=lambda _: [Provider("fmp", rows[-5:]), Provider("tiingo", rows)])
    result = prices.fetch_and_store_prices("ABC", (date.today() - start).days, service=ds)
    assert result["coverage"]["coverage_complete"] is True
    assert [call[0] for call in calls] == ["fmp", "tiingo"]
    assert {row["source"] for row in prices.read_prices("ABC", start=start)} == {"tiingo"}


def test_unavailable_api_preserves_stored_history(database):
    prices.persist_prices("ABC", [{"date": "2025-01-02", "close": 100}], source="fmp")
    provider = SimpleNamespace(name="fmp", get_price_history=lambda *args: None)
    prices.fetch_and_store_prices("ABC", 730, service=SimpleNamespace(_live_chain=lambda _: [provider]))
    assert prices.read_prices("ABC")[0]["close"] == 100


def test_minimum_two_years_including_leap_day():
    assert prices.minimum_start(date(2026, 9, 13)) == date(2024, 9, 13)
    assert prices.minimum_start(date(2024, 2, 29)) == date(2022, 2, 28)


def test_plan_includes_every_tier_legacy_benchmark_and_old_dates_without_json(database):
    with database[0]() as db:
        for ticker, tier in [("ABC", "auto_analysis"), ("DEF", "data_only")]:
            db.add(Company(ticker=ticker, company_name=ticker, sector="Unknown", industry="Unknown", universe_tier=tier))
        db.add(MemoSnapshot(ticker="OLD", version=1, generated_at=datetime(2020, 1, 2), memo_json={"body": "x" * 100000}))
        db.add(MemoSnapshot(ticker="BACK", version=1, generated_at=datetime(2000, 1, 1), as_of_date=datetime(2000, 1, 1)))
        db.commit()
    statements = []
    event.listen(database[1], "before_cursor_execute", lambda conn, cursor, statement, *args: statements.append(statement))
    plan = backfill.backfill_plan(today=date(2026, 9, 13))
    assert [target["ticker"] for target in plan["targets"]] == ["SPY", "ABC", "DEF", "OLD"]
    assert next(t for t in plan["targets"] if t["ticker"] == "OLD")["requested_start"] == "2019-12-26"
    assert plan["pending_outcome_pair_count"] == 4
    assert not any("memo_snapshots.memo_json" in statement for statement in statements)
    assert not any("market_data_syncs.report" in statement for statement in statements)


def test_backfill_reentry_uses_durable_claim(database, monkeypatch):
    from app.services import fundamental_history_service as fundamentals
    with database[0]() as db:
        db.add(Company(ticker="ABC", company_name="ABC", sector="Unknown", industry="Unknown"))
        db.add(MarketDataSync(ticker="ABC", requested_start=date(2024, 1, 1), started_at=datetime.utcnow(), status="running"))
        db.commit()
    monkeypatch.setattr(backfill, "backfill_prices", lambda *args, **kwargs: pytest.fail("duplicate API request"))
    monkeypatch.setattr(fundamentals, "backfill_fundamentals", lambda *args, **kwargs: pytest.fail("duplicate API request"))
    assert backfill.sync_ticker("ABC")["status"] == "running"


def test_sync_reports_partial_failure_without_generating_or_scoring(database, monkeypatch):
    from app.services import fundamental_history_service as fundamentals
    with database[0]() as db:
        db.add(Company(ticker="ABC", company_name="ABC", sector="Unknown", industry="Unknown"))
        db.commit()
    monkeypatch.setattr(backfill, "backfill_prices", lambda *args, **kwargs: {"success": True})
    monkeypatch.setattr(fundamentals, "backfill_fundamentals", lambda *args, **kwargs: {"success": False, "issues": [{"kind": "missing_quarters"}]})
    result = backfill.sync_ticker("ABC")
    assert result["status"] == "incomplete"
    with database[0]() as db:
        assert db.get(MarketDataSync, "ABC").report == result
        assert db.execute(select(MemoSnapshot)).scalars().all() == []
        assert db.execute(select(MemoOutcome)).scalars().all() == []


def test_superseded_import_cannot_overwrite_newer_claim(database, monkeypatch):
    from app.services import fundamental_history_service as fundamentals
    with database[0]() as db:
        db.add(Company(ticker="ABC", company_name="ABC", sector="Unknown", industry="Unknown"))
        db.commit()
    def replace_claim(*args, **kwargs):
        with database[0]() as db:
            row = db.get(MarketDataSync, "ABC")
            row.started_at += timedelta(seconds=1)
            row.report = {"belongs_to": "newer claim"}
            db.commit()
        return {"success": True}
    monkeypatch.setattr(backfill, "backfill_prices", replace_claim)
    monkeypatch.setattr(fundamentals, "backfill_fundamentals", lambda *args, **kwargs: {"success": True})
    result = backfill.sync_ticker("ABC")
    assert result["status"] == "superseded"
    with database[0]() as db:
        assert db.get(MarketDataSync, "ABC").report == {"belongs_to": "newer claim"}


def test_freshly_imported_halted_series_is_not_treated_as_current(database, monkeypatch):
    from app.config import settings
    from app.services.data_service import DataService
    start, end = date(2024, 1, 1), date(2025, 1, 1)
    prices.persist_prices("ABC", tape(start, end), source="fmp")
    monkeypatch.setattr(settings, "use_demo_data", False)
    monkeypatch.setattr(settings, "enable_live_data", True)
    ds = DataService()
    attempted = []
    monkeypatch.setattr(prices, "fetch_and_store_prices", lambda *args, **kwargs: attempted.append(args) or {})
    monkeypatch.setattr(ds, "_cached", lambda capability, key, fetcher, **kwargs: fetcher())
    rows = ds.get_price_history("ABC", days=200)
    assert attempted, "fresh import timestamp must not hide a halted historical series"
    assert rows[-1]["date"] == "2025-01-01"


def test_old_memo_can_use_durable_history_beyond_remote_ladder(database):
    from app.services.outcome_service import _evaluate_one
    generated = date(2020, 1, 2)
    target = generated + timedelta(days=30)
    rows = tape(generated, target)
    prices.persist_prices("OLD", rows, source="fmp")
    prices.persist_prices("SPY", rows, source="fmp")
    with database[0]() as db:
        snapshot = MemoSnapshot(ticker="OLD", version=1, generated_at=datetime(2020, 1, 2), memo_json={"rating_label": "Bullish"})
        db.add(snapshot)
        db.flush()
        outcome, status = _evaluate_one(snapshot, 30, db=db, today=date(2026, 9, 13), benchmark="SPY")
        assert outcome is not None, status
        assert outcome.forward_return == 0
        assert "price_window=durable_history" in outcome.note
