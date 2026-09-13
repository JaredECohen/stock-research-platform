from datetime import date, timedelta
from types import SimpleNamespace

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import DailyPrice
from app.services import price_history_service as prices

from app.services.ticker_symbols import is_multi_class, market_data_symbols, symbol_variants


def test_verified_bny_rename_keeps_share_class_and_historical_identity_separate():
    assert market_data_symbols("BK", on=date(2026, 5, 20)) == ["BK"]
    assert market_data_symbols("bk", on=date(2026, 5, 21)) == ["BNY", "BK"]
    assert symbol_variants("BK") == ["BK"]
    assert not is_multi_class("BK")


def test_acquisition_successors_are_not_guessed_as_same_security():
    assert market_data_symbols("ANSS") == ["ANSS"]
    assert market_data_symbols("AVB") == ["AVB"]
    assert market_data_symbols("BRK.B") == symbol_variants("BRK.B")


def test_backfill_uses_current_symbol_but_preserves_canonical_company_key(monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(prices, "SessionLocal", factory)
    end = date.today()
    rows = [{"date": (end - timedelta(days=offset)).isoformat(), "close": 100, "volume": 1000}
            for offset in range(35) if (end - timedelta(days=offset)).weekday() < 5]
    prices.persist_prices("SPY", rows, source="fmp")
    calls = []

    def fetch(symbol, days):
        calls.append(symbol)
        return rows if symbol == "BNY" else []

    service = SimpleNamespace(_live_chain=lambda capability: [SimpleNamespace(name="fmp", get_price_history=fetch)])
    report = prices.fetch_and_store_prices("BK", 30, service=service)
    assert report["refresh_complete"]
    assert calls == ["BNY"]
    with factory() as db:
        stored = db.execute(select(DailyPrice).where(DailyPrice.ticker == "BK")).scalars().all()
        assert stored and all(row.provider_symbol == "BNY" for row in stored)
        assert db.execute(select(DailyPrice).where(DailyPrice.ticker == "BNY")).first() is None
    engine.dispose()
