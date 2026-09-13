from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import FinancialDataRepair, FinancialPeriod
from app.services import bk_fundamental_repair as svc
from app.services import fundamental_history_service as history
from app.services import scorecard_pit

BAD = datetime(2026, 9, 13, 4, 29, 39, 595667)
OLD = datetime(2026, 9, 13, 3, 0)


@pytest.fixture
def database(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'repair.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(svc, "SessionLocal", factory)
    monkeypatch.setattr(history, "_today", lambda: date(2026, 9, 13))
    return factory


def row(*, ticker="BK", period="2025Q4", end=date(2025, 9, 30), source="fmp", value=999, fetched=BAD, line="revenue", statement="income"):
    fy, fq = history.history._parse_period(period)
    return FinancialPeriod(ticker=ticker, period=period, period_end=end, fiscal_year=fy, fiscal_quarter=fq,
        statement=statement, line_item=line, value=value, currency="USD", source=source, fetched_at=fetched,
        available_at=date(2025, 11, 1), available_at_source="provider")


def provider(monkeypatch):
    calls = []
    data = {s: [{"period": "2025Q3", "period_end": "2025-09-30", "fiscal_label_source": "provider",
                 "currency": "USD", primary: 100}] for s, primary in history.PRIMARY.items()}
    def get(symbol, start):
        calls.append((symbol, start))
        return data
    monkeypatch.setattr(history, "get_data_service", lambda: SimpleNamespace(_live_chain=lambda cap:
        [SimpleNamespace(name="fmp", get_financial_history=get)]))
    return calls, data


def test_plan_then_apply_restores_labels_and_values_and_quarantines_unknowns_without_provider_repeat(database, monkeypatch):
    calls, _ = provider(monkeypatch)
    with database() as db:
        restore = row()
        unknown = row(period="2024Q1", end=date(2024, 3, 31), value=55, source="alpha_vantage")
        unaffected = row(period="2026Q1", end=date(2026, 3, 31), fetched=OLD, value=42)
        other = row(ticker="AAPL")
        db.add_all([restore, unknown, unaffected, other])
        db.commit()
        ids = [r.id for r in [restore, unknown, unaffected, other]]
        before = {r.id: svc._snapshot(r) for r in db.execute(select(FinancialPeriod)).scalars()}
    plan = svc.prepare_bk_repair(BAD.replace(tzinfo=UTC))
    assert calls == [("BK", date(2024, 3, 31))]
    assert plan["financial_rows_modified"] == 0 and plan["plan"]["counts"] == {"restore": 1, "quarantine": 1}
    assert svc.read_bk_repair_plan(plan["plan_id"])["digest"] == plan["digest"]
    with database() as db:
        assert {r.id: svc._snapshot(r) for r in db.execute(select(FinancialPeriod)).scalars()} == before
    monkeypatch.setattr(history, "get_data_service", lambda: pytest.fail("apply must not call providers"))
    result = svc.apply_bk_repair(plan["plan_id"], plan["digest"])
    assert result["success"] and result["committed"] and result["outcome_rows_modified"] == 0
    with database() as db:
        restored = db.get(FinancialPeriod, ids[0])
        assert restored.period == "2025Q3" and restored.value == 100 and restored.source == "fmp"
        assert restored.available_at == date(2025, 11, 1) and restored.period_end == date(2025, 9, 30)
        quarantined = svc._snapshot(db.get(FinancialPeriod, ids[1]))
        assert len(quarantined["ticker"]) == 16 and quarantined["ticker"].startswith("~Q")
        assert {**quarantined, "ticker": "BK"} == before[ids[1]]
        for row_id in ids[2:]:
            assert svc._snapshot(db.get(FinancialPeriod, row_id)) == before[row_id]
        assert db.get(FinancialDataRepair, plan["plan_id"]).status == "applied"
    assert svc.apply_bk_repair(plan["plan_id"], plan["digest"])["already_applied"]


@pytest.mark.parametrize("mutation", ["value", "new_destination", "timestamp"])
def test_apply_fences_all_originals_and_destinations_before_any_write(database, monkeypatch, mutation):
    provider(monkeypatch)
    with database() as db:
        r = row()
        db.add(r)
        db.commit()
        row_id = r.id
    plan = svc.prepare_bk_repair(BAD)
    with database() as db:
        if mutation == "new_destination":
            db.add(row(period="2025Q3", fetched=OLD))
        elif mutation == "timestamp":
            db.get(FinancialPeriod, row_id).fetched_at += timedelta(microseconds=1)
        else:
            db.get(FinancialPeriod, row_id).value = 8
        db.commit()
        before = {r.id: svc._snapshot(r) for r in db.execute(select(FinancialPeriod)).scalars()}
    with pytest.raises(RuntimeError, match="version fence"):
        svc.apply_bk_repair(plan["plan_id"], plan["digest"])
    with database() as db:
        assert {r.id: svc._snapshot(r) for r in db.execute(select(FinancialPeriod)).scalars()} == before
        assert db.get(FinancialDataRepair, plan["plan_id"]).status == "planned"


def test_digest_and_tampered_durable_plan_cannot_apply(database, monkeypatch):
    provider(monkeypatch)
    with database() as db:
        db.add(row())
        db.commit()
    plan = svc.prepare_bk_repair(BAD)
    with pytest.raises(RuntimeError, match="digest"):
        svc.apply_bk_repair(plan["plan_id"], "0" * 64)
    with database() as db:
        saved = db.get(FinancialDataRepair, plan["plan_id"])
        saved.plan = {**saved.plan, "ticker": "AAPL"}
        db.commit()
    with pytest.raises(RuntimeError, match="digest"):
        svc.apply_bk_repair(plan["plan_id"], plan["digest"])


def test_destination_owned_by_unaffected_row_is_preserved_and_bad_duplicate_quarantined(database, monkeypatch):
    provider(monkeypatch)
    with database() as db:
        bad, good = row(), row(period="2025Q3", fetched=OLD, value=101)
        db.add_all([bad, good])
        db.commit()
        bad_id, good_id = bad.id, good.id
    plan = svc.prepare_bk_repair(BAD)
    assert plan["plan"]["counts"] == {"quarantine": 1}
    assert plan["plan"]["actions"][0]["reason"] == "destination_owned_by_unaffected_row"
    svc.apply_bk_repair(plan["plan_id"], plan["digest"])
    with database() as db:
        assert db.get(FinancialPeriod, good_id).value == 101
        assert db.get(FinancialPeriod, bad_id).ticker.startswith("~Q")


def test_two_affected_copies_restore_existing_canonical_and_quarantine_other(database, monkeypatch):
    provider(monkeypatch)
    with database() as db:
        old, exact = row(), row(period="2025Q3", source="alpha_vantage", value=1)
        db.add_all([old, exact])
        db.commit()
        old_id, exact_id = old.id, exact.id
    plan = svc.prepare_bk_repair(BAD)
    assert plan["plan"]["counts"] == {"quarantine": 1, "restore": 1}
    svc.apply_bk_repair(plan["plan_id"], plan["digest"])
    with database() as db:
        assert db.get(FinancialPeriod, old_id).ticker.startswith("~Q")
        assert db.get(FinancialPeriod, exact_id).value == 100
        assert db.get(FinancialPeriod, exact_id).source == "fmp"


def test_external_session_rollback_restores_every_row_and_plan_status(database, monkeypatch):
    provider(monkeypatch)
    with database() as db:
        r = row()
        db.add(r)
        db.commit()
        row_id = r.id
    plan = svc.prepare_bk_repair(BAD)
    with database() as db:
        result = svc.apply_bk_repair(plan["plan_id"], plan["digest"], db=db)
        assert not result["committed"]
        db.rollback()
    with database() as db:
        r = db.get(FinancialPeriod, row_id)
        assert r.ticker == "BK" and r.period == "2025Q4" and r.value == 999
        assert db.get(FinancialDataRepair, plan["plan_id"]).status == "planned"


def test_global_pit_availability_pass_never_mutates_quarantined_rows(database):
    with database() as db:
        r = row(ticker="~Q123456789abcde")
        r.available_at = None
        db.add(r)
        db.commit()
        row_id = r.id
        result = scorecard_pit.backfill_available_at(db=db)
        assert result["scanned"] == 0
        assert db.get(FinancialPeriod, row_id).available_at is None
