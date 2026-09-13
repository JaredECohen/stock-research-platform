from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.models import FinancialPeriod
from app.providers.fmp_provider import FMPProvider
from app.services import fundamental_history_service as svc
from app.services import history_service


@pytest.fixture
def database(tmp_path, monkeypatch):
    factory = sessionmaker(bind=create_engine(f"sqlite:///{tmp_path / 'fundamentals.db'}"), autoflush=False)
    monkeypatch.setattr(svc, "SessionLocal", factory)
    monkeypatch.setattr(svc, "_today", lambda: date(2026, 9, 13))
    return factory


def payload(*, annual=True, quarterly=True, value=100, currency="EUR"):
    periods = []
    if annual:
        periods += [(f"FY{y}", date(y, 12, 31)) for y in (2023, 2024, 2025)]
    if quarterly:
        periods += [(f"{y}Q{q}", date(y, q * 3, 30)) for y in (2024, 2025, 2026) for q in (1, 2, 3, 4)
                    if date(2024, 6, 1) <= date(y, q * 3, 30) <= date(2026, 6, 30)]
    return {s: [{"period": p, "period_end": d.isoformat(), "filing_date": (d + timedelta(days=30)).isoformat(),
                 "currency": currency, primary: value} for p, d in periods] for s, primary in svc.PRIMARY.items()}


def providers(monkeypatch, *data):
    calls = []
    chain = []
    for name, rows in data:
        def fetch(symbol, start, name=name, rows=rows):
            calls.append((name, symbol, start))
            if isinstance(rows, Exception):
                raise rows
            return rows
        chain.append(SimpleNamespace(name=name, get_financial_history=fetch))
    monkeypatch.setattr(svc, "get_data_service", lambda: SimpleNamespace(_live_chain=lambda cap: chain))
    return calls


def test_two_years_annual_and_quarterly_are_durable_and_idempotent(database, monkeypatch):
    calls = providers(monkeypatch, ("fmp", payload()))
    result = svc.backfill_fundamentals(" test ", date(2024, 9, 13))
    assert result["success"] and result["committed"]
    assert result["rows_written"] == 36
    assert result["issues"] == []
    assert result["provider"] == ["fmp"]
    second = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert second["rows_written"] == 0 and len(calls) == 1
    third = svc.backfill_fundamentals("TEST", date(2024, 9, 13), force_refresh=True)
    assert third["rows_written"] == 0 and len(calls) == 2
    rows = svc.read_stored_financials("TEST")
    assert len(rows["income"]) == 12
    assert rows["income"][0]["currency"] == "EUR"
    assert rows["income"][0]["source"] == "fmp"
    assert rows["income"][0]["available_at_source"] == "provider"


def test_partial_provider_does_not_suppress_quarterly_fallback(database, monkeypatch):
    calls = providers(monkeypatch, ("first", payload(quarterly=False)), ("second", payload(annual=False)))
    result = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert result["success"] and len(calls) == 2
    assert result["provider"] == ["first", "second"]
    assert any(i["kind"] == "provider_partial_coverage" for i in result["issues"])
    assert result["coverage"]["income"]["quarterly"]["complete"]


def test_nonfinite_missing_and_conflicting_currency_never_wipe_good_values(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    bad = payload()
    bad["income"][0]["revenue"] = float("nan")
    bad["balance"][0]["total_assets"] = None
    bad["cash"][0]["currency"] = "USD"
    bad["cash"][0]["cash_from_operations"] = 5
    providers(monkeypatch, ("fmp", bad))
    result = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert not result["success"]
    assert {"invalid_value", "missing_primary_value", "stored_value_conflict"} <= {i["kind"] for i in result["issues"]}
    with database() as db:
        annual = db.execute(select(FinancialPeriod).where(FinancialPeriod.period == "FY2023")).scalars().all()
        assert {r.value for r in annual} == {100}
        assert {r.currency for r in annual} == {"EUR"}


def test_cross_provider_conflict_is_reported_without_overwriting(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    providers(monkeypatch, ("other", payload(value=250)))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert not report["success"] and report["rows_written"] == 0
    conflicts = [i for i in report["issues"] if i["kind"] == "stored_value_conflict"]
    assert len(conflicts) == 36
    assert all(i["stored_source"] == "fmp" for i in conflicts)


def test_missing_periods_and_short_history_are_named(database, monkeypatch):
    rows = payload(quarterly=False)
    for s in rows:
        rows[s] = [r for r in rows[s] if r["period"] != "FY2024"]
    providers(monkeypatch, ("fmp", rows))
    report = svc.backfill_fundamentals("TEST", date(2023, 1, 1))
    assert not report["success"]
    assert report["coverage"]["income"]["annual"]["missing_periods"] == ["FY2024"]
    assert not report["coverage"]["income"]["annual"]["covers_start"]
    assert len([i for i in report["issues"] if i["kind"] == "coverage_gap"]) == 6


def test_external_session_keeps_commit_ownership(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload()))
    with database() as db:
        report = svc.backfill_fundamentals("TEST", date(2024, 9, 13), db=db)
        assert report["success"] and not report["committed"]
        assert db.execute(select(FinancialPeriod)).first()
        db.rollback()
    with database() as db:
        assert db.execute(select(FinancialPeriod)).first() is None


def test_provider_error_is_safe_and_fallback_remains_available(database, monkeypatch):
    providers(monkeypatch, ("first", RuntimeError("https://provider?apikey=SECRET")), ("second", payload()))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert report["success"]
    assert "SECRET" not in str(report)
    assert report["issues"][0]["error_type"] == "RuntimeError"


def test_fmp_history_has_both_cadences_and_preserves_noncalendar_fiscal_year(monkeypatch):
    provider = FMPProvider()
    calls = []
    def get(path, **params):
        calls.append((path, params))
        return [{"date": "2025-09-30", "fiscalYear": "2026", "period": "FY" if params.get("period") == "annual" else "Q1",
                 "reportedCurrency": "EUR", "revenue": 7}]
    monkeypatch.setattr(provider, "_get", get)
    result = provider.get_financial_history("test", date(2000, 1, 1))
    assert len(calls) == 6
    assert {p["period"] for _, p in calls} == {"annual", "quarter"}
    assert max(p["limit"] for _, p in calls) > 100
    assert {r["period"] for r in result["income"]} == {"FY2026", "2026Q1"}
    assert all(r["source"] == "fmp" and r["currency"] == "EUR" for r in result["income"])
    calls.clear()
    provider.get_financial_statements("test")
    assert len(calls) == 3 and all(p["limit"] == 8 and "period" not in p for _, p in calls)


def test_legacy_upsert_preserves_currency_and_ignores_nonfinite(database):
    with database() as db:
        FinancialPeriod.__table__.create(bind=db.get_bind(), checkfirst=True)
        args = dict(ticker="TEST", period="FY2025", statement="income", line_item="revenue", period_end=date(2025, 12, 31),
                    fiscal_year=2025, fiscal_quarter=None, source="fmp", currency="EUR")
        assert history_service._upsert_financial_period(db, value=3, **args)
        db.flush()
        for bad in (None, float("inf"), float("nan")):
            assert not history_service._upsert_financial_period(db, value=bad, **args)
        row = db.execute(select(FinancialPeriod)).scalar_one()
        assert row.value == 3 and row.currency == "EUR" and row.source == "fmp"


def test_coverage_read_and_annual_fallback_never_fetch(database, monkeypatch):
    calls = providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    monkeypatch.setattr(svc, "get_data_service", lambda: pytest.fail("read must not fetch"))
    report = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert report["success"] and len(calls) == 1
    annual = svc.read_stored_financials("TEST", cadence="annual")
    assert len(annual["income"]) == 3
    assert all(r["period"].startswith("FY") for r in annual["income"])


def test_persistence_failure_rolls_back_and_returns_safe_report(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload()))
    def fail(*a, **k):
        raise RuntimeError("secret connection string")
    monkeypatch.setattr(database.class_, "commit", fail)
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert not report["success"] and report["rows_written"] == 0 and not report["committed"]
    assert "secret" not in str(report)
    with database() as db:
        assert db.execute(select(FinancialPeriod)).first() is None


def test_unsourced_legacy_refresh_does_not_claim_other_provider_values(database):
    with database() as db:
        FinancialPeriod.__table__.create(bind=db.get_bind(), checkfirst=True)
        args = dict(ticker="TEST", period="FY2025", statement="income", line_item="revenue", period_end=date(2025, 12, 31),
                    fiscal_year=2025, fiscal_quarter=None, currency="EUR")
        history_service._upsert_financial_period(db, value=3, source="fmp", **args)
        db.flush()
        assert not history_service._upsert_financial_period(db, value=8, source="live", **args)
        row = db.execute(select(FinancialPeriod)).scalar_one()
        assert row.value == 3 and row.source == "fmp"


def test_fmp_unidentified_quarter_is_reported_not_relabeled_as_annual(monkeypatch):
    provider = FMPProvider()
    monkeypatch.setattr(provider, "_get", lambda *a, **k: [{"date": "2025-09-30", "reportedCurrency": "USD", "revenue": 3}])
    result = provider.get_financial_history("TEST", date(2024, 1, 1))
    assert len(result["_history_issues"]) == 3
    assert all(i["kind"] == "invalid_fiscal_quarter" for i in result["_history_issues"])
    assert all(r["cadence"] == "annual" for s in svc.LINES for r in result[s])
