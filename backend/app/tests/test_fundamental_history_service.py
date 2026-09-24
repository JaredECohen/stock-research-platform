from datetime import date, datetime, timedelta
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
    assert third["rows_written"] == third["rows_refreshed"] == 36 and len(calls) == 2
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


def test_legacy_label_lookback_does_not_require_prelisting_history(database, monkeypatch):
    FinancialPeriod.__table__.create(database.kw["bind"])
    with database() as db:
        row = FinancialPeriod(ticker="TEST", period="FY2016", period_end=date(2016, 12, 31),
                              fiscal_year=2016, statement="income", line_item="revenue",
                              source="live", value=10, currency="EUR")
        db.add(row)
        db.commit()
        old_id = row.id
    calls = providers(monkeypatch, ("fmp", payload()), ("alpha_vantage", payload(value=999)))
    result = svc.backfill_fundamentals("TEST", date(2024, 9, 13), force_refresh=True)
    assert result["success"]
    assert calls == [("fmp", "TEST", date(2016, 12, 31))]
    assert not any(i["kind"] in {"provider_partial_coverage", "stored_value_conflict"} for i in result["issues"])
    with database() as db:
        old = db.get(FinancialPeriod, old_id)
        assert (old.period, old.period_end, old.value, old.source) == ("FY2016", date(2016, 12, 31), 10, "live")


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
        return 200, [{"date": "2025-09-30", "fiscalYear": "2026", "period": "FY" if params.get("period") == "annual" else "Q1",
                      "reportedCurrency": "EUR", "revenue": 7}]
    # `_get` wraps `_get_status`, so one patch drives both FMP read paths.
    monkeypatch.setattr(provider, "_get_status", get)
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
    monkeypatch.setattr(provider, "_get_status", lambda *a, **k: (200, [{"date": "2025-09-30", "reportedCurrency": "USD", "revenue": 3}]))
    result = provider.get_financial_history("TEST", date(2024, 1, 1))
    assert len(result["_history_issues"]) == 3
    assert all(i["kind"] == "invalid_fiscal_quarter" for i in result["_history_issues"])
    assert all(r["cadence"] == "annual" for s in svc.LINES for r in result[s])


@pytest.mark.parametrize("period", ["2025-12-31", "FY20251231", "2025Q40", "2025Q0", "FY0000", "99999", "1e9"])
def test_strict_period_parser_rejects_date_and_unbounded_years(period):
    assert history_service._parse_period(period) == (None, None)


def test_invalid_incoming_and_preexisting_years_cannot_expand_coverage(database, monkeypatch):
    rows = payload()
    rows["income"].append({"period": "FY20251231", "period_end": "2025-12-31", "currency": "EUR", "revenue": 4})
    rows["income"].append({"period": "FY9999", "period_end": "2025-12-31", "currency": "EUR", "revenue": 4})
    providers(monkeypatch, ("fmp", rows))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert not report["success"]
    assert {i["period"] for i in report["issues"] if i["kind"] == "invalid_period"} == {"FY20251231", "FY9999"}
    with database() as db:
        db.add(FinancialPeriod(ticker="TEST", period="FY20251231", period_end=date(2025, 12, 31), fiscal_year=20251231,
                               statement="income", line_item="revenue", value=1, currency="EUR", source="alpha_vantage"))
        db.commit()
    coverage = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert not coverage["success"]
    assert any(i["kind"] == "invalid_stored_period" and i["period"] == "FY20251231" for i in coverage["issues"])
    assert len(coverage["coverage"]["income"]["annual"]["missing_periods"]) < 5


def test_mixed_stored_dates_and_currency_are_excluded_with_every_row_identity(database):
    with database() as db:
        FinancialPeriod.__table__.create(bind=db.get_bind(), checkfirst=True)
        for line, end, currency in [("revenue", date(2025, 12, 31), "USD"), ("net_income", date(2025, 12, 30), "EUR")]:
            db.add(FinancialPeriod(ticker="TEST", period="FY2025", period_end=end, fiscal_year=2025,
                                   statement="income", line_item=line, value=3, currency=currency, source="fmp"))
        db.commit()
    stored = svc.read_stored_financials("TEST", start_date=date(2025, 12, 31), cadence="annual")
    assert stored["income"] == []
    issue = stored["_history_issues"][0]
    assert issue["kind"] == "conflicting_stored_statement"
    assert {r["line_item"] for r in issue["rows"]} == {"revenue", "net_income"}
    assert all(r["id"] for r in issue["rows"])


def test_same_value_corroboration_preserves_owner_for_later_restatement(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    providers(monkeypatch, ("alpha_vantage", payload()))
    result = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert result["success"] and result["rows_written"] == 0
    assert {r["source"] for r in svc.read_stored_financials("TEST")["income"]} == {"fmp"}
    providers(monkeypatch, ("fmp", payload(value=101)))
    result = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert result["success"] and result["rows_written"] == 36


def test_conflicting_provider_duplicates_and_stored_period_changes_are_rejected(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    rows = payload()
    rows["income"].append({**rows["income"][0], "revenue": 999})
    rows["balance"][0]["period_end"] = "2023-12-30"
    providers(monkeypatch, ("fmp", rows))
    result = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert not result["success"] and result["rows_written"] == result["rows_refreshed"] == 34
    assert {"conflicting_provider_period", "stored_period_end_conflict"} <= {i["kind"] for i in result["issues"]}
    assert next(r for r in svc.read_stored_financials("TEST")["income"] if r["period"] == "FY2023")["revenue"] == 100


def test_alpha_explicit_report_lists_keep_annual_and_noncalendar_quarters_separate(monkeypatch):
    from app.providers.alpha_vantage_provider import AlphaVantageProvider
    provider = AlphaVantageProvider()
    requests = []
    def fetch(**params):
        requests.append(params)
        return {"annualReports": [{"fiscalDateEnding": "2025-06-28", "reportedCurrency": "EUR", "totalRevenue": "500"}],
                "quarterlyReports": [{"fiscalDateEnding": "2025-03-29", "reportedCurrency": "EUR", "totalRevenue": "110"},
                                     {"fiscalDateEnding": "2025-09-27", "reportedCurrency": "EUR", "totalRevenue": "130"}]}
    monkeypatch.setattr(provider, "_get", fetch)
    rows = provider.get_financial_history("TEST", date(2023, 1, 1))
    assert len(requests) == 3
    income = {r["period"]: r for r in rows["income"]}
    assert set(income) == {"FY2025", "2025Q3", "2026Q1"}
    assert income["FY2025"]["revenue"] == 500 and income["FY2025"]["cadence"] == "annual"
    assert income["2025Q3"]["revenue"] == 110 and income["2025Q3"]["cadence"] == "quarterly"
    assert income["2026Q1"]["period_basis"] == "extrapolated_annual_end"
    assert {r["period"] for r in rows["balance"]} == set(income)
    assert {r["period"] for r in rows["cash"]} == set(income)


def test_fmp_actual_normalizers_preserve_reported_availability_dates(monkeypatch):
    provider = FMPProvider()
    monkeypatch.setattr(provider, "_get_status", lambda *a, **k: (200, [{"date": "2025-12-31", "fiscalYear": "2025", "period": "FY" if k["period"] == "annual" else "Q4",
        "reportedCurrency": "USD", "filingDate": "2026-01-28", "acceptedDate": "2026-01-28 16:00:00"}]))
    result = provider.get_financial_history("TEST", date(2024, 1, 1))
    assert all(r["filing_date"] == "2026-01-28" and r["accepted_date"] == "2026-01-28 16:00:00" for s in svc.LINES for r in result[s])


@pytest.mark.parametrize("legacy", ["live", "unknown", ""], ids=["mode_only", "unknown_source", "empty_source"])
def test_verified_refresh_upgrades_legacy_values_with_source_and_value_audit(database, monkeypatch, legacy):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    with database() as db:
        for row in db.execute(select(FinancialPeriod)).scalars():
            row.source = legacy
        db.commit()
    providers(monkeypatch, ("fmp", payload(value=105)))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert report["success"] and report["rows_written"] == 36
    assert len(report["source_upgrades"]) == 36
    assert all(u["old_source"] == legacy and u["new_source"] == "fmp" and u["old_value"] == 100 and u["new_value"] == 105
               and u["ticker"] == "TEST" and u["id"] and u["period"] and u["line_item"] for u in report["source_upgrades"])
    assert {r["source"] for r in svc.read_stored_financials("TEST")["income"]} == {"fmp"}


@pytest.mark.parametrize("failed_result", [None, RuntimeError("secret provider URL"), {}])
def test_failed_force_refresh_does_not_claim_stored_coverage_is_fresh_success(database, monkeypatch, failed_result):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    providers(monkeypatch, ("fmp", failed_result))
    result = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert not result["success"] and not result["refresh_complete"] and result["rows_written"] == 0
    assert svc._complete(result["coverage"])
    assert any(i["kind"] == "refresh_incomplete" for i in result["issues"])
    assert "secret" not in str(result)


def test_partial_force_refresh_reports_missing_fresh_cadence_even_when_stored_complete(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    providers(monkeypatch, ("fmp", payload(quarterly=False)))
    result = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert svc._complete(result["coverage"]) and not result["success"] and not result["refresh_complete"]


def test_old_valid_anchor_cannot_expand_requested_coverage_gap_range():
    rows = {s: [{"period": "FY0001", "period_end": "0001-12-31", "currency": "USD", "source": "fmp", primary: 1},
                {"period": "FY2025", "period_end": "2025-12-31", "currency": "USD", "source": "fmp", primary: 1}]
            for s, primary in svc.PRIMARY.items()}
    result = svc._coverage(rows, date(2024, 9, 13), date(2026, 9, 13))
    assert result["income"]["annual"]["missing_periods"] == []
    assert not result["income"]["annual"]["covers_start"]


def test_unsafe_generic_financial_adapter_is_not_called(database, monkeypatch):
    unsafe = SimpleNamespace(name="generic", get_financial_statements=lambda ticker: pytest.fail("unknown cadence is unsafe"))
    monkeypatch.setattr(svc, "get_data_service", lambda: SimpleNamespace(_live_chain=lambda cap: [unsafe]))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert not report["success"]
    assert report["issues"][0] == {"kind": "history_adapter_unavailable", "provider": "generic", "symbol": "TEST"}


def test_alpha_period_derivation_survives_in_backfill_report_and_missing_anchor_is_named(database, monkeypatch):
    from app.providers.alpha_vantage_provider import AlphaVantageProvider
    provider = AlphaVantageProvider()
    monkeypatch.setattr(provider, "_get", lambda **params: {
        "annualReports": [{"fiscalDateEnding": "2025-06-28", "reportedCurrency": "USD", "totalRevenue": "100",
                           "totalAssets": "200", "operatingCashflow": "50"}],
        "quarterlyReports": [{"fiscalDateEnding": "2025-09-27", "reportedCurrency": "USD", "totalRevenue": "30",
                              "totalAssets": "210", "operatingCashflow": "15"}]})
    monkeypatch.setattr(svc, "get_data_service", lambda: SimpleNamespace(_live_chain=lambda cap: [provider]))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    inferred = [i for i in report["issues"] if i["kind"] == "fiscal_period_derivation" and i["period"] == "2026Q1"]
    assert len(inferred) == 3
    assert all(i["provider"] == "alpha_vantage" and i["symbol"] == "TEST" and i["anchor_date"] == "2026-06-28"
               and i["period_basis"] == "extrapolated_annual_end" for i in inferred)
    assert next(r for r in svc.read_stored_financials("TEST")["income"] if r["period"] == "2026Q1")["revenue"] == 30
    monkeypatch.setattr(provider, "_get", lambda **params: {"quarterlyReports": [{"fiscalDateEnding": "2025-09-27"}]})
    raw = provider.get_financial_history("TEST", date(2024, 9, 13))
    assert not any(raw[s] for s in svc.LINES)
    assert len([i for i in raw["_history_issues"] if i["kind"] == "fiscal_quarter_unresolved"]) == 3


def test_stored_period_aliases_merge_once_and_conflicting_values_are_excluded(database):
    with database() as db:
        FinancialPeriod.__table__.create(bind=db.get_bind(), checkfirst=True)
        for period in ("2025", "FY2025"):
            db.add(FinancialPeriod(ticker="TEST", period=period, period_end=date(2025, 12, 31), fiscal_year=2025,
                                   statement="income", line_item="revenue", value=100, currency="EUR", source="fmp"))
        db.commit()
    rows = svc.read_stored_financials("TEST", cadence="annual")
    assert len(rows["income"]) == 1 and rows["income"][0]["period"] == "FY2025"
    with database() as db:
        db.execute(select(FinancialPeriod).where(FinancialPeriod.period == "2025")).scalar_one().value = 200
        db.commit()
    rows = svc.read_stored_financials("TEST", cadence="annual")
    assert rows["income"] == []
    assert {r["period"] for r in rows["_history_issues"][0]["rows"]} == {"2025", "FY2025"}


def test_backfill_updates_standalone_canonical_without_duplicate_and_repairs_null_end(database, monkeypatch):
    with database() as db:
        FinancialPeriod.__table__.create(bind=db.get_bind(), checkfirst=True)
        db.add(FinancialPeriod(ticker="TEST", period="FY2025", period_end=None, fiscal_year=2025,
                               statement="income", line_item="revenue", value=90, currency="EUR", source="live"))
        db.commit()
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert report["success"] and report["rows_written"] == 36
    assert any(i["kind"] == "invalid_stored_period" and i["resolved"] for i in report["issues"])
    with database() as db:
        rows = db.execute(select(FinancialPeriod).where(FinancialPeriod.statement == "income", FinancialPeriod.fiscal_year == 2025,
                                                       FinancialPeriod.fiscal_quarter.is_(None))).scalars().all()
        assert len(rows) == 1 and rows[0].period == "FY2025" and rows[0].value == 100 and rows[0].source == "fmp"
        assert rows[0].period_end == date(2025, 12, 31)


def test_duplicate_stored_aliases_are_not_arbitrarily_restated(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    with database() as db:
        db.add(FinancialPeriod(ticker="TEST", period="2025", period_end=date(2025, 12, 31), fiscal_year=2025,
                               statement="income", line_item="revenue", value=100, currency="EUR", source="fmp"))
        db.commit()
    providers(monkeypatch, ("fmp", payload(value=101)))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert not report["success"]
    issue = next(i for i in report["issues"] if i["kind"] == "stored_period_alias_conflict")
    assert {r["period"] for r in issue["rows"]} == {"FY2025", "2025"}
    assert next(r for r in svc.read_stored_financials("TEST")["income"] if r["period"] == "FY2025")["revenue"] == 100


def test_quarter_freshness_names_missing_later_quarter():
    rows = {"income": [{"period": "2026Q1", "period_end": "2026-03-31", "currency": "USD", "source": "fmp", "revenue": 10}]}
    bucket = svc._coverage(rows, date(2024, 9, 13), date(2026, 9, 13))["income"]["quarterly"]
    assert bucket["stale"] and bucket["stale_threshold_days"] == 140
    rows["income"][0].update(period="2025Q3", period_end="2025-09-30")
    bucket = svc._coverage(rows, date(2024, 9, 13), date(2026, 3, 1))["income"]["quarterly"]
    assert not bucket["stale"] and bucket["stale_threshold_days"] == 185


def test_boolean_values_are_rejected_before_real_provider_mapping(monkeypatch):
    from app.providers.alpha_vantage_provider import AlphaVantageProvider
    fmp = FMPProvider()
    monkeypatch.setattr(fmp, "_get_status", lambda *a, **k: (200, [{"date": "2025-12-31", "fiscalYear": "2025",
        "period": "FY" if k["period"] == "annual" else "Q4", "reportedCurrency": "USD", "revenue": True}]))
    raw = fmp.get_financial_history("TEST", date(2024, 1, 1))
    assert all(r["revenue"] is None for r in raw["income"])
    assert any(i["raw_field"] == "revenue" and i["reason"] == "boolean_value" for i in raw["_history_issues"])
    alpha = AlphaVantageProvider()
    monkeypatch.setattr(alpha, "_get", lambda **k: {"annualReports": [{"fiscalDateEnding": "2025-12-31",
        "reportedCurrency": "USD", "totalRevenue": True}], "quarterlyReports": []})
    raw = alpha.get_financial_history("TEST", date(2024, 1, 1))
    assert raw["income"][0]["revenue"] is None
    assert any(i.get("raw_field") == "totalRevenue" and i["reason"] == "boolean_value" for i in raw["_history_issues"])


def test_complete_but_week_old_primary_values_trigger_provider_refresh(database, monkeypatch):
    calls = providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    old = datetime.utcnow() - timedelta(days=8)
    with database() as db:
        for row in db.execute(select(FinancialPeriod)).scalars():
            row.fetched_at = old
        db.commit()
    coverage = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert svc._complete(coverage["coverage"]) and not coverage["success"]
    assert len([i for i in coverage["issues"] if i["kind"] == "stored_fetch_stale"]) == 6
    assert coverage["coverage"]["income"]["annual"]["refresh_ttl_days"] == 7
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert report["success"] and len(calls) == 2
    assert report["rows_written"] == report["rows_refreshed"] == 36
    assert all(b["fresh"] for s in report["coverage"].values() for b in s.values())
    again = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert again["rows_written"] == 0 and len(calls) == 2


def test_primary_fetch_freshness_is_not_hidden_by_a_fresh_optional_line(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    with database() as db:
        row = db.execute(select(FinancialPeriod).where(FinancialPeriod.statement == "income", FinancialPeriod.period == "FY2025")).scalar_one()
        row.fetched_at = datetime.utcnow() - timedelta(days=8)
        db.add(FinancialPeriod(ticker="TEST", statement="income", period="FY2025", period_end=date(2025, 12, 31), fiscal_year=2025,
                               line_item="net_income", value=3, currency="EUR", source="fmp", fetched_at=datetime.utcnow()))
        db.commit()
    report = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert not report["coverage"]["income"]["annual"]["fresh"] and not report["success"]


@pytest.mark.parametrize("raw,expected", [
    ({"shortTermDebt": 0, "longTermDebt": 0}, (0, 0, 0)),
    ({"shortTermDebt": "0", "longTermDebt": "12"}, (0, 12, 12)),
    ({"shortTermDebt": None, "longTermDebt": 12}, (None, 12, None)),
    ({"shortTermDebt": 12}, (12, None, None)),
    ({"shortTermDebt": True, "longTermDebt": False}, (None, None, None)),
    ({"shortTermDebt": -5, "longTermDebt": 5}, (-5, 5, 0)),
    ({"shortTermDebt": 0, "longTermDebt": 6210000000, "totalDebt": 6648000000}, (0, 6210000000, 6648000000)),
    ({"shortTermDebt": 12, "longTermDebt": 4, "totalDebt": 0}, (12, 4, 0)),
])
def test_fmp_debt_mapping_preserves_zero_missing_and_reported_total(raw, expected):
    row = FMPProvider._balance_row(raw)
    assert (row["short_term_debt"], row["long_term_debt"], row["total_debt"]) == expected


def test_fmp_zero_common_dividends_do_not_fall_through_to_other_dividends():
    assert FMPProvider._cash_row({"commonDividendsPaid": 0, "netDividendsPaid": -12})["dividends_paid"] == 0
    assert FMPProvider._cash_row({"netDividendsPaid": -12})["dividends_paid"] == -12
    assert FMPProvider._cash_row({})["dividends_paid"] is None


def test_optional_null_warnings_do_not_block_coverage_or_trigger_repeated_fetch(database, monkeypatch):
    calls = providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    with database() as db:
        for year in (2018, 2025):
            db.add(FinancialPeriod(ticker="TEST", period=f"FY{year}", period_end=date(year, 12, 31), fiscal_year=year,
                                   statement="balance", line_item="short_term_debt", value=None, currency="EUR", source="live"))
        db.commit()
    coverage = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert coverage["success"]
    assert len(coverage["issues"]) == 2 and all(i["kind"] == "missing_stored_optional_value" and i["id"] for i in coverage["issues"])
    skipped = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert skipped["success"] and skipped["rows_written"] == 0 and len(calls) == 1
    fresh = payload()
    fresh["balance"][-1]["short_term_debt"] = 0
    next(r for r in fresh["balance"] if r["period"] == "FY2025")["short_term_debt"] = 0
    providers(monkeypatch, ("fmp", fresh))
    refreshed = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert refreshed["success"]
    assert any(i["kind"] == "missing_stored_optional_value" and i["period"] == "FY2025" and i["resolved"] for i in refreshed["issues"])
    with database() as db:
        old = db.execute(select(FinancialPeriod).where(FinancialPeriod.statement == "balance", FinancialPeriod.period == "FY2018")).scalar_one()
        assert old.value is None  # Retained as unknown, never guessed to be zero.


@pytest.mark.parametrize("line,value,currency,kind", [
    ("total_assets", None, "EUR", "missing_stored_primary_value"),
    ("short_term_debt", float("inf"), "EUR", "invalid_stored_value_or_currency"),
    ("short_term_debt", 0, "", "invalid_stored_value_or_currency"),
])
def test_primary_missing_nonfinite_and_unidentified_currency_remain_blocking(database, monkeypatch, line, value, currency, kind):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    with database() as db:
        row = db.execute(select(FinancialPeriod).where(FinancialPeriod.statement == "balance", FinancialPeriod.period == "FY2025")).scalar_one()
        if line == "total_assets":
            row.value = value
        else:
            db.add(FinancialPeriod(ticker="TEST", period="FY2025", period_end=date(2025, 12, 31), fiscal_year=2025,
                                   statement="balance", line_item=line, value=value, currency=currency, source="fmp"))
        db.commit()
    report = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert not report["success"] and svc._has_blockers(report["issues"])
    assert any(i["kind"] == kind and i["line_item"] == line for i in report["issues"])


def test_legacy_default_currency_is_corrected_to_verified_provider_currency_with_audit(database, monkeypatch):
    with database() as db:
        FinancialPeriod.__table__.create(bind=db.get_bind(), checkfirst=True)
        # Primary is absent: unrelated old USD metadata must not block adding it.
        db.add(FinancialPeriod(ticker="TEST", period="FY2025", period_end=date(2025, 12, 31), fiscal_year=2025,
                               statement="income", line_item="net_income", value=30, currency="USD", source="live"))
        db.commit()
    fresh = payload(currency="EUR")
    next(r for r in fresh["income"] if r["period"] == "FY2025")["net_income"] = 31
    providers(monkeypatch, ("fmp", fresh))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert report["success"]
    upgrade = report["source_upgrades"][0]
    assert (upgrade["old_source"], upgrade["new_source"], upgrade["old_currency"], upgrade["new_currency"], upgrade["old_value"], upgrade["new_value"]) == ("live", "fmp", "USD", "EUR", 30, 31)
    providers(monkeypatch, ("fmp", payload(currency="USD")))
    protected = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert not protected["success"] and protected["rows_written"] == 0
    assert {r["currency"] for r in svc.read_stored_financials("TEST")["income"]} == {"EUR"}


def fiscal_payload():
    rows = payload(annual=False)
    for statement, primary in svc.PRIMARY.items():
        rows[statement] += [{"period": f"FY{year}", "period_end": date(year + 1, 2, 1).isoformat(), "currency": "EUR",
                            "filing_date": date(year + 1, 3, 3).isoformat(), "fiscal_label_source": "provider", primary: 101}
                           for year in range(2017, 2026)]
    return rows


def seed_shifted_legacy(database, *, with_optional=False):
    ids = {}
    with database() as db:
        FinancialPeriod.__table__.create(bind=db.get_bind(), checkfirst=True)
        for year in range(2018, 2027):
            for statement, primary in svc.PRIMARY.items():
                row = FinancialPeriod(ticker="TEST", period=f"FY{year}", period_end=date(year, 2, 1), fiscal_year=year,
                    statement=statement, line_item=primary, value=100, currency="EUR", source="live",
                    available_at=date(year, 3, 20), available_at_source="provider")
                db.add(row)
                db.flush()
                ids[(statement, year)] = row.id
        if with_optional:
            # LULU-like partial prior import: known correct debt at FY2025,
            # alongside old legacy FY2025 primary facts ending a year earlier.
            db.add(FinancialPeriod(ticker="TEST", period="FY2025", period_end=date(2026, 2, 1), fiscal_year=2025,
                statement="balance", line_item="short_term_debt", value=0, currency="EUR", source="fmp"))
            db.add(FinancialPeriod(ticker="TEST", period="FY2026", period_end=date(2026, 2, 1), fiscal_year=2026,
                statement="balance", line_item="short_term_debt", value=None, currency="EUR", source="live"))
        db.commit()
    return ids


def test_provider_confirmed_fiscal_chain_relabels_atomically_with_ids_and_availability(database, monkeypatch):
    ids = seed_shifted_legacy(database)
    calls = providers(monkeypatch, ("fmp", fiscal_payload()))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert report["success"] and report["rows_relabelled"] == 27
    assert calls[0][2] == date(2018, 2, 1) == date.fromisoformat(report["provider_requested_start"])
    assert len(report["period_relabels"]) == 27
    with database() as db:
        for (_statement, year), row_id in ids.items():
            row = db.get(FinancialPeriod, row_id)
            assert row.period == f"FY{year - 1}" and row.period_end == date(year, 2, 1)
            assert row.fiscal_year == year - 1 and row.value == 101 and row.source == "fmp"
            assert row.available_at == date(year, 3, 20)
    second = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert second["success"] and second["rows_relabelled"] == second["rows_written"] == 0 and len(calls) == 1


def test_partial_known_rows_and_optional_null_duplicates_do_not_block_other_components(database, monkeypatch):
    ids = seed_shifted_legacy(database, with_optional=True)
    fresh = fiscal_payload()
    for row in fresh["balance"]:
        if row["period"] == "FY2025":
            row["short_term_debt"] = 0
    providers(monkeypatch, ("fmp", fresh))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert report["success"] and report["rows_relabelled"] == 27
    assert any(i["kind"] == "legacy_period_relabel_optional_null" and i["period"] == "FY2026" for i in report["issues"])
    with database() as db:
        known = db.execute(select(FinancialPeriod).where(FinancialPeriod.period == "FY2025", FinancialPeriod.line_item == "short_term_debt")).scalar_one()
        assert known.source == "fmp" and known.value == 0 and known.period_end == date(2026, 2, 1)
        missing = db.execute(select(FinancialPeriod).where(FinancialPeriod.period == "FY2026", FinancialPeriod.line_item == "short_term_debt")).scalar_one()
        assert missing.value is None and missing.source == "live"
        assert db.get(FinancialPeriod, ids[("balance", 2025)]).period == "FY2024"


def test_known_provider_destination_blocks_dependent_chain_without_overwrite(database, monkeypatch):
    ids = seed_shifted_legacy(database)
    with database() as db:
        db.get(FinancialPeriod, ids[("income", 2024)]).source = "other_verified"
        db.commit()
    providers(monkeypatch, ("fmp", fiscal_payload()))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert not report["success"]
    conflicts = [i for i in report["issues"] if i["kind"] == "legacy_period_relabel_conflict"]
    assert {i["id"] for i in conflicts} >= {ids[("income", 2025)], ids[("income", 2026)]}
    with database() as db:
        known = db.get(FinancialPeriod, ids[("income", 2024)])
        assert (known.period, known.period_end, known.source, known.value) == ("FY2024", date(2024, 2, 1), "other_verified", 100)
        assert db.get(FinancialPeriod, ids[("income", 2025)]).period == "FY2025"


def test_relabel_and_later_write_failure_roll_back_together(database, monkeypatch):
    ids = seed_shifted_legacy(database)
    providers(monkeypatch, ("fmp", fiscal_payload()))
    monkeypatch.setattr(history_service, "_upsert_financial_period", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("failed")))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert not report["success"] and not report["committed"] and report["rows_relabelled"] == 0
    with database() as db:
        for (_, year), row_id in ids.items():
            row = db.get(FinancialPeriod, row_id)
            assert row.period == f"FY{year}" and row.source == "live" and row.value == 100


def test_ambiguous_provider_same_end_labels_are_rejected_before_relabel(database, monkeypatch):
    ids = seed_shifted_legacy(database)
    fresh = fiscal_payload()
    row = next(row for row in fresh["income"] if row["period"] == "FY2025")
    fresh["income"].append({**row, "period": "FY2026"})
    providers(monkeypatch, ("fmp", fresh))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert not report["success"] and any(i["kind"] == "ambiguous_provider_period_end" for i in report["issues"])
    with database() as db:
        row = db.get(FinancialPeriod, ids[("income", 2026)])
        assert row.period == "FY2026" and row.source == "live" and row.value == 100


def test_normal_fmp_statement_read_honors_reported_fiscal_year_to_prevent_recurrence(monkeypatch):
    provider = FMPProvider()
    calls = []
    def fetch(path, **params):
        calls.append((path, params))
        return [{"date": "2026-02-01", "fiscalYear": "2025", "period": "FY", "reportedCurrency": "USD",
                 "revenue": 10, "totalAssets": 20, "operatingCashFlow": 3}]
    monkeypatch.setattr(provider, "_get", fetch)
    raw = provider.get_financial_statements("HD")
    assert len(calls) == 3 and all(params["limit"] == 8 for _, params in calls)
    assert {row["period"] for statement in svc.LINES for row in raw[statement]} == {"FY2025"}
    assert FMPProvider._period_label("2026-02-01", "Q4", "2025") == "2025Q4"
    assert FMPProvider._period_label("2026-02-01", "FY", "20260201") == "FY2026"


def test_external_session_owns_relabel_commit_and_rollback(database, monkeypatch):
    ids = seed_shifted_legacy(database)
    providers(monkeypatch, ("fmp", fiscal_payload()))
    with database() as db:
        report = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True, db=db)
        assert report["success"] and not report["committed"] and report["rows_relabelled"] == 27
        db.rollback()
    with database() as db:
        assert all(db.get(FinancialPeriod, row_id).period == f"FY{year}" for (_, year), row_id in ids.items())


def test_duplicate_legacy_destination_is_preserved_after_unique_provider_fact_is_stored(database, monkeypatch):
    ids = seed_shifted_legacy(database)
    with database() as db:
        extra = FinancialPeriod(ticker="TEST", period="FY2027", period_end=date(2026, 2, 1), fiscal_year=2027,
            statement="income", line_item="revenue", value=90, currency="EUR", source="live")
        db.add(extra)
        db.commit()
        extra_id = extra.id
    providers(monkeypatch, ("fmp", fiscal_payload()))
    result = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert result["success"]
    blocked = {i["id"] for i in result["issues"] if i["kind"] == "legacy_period_relabel_preserved_duplicate"}
    assert {ids[("income", 2026)], extra_id} <= blocked
    coverage = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert coverage["success"]
    excluded = [i for i in coverage["issues"] if i["kind"] == "legacy_duplicate_observation_excluded"]
    assert {i["id"] for i in excluded} == {ids[("income", 2026)], extra_id}
    assert len({i["replacement"]["id"] for i in excluded}) == 1
    with database() as db:
        assert db.get(FinancialPeriod, extra_id).value == 90
        assert db.get(FinancialPeriod, ids[("income", 2026)]).value == 100


def seed_undated_alias(database, *, source="live", canonical=False):
    with database() as db:
        FinancialPeriod.__table__.create(db.get_bind(), checkfirst=True)
        row = FinancialPeriod(ticker="TEST", period="FY2023" if canonical else "2023", statement="income",
            line_item="revenue", value=777, currency="USD", period_end=None, fiscal_year=2023,
            fiscal_quarter=None, source=source, available_at=date(2024, 2, 1), fetched_at=datetime(2024, 3, 1))
        db.add(row)
        db.commit()
        return row.id


@pytest.mark.parametrize("existing_canonical", [False, True])
def test_undated_legacy_alias_is_preserved_beside_confirmed_canonical_fact(database, monkeypatch, existing_canonical):
    calls = providers(monkeypatch, ("fmp", payload()))
    if existing_canonical:
        svc.backfill_fundamentals("TEST", date(2024, 9, 13))
        with database() as db:
            row = db.execute(select(FinancialPeriod).where(FinancialPeriod.period == "FY2023", FinancialPeriod.statement == "income")).scalar_one()
            row.source = "live"
            db.commit()
    alias_id = seed_undated_alias(database)
    before = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert not before["success"]
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert report["success"]
    with database() as db:
        alias = db.get(FinancialPeriod, alias_id)
        assert (alias.period, alias.period_end, alias.value, alias.source, alias.currency) == ("2023", None, 777, "live", "USD")
        assert alias.available_at == date(2024, 2, 1) and alias.fetched_at == datetime(2024, 3, 1)
        canonical = db.execute(select(FinancialPeriod).where(FinancialPeriod.period == "FY2023", FinancialPeriod.statement == "income")).scalar_one()
        assert canonical.id != alias_id and canonical.source == "fmp" and canonical.value == 100
        assert canonical.period_end == date(2023, 12, 31)
    coverage = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    warning = next(i for i in coverage["issues"] if i["kind"] == "unusable_legacy_observation")
    assert coverage["success"] and warning["id"] == alias_id and warning["value"] == 777
    assert warning["usable_canonical_id"] == canonical.id and warning["usable_canonical_source"] == "fmp"
    rows = svc.read_stored_financials("TEST")
    annual = [r for r in rows["income"] if r["period"] == "FY2023"]
    assert len(annual) == 1 and annual[0]["revenue"] == 100
    n_calls = len(calls)
    again = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert again["success"] and again["rows_written"] == 0 and len(calls) == n_calls


def test_undated_legacy_alias_without_provider_primary_remains_blocking(database, monkeypatch):
    alias_id = seed_undated_alias(database)
    rows = payload()
    rows["income"] = [r for r in rows["income"] if r["period"] != "FY2023"]
    providers(monkeypatch, ("fmp", rows))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert not report["success"]
    assert any(i["kind"] == "invalid_stored_period" and i["id"] == alias_id and not i.get("resolved") for i in report["issues"])
    assert not any(i["kind"] == "unusable_legacy_observation" for i in report["issues"])


def test_known_provider_undated_alias_is_not_discarded_to_resolve_conflict(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    alias_id = seed_undated_alias(database, source="alpha_vantage")
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert not report["success"]
    assert any(i["kind"] == "stored_period_alias_conflict" for i in report["issues"])
    assert any(i["kind"] == "invalid_stored_period" and i["id"] == alias_id for i in report["issues"])
    with database() as db:
        assert db.get(FinancialPeriod, alias_id).value == 777


def test_invalid_exact_canonical_occupant_in_alias_group_stays_blocking(database, monkeypatch):
    alias_id = seed_undated_alias(database)
    canonical_id = seed_undated_alias(database, canonical=True)
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert not report["success"]
    assert any(i["kind"] == "stored_period_alias_conflict" for i in report["issues"])
    with database() as db:
        for row_id in (alias_id, canonical_id):
            row = db.get(FinancialPeriod, row_id)
            assert row.period_end is None and row.value == 777 and row.source == "live"


def test_fundamentals_keep_canonical_bk_routing_separate_from_price_rename(database, monkeypatch):
    calls = providers(monkeypatch, ("fmp", payload()))
    result = svc.backfill_fundamentals("BK", date(2024, 9, 13))
    assert result["success"] and [symbol for _, symbol, _ in calls] == ["BK"]


@pytest.mark.parametrize("currency", ["None", "NULL", "N/A", "NAN", "UNKNOWN", "XXX", "XTS"])
def test_placeholder_currency_is_rejected_on_incoming_and_stored_reads(database, monkeypatch, currency):
    providers(monkeypatch, ("fmp", payload(currency=currency)))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13))
    assert not report["success"] and report["rows_written"] == 0
    assert any(i["kind"] == "missing_or_invalid_currency" and i["currency"] == currency.upper() for i in report["issues"])
    with database() as db:
        db.add(FinancialPeriod(ticker="TEST", period="FY2025", fiscal_year=2025, period_end=date(2025, 12, 31),
            statement="income", line_item="revenue", value=100, currency=currency, source="alpha_vantage"))
        db.commit()
    stored = svc.read_stored_financials("TEST")
    assert not stored["income"]
    assert any(i["kind"] == "invalid_stored_value_or_currency" for i in stored["_history_issues"])
    assert not svc._complete(svc._coverage(payload(currency=currency), date(2024, 9, 13), date(2026, 9, 13)))
