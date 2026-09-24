"""Read-only handling of preserved legacy facts under a stale fiscal label."""
from datetime import date, datetime

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.models import FinancialPeriod
from app.services import fundamental_history_service as svc


@pytest.fixture
def database(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy-read.db'}")
    FinancialPeriod.__table__.create(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(svc, "SessionLocal", factory)
    monkeypatch.setattr(svc, "_today", lambda: date(2026, 9, 13))
    monkeypatch.setattr(svc, "get_data_service", lambda: pytest.fail("read must not access providers"))
    yield factory, engine
    engine.dispose()


def fact(period, end, statement, line, value, *, source="fmp", currency="USD"):
    fy, fq = svc.history._parse_period(period)
    return FinancialPeriod(ticker="TEST", period=period, period_end=end, fiscal_year=fy, fiscal_quarter=fq,
        statement=statement, line_item=line, value=value, currency=currency, source=source,
        available_at=end, available_at_source="provider", fetched_at=datetime.utcnow())


def seed(database, *, missing_primary=False):
    with database[0]() as db:
        for statement, primary in svc.PRIMARY.items():
            for year in (2023, 2024, 2025):
                if not (missing_primary and statement == "balance" and year == 2025):
                    db.add(fact(f"FY{year}", date(year + 1, 2, 1), statement, primary, 100))
            for year, quarter in ((2024, 2), (2024, 3), (2024, 4), (2025, 1), (2025, 2), (2025, 3), (2025, 4), (2026, 1), (2026, 2)):
                db.add(fact(f"{year}Q{quarter}", date(year, quarter * 3, 30), statement, primary, 100))
        legacy, named = [], []
        for line, value in (("short_term_debt", 5), ("total_debt", 7)):
            legacy.append(fact("FY2026", date(2026, 2, 1), "balance", line, value, source="live", currency="EUR"))
            named.append(fact("FY2025", date(2026, 2, 1), "balance", line, 0))
        db.add_all(legacy + named)
        db.commit()
        return [r.id for r in legacy], [r.id for r in named]


def all_stored_fields(database):
    with database[0]() as db:
        return [{c.name: getattr(row, c.name) for c in FinancialPeriod.__table__.columns}
                for row in db.execute(select(FinancialPeriod).order_by(FinancialPeriod.id)).scalars()]


def test_populated_legacy_duplicates_are_excluded_without_any_stored_change(database, caplog):
    legacy, named = seed(database)
    before = all_stored_fields(database)
    statements = []
    event.listen(database[1], "before_cursor_execute", lambda conn, cursor, sql, *args: statements.append(sql))
    rows = svc.read_stored_financials("TEST")
    result = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert result["success"]
    latest = next(r for r in rows["balance"] if r["period"] == "FY2025")
    assert latest["short_term_debt"] == latest["total_debt"] == 0
    assert latest["currency"] == "USD" and latest["line_sources"]["total_debt"] == "fmp"
    assert not any(r["period"] == "FY2026" for r in rows["balance"])
    warnings = [r for r in rows["_history_issues"] if r["kind"] == "legacy_duplicate_observation_excluded"]
    assert {r["id"] for r in warnings} == set(legacy)
    assert {r["replacement"]["id"] for r in warnings} == set(named)
    for issue in warnings:
        assert issue["source"] == "live" and issue["currency"] == "EUR"
        assert issue["value"] in {5, 7} and issue["replacement"]["value"] == 0
        assert issue["period_end"] == issue["replacement"]["period_end"] == "2026-02-01"
        assert issue["replacement"]["source"] == "fmp"
    assert all_stored_fields(database) == before
    assert not any(sql.lstrip().upper().startswith(("UPDATE", "INSERT", "DELETE")) for sql in statements)
    assert "legacy_duplicate_observation_excluded" in caplog.text
    assert all(str(row_id) in caplog.text for row_id in legacy + named)


def test_multiple_valid_provider_candidates_prefer_the_primary(database):
    # R3 (FMP primary): with FMP and Alpha Vantage both observing the legacy
    # line's period end, FMP's row is the replacement; AV's alias inside FMP's
    # period is named as superseded (R1). Nothing stored changes.
    legacy, named = seed(database)
    with database[0]() as db:
        alpha = fact("2025", date(2026, 2, 1), "balance", "short_term_debt", 0, source="alpha_vantage")
        db.add(alpha)
        db.commit()
        alpha_id = alpha.id
    before = all_stored_fields(database)
    result = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert result["success"]
    excluded = next(i for i in result["issues"] if i["kind"] == "legacy_duplicate_observation_excluded" and i["id"] == legacy[0])
    assert excluded["replacement"]["id"] == named[0] and excluded["replacement"]["source"] == "fmp"
    assert any(i["kind"] == "superseded_by_primary" and i["id"] == alpha_id for i in result["issues"])
    assert not any(i["kind"] == "duplicate_stored_period_end" for i in result["issues"])
    assert all_stored_fields(database) == before


def test_multiple_non_primary_candidates_are_ambiguous_and_stay_blocking(database):
    legacy, named = seed(database)
    with database[0]() as db:
        for row in db.execute(select(FinancialPeriod).where(FinancialPeriod.source == "fmp")).scalars():
            row.source = "other_verified"
        db.add(fact("2025", date(2026, 2, 1), "balance", "short_term_debt", 0, source="alpha_vantage"))
        db.commit()
    before = all_stored_fields(database)
    result = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert not result["success"]
    assert not any(i["kind"] == "legacy_duplicate_observation_excluded" and i["id"] == legacy[0] for i in result["issues"])
    assert any(i["kind"] == "duplicate_stored_period_end" for i in result["issues"])
    assert all_stored_fields(database) == before


def test_known_provider_contradiction_is_never_excluded_as_legacy(database):
    # Still not excluded *as legacy*: the named AV row under a label FMP does
    # not use at that period end is superseded by FMP's label (R2), named
    # with its identity and value, nonblocking. Nothing stored changes.
    legacy, _ = seed(database)
    with database[0]() as db:
        db.get(FinancialPeriod, legacy[0]).source = "alpha_vantage"
        db.commit()
    before = all_stored_fields(database)
    result = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert result["success"]
    assert not any(i["kind"] == "duplicate_stored_period_end" for i in result["issues"])
    assert not any(i["kind"] == "legacy_duplicate_observation_excluded" and i["id"] == legacy[0] for i in result["issues"])
    superseded = next(i for i in result["issues"] if i["kind"] == "label_superseded_by_primary" and i["id"] == legacy[0])
    assert (superseded["source"], superseded["value"], superseded["period"], superseded["primary_period"]) == (
        "alpha_vantage", 5, "FY2026", "FY2025")
    assert all_stored_fields(database) == before


def test_optional_duplicate_exclusion_cannot_create_missing_primary_coverage(database):
    seed(database, missing_primary=True)
    result = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert not result["success"]
    assert result["coverage"]["balance"]["annual"]["stale"]
    assert any(i["kind"] == "legacy_duplicate_observation_excluded" for i in result["issues"])


@pytest.mark.parametrize("change", ["end", "cadence", "missing_value", "missing_currency"])
def test_only_exact_valid_observation_identity_can_replace_legacy(database, change):
    legacy, named = seed(database)
    with database[0]() as db:
        row = db.get(FinancialPeriod, named[0])
        if change == "end":
            row.period_end = date(2026, 1, 31)
        elif change == "cadence":
            row.period = "2025Q4"
            row.fiscal_quarter = 4
        elif change == "missing_value":
            row.value = None
        elif change == "missing_currency":
            row.currency = ""
        db.commit()
    before = all_stored_fields(database)
    result = svc.read_stored_financials("TEST")
    assert not any(i["kind"] == "legacy_duplicate_observation_excluded" and i["id"] == legacy[0] for i in result.get("_history_issues", []))
    assert all_stored_fields(database) == before


def test_unconfirmed_legacy_duplicates_remain_a_blocking_read_gap(database):
    legacy, named = seed(database)
    with database[0]() as db:
        for row_id in named:
            db.get(FinancialPeriod, row_id).source = "live"
        # No FMP row may observe that period end either: a primary label
        # would supersede the legacy ones (R2) instead of leaving a gap.
        for row in db.execute(select(FinancialPeriod).where(FinancialPeriod.period == "FY2025",
                                                            FinancialPeriod.statement == "balance")).scalars():
            row.source = "live"
        db.commit()
    before = all_stored_fields(database)
    result = svc.fundamental_coverage("TEST", date(2024, 9, 13))
    assert not result["success"]
    assert any(i["kind"] == "duplicate_stored_period_end" for i in result["issues"])
    assert not any(i["kind"] == "legacy_duplicate_observation_excluded" for i in result["issues"])
    assert all_stored_fields(database) == before


def test_backfill_quarantines_collision_rows_and_reports_success_only_with_confirmed_read(database, monkeypatch):
    from types import SimpleNamespace

    legacy, named = seed(database)
    before = {row["id"]: row for row in all_stored_fields(database)}
    payload = {statement: [] for statement in svc.LINES}
    with database[0]() as db:
        rows = list(db.execute(select(FinancialPeriod).where(FinancialPeriod.source == "fmp")).scalars())
        grouped = {}
        for row in rows:
            item = grouped.setdefault((row.statement, row.period), {
                "period": row.period, "period_end": row.period_end.isoformat(),
                "currency": row.currency, "fiscal_label_source": "provider"})
            item[row.line_item] = row.value
        for (statement, _), item in grouped.items():
            payload[statement].append(item)
    provider = SimpleNamespace(name="fmp", get_financial_history=lambda ticker, start: payload)
    monkeypatch.setattr(svc, "get_data_service", lambda: SimpleNamespace(_live_chain=lambda cap: [provider]))
    report = svc.backfill_fundamentals("TEST", date(2024, 9, 13), True)
    assert report["success"] and report["committed"] and report["rows_relabelled"] == 0
    # FMP-primary: the legacy labels contradicting FMP's FY2025 observation are
    # quarantined (was: preserved in place and excluded on every read).
    conflicts = [i for i in report["issues"] if i["kind"] == "legacy_period_relabel_conflict"]
    assert {i["id"] for i in conflicts} == set(legacy)
    assert all(i["resolution"] == "quarantined_superseded_by_primary" for i in conflicts)
    assert {q["id"]: q["reason"] for q in report["quarantined"]} == {
        row_id: "label_conflicts_with_primary_observation" for row_id in legacy}
    after = {row["id"]: row for row in all_stored_fields(database)}
    for row_id in legacy:
        assert after[row_id]["ticker"].startswith("~Q") and len(after[row_id]["ticker"]) == 16
        assert {k: v for k, v in after[row_id].items() if k != "ticker"} == {
            k: v for k, v in before[row_id].items() if k != "ticker"}
    # Provider verification may refresh its own metadata; preserved duplicates
    # and named factual identities/values/currency/availability do not change.
    assert all({k: v for k, v in after[row_id].items() if k != "fetched_at"} ==
               {k: v for k, v in before[row_id].items() if k != "fetched_at"} for row_id in named)
    assert svc.fundamental_coverage("TEST", date(2024, 9, 13))["success"]
