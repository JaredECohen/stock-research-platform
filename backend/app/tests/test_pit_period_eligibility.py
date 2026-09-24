"""Regression for observed LULU dates, strict unknowns, provenance and DB parity."""
import json
from datetime import date, datetime

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.finance import scorecard_features as features
from app.finance.pit_eligibility import date_exclusion_reasons
from app.models import FinancialPeriod
from app.services import scorecard_pit, scorecard_service


def _tuple(*, ident=35607, end="2026-02-01", available="2025-04-18", value=298724000.0):
    return ("balance", "short_term_debt", "FY2025", end, 2025, None, value, available,
            {"id": ident, "ticker": "LULU", "source": "fmp", "currency": "USD",
             "available_at_source": "lag_rule", "fetched_at": "2026-09-13T05:14:55.006021"})


@pytest.mark.parametrize("as_of", [date(2025, 4, 18), date(2026, 2, 1), date(2026, 9, 13)])
def test_observed_lulu_availability_never_admits_impossible_period(as_of):
    # Exact tuple fields from the final production coverage read; no live call.
    row = _tuple()
    snap = features.pit_snapshot([row], as_of)
    assert snap.points == ()
    assert snap.notes["available_before_period_end"] == 1
    assert snap.notes["pit_excluded"] == 1
    audit = snap.excluded_rows[0]
    assert audit["id"] == 35607 and audit["source"] == "fmp" and audit["ticker"] == "LULU"
    assert audit["period_end"] == "2026-02-01" and audit["available_at"] == "2025-04-18"
    assert audit["available_at_source"] == "lag_rule" and audit["value"] == 298724000.0
    assert ("period_end_after_as_of" in audit["reasons"]) == (as_of < date(2026, 2, 1))
    detail = features.compute_features_detailed(snap, {}, "Technology")
    assert detail.context["pit_exclusions"] == list(snap.excluded_rows)


def test_known_dates_are_inclusive_and_eight_field_callers_still_work():
    row = _tuple(end="2025-02-02", available="2025-04-18")[:8]
    snap = features.pit_snapshot([row], date(2025, 4, 18))
    assert snap.latest.balance == {"short_term_debt": 298724000.0}
    assert not snap.excluded_rows
    assert not date_exclusion_reasons(date(2025, 4, 18), datetime(2025, 4, 18, 16), date(2025, 4, 18))


@pytest.mark.parametrize("end,available,reason", [
    (None, "2025-04-18", "missing_period_end"),
    ("bad-date", "2025-04-18", "missing_period_end"),
    ("2025-02-02", None, "missing_available_at"),
    ("2025-02-02", "2025-05-18", "available_after_as_of"),
    ("2026-02-01", "2026-04-18", "period_end_after_as_of"),
])
def test_missing_and_future_dates_are_explained_without_fiscal_date_guesses(end, available, reason):
    snap = features.pit_snapshot([_tuple(end=end, available=available)], date(2025, 4, 18))
    assert not snap.points and reason in snap.excluded_rows[0]["reasons"]
    assert snap.excluded_rows[0]["period_end"] == end


def test_exclusion_audit_is_uncapped_order_independent_and_part_of_skip_identity():
    rows = [_tuple(ident=i) for i in range(150)]
    snap = features.pit_snapshot(rows, date(2025, 4, 18))
    reverse = features.pit_snapshot(reversed(rows), date(2025, 4, 18))
    assert len(snap.excluded_rows) == 150 and {r["id"] for r in snap.excluded_rows} == set(range(150))
    assert snap == reverse
    assert features.inputs_hash(snap, {}) == features.inputs_hash(reverse, {})
    assert features.inputs_hash(snap, {}) != features.inputs_hash(features.pit_snapshot(rows[:-1], snap.as_of), {})
    json.dumps(features.compute_features_detailed(snap, {}, "Technology").context, allow_nan=False)


def test_database_reader_and_execution_stream_exclude_same_rows_without_writes():
    engine = create_engine("sqlite://")
    FinancialPeriod.__table__.create(engine)
    with Session(engine) as db:
        for ident, end, available in [(35607, date(2026, 2, 1), date(2025, 4, 18)),
                                      (35609, date(2026, 2, 1), date(2025, 4, 18)),
                                      (99, date(2025, 2, 2), date(2025, 4, 18)),
                                      (100, None, date(2025, 4, 18))]:
            db.add(FinancialPeriod(id=ident, ticker="LULU", statement="balance", line_item=f"line_{ident}",
                period="FY2025", fiscal_year=2025, value=298724000.0, period_end=end,
                available_at=available, available_at_source="lag_rule", currency="USD", source="fmp"))
        db.commit()
        before = [{c.key: getattr(row, c.key) for c in FinancialPeriod.__table__.columns}
                  for row in db.scalars(select(FinancialPeriod).order_by(FinancialPeriod.id))]
        public = scorecard_pit.snapshot_as_of("LULU", date(2025, 4, 18), db=db)
        ticker, rows = next(scorecard_service._iter_period_rows(db, ["LULU"]))
        execution = features.pit_snapshot(rows, date(2025, 4, 18))
        assert ticker == "LULU" and {r["id"] for r in public["excluded_rows"]} == {35607, 35609, 100}
        assert {r["id"] for r in execution.excluded_rows} == {35607, 35609, 100}
        assert public["periods"][0]["balance"] == execution.latest.balance == {"line_99": 298724000.0}
        assert all(r["source"] == "fmp" and r["available_at_source"] == "lag_rule" for r in execution.excluded_rows)
        after = [{c.key: getattr(row, c.key) for c in FinancialPeriod.__table__.columns}
                 for row in db.scalars(select(FinancialPeriod).order_by(FinancialPeriod.id))]
        assert before == after and not db.dirty and not db.new and not db.deleted


def test_lulu_invalid_availability_flagged_not_reinserted(tmp_path, monkeypatch):
    """The FMP takeover must not re-date or re-insert LULU 35607/35609.

    Their stored availability (2025-04-18) precedes their period end
    (2026-02-01), so the PIT readers exclude them. A re-pull flags them
    (nonblocking) and leaves them exactly as stored: no replacement row with
    a new availability date may make the period PIT-eligible (integration
    critique item 5; docs/ops/scorecard-pit-eligibility.md).
    """
    from types import SimpleNamespace

    from sqlalchemy.orm import sessionmaker

    from app.database import Base
    from app.models import FinancialDataRepair
    from app.services import fundamental_history_service as svc

    engine = create_engine(f"sqlite:///{tmp_path / 'lulu.db'}")
    Base.metadata.create_all(engine, tables=[FinancialPeriod.__table__, FinancialDataRepair.__table__])
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(svc, "SessionLocal", factory)
    monkeypatch.setattr(svc, "_today", lambda: date(2026, 9, 13))
    debt = ("short_term_debt", "total_debt")
    with factory() as db:
        for ident, line in zip((35607, 35609), debt, strict=True):
            db.add(FinancialPeriod(id=ident, ticker="LULU", statement="balance", line_item=line, period="FY2025",
                                   fiscal_year=2025, value=298724000.0, period_end=date(2026, 2, 1),
                                   available_at=date(2025, 4, 18), available_at_source="lag_rule", currency="USD",
                                   source="fmp", fetched_at=datetime(2026, 9, 13, 5, 14, 55)))
        db.commit()
    fmp = {s: [] for s in svc.LINES}
    for year in (2023, 2024, 2025):
        for statement, primary in svc.PRIMARY.items():
            row = {"period": f"FY{year}", "period_end": date(year + 1, 2, 1).isoformat(), "currency": "USD",
                   "filing_date": date(year + 1, 3, 27).isoformat(), "fiscal_label_source": "provider", primary: 1.0}
            if statement == "balance" and year == 2025:
                row.update(short_term_debt=298724000.0, total_debt=298724000.0)
            fmp[statement].append(row)
    provider = SimpleNamespace(name="fmp", get_financial_history=lambda symbol, start: fmp)
    monkeypatch.setattr(svc, "get_data_service", lambda: SimpleNamespace(_live_chain=lambda cap: [provider]))
    report = svc.backfill_fundamentals("LULU", date(2024, 9, 13), True)
    flags = [i for i in report["issues"] if i["kind"] == "primary_availability_precedes_period_end"]
    assert {f["id"] for f in flags} == {35607, 35609} and not svc._has_blockers(flags)
    assert report["rows_quarantined"] == 0
    with factory() as db:
        rows = list(db.scalars(select(FinancialPeriod).where(FinancialPeriod.ticker == "LULU",
                                                             FinancialPeriod.line_item.in_(debt))))
        assert {(r.id, r.available_at, r.period_end) for r in rows} == {
            (35607, date(2025, 4, 18), date(2026, 2, 1)), (35609, date(2025, 4, 18), date(2026, 2, 1))}
        _, stream = next(scorecard_service._iter_period_rows(db, ["LULU"]))
    snap = features.pit_snapshot(stream, date(2026, 9, 13))
    assert {r["id"] for r in snap.excluded_rows} >= {35607, 35609}
    assert not set(debt) & set(snap.latest.balance)
