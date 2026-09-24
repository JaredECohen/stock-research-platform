"""FIX-006 / W5a: FMP owns every observation it reports (owner decision 2026-09-24).

Each case below is a shape from the September data audit (MDT period-end
drift, MSFT/NVDA undated bare-year aliases, LULU label and availability
conflicts, ANSS/ARM secondary rows, BK definition breaks). Conflicting rows
are quarantined under a `~Q` namespace with a durable before-image, never
deleted; equal observations are adopted; every in-place change is audited and
reversible. Nothing here reaches a network or an LLM.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, func, select, update
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.finance import scorecard_features
from app.models import FinancialDataRepair, FinancialPeriod
from app.services import (
    bk_fundamental_repair,
    cycle_position,
    fundamentals_series_service,
    history_service,
    scorecard_pit,
)
from app.services import fundamental_history_service as svc
from app.services import fundamental_quarantine as fq

START = date(2024, 9, 13)


@pytest.fixture
def database(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'takeover.db'}")
    Base.metadata.create_all(engine, tables=[FinancialPeriod.__table__, FinancialDataRepair.__table__])
    factory = sessionmaker(bind=engine, autoflush=False)
    for module in (svc, fq, cycle_position):
        monkeypatch.setattr(module, "SessionLocal", factory)
    monkeypatch.setattr(svc, "_today", lambda: date(2026, 9, 13))
    yield factory
    engine.dispose()


def periods(*, annual=True, quarterly=True):
    out = []
    if annual:
        out += [(f"FY{y}", date(y, 12, 31)) for y in (2023, 2024, 2025)]
    if quarterly:
        out += [(f"{y}Q{q}", date(y, q * 3, 30)) for y in (2024, 2025, 2026) for q in (1, 2, 3, 4)
                if date(2024, 6, 1) <= date(y, q * 3, 30) <= date(2026, 6, 30)]
    return out


def payload(*, value=100.0, currency="USD", annual=True, quarterly=True, extra=None, skip=()):
    rows = {s: [] for s in svc.LINES}
    for period, end in periods(annual=annual, quarterly=quarterly):
        if period in skip:
            continue
        for statement, primary in svc.PRIMARY.items():
            row = {"period": period, "period_end": end.isoformat(), "currency": currency,
                   "filing_date": (end + timedelta(days=30)).isoformat(), "fiscal_label_source": "provider",
                   primary: value}
            row.update((extra or {}).get((period, statement), {}))
            rows[statement].append(row)
    return rows


def providers(monkeypatch, *data):
    calls = []
    chain = []
    for name, rows in data:
        def fetch(symbol, start, name=name, rows=rows):
            calls.append((name, symbol, start))
            return rows
        chain.append(SimpleNamespace(name=name, get_financial_history=fetch))
    monkeypatch.setattr(svc, "get_data_service", lambda: SimpleNamespace(_live_chain=lambda cap: chain))
    return calls


def add(database, ticker="TEST", **fields):
    fy, fq_ = history_service._parse_period(fields["period"])
    defaults = dict(ticker=ticker, fiscal_year=fy, fiscal_quarter=fq_, currency="USD", source="live",
                    fetched_at=datetime(2026, 9, 13, 3), statement="income", line_item="revenue")
    with database() as db:
        row = FinancialPeriod(**{**defaults, **fields})
        db.add(row)
        db.commit()
        return row.id


def fields(database, row_id):
    with database() as db:
        row = db.get(FinancialPeriod, row_id)
        return {c.name: getattr(row, c.name) for c in FinancialPeriod.__table__.columns}


def all_rows(database):
    with database() as db:
        return [{c.name: getattr(r, c.name) for c in FinancialPeriod.__table__.columns}
                for r in db.execute(select(FinancialPeriod).order_by(FinancialPeriod.id)).scalars()]


def repairs(database, kind=None):
    with database() as db:
        rows = list(db.execute(select(FinancialDataRepair).order_by(FinancialDataRepair.created_at)).scalars())
        for r in rows:
            db.expunge(r)
    return [r for r in rows if kind is None or r.plan.get("kind") == kind]


def live_rows(database, ticker="TEST", **where):
    with database() as db:
        stmt = select(FinancialPeriod).where(FinancialPeriod.ticker == ticker)
        for key, value in where.items():
            stmt = stmt.where(getattr(FinancialPeriod, key) == value)
        return list(db.execute(stmt).scalars())


def assert_quarantined_unchanged(database, row_id, before):
    after = fields(database, row_id)
    assert after["ticker"].startswith("~Q") and len(after["ticker"]) == 16
    assert {k: v for k, v in after.items() if k != "ticker"} == {k: v for k, v in before.items() if k != "ticker"}


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

def test_primary_adopts_equal_secondary_and_keeps_availability(database, monkeypatch):
    providers(monkeypatch, ("alpha_vantage", payload()))
    assert svc.backfill_fundamentals("TEST", START)["success"]
    before = {r["id"]: r for r in all_rows(database)}
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert report["success"] and report["rows_quarantined"] == 0 and report["rows_inserted"] == 0
    assert len(report["adoptions"]) == 36
    after = {r["id"]: r for r in all_rows(database)}
    assert set(after) == set(before)  # same ids: adopted in place, nothing inserted
    for row_id, row in after.items():
        assert row["source"] == "fmp" and row["value"] == before[row_id]["value"]
        assert (row["available_at"], row["available_at_source"]) == (before[row_id]["available_at"], before[row_id]["available_at_source"])
    [audit] = repairs(database, "primary_adoption")
    assert {a["before"]["source"] for a in audit.plan["actions"]} == {"alpha_vantage"}
    assert {a["after"]["source"] for a in audit.plan["actions"]} == {"fmp"}


def test_primary_value_conflict_quarantines_with_full_before_image(database, monkeypatch):
    providers(monkeypatch, ("alpha_vantage", payload(value=90)))
    svc.backfill_fundamentals("TEST", START)
    before = {r["id"]: r for r in all_rows(database)}
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert report["success"] and report["rows_quarantined"] == 36 == report["rows_inserted"]
    assert {q["reason"] for q in report["quarantined"]} == {"value_conflicts_with_primary"}
    for row_id, row in before.items():
        assert_quarantined_unchanged(database, row_id, row)
    [audit] = repairs(database, "fmp_primary_takeover")
    assert bk_fundamental_repair._digest(audit.plan) == audit.digest and audit.status == "applied"
    namespace = audit.plan["namespace"]
    assert namespace.startswith("~Q") and len(namespace) == 16
    assert {a["id"]: a["before"] for a in audit.plan["actions"]} == {
        i: {k: (v.isoformat() if isinstance(v, (date, datetime)) else v) for k, v in r.items()} for i, r in before.items()}
    assert audit.plan["memo_rows_modified"] == audit.plan["outcome_rows_modified"] == 0
    replacements = {row.id for row in live_rows(database)}
    assert set(audit.result["replacement_ids"].values()) == replacements
    carried = {(r["period"], r["statement"]): r["available_at"] for r in before.values()}
    assert all(carried[(r.period, r.statement)] == r.available_at for r in live_rows(database))
    assert {r.source for r in live_rows(database)} == {"fmp"} and {r.value for r in live_rows(database)} == {100}


def test_mdt_period_end_conflict_is_quarantined_and_fmp_date_stored(database, monkeypatch):
    legacy = {s: add(database, period="FY2024", period_end=date(2024, 4, 30), statement=s, line_item=p, value=5.0,
                     available_at=date(2024, 6, 20), available_at_source="provider")
              for s, p in svc.PRIMARY.items()}
    before = {i: fields(database, i) for i in legacy.values()}
    fmp = payload(extra={("FY2024", s): {"period_end": "2024-04-26"} for s in svc.PRIMARY})
    providers(monkeypatch, ("fmp", fmp))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert {q["id"]: q["reason"] for q in report["quarantined"]} == {
        i: "period_end_conflicts_with_primary" for i in legacy.values()}
    for row_id, row in before.items():
        assert_quarantined_unchanged(database, row_id, row)
    stored = live_rows(database, period="FY2024")
    assert {(r.period_end, r.source, r.value) for r in stored} == {(date(2024, 4, 26), "fmp", 100.0)}
    # Availability 2024-06-20 is after FMP's period end, so it is carried.
    assert {r.available_at for r in stored} == {date(2024, 6, 20)}


def test_undated_bare_year_aliases_are_quarantined(database, monkeypatch):
    aliases = [add(database, period=str(y), period_end=None, value=777.0, available_at=None) for y in (2023, 2024, 2025)]
    before = {i: fields(database, i) for i in aliases}
    assert any(i["kind"] == "invalid_stored_period" for i in svc.read_stored_financials("TEST")["_history_issues"])
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert report["success"]
    assert {q["id"]: q["reason"] for q in report["quarantined"]} == {i: "alias_superseded_by_primary" for i in aliases}
    for row_id, row in before.items():
        assert_quarantined_unchanged(database, row_id, row)
    assert "_history_issues" not in svc.read_stored_financials("TEST")


def test_lulu_label_duplicates_quarantined_and_impossible_availability_flagged(database, monkeypatch):
    # LULU: legacy FY2026 debt rows duplicate FMP FY2025 at 2026-02-01, and
    # FMP's own FY2025 rows carry availability before their period end.
    fmp_rows = [add(database, period="FY2025", period_end=date(2026, 2, 1), statement="balance", line_item=line,
                    value=0.0, source="fmp", available_at=date(2025, 4, 18), available_at_source="lag_rule")
                for line in ("short_term_debt", "total_debt")]
    legacy_rows = [add(database, period="FY2026", period_end=date(2026, 2, 1), statement="balance", line_item=line,
                       value=value, currency="EUR", available_at=date(2026, 3, 20)) for line, value in
                   (("short_term_debt", 5.0), ("total_debt", 7.0))]
    before = {i: fields(database, i) for i in fmp_rows + legacy_rows}
    fmp = payload(annual=False)
    for year in (2023, 2024, 2025):  # January fiscal year ends, FMP fiscalYear labels
        for statement, primary in svc.PRIMARY.items():
            row = {"period": f"FY{year}", "period_end": date(year + 1, 2, 1).isoformat(), "currency": "USD",
                   "fiscal_label_source": "provider", primary: 100.0}
            if statement == "balance" and year == 2025:
                row.update(short_term_debt=0.0, total_debt=0.0)
            fmp[statement].append(row)
    providers(monkeypatch, ("fmp", fmp))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert {q["id"]: q["reason"] for q in report["quarantined"]} == {
        i: "label_conflicts_with_primary_observation" for i in legacy_rows}
    for row_id in legacy_rows:
        assert_quarantined_unchanged(database, row_id, before[row_id])
    flags = [i for i in report["issues"] if i["kind"] == "primary_availability_precedes_period_end"]
    assert {f["id"] for f in flags} == set(fmp_rows)
    assert all(f["action"] == "flagged_not_moved" for f in flags) and not svc._has_blockers(flags)
    # Flagged, not moved and not re-inserted: stored exactly as before.
    for row_id in fmp_rows:
        assert fields(database, row_id) == {**before[row_id], "fetched_at": fields(database, row_id)["fetched_at"]}
    assert len(live_rows(database, period="FY2025", line_item="short_term_debt")) == 1


def test_secondary_rows_in_primary_periods_are_skipped_nonblocking(database, monkeypatch):
    fmp = payload()
    alpha = payload(value=90, extra={("FY2025", "income"): {"net_income": 12.0}})
    providers(monkeypatch, ("fmp", fmp))
    svc.backfill_fundamentals("TEST", START)
    before = all_rows(database)
    providers(monkeypatch, ("alpha_vantage", alpha))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert report["success"] and report["rows_written"] == 0 and report["secondary_rows_skipped"] == 36
    disagreements = [i for i in report["issues"] if i["kind"] == "secondary_disagrees_with_primary"]
    assert len(disagreements) == 36 and not svc._has_blockers(disagreements)
    # AV never fills a line inside a period FMP reports, even a line FMP lacks.
    assert all_rows(database) == before


def test_no_non_fmp_duplicate_left_in_fmp_periods(database, monkeypatch):
    # AV rows fill a period's extra line and relabel a period; after FMP takes
    # over, the raw readers (no precedence logic) see only FMP in FMP periods.
    providers(monkeypatch, ("alpha_vantage", payload(value=90, extra={
        ("FY2025", "income"): {"operating_income": 9.0}, ("2026Q2", "income"): {"operating_income": 3.0}})))
    svc.backfill_fundamentals("TEST", START)
    add(database, period="2025", period_end=date(2025, 12, 31), value=91.0, source="alpha_vantage",
        available_at=date(2026, 2, 1))
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert report["success"]
    reasons = {q["reason"] for q in report["quarantined"]}
    assert {"value_conflicts_with_primary", "secondary_row_in_primary_period", "alias_superseded_by_primary"} <= reasons
    with database() as db:
        history = history_service.get_financial_history("TEST", ["revenue", "operating_income"], limit=100, db=db)
    assert history["operating_income"] == []
    assert len(history["revenue"]) == 12 and {r["value"] for r in history["revenue"]} == {100.0}
    assert all(r.source == "fmp" for r in live_rows(database))
    # cycle_position's quarterly margin reader sees no AV operating income.
    assert cycle_position._quarterly_op_margin_series("TEST") == []


def test_secondary_fills_periods_primary_lacks(database, monkeypatch):
    # ARM shape: AV holds pre-listing periods FMP does not report.
    early = {s: [{"period": "FY2022", "period_end": "2022-12-31", "currency": "USD", p: 50.0}] for s, p in svc.PRIMARY.items()}
    providers(monkeypatch, ("alpha_vantage", early))
    svc.backfill_fundamentals("TEST", date(2022, 1, 1))
    kept = {r["id"] for r in all_rows(database)}
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert report["success"] and report["rows_quarantined"] == 0
    assert {r.id for r in live_rows(database, source="alpha_vantage")} == kept


def test_non_primary_pairs_keep_existing_conflict_rules(database, monkeypatch):
    providers(monkeypatch, ("first", payload()))
    svc.backfill_fundamentals("TEST", START)
    providers(monkeypatch, ("second", payload(value=250)))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert not report["success"] and report["rows_written"] == 0 and report["rows_quarantined"] == 0
    assert len([i for i in report["issues"] if i["kind"] == "stored_value_conflict"]) == 36
    assert repairs(database) == []


def test_legacy_nonconflicting_optional_line_is_retained(database, monkeypatch):
    kept = add(database, period="FY2025", period_end=date(2025, 12, 31), line_item="r_and_d", value=4.0)
    null_line = add(database, period="FY2025", period_end=date(2025, 12, 31), statement="balance",
                    line_item="goodwill", value=None)
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert report["success"] and report["rows_quarantined"] == 0
    assert fields(database, kept)["ticker"] == "TEST" and fields(database, null_line)["ticker"] == "TEST"
    fy2025 = next(r for r in svc.read_stored_financials("TEST")["income"] if r["period"] == "FY2025")
    assert fy2025["r_and_d"] == 4.0 and fy2025["line_sources"] == {"revenue": "fmp", "r_and_d": "live"}


def test_fetch_window_covers_old_named_secondary_rows(database, monkeypatch):
    add(database, period="FY2015", period_end=date(2015, 12, 31), value=1.0, source="alpha_vantage")
    calls = providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert calls[0][2] == date(2015, 12, 31) == date.fromisoformat(report["provider_requested_start"])


def test_dry_run_persists_nothing_but_reports_the_plan(database, monkeypatch):
    providers(monkeypatch, ("alpha_vantage", payload(value=90)))
    svc.backfill_fundamentals("TEST", START)
    add(database, period="2024", period_end=None, value=1.0)
    before = all_rows(database)
    with database() as db:
        repair_count = db.execute(select(func.count()).select_from(FinancialDataRepair)).scalar_one()
    providers(monkeypatch, ("fmp", payload(value=100)))
    report = svc.backfill_fundamentals("TEST", START, True, dry_run=True)
    assert report["dry_run"] and not report["committed"]
    assert report["rows_quarantined"] == 37 and report["rows_inserted"] == 36
    assert {q["reason"] for q in report["quarantined"]} == {"value_conflicts_with_primary", "alias_superseded_by_primary"}
    assert all_rows(database) == before
    with database() as db:
        assert db.execute(select(func.count()).select_from(FinancialDataRepair)).scalar_one() == repair_count
    with database() as db, pytest.raises(ValueError):
        svc.backfill_fundamentals("TEST", START, True, dry_run=True, db=db)


def test_quarantine_fence_rolls_back_on_concurrent_change(database, monkeypatch):
    providers(monkeypatch, ("alpha_vantage", payload(value=90)))
    svc.backfill_fundamentals("TEST", START)
    before = all_rows(database)
    original = fq._move

    def racing(db, row, **kwargs):
        # Another writer changes the row after it was read and judged.
        db.execute(update(FinancialPeriod).where(FinancialPeriod.id == row.id).values(value=12345.0)
                   .execution_options(synchronize_session=False))
        return original(db, row, **kwargs)

    monkeypatch.setattr(fq, "_move", racing)
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert not report["success"] and not report["committed"] and report["rows_quarantined"] == 0
    assert any(i["kind"] == "persistence_or_read_error" and i["error_type"] == "QuarantineFenceError" for i in report["issues"])
    assert all_rows(database) == before and repairs(database) == []
    # Nothing it listed persisted, so no consumer may count it as done.
    assert report["quarantined"] == [] and report["adoptions"] == [] and report["restatements"] == []
    assert len(report["rolled_back_plan"]["quarantined"]) == 36
    from app.services.fmp_repull_ledger import compact_report
    assert compact_report({"status": "incomplete", "fundamentals": report})["rows_quarantined"] == 0


def test_expected_quarantine_mismatch_aborts_whole_ticker(database, monkeypatch):
    providers(monkeypatch, ("alpha_vantage", payload(value=90)))
    svc.backfill_fundamentals("TEST", START)
    before = all_rows(database)
    providers(monkeypatch, ("fmp", payload()))
    reviewed = {r["id"]: "value_conflicts_with_primary" for r in before[:-1]}  # one row short
    plan = {"quarantine": reviewed, "primary_received": True}
    report = svc.backfill_fundamentals("TEST", START, True, expected_plan=plan)
    assert not report["success"] and report["rows_written"] == 0
    mismatch = next(i for i in report["issues"] if i["kind"] == "repull_plan_mismatch")
    assert mismatch["differs"] == ["quarantine"]
    assert set(mismatch["actual"]["quarantine"]) - set(mismatch["expected"]["quarantine"]) == {str(before[-1]["id"])}
    assert all_rows(database) == before and repairs(database) == []
    exact = {"quarantine": {r["id"]: "value_conflicts_with_primary" for r in before}, "primary_received": True}
    assert svc.backfill_fundamentals("TEST", START, True, expected_plan=exact)["rows_quarantined"] == 36


def test_unattended_label_shift_writes_planned_repair_and_moves_nothing(database, monkeypatch):
    legacy = [add(database, period="FY2024", period_end=date(2024, 4, 30), statement=s, line_item=p, value=5.0)
              for s, p in svc.PRIMARY.items()]
    providers(monkeypatch, ("fmp", payload(extra={("FY2024", s): {"period_end": "2024-04-26"} for s in svc.PRIMARY})))
    before = all_rows(database)
    report = svc.backfill_fundamentals("TEST", START, True, mode="unattended")
    assert not report["success"] and report["rows_quarantined"] == 0
    planned = next(i for i in report["issues"] if i["kind"] == "primary_takeover_planned")
    assert planned["reason"] == "unsafe_reasons" and planned["repair_id"] == report["planned_repair_id"]
    moved_or_changed = [r for r in all_rows(database) if r["id"] in legacy]
    assert moved_or_changed == [r for r in before if r["id"] in legacy]
    # FMP's replacement for the blocked key is not written beside the rows.
    assert {r.source for r in live_rows(database, period="FY2024")} == {"live"}
    [plan] = repairs(database, "fmp_primary_takeover")
    assert plan.status == "planned" and {a["id"] for a in plan.plan["actions"]} == set(legacy)
    # Reviewed and applied through the bk-repair apply path, fenced.
    result = fq.apply_planned(plan.id, plan.digest)
    assert result["rows_quarantined"] == 3
    for row in before:
        if row["id"] in legacy:
            assert_quarantined_unchanged(database, row["id"], row)
    later = svc.backfill_fundamentals("TEST", START, True, mode="unattended")
    assert later["success"] and {r.period_end for r in live_rows(database, period="FY2024")} == {date(2024, 4, 26)}


def test_unattended_small_safe_value_conflict_applies_immediately(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", START)
    drifted = add(database, period="FY2025", period_end=date(2025, 12, 31), line_item="net_income", value=3.0,
                  source="alpha_vantage")
    fmp = payload(extra={("FY2025", "income"): {"net_income": 4.0}})
    providers(monkeypatch, ("fmp", fmp))
    report = svc.backfill_fundamentals("TEST", START, True, mode="unattended")
    assert report["success"] and report["rows_quarantined"] == 1 and report["planned_repair_id"] is None
    assert fields(database, drifted)["ticker"].startswith("~Q")


def test_restore_quarantine_round_trip(database, monkeypatch):
    providers(monkeypatch, ("alpha_vantage", payload(value=90)))
    svc.backfill_fundamentals("TEST", START)
    before = all_rows(database)
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True)
    replacements = {r.id for r in live_rows(database)}
    result = fq.restore_quarantine(report["quarantine_repair_id"])
    assert result["status"] == "restored" and result["rows_restored"] == 36
    assert set(result["displaced_ids"]) == replacements
    assert [r for r in all_rows(database) if r["id"] not in replacements] == before
    for row_id in replacements:
        assert fields(database, row_id)["ticker"].startswith("~Q")  # moved aside, not deleted
    [original] = repairs(database, "fmp_primary_takeover")
    [displaced] = repairs(database, "restore_displaced")
    assert original.status == "restored" and displaced.status == "applied"
    assert displaced.plan["restores"] == original.id
    assert fq.restore_quarantine(report["quarantine_repair_id"])["already_restored"]


def test_restore_refuses_a_changed_quarantined_row(database, monkeypatch):
    providers(monkeypatch, ("alpha_vantage", payload(value=90)))
    svc.backfill_fundamentals("TEST", START)
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True)
    with database() as db:
        db.get(FinancialPeriod, report["quarantined"][0]["id"]).value = 1.0
        db.commit()
    snapshot = all_rows(database)
    with pytest.raises(fq.QuarantineFenceError):
        fq.restore_quarantine(report["quarantine_repair_id"])
    assert all_rows(database) == snapshot


def test_adoption_and_restatement_audited_and_restorable(database, monkeypatch):
    providers(monkeypatch, ("alpha_vantage", payload()))
    svc.backfill_fundamentals("TEST", START)
    original = {r["id"]: r for r in all_rows(database)}
    providers(monkeypatch, ("fmp", payload()))
    adopted = svc.backfill_fundamentals("TEST", START, True)
    providers(monkeypatch, ("fmp", payload(value=101)))
    restated = svc.backfill_fundamentals("TEST", START, True)
    assert len(restated["restatements"]) == 36 and restated["restatement_repair_id"]
    assert all(r["old_value"] == 100 and r["new_value"] == 101 for r in restated["restatements"])
    [audit] = repairs(database, "restatement")
    assert {a["before"]["value"] for a in audit.plan["actions"]} == {100.0}
    assert {a["after"]["value"] for a in audit.plan["actions"]} == {101.0}
    fq.restore_repair(restated["restatement_repair_id"])
    assert {r["value"] for r in all_rows(database)} == {100.0}
    fq.restore_repair(adopted["adoption_repair_id"])
    assert all_rows(database) == list(original.values())


def test_quarantined_rows_invisible_to_ticker_reads_and_pit_backfill(database, monkeypatch):
    alias = add(database, period="2024", period_end=None, value=1.0, available_at=None)
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", START, True)
    namespace = fields(database, alias)["ticker"]
    with database() as db:
        assert all(r["value"] == 100 for r in history_service.get_financial_history("TEST", ["revenue"], db=db)["revenue"])
        filled = scorecard_pit.backfill_available_at(db=db)
        db.commit()
        snapshot = scorecard_pit.snapshot_as_of("TEST", date(2026, 9, 13), db=db)
    assert filled["scanned"] == 0  # the NULL-availability quarantined row is excluded
    assert fields(database, alias)["available_at"] is None and fields(database, alias)["ticker"] == namespace
    assert all(r["value"] == 100 for r in snapshot["rows"])


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------

def test_read_precedence_prefers_primary(database):
    end = date(2026, 2, 1)
    fmp_id = add(database, period="FY2025", period_end=end, statement="balance", line_item="total_assets", value=100.0,
                 source="fmp")
    # R1: a named secondary line and a legacy line FMP reports, inside FMP's period.
    alpha_line = add(database, period="2025", period_end=end, statement="balance", line_item="goodwill", value=9.0,
                     source="alpha_vantage")
    legacy_line = add(database, period="2025", period_end=end, statement="balance", line_item="total_assets", value=99.0)
    # R2: another label at the same period end.
    other_label = add(database, period="FY2026", period_end=end, statement="balance", line_item="short_term_debt",
                      value=5.0, source="alpha_vantage")
    # R3: a legacy line under a stale label with two named observations of
    # its period end (FMP and AV); the primary one is its replacement.
    fmp_debt = add(database, period="FY2025", period_end=end, statement="balance", line_item="total_debt", value=7.0,
                   source="fmp")
    alpha_debt = add(database, period="2025", period_end=end, statement="balance", line_item="total_debt", value=7.5,
                     source="alpha_vantage")
    stale = add(database, period="FY2026", period_end=end, statement="balance", line_item="total_debt", value=6.0)
    before = all_rows(database)
    stored = svc.read_stored_financials("TEST")
    [row] = [r for r in stored["balance"] if r["period"] == "FY2025"]
    assert row["total_assets"] == 100.0 and row["total_debt"] == 7.0 and "goodwill" not in row
    assert row["line_sources"] == {"total_assets": "fmp", "total_debt": "fmp"}
    issues = stored["_history_issues"]
    superseded = {i["id"]: i for i in issues if i["kind"] == "superseded_by_primary"}
    assert set(superseded) == {alpha_line, legacy_line, alpha_debt} and superseded[legacy_line]["primary_value"] == 100.0
    assert [i["id"] for i in issues if i["kind"] == "label_superseded_by_primary"] == [other_label]
    [excluded] = [i for i in issues if i["kind"] == "legacy_duplicate_observation_excluded"]
    assert excluded["id"] == stale and excluded["replacement"]["id"] == fmp_debt
    assert not svc._has_blockers(issues)
    assert all_rows(database) == before and fmp_id


def test_bk_definition_break_flag_nonblocking_and_retires_when_value_changes(database):
    ids = [add(database, ticker="BK", period=period, period_end=end, line_item=line, value=value, source="fmp")
           for period, end, line, value in [
               ("2026Q1", date(2026, 3, 31), "revenue", 9.863e9),
               ("2026Q2", date(2026, 6, 30), "revenue", 5.698e9),
               ("2026Q2", date(2026, 6, 30), "eps_diluted", 2.43),
               ("2026Q2", date(2026, 6, 30), "weighted_avg_shares_diluted", 698.164e6)]]
    flags = [i for i in svc.read_stored_financials("BK")["_history_issues"] if i["kind"] == "provider_definition_break"]
    assert {(f["period"], f["line_item"]) for f in flags} == {
        ("2026Q1", "revenue"), ("2026Q2", "eps_diluted"), ("2026Q2", "weighted_avg_shares_diluted")}
    assert not svc._has_blockers(flags)
    assert all(f["evidence"] == "docs/reviews/2026-09-13-bk-provider-comparability.json" for f in flags)
    with database() as db:
        db.get(FinancialPeriod, ids[0]).value = 5.409e9  # FMP corrected to net revenue
        db.commit()
    flags = [i for i in svc.read_stored_financials("BK")["_history_issues"] if i["kind"] == "provider_definition_break"]
    assert ("2026Q1", "revenue") not in {(f["period"], f["line_item"]) for f in flags}


def test_entitlement_denial_under_one_spelling_is_resolved_by_another(database, monkeypatch):
    """BRK.B: FMP refuses the dotted spelling (402) and answers BRK-B. That is
    a symbol-coverage refusal, resolved; only unresolved denials are a gap."""
    def fetch(symbol, start):
        if symbol == "BRK.B":
            return {s: [] for s in svc.LINES} | {"_history_issues": [
                {"kind": "provider_entitlement_denied", "status": 402, "endpoint": "/income-statement",
                 "statement": "income", "cadence": cadence} for cadence in ("annual", "quarterly")]}
        return payload()
    monkeypatch.setattr(svc, "get_data_service", lambda: SimpleNamespace(
        _live_chain=lambda cap: [SimpleNamespace(name="fmp", get_financial_history=fetch)]))
    report = svc.backfill_fundamentals("BRK.B", START)
    denied = [i for i in report["issues"] if i["kind"] == "provider_entitlement_denied"]
    assert len(denied) == 2 and all(i["symbol"] == "BRK.B" and i["resolved"] and i["resolved_by_symbol"] == "BRK-B"
                                    for i in denied)
    assert report["success"]


def test_primary_provider_constant_is_shared_by_every_tie_break():
    assert svc.PRIMARY_PROVIDER == fq.PRIMARY_PROVIDER == "fmp"
    assert scorecard_features._PRIMARY_PROVIDER == fundamentals_series_service._PRIMARY_PROVIDER == svc.PRIMARY_PROVIDER


# ---------------------------------------------------------------------------
# Unattended guard limits, planned repairs, fences (review findings)
# ---------------------------------------------------------------------------

def test_unattended_safe_reasons_over_the_row_cap_are_planned(database, monkeypatch):
    providers(monkeypatch, ("alpha_vantage", payload(value=90)))
    svc.backfill_fundamentals("TEST", START)
    before = all_rows(database)
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True, mode="unattended")
    # 36 value conflicts: every reason is safe, but 36 > QUARANTINE_AUTO_MAX_ROWS.
    assert {q["reason"] for q in report["quarantined"]} <= svc.SAFE_AUTO_REASONS
    planned = next(i for i in report["issues"] if i["kind"] == "primary_takeover_planned")
    assert planned["reason"] == "exceeds_unattended_limits" and planned["rows"] == 36 > svc.QUARANTINE_AUTO_MAX_ROWS
    assert report["rows_quarantined"] == 0 and all_rows(database) == before
    [plan] = repairs(database, "fmp_primary_takeover")
    assert plan.status == "planned"


def test_unattended_safe_reasons_over_the_share_cap_are_planned_and_block(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", START)
    drifted = [add(database, period=p, period_end=date(int(p[2:]), 12, 31), line_item="net_income", value=3.0,
                   source="alpha_vantage") for p in ("FY2023", "FY2024", "FY2025")]
    before = all_rows(database)
    fmp = payload(extra={(p, "income"): {"net_income": 4.0} for p in ("FY2023", "FY2024", "FY2025")})
    providers(monkeypatch, ("fmp", fmp))
    report = svc.backfill_fundamentals("TEST", START, True, mode="unattended")
    # 3 rows is under the 20-row cap but over 5% of the ticker's 39 rows.
    assert len(before) == 39 and 3 > svc.QUARANTINE_AUTO_MAX_SHARE * 39
    planned = next(i for i in report["issues"] if i["kind"] == "primary_takeover_planned")
    assert planned["reason"] == "exceeds_unattended_limits" and planned["rows"] == 3
    assert all(fields(database, i)["ticker"] == "TEST" for i in drifted)
    # Coverage is complete and fresh: the planned repair alone blocks success.
    blocking = {i["kind"] for i in report["issues"] if not i.get("resolved") and i["kind"] in svc.BLOCKING_ISSUES}
    assert blocking == {"primary_takeover_planned"} and not report["success"]


def test_apply_planned_refuses_wrong_digest_changed_row_and_non_planned_status(database, monkeypatch):
    legacy = [add(database, period="FY2024", period_end=date(2024, 4, 30), statement=s, line_item=p, value=5.0)
              for s, p in svc.PRIMARY.items()]
    providers(monkeypatch, ("fmp", payload(extra={("FY2024", s): {"period_end": "2024-04-26"} for s in svc.PRIMARY})))
    svc.backfill_fundamentals("TEST", START, True, mode="unattended")
    [plan] = repairs(database, "fmp_primary_takeover")
    snapshot = all_rows(database)
    with pytest.raises(RuntimeError, match="digest"):
        fq.apply_planned(plan.id, "0" * 64)
    assert all_rows(database) == snapshot and repairs(database, "fmp_primary_takeover")[0].status == "planned"
    with database() as db:
        db.get(FinancialPeriod, legacy[0]).value = 6.0  # changed after review
        db.commit()
    changed = all_rows(database)
    with pytest.raises(fq.QuarantineFenceError):
        fq.apply_planned(plan.id, plan.digest)
    assert all_rows(database) == changed and repairs(database, "fmp_primary_takeover")[0].status == "planned"
    with database() as db:
        db.get(FinancialPeriod, legacy[0]).value = 5.0
        db.commit()
    assert fq.apply_planned(plan.id, plan.digest)["rows_quarantined"] == 3
    fq.restore_repair(plan.id)
    restored = all_rows(database)
    with pytest.raises(RuntimeError, match="planned"):
        fq.apply_planned(plan.id, plan.digest)  # a restored plan is never re-applied
    assert all_rows(database) == restored


def test_bk_repair_apply_route_dispatches_planned_takeovers_and_refuses_records(database, monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "admin_api_token", "test-takeover-token")
    http, headers = TestClient(app), {"Authorization": "Bearer test-takeover-token"}
    for statement, line in svc.PRIMARY.items():
        add(database, period="FY2024", period_end=date(2024, 4, 30), statement=statement, line_item=line, value=5.0)
    shifted = payload(extra={("FY2024", statement): {"period_end": "2024-04-26"} for statement in svc.PRIMARY})
    providers(monkeypatch, ("fmp", shifted))
    svc.backfill_fundamentals("TEST", START, True, mode="unattended")
    [plan] = repairs(database, "fmp_primary_takeover")
    url = "/api/admin/market-data/bk-repair/{}/apply"
    assert http.post(url.format(plan.id), headers=headers, json={"digest": "0" * 64}).status_code == 409
    applied = http.post(url.format(plan.id), headers=headers, json={"digest": plan.digest})
    assert applied.status_code == 200 and applied.json()["rows_quarantined"] == 3
    # Ledgers and in-place audits are records, not applicable plans.
    svc.backfill_fundamentals("TEST", START, True)
    providers(monkeypatch, ("fmp", payload(value=101, extra={
        ("FY2024", statement): {"period_end": "2024-04-26"} for statement in svc.PRIMARY})))
    svc.backfill_fundamentals("TEST", START, True)
    [audit] = repairs(database, "restatement")
    with database() as db:
        db.add(FinancialDataRepair(id="fmp-repull-2026-09-dry-run", digest="d" * 64, status="complete",
                                   plan={"kind": "fmp_repull_ledger"}, result={}, created_at=datetime(2026, 9, 24)))
        db.commit()
    snapshot = all_rows(database)
    for record_id, digest in (("fmp-repull-2026-09-dry-run", "d" * 64), (audit.id, audit.digest)):
        assert http.post(url.format(record_id), headers=headers, json={"digest": digest}).status_code == 400
    assert all_rows(database) == snapshot


def test_restore_of_an_adoption_refuses_a_row_restated_since(database, monkeypatch):
    providers(monkeypatch, ("alpha_vantage", payload()))
    svc.backfill_fundamentals("TEST", START)
    providers(monkeypatch, ("fmp", payload()))
    adopted = svc.backfill_fundamentals("TEST", START, True)
    providers(monkeypatch, ("fmp", payload(value=101)))
    svc.backfill_fundamentals("TEST", START, True)
    snapshot = all_rows(database)
    # The adoption's after-image (fmp, 100) no longer matches: restoring it
    # would overwrite FMP's later restatement with the stale before-image.
    with pytest.raises(fq.QuarantineFenceError):
        fq.restore_repair(adopted["adoption_repair_id"])
    assert all_rows(database) == snapshot
    assert repairs(database, "primary_adoption")[0].status == "applied"


def test_restoring_an_adoption_also_reverses_its_relabel(database, monkeypatch):
    # A legacy FY2024 row whose period end is FMP's FY2025 is relabelled to
    # FY2025 and adopted in the same run; FMP's own FY2024 then takes the
    # old key. Restore puts the row back as it was and moves FMP's aside.
    legacy = add(database, period="FY2024", period_end=date(2025, 12, 31), value=100.0,
                 available_at=date(2026, 2, 15), available_at_source="provider")
    original = fields(database, legacy)
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert [(r["id"], r["old_period"], r["new_period"]) for r in report["period_relabels"]] == [
        (legacy, "FY2024", "FY2025")]
    assert legacy in {a["id"] for a in report["adoptions"]}
    [audit] = repairs(database, "primary_adoption")
    assert next(a for a in audit.plan["actions"] if a["id"] == legacy)["before"]["period"] == "FY2024"
    [fmp_fy2024] = live_rows(database, period="FY2024", statement="income", line_item="revenue")
    result = fq.restore_repair(report["adoption_repair_id"])
    assert fields(database, legacy) == original
    assert result["displaced_ids"] == [fmp_fy2024.id] and fields(database, fmp_fy2024.id)["ticker"].startswith("~Q")
    [displaced] = repairs(database, "restore_displaced")
    assert displaced.plan["restores"] == report["adoption_repair_id"]


def test_executed_plan_must_match_adoptions_restatements_and_relabels(database, monkeypatch):
    legacy = add(database, period="FY2024", period_end=date(2025, 12, 31), value=100.0)
    providers(monkeypatch, ("fmp", payload()))
    dry = svc.backfill_fundamentals("TEST", START, True, dry_run=True)
    plan = svc.plan_identity(dry)
    assert plan["relabels"] == {str(legacy): "FY2025"} and plan["adoptions"] == [legacy]
    assert plan["primary_received"] is True
    before = all_rows(database)
    for component, wrong in (("relabels", {}), ("adoptions", []), ("restatements", {str(legacy): 7.0})):
        report = svc.backfill_fundamentals("TEST", START, True, expected_plan={**plan, component: wrong})
        mismatch = next(i for i in report["issues"] if i["kind"] == "repull_plan_mismatch")
        assert component in mismatch["differs"] and not report["success"]
        assert all_rows(database) == before and repairs(database) == []
    assert svc.backfill_fundamentals("TEST", START, True, expected_plan=plan)["success"]
    assert fields(database, legacy)["period"] == "FY2025"


def test_named_currency_conflict_is_a_safe_unattended_quarantine(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload()))
    svc.backfill_fundamentals("TEST", START)
    eur = add(database, period="FY2025", period_end=date(2025, 12, 31), line_item="net_income", value=4.0,
              currency="EUR", source="alpha_vantage")
    providers(monkeypatch, ("fmp", payload(extra={("FY2025", "income"): {"net_income": 4.0}})))
    report = svc.backfill_fundamentals("TEST", START, True, mode="unattended")
    assert {q["id"]: q["reason"] for q in report["quarantined"]} == {eur: "currency_conflicts_with_primary"}
    assert report["planned_repair_id"] is None and fields(database, eur)["ticker"].startswith("~Q")
    [replacement] = live_rows(database, period="FY2025", line_item="net_income")
    assert (replacement.source, replacement.currency, replacement.value) == ("fmp", "USD", 4.0)


def test_invalid_availability_is_a_safe_unattended_quarantine_never_carried(database, monkeypatch):
    providers(monkeypatch, ("fmp", payload(skip=("FY2025",))))
    svc.backfill_fundamentals("TEST", START)
    impossible = date(2025, 6, 1)  # before the FY2025 period end
    equal = add(database, period="FY2025", period_end=date(2025, 12, 31), value=100.0, source="alpha_vantage",
                available_at=impossible, available_at_source="provider")
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True, mode="unattended")
    assert {q["id"]: q["reason"] for q in report["quarantined"]} == {equal: "invalid_availability_precedes_period_end"}
    assert report["planned_repair_id"] is None and fields(database, equal)["ticker"].startswith("~Q")
    [replacement] = live_rows(database, period="FY2025", line_item="revenue")
    # FMP's filing date (period end + 30 days), not the impossible date.
    assert (replacement.source, replacement.available_at) == ("fmp", date(2026, 1, 30))


def test_value_conflict_replacement_does_not_carry_an_invalid_availability(database, monkeypatch):
    conflict = add(database, period="FY2025", period_end=date(2025, 12, 31), statement="balance",
                   line_item="total_assets", value=90.0, source="alpha_vantage", available_at=date(2025, 6, 1),
                   available_at_source="provider")
    valid = add(database, period="FY2024", period_end=date(2024, 12, 31), value=90.0, source="alpha_vantage",
                available_at=date(2025, 2, 20), available_at_source="provider")
    providers(monkeypatch, ("fmp", payload()))
    report = svc.backfill_fundamentals("TEST", START, True)
    assert {q["id"]: q["reason"] for q in report["quarantined"]} == {
        conflict: "value_conflicts_with_primary", valid: "value_conflicts_with_primary"}
    [invalid_replacement] = live_rows(database, period="FY2025", statement="balance", line_item="total_assets")
    [carried] = live_rows(database, period="FY2024", statement="income", line_item="revenue")
    assert invalid_replacement.available_at == date(2026, 1, 30)
    assert carried.available_at == date(2025, 2, 20)  # a valid original availability is carried


def test_entitlement_denials_resolve_only_under_another_spelling_of_the_same_provider(database, monkeypatch):
    from app.services.fmp_repull_ledger import compact_report

    def denial(statement, cadence="quarterly"):
        return {"kind": "provider_entitlement_denied", "status": 402, "statement": statement, "cadence": cadence,
                "endpoint": {"income": "/income-statement", "cash": "/cash-flow-statement"}[statement]}

    def served(symbol, start):
        if symbol == "BRK.B":
            return {s: [] for s in svc.LINES} | {"_history_issues": [denial("income", "annual")]}
        return payload(quarterly=False) | {"_history_issues": [denial("cash")]}

    fallback = payload()
    chain = [SimpleNamespace(name="fmp", get_financial_history=served),
             SimpleNamespace(name="alpha_vantage", get_financial_history=lambda symbol, start: fallback)]
    monkeypatch.setattr(svc, "get_data_service", lambda: SimpleNamespace(_live_chain=lambda cap: chain))
    single = svc.backfill_fundamentals("TEST", START, dry_run=True)
    [cash] = [i for i in single["issues"] if i["kind"] == "provider_entitlement_denied"]
    # The same spelling answered other statements: a real endpoint gap. A
    # fallback provider's answer does not resolve FMP's refusal either.
    assert cash["symbol"] == "TEST" and not cash.get("resolved")
    assert any(a["provider"] == "alpha_vantage" and a["received"] for a in single["attempts"])
    assert compact_report({"dry_run": True, "fundamentals": single})["entitlement_denied"] == [
        "/cash-flow-statement:quarterly:402:TEST"]
    brk = svc.backfill_fundamentals("BRK.B", START, dry_run=True)
    denied = {i["symbol"]: i for i in brk["issues"] if i["kind"] == "provider_entitlement_denied"}
    assert denied["BRK.B"]["resolved_by_symbol"] == "BRK-B" and not denied["BRK-B"].get("resolved")
    record = compact_report({"dry_run": True, "fundamentals": brk})
    assert record["entitlement_denied"] == ["/cash-flow-statement:quarterly:402:BRK-B"]
    assert record["entitlement_resolved_by_symbol"] == ["/income-statement:BRK.B->BRK-B"]
