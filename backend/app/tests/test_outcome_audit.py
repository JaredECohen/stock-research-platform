"""Audit real stored rows without trusting the weekday heuristic as proof."""
from datetime import UTC, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models import MemoOutcome, MemoSnapshot
from app.services.outcome_audit import _classify, audit_outcomes

GENERATED = datetime(2026, 1, 1, 23, 30)


@pytest.fixture
def db():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _seed(db, *, horizon=30, elapsed=timedelta(days=100), note="", orphan=False, version=1):
    snap = MemoSnapshot(ticker="AUDIT", version=version, generated_at=GENERATED, memo_json={})
    if not orphan:
        db.add(snap)
        db.flush()
    outcome = MemoOutcome(
        memo_snapshot_id=999999 if orphan else snap.id, ticker="AUDIT",
        horizon_days=horizon, evaluated_at=GENERATED + elapsed,
        price_at_memo=100, forward_return=-0.2, benchmark_return=0.1,
        alpha=-0.3, thesis_held=False, note=note,
    )
    db.add(outcome)
    db.commit()
    return outcome


def _raw(*, horizon=30, generated=GENERATED, evaluated=None, note=""):
    return {
        "generated_at": generated, "evaluated_at": evaluated or generated + timedelta(days=100),
        "horizon_days": horizon, "snapshot_id": 1, "snapshot_ticker": "AUDIT",
        "ticker": "AUDIT", "as_of_date": None, "note": note,
    }


@pytest.mark.parametrize("horizon,threshold", [(30, 84), (90, 168), (180, 294), (365, 553)])
@pytest.mark.parametrize("microseconds,expected", [(-1, False), (0, False), (1, True)])
def test_strict_threshold_keeps_sub_day_and_microsecond_precision(horizon, threshold, microseconds, expected):
    row = _classify(_raw(
        horizon=horizon,
        evaluated=GENERATED + timedelta(days=threshold, microseconds=microseconds),
    ))
    assert row["threshold_days"] == threshold
    assert row["late_evaluation_candidate"] is expected


def test_aware_offsets_and_naive_utc_are_the_same_instant():
    utc = UTC
    ny = timezone(timedelta(hours=-4))
    generated = datetime(2026, 3, 7, 23, 30, tzinfo=ny)
    evaluated = generated.astimezone(utc) + timedelta(days=84, microseconds=1)
    aware = _classify(_raw(generated=generated, evaluated=evaluated))
    naive = _classify(_raw(generated=generated.astimezone(utc).replace(tzinfo=None), evaluated=evaluated))
    assert aware == naive
    assert aware["generated_at"] == "2026-03-08T03:30:00Z"
    assert aware["late_evaluation_candidate"] is True


@pytest.mark.parametrize("offset,expected", [(-8, True), (-7, False), (7, False), (8, True)])
def test_recorded_baseline_uses_independent_date_evidence(offset, expected):
    baseline = (GENERATED.date() + timedelta(days=offset)).isoformat()
    row = _classify(_raw(evaluated=GENERATED + timedelta(days=30), note=f"horizon=30d, baseline={baseline}"))
    assert row["late_evaluation_candidate"] is False
    assert row["recorded_baseline_offset_days"] == offset
    assert (row["status"] == "candidate") is expected


def test_missing_invalid_and_backtest_metadata_are_explicit():
    for overrides, reason in [
        ({"snapshot_id": None, "generated_at": None}, "missing_snapshot"),
        ({"evaluated_at": None}, "missing_or_invalid_timestamp"),
        ({"horizon_days": 0}, "invalid_horizon"),
        ({"evaluated_at": GENERATED - timedelta(seconds=1)}, "evaluated_before_generated"),
        ({"note": "baseline=not-a-date"}, "invalid_recorded_baseline"),
        ({"as_of_date": GENERATED}, "backtest_outcome"),
        ({"snapshot_ticker": "WRONG"}, "snapshot_ticker_mismatch"),
    ]:
        row = _classify({**_raw(evaluated=GENERATED + timedelta(days=30)), **overrides})
        assert row["status"] == "indeterminate"
        assert reason in row["reasons"]


def test_every_outcome_including_orphans_is_returned_without_writes(db):
    # More than the original production census and common default caps.
    for version in range(1, 712):
        _seed(db, version=version)
    orphan = _seed(db, orphan=True)
    before = db.execute(select(*MemoOutcome.__table__.c)).all()
    statements = []

    def select_only(conn, cursor, statement, parameters, context, executemany):
        assert statement.lstrip().upper().startswith("SELECT"), statement
        statements.append(statement)

    event.listen(db.get_bind(), "before_cursor_execute", select_only)
    try:
        report = audit_outcomes(db)
    finally:
        event.remove(db.get_bind(), "before_cursor_execute", select_only)
    assert len(statements) == 1
    assert "memo_json" not in statements[0]
    assert report["total_rows"] == report["returned_rows"] == len(report["rows"]) == 712
    assert report["counts"]["candidate"] == 711
    assert report["counts"]["missing_snapshot"] == 1
    assert report["counts"]["indeterminate"] == 1
    assert report["excluded_rows"] == 0 and report["truncated"] is False
    assert [row["id"] for row in report["rows"]] == [row.id for row in before]
    missing = next(row for row in report["rows"] if row["id"] == orphan.id)
    assert missing["generated_at"] is None
    assert missing["late_evaluation_candidate"] is None
    assert db.execute(select(*MemoOutcome.__table__.c)).all() == before


def test_empty_database_has_explicit_zero_counts(db):
    report = audit_outcomes(db)
    assert report["total_rows"] == 0
    assert report["rows"] == []
    assert all(value == 0 for value in report["counts"].values())


def test_weekday_model_has_no_missed_drift_over_tolerance_but_partial_response_does():
    # Independent replay: build actual weekday tapes and apply the OLD
    # baseline fallback, rather than restating the threshold formula.
    origin = GENERATED.date() - timedelta(days=1000)
    tape = [origin + timedelta(days=n) for n in range(2500)
            if (origin + timedelta(days=n)).weekday() < 5]
    checked = contaminated = false_positives = 0
    for weekday in range(7):
        generated = GENERATED + timedelta(days=weekday)
        for horizon in (30, 90, 180, 365):
            for age in range(horizon, horizon + 250):
                evaluated = generated + timedelta(days=age)
                rows = [d for d in tape if d <= evaluated.date()][-(horizon + 30):]
                baseline = next((d for d in rows if d >= generated.date()), None)
                target = next((d for d in reversed(rows) if d <= generated.date() + timedelta(days=horizon)), None)
                if baseline is None or target is None:
                    continue  # old evaluator would not have persisted a row
                drifted = (baseline - generated.date()).days > 7
                flagged = _classify(_raw(horizon=horizon, generated=generated, evaluated=evaluated))["late_evaluation_candidate"]
                assert not drifted or flagged
                checked += 1
                contaminated += int(drifted)
                false_positives += int(flagged and not drifted)
    assert checked > 1000 and contaminated > 100 and false_positives > 0

    # A short fallback can start well after the memo even at 45 days old.
    # Prove why the endpoint must never call below-threshold rows "clean".
    evaluated = GENERATED + timedelta(days=45)
    rows = [d for d in tape if d <= evaluated.date()][-20:]
    baseline = next(d for d in rows if d >= GENERATED.date())
    assert (baseline - GENERATED.date()).days > 7
    assert baseline <= GENERATED.date() + timedelta(days=30)
    assert _classify(_raw(evaluated=evaluated))["late_evaluation_candidate"] is False


def test_admin_endpoint_is_guarded_and_serves_actual_audit(db, monkeypatch):
    from app.config import settings
    from app.main import app

    _seed(db)
    monkeypatch.setattr(settings, "admin_api_token", "test-audit-token")
    app.dependency_overrides[get_db] = lambda: db
    try:
        # Schema is initialized by the fixture; the handler needs no worker
        # or universe seeding to exercise its actual authenticated HTTP path.
        client = TestClient(app)
        assert client.get("/api/admin/outcome-audit").status_code == 401
        response = client.get("/api/admin/outcome-audit", headers={"Authorization": "Bearer test-audit-token"})
        assert response.status_code == 200
        report = response.json()
        assert report["read_only"] is True
        assert report["counts"]["late_evaluation_candidates"] == 1
        assert "not certified clean" in " ".join(report["limitations"])
        assert report["rows"][0]["alpha"] == -0.3
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_audit_reports_eligibility_without_filtering(db):
    """W6: every row is listed WITH its eligibility; nothing is filtered out.

    Still one SELECT and still no memo body, so the before/after receipt can
    prove exclusion changed no stored outcome.
    """
    from app.services.outcome_eligibility import REASON_DEV_COPY, REASON_LIVE
    from app.tests.eligibility_helpers import mark

    eligible = _seed(db, version=1)
    excluded = _seed(db, version=2)
    unclassified = _seed(db, version=3)
    mark(db, eligible.memo_snapshot_id)
    mark(db, excluded.memo_snapshot_id, eligible=False, reason=REASON_DEV_COPY)
    statements = []

    def select_only(conn, cursor, statement, parameters, context, executemany):
        assert statement.lstrip().upper().startswith("SELECT"), statement
        statements.append(statement)

    event.listen(db.get_bind(), "before_cursor_execute", select_only)
    try:
        report = audit_outcomes(db)
    finally:
        event.remove(db.get_bind(), "before_cursor_execute", select_only)
    assert len(statements) == 1 and "memo_json" not in statements[0]
    assert report["total_rows"] == 3
    rows = {row["id"]: row for row in report["rows"]}
    assert (rows[eligible.id]["track_record_eligible"], rows[eligible.id]["eligibility_reason"]) == (True, REASON_LIVE)
    assert rows[eligible.id]["generation_mode_observed"] == "live"
    assert (rows[excluded.id]["track_record_eligible"], rows[excluded.id]["eligibility_reason"]) == (
        False, REASON_DEV_COPY)
    assert rows[unclassified.id]["track_record_eligible"] is None
    tre = report["track_record_eligibility"]
    assert (tre["eligible"], tre["excluded"], tre["unclassified"]) == (1, 1, 1)
    assert tre["by_reason"] == {REASON_DEV_COPY: 1, REASON_LIVE: 1, "unclassified": 1}
    # The enumerated-set check (this sqlite has none of the production ids).
    check = tre["dev_copy_set"]
    assert check["expected_snapshots"] == 300 and check["matches"] is False
    assert check["unexpected_snapshot_ids"] == [excluded.memo_snapshot_id]
    assert "eligible rows (W6)" in report["limitations"][-1]
    assert set(report["counts"]) == {
        "candidate", "indeterminate", "not_flagged", "late_evaluation_candidates", "missing_snapshot",
        "recorded_baseline_outside_tolerance", "without_baseline_metadata",
    }


def test_dev_copy_receipt_matches_missing_and_ignores_inherited_patches(db):
    """The post-deploy receipt: the directly classified dev-copy set equals
    the enumerated 300 snapshots / 600 rows; a snapshot whose outcomes are
    absent is named as missing; a patch that INHERITS the dev-copy reason is
    neither observed nor unexpected (it is outside the 300/600 count)."""
    from sqlalchemy import delete

    from app.services import outcome_eligibility
    from app.services.outcome_eligibility_evidence import DEV_COPY_SNAPSHOTS
    from app.tests.eligibility_helpers import add_outcome, add_snapshot

    snaps = {}
    for sid, ticker, generated in DEV_COPY_SNAPSHOTS:
        at = datetime.fromisoformat(generated)
        snap = add_snapshot(db, id=sid, ticker=ticker, version=sid, generated_at=at, mode="demo",
                            memo_generated_at=generated)
        for horizon in (30, 90):
            add_outcome(db, snap, horizon=horizon, forward_return=0.1, alpha=0.02)
        snaps[sid] = snap
    parent_id, parent_ticker, _ = DEV_COPY_SNAPSHOTS[0]
    patch = add_snapshot(db, id=900, ticker=parent_ticker, version=100000, parent_version=parent_id,
                         trigger="incremental_patch", generated_at=datetime(2026, 6, 20), mode="live")
    add_outcome(db, patch, horizon=30, forward_return=0.1, alpha=0.02)
    db.commit()
    summary = outcome_eligibility.classify_pending(db=db)
    assert summary["by_reason"] == {outcome_eligibility.REASON_DEV_COPY: 301}

    check = audit_outcomes(db)["track_record_eligibility"]["dev_copy_set"]
    assert check == {
        "expected_snapshots": 300, "observed_snapshots": 300, "observed_rows": 600,
        "matches": True, "missing_snapshot_ids": [], "unexpected_snapshot_ids": [],
    }
    rows = {row["memo_snapshot_id"]: row for row in audit_outcomes(db)["rows"]}
    assert rows[patch.id]["eligibility_inherited_from"] == parent_id

    # A superset is not a match: one extra direct dev-copy row on top of the
    # full set (only a hand-written ledger row can do this; the sweep refuses).
    from app.tests.eligibility_helpers import mark
    extra = add_snapshot(db, id=901, ticker="EXTRA", generated_at=datetime(2026, 5, 4), mode="demo")
    add_outcome(db, extra, horizon=30, forward_return=0.1, alpha=0.02)
    db.commit()
    mark(db, extra.id, eligible=False, reason=outcome_eligibility.REASON_DEV_COPY)
    check = audit_outcomes(db)["track_record_eligibility"]["dev_copy_set"]
    assert check["matches"] is False and check["unexpected_snapshot_ids"] == [extra.id]
    db.execute(delete(MemoOutcome).where(MemoOutcome.memo_snapshot_id == extra.id))

    dropped = DEV_COPY_SNAPSHOTS[-1][0]
    db.execute(delete(MemoOutcome).where(MemoOutcome.memo_snapshot_id == dropped))
    db.commit()
    check = audit_outcomes(db)["track_record_eligibility"]["dev_copy_set"]
    assert check["matches"] is False
    assert (check["observed_snapshots"], check["observed_rows"]) == (299, 598)
    assert check["missing_snapshot_ids"] == [dropped]
    assert check["unexpected_snapshot_ids"] == []
