"""Wave 4A tests — realized-outcome tracking.

Covers:
- `_thesis_held` direction logic (Bullish wants positive return, Bearish
  wants negative; Neutral returns None).
- `_close_on_or_before` / `_close_on_or_after` price helpers handle
  empty / future-only / missing cases cleanly.
- `evaluate_all_due` is idempotent — running twice yields zero new rows.
- Backtest snapshots (`as_of_date` set) are skipped.
- Horizons that haven't come of age are skipped.
- `track_record` filters + aggregates correctly.
- Reflection entries are written to the company memory file for the
  long horizons (90d / 365d) but NOT for short horizons (30d / 180d).
- Admin endpoint serves the track-record query.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select

from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import Company, MemoOutcome, MemoPostmortem, MemoSnapshot
from app.services import outcome_eligibility as oe
from app.services import outcome_service
from app.services.outcome_eligibility_evidence import DEV_COPY_SNAPSHOTS
from app.tests.eligibility_helpers import add_outcome, add_snapshot, classify_all, isolated_sessions, mark


def _seed_snapshot(
    ticker: str = "TSTONE", *, version: int = 1,
    rating: str = "Bullish", confidence: float = 70.0,
    days_ago: int = 120, as_of_date=None, regime: str = "",
) -> MemoSnapshot:
    """Insert a memo snapshot N days ago for outcome testing."""
    with SessionLocal() as db:
        outcome_service._ensure_table(db)
        from app.services.memo_store import _ensure_table as _ensure_memo
        _ensure_memo(db)
        # Clear any prior rows so the test stays deterministic.
        db.query(MemoOutcome).filter(MemoOutcome.ticker == ticker).delete()
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker == ticker).delete()
        snap = MemoSnapshot(
            ticker=ticker, version=version, parent_version=None,
            trigger="first_run",
            memo_json={
                "ticker": ticker, "rating_label": rating,
                "confidence_score": confidence, "sector": "Technology",
                # W6: eligibility is fail-closed on generation_mode.
                "generation_mode": "live",
                "macro_regime_at_memo": regime,
                "agent_influence": {"valuation": 0.5},
            },
            revision_log=[], generated_at=datetime.utcnow() - timedelta(days=days_ago),
            as_of_date=as_of_date,
        )
        db.add(snap)
        db.commit()
        db.refresh(snap)
        db.expunge(snap)
        return snap


def _stub_prices(rows_by_ticker: dict[str, list[dict[str, Any]]]):
    """Patch market_data_service.get_price_series to return stub data."""
    from app.services import market_data_service
    def fake(ticker: str, days: int = 252):
        return rows_by_ticker.get(ticker.upper(), [])
    return patch.object(market_data_service, "get_price_series", side_effect=fake)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_thesis_held_directions():
    assert outcome_service._thesis_held("Bullish", 0.10) is True
    assert outcome_service._thesis_held("Bullish", -0.10) is False
    assert outcome_service._thesis_held("Bearish", -0.05) is True
    assert outcome_service._thesis_held("Bearish", 0.05) is False
    assert outcome_service._thesis_held("Mixed Positive", 0.01) is True
    assert outcome_service._thesis_held("Neutral", 0.10) is None
    assert outcome_service._thesis_held("", 0.0) is None


def test_close_on_or_before_picks_latest_in_window():
    rows = [
        {"date": "2024-01-01", "close": 100.0},
        {"date": "2024-01-15", "close": 105.0},
        {"date": "2024-02-01", "close": 110.0},
    ]
    assert outcome_service._close_on_or_before(rows, "2024-01-20") == 105.0
    assert outcome_service._close_on_or_before(rows, "2023-12-01") is None


def test_close_on_or_after_picks_earliest_in_window():
    rows = [
        {"date": "2024-01-01", "close": 100.0},
        {"date": "2024-02-01", "close": 110.0},
    ]
    assert outcome_service._close_on_or_after(rows, "2024-01-15") == 110.0
    assert outcome_service._close_on_or_after(rows, "2024-03-01") is None


# ---------------------------------------------------------------------------
# evaluate_all_due
# ---------------------------------------------------------------------------

def test_evaluate_all_due_writes_outcomes_for_due_horizons():
    snap = _seed_snapshot(
        "TSTONE", days_ago=120, rating="Bullish", regime="soft_landing",
    )
    today = (snap.generated_at + timedelta(days=120)).date()
    g_iso = snap.generated_at.date().isoformat()
    target_30 = (snap.generated_at.date() + timedelta(days=30)).isoformat()
    target_90 = (snap.generated_at.date() + timedelta(days=90)).isoformat()
    prices = {
        "TSTONE": [
            {"date": g_iso, "close": 100.0},
            {"date": target_30, "close": 110.0},   # +10% at 30d
            {"date": target_90, "close": 120.0},   # +20% at 90d
            {"date": today.isoformat(), "close": 130.0},
        ],
        "SPY": [
            {"date": g_iso, "close": 500.0},
            {"date": target_30, "close": 510.0},   # +2%
            {"date": target_90, "close": 530.0},   # +6%
            {"date": today.isoformat(), "close": 540.0},
        ],
    }
    with _stub_prices(prices):
        res = outcome_service.evaluate_all_due(today=today)
    assert res["written"] >= 2  # at least 30d + 90d
    rows = outcome_service.get_outcomes_for_snapshot(snap.id)
    horizons = {r["horizon_days"] for r in rows}
    assert {30, 90} <= horizons
    # 365d horizon hasn't come of age (today is 120 days post-memo) → not written.
    assert 365 not in horizons
    by_h = {r["horizon_days"]: r for r in rows}
    assert by_h[30]["forward_return"] == 0.10
    assert by_h[30]["thesis_held"] is True
    assert by_h[90]["forward_return"] == 0.20
    assert by_h[30]["regime_at_memo"] == "soft_landing"
    # Alpha = ticker − benchmark.
    assert abs(by_h[30]["alpha"] - (0.10 - 0.02)) < 1e-9


def test_evaluate_all_due_idempotent_on_second_run():
    snap = _seed_snapshot("TSTONE2", days_ago=100, rating="Bullish")
    today = (snap.generated_at + timedelta(days=100)).date()
    g_iso = snap.generated_at.date().isoformat()
    prices = {
        "TSTONE2": [
            {"date": g_iso, "close": 100.0},
            {"date": (snap.generated_at.date() + timedelta(days=30)).isoformat(), "close": 105.0},
            {"date": (snap.generated_at.date() + timedelta(days=90)).isoformat(), "close": 115.0},
        ],
        "SPY": [
            {"date": g_iso, "close": 500.0},
            {"date": (snap.generated_at.date() + timedelta(days=30)).isoformat(), "close": 510.0},
            {"date": (snap.generated_at.date() + timedelta(days=90)).isoformat(), "close": 520.0},
        ],
    }
    with _stub_prices(prices):
        first = outcome_service.evaluate_all_due(today=today)
        second = outcome_service.evaluate_all_due(today=today)
    assert first["written"] >= 1
    assert second["written"] == 0  # idempotent


def test_due_outcome_without_prices_is_reported_as_unavailable():
    """Missing provider config is failed work, not a successful no-op.

    Production reproduced this with hundreds of due pairs and no price
    provider configured on the worker: ``written=0 errors=0`` used to hide
    the outage indefinitely.
    """
    snap = _seed_snapshot("TSTNOPX", days_ago=40, rating="Bullish")
    today = (snap.generated_at + timedelta(days=40)).date()
    with _stub_prices({}):
        res = outcome_service.evaluate_all_due(horizons=[30], today=today)
    assert res["written"] == 0
    assert res["data_unavailable"] >= 1
    assert res["ticker_prices_unavailable"] >= 1
    assert res["due"] >= res["data_unavailable"]


def test_outcome_loop_marks_missing_due_data_as_failed(monkeypatch):
    from app.monitoring import outcome_loop

    result = {
        "evaluated": 12,
        "due": 3,
        "written": 0,
        "already_recorded": 0,
        "data_unavailable": 3,
        "reflections": 0,
        "errors": 0,
    }
    calls = []
    monkeypatch.setattr(outcome_loop, "evaluate_all_due", lambda: result)
    monkeypatch.setattr(
        outcome_loop, "record_run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert outcome_loop.run_once() == result
    assert calls[0][0] == ("outcome_loop",)
    assert calls[0][1]["success"] is False
    assert "unavailable=3" in calls[0][1]["note"]


def test_postmortem_feedback_is_persisted_from_memo_json(monkeypatch):
    """A scored outcome must become durable feedback for future PM prompts."""
    from app.services import postmortem_service

    snap = _seed_snapshot(
        "TSTFEED", days_ago=120, rating="Bullish", regime="soft_landing",
    )
    g_iso = snap.generated_at.date().isoformat()
    target_iso = (snap.generated_at.date() + timedelta(days=90)).isoformat()
    today = (snap.generated_at + timedelta(days=120)).date()
    prices = {
        "TSTFEED": [
            {"date": g_iso, "close": 100.0},
            {"date": target_iso, "close": 120.0},
        ],
        "SPY": [
            {"date": g_iso, "close": 500.0},
            {"date": target_iso, "close": 530.0},
        ],
    }
    with _stub_prices(prices):
        result = outcome_service.evaluate_all_due(horizons=[90], today=today)
    assert result["written"] >= 1

    # Keep the test deterministic and zero-cost.  The production service
    # intentionally has a deterministic lesson fallback when no LLM answer
    # is available; the DB row is what influence_feedback reads later.
    monkeypatch.setattr(postmortem_service, "_llm_postmortem", lambda *a, **k: None)
    from app.config import settings
    monkeypatch.setattr(settings, "enable_long_term_memory", False)
    report = postmortem_service.run_postmortems(horizon_days=90, limit=200)
    assert report["written"] >= 1

    with SessionLocal() as db:
        row = db.query(MemoPostmortem).filter(
            MemoPostmortem.memo_snapshot_id == snap.id,
            MemoPostmortem.horizon_days == 90,
        ).one()
    assert row.ticker == "TSTFEED"
    assert row.regime_at_memo == "soft_landing"
    assert row.lesson
    assert row.verdict in {"right", "wrong", "mixed", "pending"}

    from app.services.influence_feedback import specialist_reliability
    feedback = specialist_reliability(lookback=30)
    assert feedback["n"] >= 1
    assert "valuation" in feedback["per_agent"]


def test_backtest_snapshots_are_skipped():
    bt_date = datetime.utcnow() - timedelta(days=200)
    snap = _seed_snapshot(
        "TSTBT", days_ago=200, rating="Bullish", as_of_date=bt_date,
    )
    today = (snap.generated_at + timedelta(days=200)).date()
    prices = {
        "TSTBT": [{"date": "2099-01-01", "close": 100.0}],
        "SPY": [{"date": "2099-01-01", "close": 500.0}],
    }
    with _stub_prices(prices):
        outcome_service.evaluate_all_due(today=today)
    rows = outcome_service.get_outcomes_for_snapshot(snap.id)
    assert rows == []  # backtest → no outcomes


def test_horizons_not_yet_due_are_skipped():
    snap = _seed_snapshot("TSTHALF", days_ago=20, rating="Bullish")
    today = (snap.generated_at + timedelta(days=20)).date()
    prices = {
        "TSTHALF": [
            {"date": snap.generated_at.date().isoformat(), "close": 100.0},
            {"date": today.isoformat(), "close": 105.0},
        ],
        "SPY": [
            {"date": snap.generated_at.date().isoformat(), "close": 500.0},
            {"date": today.isoformat(), "close": 510.0},
        ],
    }
    with _stub_prices(prices):
        outcome_service.evaluate_all_due(today=today)
    rows = outcome_service.get_outcomes_for_snapshot(snap.id)
    # Only horizons ≤ 20d would be due — but our DEFAULT_HORIZONS starts at 30.
    assert rows == []


# ---------------------------------------------------------------------------
# track_record
# ---------------------------------------------------------------------------

def test_track_record_aggregates_and_filters_by_ticker():
    snap_a = _seed_snapshot("TSTAA", days_ago=120, rating="Bullish")
    snap_b = _seed_snapshot("TSTBB", days_ago=120, rating="Bearish")
    g_a = snap_a.generated_at.date().isoformat()
    g_b = snap_b.generated_at.date().isoformat()
    target_a = (snap_a.generated_at.date() + timedelta(days=90)).isoformat()
    target_b = (snap_b.generated_at.date() + timedelta(days=90)).isoformat()
    today = (snap_a.generated_at + timedelta(days=100)).date()
    prices = {
        "TSTAA": [
            {"date": g_a, "close": 100.0},
            {"date": target_a, "close": 120.0},  # +20%, Bullish HELD
        ],
        "TSTBB": [
            {"date": g_b, "close": 100.0},
            {"date": target_b, "close": 80.0},   # -20%, Bearish HELD
        ],
        "SPY": [
            {"date": g_a, "close": 500.0},
            {"date": target_a, "close": 520.0},  # +4%
        ],
    }
    with _stub_prices(prices):
        outcome_service.evaluate_all_due(today=today)
    overall = outcome_service.track_record(horizon_days=90)
    assert overall["total"] >= 2
    assert overall["thesis_hit_rate"] == 1.0  # both held

    only_a = outcome_service.track_record(ticker="TSTAA", horizon_days=90)
    assert only_a["total"] == 1
    assert only_a["thesis_hit_rate"] == 1.0


def test_track_record_neutral_excluded_from_directional_count():
    snap = _seed_snapshot("TSTNEU", days_ago=120, rating="Neutral")
    g_iso = snap.generated_at.date().isoformat()
    target_iso = (snap.generated_at.date() + timedelta(days=90)).isoformat()
    today = (snap.generated_at + timedelta(days=120)).date()
    prices = {
        "TSTNEU": [
            {"date": g_iso, "close": 100.0},
            {"date": target_iso, "close": 105.0},
        ],
        "SPY": [
            {"date": g_iso, "close": 500.0},
            {"date": target_iso, "close": 510.0},
        ],
    }
    with _stub_prices(prices):
        outcome_service.evaluate_all_due(today=today)
    tr = outcome_service.track_record(ticker="TSTNEU", horizon_days=90)
    assert tr["total"] == 1
    # Neutral has no direction → not counted in directional_evaluations.
    assert tr["directional_evaluations"] == 0
    assert tr["thesis_hit_rate"] is None


# ---------------------------------------------------------------------------
# Reflection writes
# ---------------------------------------------------------------------------

def test_long_horizons_write_reflection_to_memory(tmp_path, monkeypatch):
    from app.config import settings
    from app.memory.longterm import company_memory_path
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))

    snap = _seed_snapshot("TSTRFL", days_ago=120, rating="Bullish")
    g_iso = snap.generated_at.date().isoformat()
    target_iso = (snap.generated_at.date() + timedelta(days=90)).isoformat()
    today = (snap.generated_at + timedelta(days=120)).date()
    prices = {
        "TSTRFL": [
            {"date": g_iso, "close": 100.0},
            {"date": target_iso, "close": 110.0},
        ],
        "SPY": [
            {"date": g_iso, "close": 500.0},
            {"date": target_iso, "close": 510.0},
        ],
    }
    with _stub_prices(prices):
        result = outcome_service.evaluate_all_due(today=today)
    assert result["reflections"] >= 1
    path = company_memory_path("TSTRFL")
    assert path.exists()
    text = path.read_text()
    assert "outcome:90d" in text


def test_short_horizons_do_not_write_reflection(tmp_path, monkeypatch):
    from app.config import settings
    from app.memory.longterm import company_memory_path
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))

    # Only 30d horizon will be due; no reflection should be written.
    snap = _seed_snapshot("TSTSHORT", days_ago=40, rating="Bullish")
    g_iso = snap.generated_at.date().isoformat()
    target_iso = (snap.generated_at.date() + timedelta(days=30)).isoformat()
    today = (snap.generated_at + timedelta(days=40)).date()
    prices = {
        "TSTSHORT": [
            {"date": g_iso, "close": 100.0},
            {"date": target_iso, "close": 102.0},
        ],
        "SPY": [
            {"date": g_iso, "close": 500.0},
            {"date": target_iso, "close": 505.0},
        ],
    }
    with _stub_prices(prices):
        outcome_service.evaluate_all_due(today=today)
    path = company_memory_path("TSTSHORT")
    # File may or may not exist (if it doesn't, definitely no 30d outcome entry was written).
    if path.exists():
        text = path.read_text()
        assert "outcome:30d" not in text


# ---------------------------------------------------------------------------
# Admin endpoint
# ---------------------------------------------------------------------------

def test_admin_track_record_endpoint_returns_aggregates():
    snap = _seed_snapshot("TSTADM", days_ago=120, rating="Bullish")
    g_iso = snap.generated_at.date().isoformat()
    target_iso = (snap.generated_at.date() + timedelta(days=90)).isoformat()
    today = (snap.generated_at + timedelta(days=120)).date()
    prices = {
        "TSTADM": [
            {"date": g_iso, "close": 100.0},
            {"date": target_iso, "close": 110.0},
        ],
        "SPY": [
            {"date": g_iso, "close": 500.0},
            {"date": target_iso, "close": 510.0},
        ],
    }
    with _stub_prices(prices):
        outcome_service.evaluate_all_due(today=today)
    c = TestClient(app)
    r = c.get("/api/admin/track-record?horizon_days=90&ticker=TSTADM")
    assert r.status_code == 200
    body = r.json()
    assert body["horizon_days"] == 90
    assert body["total"] >= 1


# ---------------------------------------------------------------------------
# W6 / FIX-007 — eligibility-aware evaluation and the provisional track record.
# Exact counts, so every test here runs on its own sqlite engine.
# ---------------------------------------------------------------------------

W6_TODAY = date(2026, 9, 24)


@pytest.fixture
def w6(tmp_path, monkeypatch):
    sessions, engine = isolated_sessions(tmp_path, monkeypatch, outcome_service)
    monkeypatch.setattr(settings, "enable_long_term_memory", False)
    yield sessions, engine
    engine.dispose()


def _dev_copy_nvda(db) -> MemoSnapshot:
    sid, ticker, generated = next(row for row in DEV_COPY_SNAPSHOTS if row[1] == "NVDA")
    return add_snapshot(
        db, id=sid, ticker=ticker, version=sid, generated_at=datetime.fromisoformat(generated),
        mode="demo", memo_generated_at=generated,
    )


def _series(generated: datetime, closes: dict[int, float]) -> list[dict[str, Any]]:
    return [
        {"date": (generated.date() + timedelta(days=offset)).isoformat(), "close": close}
        for offset, close in sorted(closes.items())
    ]


def test_evaluate_all_due_skips_ineligible_snapshots(w6, monkeypatch):
    sessions, _ = w6
    with sessions() as db:
        dev = _dev_copy_nvda(db)
        live = add_snapshot(db, ticker="LIVEW6", generated_at=datetime(2026, 5, 1, 13))
        db.commit()
    requests: list[str] = []
    series = {
        "LIVEW6": _series(live.generated_at, {0: 100.0, 30: 110.0, 90: 120.0}),
        "SPY": _series(live.generated_at, {0: 500.0, 30: 505.0, 90: 510.0}),
        "NVDA": _series(dev.generated_at, {0: 100.0, 30: 150.0, 90: 200.0}),
    }

    def prices(ticker, days=252):
        requests.append(ticker)
        return series.get(ticker, [])

    reflected: list[str] = []
    from app.services import market_data_service
    monkeypatch.setattr(market_data_service, "get_price_series", prices)
    monkeypatch.setattr(outcome_service, "_maybe_write_reflection",
                        lambda snap, out: reflected.append(snap.ticker) or True)
    report = outcome_service.evaluate_all_due(today=W6_TODAY)
    assert report["written"] == 2 and report["due"] == 2
    assert report["ineligible"] == 4
    assert report["ineligible_snapshots_by_reason"] == {oe.REASON_DEV_COPY: 1}
    assert report["unclassified"] == 0 and report["data_unavailable"] == 0
    assert "NVDA" not in requests, "an ineligible snapshot must not cost a price fetch"
    assert reflected == ["LIVEW6"]
    with sessions() as db:
        assert set(db.execute(select(MemoOutcome.memo_snapshot_id)).scalars()) == {live.id}


def test_failed_sweep_still_scores_classified_snapshots(w6, monkeypatch):
    """A sweep that aborts (ExclusionSetMismatch) turns the loop red but does
    not stop scoring: snapshots with an existing ledger row are still
    evaluated, and the one the sweep could not classify stays unscored.

    Before, the exception escaped evaluate_all_due ahead of the scan, so one
    unexpected snapshot stopped all outcome scoring until a redeploy.
    """
    from app.monitoring import outcome_loop
    from app.services import market_data_service

    sessions, _ = w6
    dev_ids = {row[0] for row in DEV_COPY_SNAPSHOTS}
    with sessions() as db:
        live = add_snapshot(db, ticker="LIVEW6", generated_at=datetime(2026, 5, 1, 13))
        db.commit()
        classify_all(db)
        # Demo, copy-era id and date, but not in the enumerated evidence.
        stray = add_snapshot(db, id=next(i for i in range(400, 583) if i not in dev_ids), ticker="STRAYW6",
                             generated_at=datetime(2026, 5, 4), mode="demo")
        db.commit()
    series = {
        "LIVEW6": _series(live.generated_at, {0: 100.0, 30: 110.0, 90: 120.0}),
        "SPY": _series(live.generated_at, {0: 500.0, 30: 505.0, 90: 510.0}),
    }
    requests: list[str] = []
    monkeypatch.setattr(market_data_service, "get_price_series",
                        lambda ticker, days=252: requests.append(ticker) or series.get(ticker, []))
    calls: list[int] = []
    real = oe.classify_pending

    def counting(**kwargs):
        calls.append(1)
        return real(**kwargs)

    monkeypatch.setattr(oe, "classify_pending", counting)
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(outcome_loop, "record_run", lambda *a, **k: recorded.append(k))
    monkeypatch.setattr(outcome_service, "_maybe_write_reflection", lambda snap, out: False)
    monkeypatch.setattr(outcome_loop, "evaluate_all_due",
                        lambda: outcome_service.evaluate_all_due(today=W6_TODAY))
    res = outcome_loop.run_once()
    assert res["written"] == 2, "the classified live snapshot is still scored"
    assert res["classification_error"].startswith("ExclusionSetMismatch")
    assert f"STRAYW6#{stray.id}" in res["classification_error"]
    assert res["unclassified_snapshot_ids"] == [stray.id] and res["unclassified"] == 2
    assert "STRAYW6" not in requests
    assert len(calls) == 1, "a failed sweep is not retried once per unclassified snapshot"
    assert recorded[0]["success"] is False
    assert "classification_error=ExclusionSetMismatch" in recorded[0]["note"]
    with sessions() as db:
        assert set(db.execute(select(MemoOutcome.memo_snapshot_id)).scalars()) == {live.id}
        assert oe.lookup(db, stray.id) is None, "the aborted sweep wrote nothing"


def test_sweep_failure_alone_turns_loop_red(w6, monkeypatch):
    from app.monitoring import outcome_loop

    def fail(**kwargs):
        raise oe.ExclusionSetMismatch("synthetic")

    monkeypatch.setattr(oe, "classify_pending", fail)
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(outcome_loop, "record_run", lambda *a, **k: recorded.append(k))
    res = outcome_loop.run_once()
    assert res["errors"] == 0 and res["unclassified"] == 0 and res["data_unavailable"] == 0
    assert recorded[0]["success"] is False
    assert "classification_error=ExclusionSetMismatch: synthetic" in recorded[0]["note"]


def test_unclassified_snapshot_turns_loop_red(w6, monkeypatch):
    from app.monitoring import outcome_loop
    from app.services import market_data_service

    sessions, _ = w6
    with sessions() as db:
        snap = add_snapshot(db, ticker="UNCLW6", generated_at=datetime.utcnow() - timedelta(days=100))
        db.commit()
    monkeypatch.setattr(oe, "classify_pending", lambda **kwargs: {"classified": 0})
    requests: list[str] = []
    monkeypatch.setattr(market_data_service, "get_price_series",
                        lambda ticker, days=252: requests.append(ticker) or [])
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(outcome_loop, "record_run", lambda *a, **k: recorded.append(k))
    res = outcome_loop.run_once()
    assert res["unclassified"] == 2                       # 30d and 90d are due; 180d/365d are not
    assert res["unclassified_snapshot_ids"] == [snap.id]
    assert res["data_unavailable"] == 0 and requests == []
    assert recorded[0]["success"] is False
    assert "unclassified=2" in recorded[0]["note"]
    assert f"unclassified_snapshot_ids={snap.id}" in recorded[0]["note"]


def test_unclassified_only_when_due(w6, monkeypatch):
    """A snapshot with nothing due is not a failure; one that arrives after the
    sweep is classified inline before it is judged (the evaluator-fence race)."""
    from app.services import market_data_service

    sessions, _ = w6
    monkeypatch.setattr(market_data_service, "get_price_series", lambda ticker, days=252: [])
    with sessions() as db:
        add_snapshot(db, ticker="FRESHW6", generated_at=datetime(2026, 9, 20))
        db.commit()
    real = oe.classify_pending
    monkeypatch.setattr(oe, "classify_pending", lambda **kwargs: {"classified": 0})
    report = outcome_service.evaluate_all_due(today=W6_TODAY)
    assert report["unclassified"] == 0 and report["not_due"] == 4
    assert report["unclassified_snapshot_ids"] == []

    arrivals: list[int] = []

    def sweep_then_race(**kwargs):
        out = real(**kwargs)
        if not arrivals:
            with sessions() as other:
                late = add_snapshot(other, ticker="LATEW6", generated_at=datetime(2026, 5, 1))
                other.commit()
                arrivals.append(late.id)
        return out

    monkeypatch.setattr(oe, "classify_pending", sweep_then_race)
    report = outcome_service.evaluate_all_due(today=W6_TODAY)
    assert arrivals
    assert report["unclassified"] == 0
    assert report["ticker_prices_unavailable"] == 2       # LATEW6 was judged, not skipped
    with sessions() as db:
        assert oe.lookup(db, arrivals[0]) == oe.Classification(True, oe.REASON_LIVE)


def _tr(**kwargs):
    return outcome_service.track_record(**kwargs)


def test_track_record_counts_only_eligible_rows(w6):
    sessions, _ = w6
    with sessions() as db:
        live = add_snapshot(db, ticker="ELIG", generated_at=datetime(2026, 5, 20))
        add_outcome(db, live, horizon=90, forward_return=0.10, alpha=0.02)
        dev = _dev_copy_nvda(db)
        add_outcome(db, dev, horizon=90, forward_return=0.50, alpha=0.40)
        db.commit()
        classify_all(db)
        late = add_snapshot(db, ticker="LATE", generated_at=datetime(2026, 6, 1))
        add_outcome(db, late, horizon=90, forward_return=-0.3, alpha=-0.3)
        db.commit()
    tr = _tr(horizon_days=90)
    assert tr["total"] == 1 and tr["directional_evaluations"] == 1
    assert tr["thesis_hit_rate"] == 1.0
    assert tr["avg_forward_return"] == pytest.approx(0.10) and tr["avg_alpha"] == pytest.approx(0.02)
    assert tr["eligibility"] == {
        "rule_version": oe.RULE_VERSION, "eligible": 1, "excluded": 1, "unclassified": 1,
        "excluded_by_reason": {oe.REASON_DEV_COPY: 1},
        "eligible_by_reason": {oe.REASON_LIVE: 1},
    }
    assert tr["benchmark"] == "SPY"


def test_track_record_alpha_block(w6):
    sessions, _ = w6
    rows: list[tuple[str, str, float | None]] = [("ONE", "Bullish", a) for a in (-0.10, -0.08, -0.06, -0.04, -0.02)]
    rows += [("TWO", "Bullish", 0.05), ("THREE", "Bullish", 0.06), ("FOUR", "Very Bullish", 0.07)]
    rows += [("FIVE", "Bearish", -0.03)]         # trailed SPY on a short call: beat it
    rows += [("SIX", "Neutral", 0.50)]           # no direction: not in the adjusted set
    rows += [("SEVEN", "Bullish", None)]         # benchmark missing
    with sessions() as db:
        versions: dict[str, int] = {}
        for ticker, rating, alpha in rows:
            versions[ticker] = versions.get(ticker, 0) + 1
            snap = add_snapshot(db, ticker=ticker, version=versions[ticker], rating=rating,
                                generated_at=datetime(2026, 5, 20) + timedelta(hours=versions[ticker]))
            add_outcome(db, snap, horizon=90, forward_return=(alpha or 0.0) + 0.01, alpha=alpha)
        db.commit()
        classify_all(db)
    block = _tr(horizon_days=90)["alpha"]
    raw = [a for _, _, a in rows if a is not None]
    assert block["n"] == 10 and block["unavailable"] == 1
    assert block["mean"] == pytest.approx(sum(raw) / len(raw))
    assert block["directional_n"] == 9
    assert block["directional_median"] == pytest.approx(-0.02)
    assert block["beat_benchmark_rate"] == pytest.approx(4 / 9)
    # One vote per company: ONE's five overlapping memos count once.
    assert block["company_weighted_median"] == pytest.approx(0.05)
    assert block["company_weighted_median"] != block["directional_median"]


def test_track_record_base_rate_and_mix(w6):
    sessions, _ = w6
    seeds = [
        ("B1", "Bullish", 0.10, 0.05, "llm_pm"),
        ("B2", "Bullish", -0.05, -0.08, "keyword_pm"),
        ("B3", "Bearish", -0.02, -0.04, "keyword_pm"),
        ("N1", "Neutral", 0.30, 0.20, "patch"),
        ("U1", "", 0.10, 0.01, "keyword_pm"),
    ]
    with sessions() as db:
        for ticker, rating, fwd, alpha, source in seeds:
            snap = add_snapshot(db, ticker=ticker, rating=rating, generated_at=datetime(2026, 5, 20))
            add_outcome(db, snap, horizon=90, forward_return=fwd, alpha=alpha, rating=rating)
            db.commit()
            mark(db, snap.id, rating_source=source)
    tr = _tr(horizon_days=90)
    assert tr["thesis_hit_rate"] == pytest.approx(2 / 3)
    assert tr["base_rate"] == {
        "always_bullish_hit_rate": pytest.approx(1 / 3), "always_bullish_n": 3,
        "positive_alpha_rate": pytest.approx(3 / 5), "positive_alpha_n": 5,
    }
    assert tr["rating_mix"] == {"Bearish": 1, "Bullish": 2, "Neutral": 1, "Unrated": 1}
    assert tr["rating_mix_by_source"] == {
        "keyword_pm": {"Bearish": 1, "Bullish": 1, "Unrated": 1},
        "llm_pm": {"Bullish": 1},
        "patch": {"Neutral": 1},
    }


def test_base_rate_uses_directional_denominator(w6):
    """Neutral rows are not calls; counting them made "always Bullish" look
    like a hard bar to clear (the pre-W6 figure used every row)."""
    sessions, _ = w6
    with sessions() as db:
        for i, (rating, fwd) in enumerate([("Bullish", -0.10), ("Neutral", 0.2), ("Neutral", 0.2), ("Neutral", 0.2)]):
            snap = add_snapshot(db, ticker=f"D{i}", rating=rating, generated_at=datetime(2026, 5, 20))
            add_outcome(db, snap, horizon=90, forward_return=fwd, alpha=fwd)
        db.commit()
        classify_all(db)
    base = _tr(horizon_days=90)["base_rate"]
    assert base["always_bullish_n"] == 1
    assert base["always_bullish_hit_rate"] == 0.0


def test_track_record_coverage_and_universe(w6):
    sessions, _ = w6
    with sessions() as db:
        for t in ("AA", "BB", "CC", "DD"):
            db.add(Company(ticker=t, company_name=t, sector="Tech", industry="Soft", is_etf=False))
        db.add(Company(ticker="ETFX", company_name="ETF", sector="ETF", industry="ETF", is_etf=True))
        a = add_snapshot(db, ticker="AA", generated_at=datetime(2026, 5, 20))
        b = add_snapshot(db, ticker="BB", generated_at=datetime(2026, 5, 20), rating="Neutral")
        add_outcome(db, a, horizon=30, forward_return=0.1, alpha=0.01)
        add_outcome(db, b, horizon=30, forward_return=0.1, alpha=0.01)
        add_outcome(db, a, horizon=90, forward_return=0.1, alpha=0.01)
        db.commit()
        classify_all(db)
    cov = _tr(horizon_days=30)["coverage"]
    assert cov["universe_companies"] == 4
    assert cov["memos_any_horizon"] == 2 and cov["companies_any_horizon"] == 2
    by_h = {row["horizon_days"]: row for row in cov["horizons"]}
    assert sorted(by_h) == [30, 90, 180, 365]
    assert by_h[30] == {"horizon_days": 30, "memos": 2, "companies": 2, "directional": 1,
                        "universe_pct": 0.5, "late_evaluation_candidates": 0}
    assert by_h[90]["companies"] == 1 and by_h[180]["memos"] == 0
    with sessions() as db:
        db.query(Company).delete()
        db.commit()
    assert _tr(horizon_days=30)["coverage"]["horizons"][0]["universe_pct"] is None


def test_late_evaluation_disclosed(w6):
    """FIX-007's late-evaluation candidates are counted where they are shown."""
    sessions, _ = w6
    gen = datetime(2026, 5, 20)
    with sessions() as db:
        a = add_snapshot(db, ticker="LATEA", generated_at=gen)
        b = add_snapshot(db, ticker="LATEB", generated_at=gen)
        add_outcome(db, a, horizon=30, forward_return=0.1, alpha=0.01, evaluated_at=gen + timedelta(days=100))
        add_outcome(db, b, horizon=30, forward_return=0.1, alpha=0.01, evaluated_at=gen + timedelta(days=31))
        db.commit()
        classify_all(db)
    cov = _tr(horizon_days=30)["coverage"]
    assert cov["late_evaluation_candidates"] == 1
    assert cov["horizons"][0]["late_evaluation_candidates"] == 1
    assert outcome_service.is_late_evaluation(gen, gen + timedelta(days=84), 30) is False
    assert outcome_service.is_late_evaluation(gen, gen + timedelta(days=84, microseconds=1), 30) is True


def test_track_record_provisional_thresholds(w6, monkeypatch):
    sessions, _ = w6
    monkeypatch.setattr(outcome_service, "PROVISIONAL_MIN_COMPANIES", 2)
    monkeypatch.setattr(outcome_service, "PROVISIONAL_MIN_DIRECTIONAL", 3)
    with sessions() as db:
        for v in (1, 2):
            snap = add_snapshot(db, ticker="PRV1", version=v, generated_at=datetime(2026, 5, 20, v))
            add_outcome(db, snap, horizon=90, forward_return=0.1, alpha=0.01)
        db.commit()
        classify_all(db)
    below = _tr(horizon_days=90)["provisional"]
    assert below == {"is_provisional": True,
                     "reasons": ["companies_below_threshold", "directional_below_threshold"],
                     "min_companies": 2, "min_directional": 3, "companies": 1, "directional": 2}
    with sessions() as db:
        snap = add_snapshot(db, ticker="PRV2", generated_at=datetime(2026, 5, 20))
        add_outcome(db, snap, horizon=90, forward_return=0.1, alpha=0.01)
        db.commit()
        classify_all(db)
    at = _tr(horizon_days=90)["provisional"]
    assert at["is_provisional"] is False and at["reasons"] == []


def test_track_record_sector_filter_reads_no_memo_bodies(w6):
    sessions, engine = w6
    with sessions() as db:
        tech = add_snapshot(db, ticker="SECT", generated_at=datetime(2026, 5, 20), sector="Technology")
        energy = add_snapshot(db, ticker="SECE", generated_at=datetime(2026, 5, 20), sector="Energy")
        for snap in (tech, energy):
            add_outcome(db, snap, horizon=90, forward_return=0.1, alpha=0.01)
        db.commit()
        classify_all(db)
    statements: list[str] = []

    def capture(conn, cursor, statement, params, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        tr = _tr(horizon_days=90, sector="energy")
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert tr["total"] == 1 and tr["sector_filter"] == "energy"
    assert statements and not any("memo_json" in s for s in statements)


@pytest.mark.parametrize("rating", [
    "Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish",
    "Mixed Positive", "Mixed Negative", "", "bullish ", "Unrated",
])
def test_direction_agrees_with_thesis_held(rating):
    direction = outcome_service._direction(rating)
    up, down = outcome_service._thesis_held(rating, 0.1), outcome_service._thesis_held(rating, -0.1)
    if direction is None:
        assert up is None and down is None
    else:
        assert (up, down) == ((True, False) if direction == 1 else (False, True))


def test_outcome_rows_unchanged_by_exclusion(w6):
    """Exclusion is a read-side filter: no stored outcome value moves, and the
    browser-called endpoint writes nothing."""
    sessions, engine = w6
    with sessions() as db:
        live = add_snapshot(db, ticker="KEEPL", generated_at=datetime(2026, 5, 20))
        dev = _dev_copy_nvda(db)
        for snap in (live, dev):
            add_outcome(db, snap, horizon=90, forward_return=0.1, alpha=0.02)
        db.commit()
        before = db.execute(select(*MemoOutcome.__table__.c).order_by(MemoOutcome.id)).all()
        classify_all(db)
    writes: list[str] = []

    def capture(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().split(None, 1)[0].upper() in {"INSERT", "UPDATE", "DELETE", "CREATE", "ALTER"}:
            writes.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        body = TestClient(app).get("/api/admin/track-record?horizon_days=90").json()
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert writes == []
    assert body["total"] == 1 and body["eligibility"]["excluded"] == 1
    with sessions() as db:
        assert db.execute(select(*MemoOutcome.__table__.c).order_by(MemoOutcome.id)).all() == before
