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
from typing import Any, Dict, List
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models import MemoOutcome, MemoSnapshot
from app.services import outcome_service


def _seed_snapshot(
    ticker: str = "TSTONE", *, version: int = 1,
    rating: str = "Bullish", confidence: float = 70.0,
    days_ago: int = 120, as_of_date=None,
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
            },
            revision_log=[], generated_at=datetime.utcnow() - timedelta(days=days_ago),
            as_of_date=as_of_date,
        )
        db.add(snap)
        db.commit()
        db.refresh(snap)
        db.expunge(snap)
        return snap


def _stub_prices(rows_by_ticker: Dict[str, List[Dict[str, Any]]]):
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
    snap = _seed_snapshot("TSTONE", days_ago=120, rating="Bullish")
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
        res = outcome_service.evaluate_all_due(today=today)
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
        outcome_service.evaluate_all_due(today=today)
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
