"""Wave 1B — universe_tier in /api/stocks response.

Verifies the list endpoint returns the per-ticker tier so the frontend
can render the tier badge + analyze gate without an extra round-trip.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models import Company


def _ensure_started():
    """conftest sets ENABLE_LIVE_DATA=false. Hit /health to fire startup."""
    from app.database import init_db
    from app.tests.fixtures.seed_demo_data import run_full_seed
    init_db()
    run_full_seed()
    return TestClient(app)


def test_list_stocks_returns_universe_tier_per_ticker():
    c = _ensure_started()
    r = c.get("/api/stocks")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert data
    for row in data:
        assert "universe_tier" in row
        assert row["universe_tier"] in {"data_only", "auto_analysis", "analyzed_on_demand"}


def test_tier1_names_marked_auto_analysis_in_list_response():
    c = _ensure_started()
    r = c.get("/api/stocks")
    by_ticker = {row["ticker"]: row for row in r.json()}
    # MSFT is one of the always-tier-1 names per universe_tier1.json.
    assert by_ticker["MSFT"]["universe_tier"] == "auto_analysis"
    # NVDA is also tier-1.
    assert by_ticker["NVDA"]["universe_tier"] == "auto_analysis"


def test_data_only_ticker_in_list_response():
    """A demo-set ticker that's NOT in tier1 should show as data_only."""
    c = _ensure_started()
    # Reset BAC to data_only in case a prior test promoted it.
    with SessionLocal() as db:
        bac = db.get(Company, "BAC")
        if bac is not None:
            bac.universe_tier = "data_only"
            db.commit()
    r = c.get("/api/stocks")
    by_ticker = {row["ticker"]: row for row in r.json()}
    if "BAC" in by_ticker:
        assert by_ticker["BAC"]["universe_tier"] == "data_only"
