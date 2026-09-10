"""FEAT-001 under the login wall: the plan shapes the request.

Free 2 companies × 2 metrics × 5 years, Pro 5 × 4 × full history
(DEVPLAN). Companies/metrics beyond the plan are a structured 402 with
the limits, the request and what Pro unlocks; years are capped silently
and reported as `capped_by_plan`. Anonymous is 401 like every other
customer route. Nothing here is metered — the explorer is a read — so
no `usage_events` row may appear.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models import FinancialPeriod, UsageEvent
from app.schemas import SeriesResponse
from app.services import fundamentals_series_service as fss
from app.services import history_service
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth
from app.tests.factories import seed_annual_periods
from app.tests.gating_helpers import assert_structured, free_user, pro_user, purge_rate_windows, user_id_for

NOW = datetime(2025, 6, 1, 12, 0, 0)   # the service clock; see test_fundamentals_api
FRESH = datetime(2025, 5, 30)
TICKERS = ("FAEA", "FAEB", "FAEC", "FAED", "FAEE")
LINES = {"revenue": 500.0, "gross_profit": 200.0, "net_income": 50.0, "cash_from_operations": 90.0}
FREE_SHAPE = {"max_companies": 2, "max_metrics": 2, "max_years": 5}
PRO_SHAPE = {"max_companies": 5, "max_metrics": 4, "max_years": None}


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture()
def auth_on(monkeypatch, clerk):
    purge_rate_windows()
    yield from enable_auth(monkeypatch, clerk)


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(fss, "_utcnow", lambda: NOW)


def _wipe(db) -> None:
    db.query(FinancialPeriod).filter(FinancialPeriod.ticker.in_(TICKERS)).delete(synchronize_session=False)
    db.commit()


@pytest.fixture()
def seeded():
    """Eight fiscal years for five throwaway tickers."""
    with SessionLocal() as db:
        history_service._ensure_tables(db)
        _wipe(db)
        for t in TICKERS:
            seed_annual_periods(db, t, {fy: LINES for fy in range(2017, 2025)}, fetched_at=FRESH)
        db.commit()
    yield
    with SessionLocal() as db:
        _wipe(db)


def _series(client, token=None, **body):
    return client.post("/api/fundamentals/series", json=body, headers=bearer(token) if token else {})


# ---------------------------------------------------------------------------
# Anonymous
# ---------------------------------------------------------------------------

def test_anonymous_is_401_on_both_routes(auth_on, client, seeded):
    detail = assert_structured(client.get("/api/fundamentals/catalog"), code="auth_required", status=401)
    assert detail["message"]
    resp = _series(client, tickers=["FAEA"], metrics=["revenue"])
    assert_structured(resp, code="auth_required", status=401)
    assert resp.headers.get("www-authenticate") == "Bearer"


# ---------------------------------------------------------------------------
# Free
# ---------------------------------------------------------------------------

def test_free_can_read_the_catalog(auth_on, client):
    _sub, tok = free_user(auth_on)
    assert client.get("/api/fundamentals/catalog", headers=bearer(tok)).status_code == 200


@pytest.mark.parametrize("tickers,metrics", [
    (["FAEA", "FAEB", "FAEC"], ["revenue"]),
    (["FAEA"], ["revenue", "net_margin", "gross_margin"]),
    (["FAEA", "FAEB", "FAEC", "FAED", "FAEE"], ["revenue", "net_margin", "gross_margin", "pe_ttm"]),
])
def test_free_shape_exceeded_is_a_structured_402(auth_on, client, seeded, tickers, metrics):
    _sub, tok = free_user(auth_on)
    resp = _series(client, tok, tickers=tickers, metrics=metrics, years=3)
    detail = assert_structured(resp, code="plan_required", status=402)
    assert detail["feature"] == "fundamentals_explorer" and detail["plan"] == "free"
    assert detail["upgrade_url"] == "/pricing"
    assert detail["extra"]["limits"] == FREE_SHAPE
    assert detail["extra"]["requested"] == {"companies": len(tickers), "metrics": len(metrics), "years": 3}
    assert detail["extra"]["upgrade"] == {"plan": "pro", "limits": PRO_SHAPE, "url": "/pricing"}
    # The message carries the numbers, so the UpgradePrompt can quote them.
    assert "2 companies × 2 metrics × 5 years" in detail["message"]
    assert f"{len(tickers)} companies × {len(metrics)} metrics" in detail["message"]
    assert "Pro draws 5 × 4 × full history" in detail["message"]


@pytest.mark.parametrize("requested,applied,capped", [
    (None, 5, True),   # "max" on Free is the plan's 5 years
    (10, 5, True),     # a Pro user's shared URL still renders, capped
    (5, 5, False),
    (3, 3, False),
])
def test_free_years_are_capped_not_refused(auth_on, client, seeded, requested, applied, capped):
    _sub, tok = free_user(auth_on)
    body = {"tickers": ["FAEA", "FAEB"], "metrics": ["revenue", "gross_margin"]}
    if requested is not None:
        body["years"] = requested
    resp = _series(client, tok, **body)
    assert resp.status_code == 200, resp.text
    out = SeriesResponse.model_validate(resp.json())
    assert out.limits.applied.model_dump() == {"companies": 2, "metrics": 2, "years": applied}
    assert out.limits.capped_by_plan is capped
    assert out.periods == [f"FY{fy}" for fy in range(2025 - applied, 2025)]


def test_the_explorer_is_not_metered(auth_on, client, seeded):
    _sub, tok = free_user(auth_on)
    assert _series(client, tok, tickers=["FAEA"], metrics=["revenue"]).status_code == 200
    assert client.get("/api/fundamentals/catalog", headers=bearer(tok)).status_code == 200
    uid = user_id_for(client, tok)
    with SessionLocal() as db:
        assert db.query(UsageEvent).filter(UsageEvent.user_id == uid).count() == 0


# ---------------------------------------------------------------------------
# Pro
# ---------------------------------------------------------------------------

def test_pro_gets_five_by_four_by_full_history(auth_on, client, seeded):
    _sub, tok = pro_user(client, auth_on)
    resp = _series(client, tok, tickers=list(TICKERS), metrics=["revenue", "net_margin", "gross_margin", "pe_ttm"])
    assert resp.status_code == 200, resp.text
    out = SeriesResponse.model_validate(resp.json())
    assert out.limits.applied.model_dump() == {"companies": 5, "metrics": 4, "years": None}
    assert out.limits.capped_by_plan is False
    assert out.periods == [f"FY{fy}" for fy in range(2017, 2025)]
    assert len(out.series) == 20
    # A Pro user asking for a range is honoured verbatim.
    resp = _series(client, tok, tickers=["FAEA"], metrics=["revenue"], years=3)
    out = SeriesResponse.model_validate(resp.json())
    assert out.limits.applied.years == 3 and out.limits.capped_by_plan is False


def test_pro_still_gets_the_absolute_ceilings(auth_on, client, seeded):
    """The plan shape is the absolute ceiling for Pro, so the 422 comes
    from the request model, not a 402 from the gate."""
    _sub, tok = pro_user(client, auth_on)
    resp = _series(client, tok, tickers=[*TICKERS, "ZZFA9"], metrics=["revenue"])
    assert resp.status_code == 422, resp.text
