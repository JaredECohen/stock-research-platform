"""FEAT-001 — failure paths across the series and commentary routes.

The explorer's promise is that a failure somewhere else degrades into a
stated reason, never a 500 and never a silent zero:

  - the price chain raising (or swallowing a provider error) → the
    series still answers 200 and every market-derived point is
    `no_price` with a warning;
  - a company with no `financial_periods` rows → `not_backfilled` with
    the research remedy, and commentary on that chart is a free
    degraded body (there is nothing to interpret);
  - the usage meter's database failing → commentary is 503
    `usage_unavailable` with nothing charged and no LLM call, while the
    series read (which never touches the meter) still works;
  - the LLM breaker open → a degraded commentary whose `degraded` agrees
    with `/api/providers/status` `llm.degraded`, and closing it agrees
    the other way.

Deterministic and network-free: throwaway tickers, patched providers,
patched `llm.chat_json`, a fake key that only the patch could read.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.agents import llm
from app.auth import ratelimit, usage
from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import ChartCommentary, Company, FinancialPeriod
from app.schemas import CommentaryOut, SeriesResponse
from app.services import chart_commentary as cc
from app.services import fundamentals_series_service as fss
from app.services import history_service
from app.services.data_service import get_data_service
from app.services.fundamentals_series_service import build_series
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth
from app.tests.factories import seed_annual_periods
from app.tests.gating_helpers import (
    assert_structured,
    free_user,
    purge_memos,
    purge_rate_windows,
    store_memo,
    user_id_for,
)

NOW = datetime(2025, 6, 1, 12, 0, 0)
FRESH = datetime(2025, 5, 30)
SEEDED = "FFPA"
EMPTY = "FFPZ"    # in `companies`, no statement rows
LINES = {"revenue": 1000.0, "gross_profit": 600.0, "net_income": 100.0, "weighted_avg_shares_diluted": 50.0}
FAKE_OPENAI = "sk-test-failure-paths-not-a-real-key-0000"
CANNED = {"memo_view": [{"ticker": SEEDED, "text": "The memo's view is consistent with the data."}], "caveats": []}


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
    monkeypatch.setattr(cc, "_utcnow", lambda: NOW)


@pytest.fixture(autouse=True)
def _clean():
    def wipe():
        with SessionLocal() as db:
            db.query(ChartCommentary).delete(synchronize_session=False)
            db.query(FinancialPeriod).filter(FinancialPeriod.ticker.in_((SEEDED, EMPTY))).delete(synchronize_session=False)
            db.query(Company).filter(Company.ticker == EMPTY).delete(synchronize_session=False)
            # `authorize()` takes the two-in-flight lease before it reserves
            # the meter and does not give it back when `usage.reserve`
            # raises (the 503 test below), so the lease outlives the test
            # with the service's pinned 2025 clock on it. Left in place it
            # counts as an already-expired row for `test_ratelimit_db`'s
            # GC test; drop every chart_commentary lease on the way out.
            db.query(ratelimit.ActiveAction).filter(ratelimit.ActiveAction.feature == "chart_commentary").delete(
                synchronize_session=False,
            )
            db.commit()
        purge_memos(SEEDED, EMPTY)
    wipe()
    llm.reset_circuit_breaker()
    llm.reset_failover_state()
    yield
    wipe()
    llm.reset_circuit_breaker()
    llm.reset_failover_state()


@pytest.fixture()
def seeded():
    with SessionLocal() as db:
        history_service._ensure_tables(db)
        seed_annual_periods(db, SEEDED, {fy: LINES for fy in range(2019, 2025)}, fetched_at=FRESH)
        db.add(Company(ticker=EMPTY, company_name="Empty Co", sector="Technology", industry="Software"))
        db.commit()


@pytest.fixture()
def memo():
    snap = store_memo(SEEDED)
    with SessionLocal() as db:
        row = db.get(type(snap), snap.id)
        row.generated_at = datetime(2025, 3, 1)
        db.commit()
    return snap


@pytest.fixture()
def llm_available(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", FAKE_OPENAI)
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "llm_provider", "auto")


def _fp(tickers, metrics, years=None):
    with SessionLocal() as db:
        return build_series(list(tickers), list(metrics), years=years, db=db).fingerprint


def _series(client, token=None, **body):
    return client.post("/api/fundamentals/series", json=body, headers=bearer(token) if token else {})


def _commentary(client, token=None, **body):
    return client.post("/api/fundamentals/commentary", json=body, headers=bearer(token) if token else {})


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def test_price_chain_raising_degrades_to_no_price(client, seeded, monkeypatch):
    """The provider chain surfaces an exception (a wrapper that forgot
    to catch, a cache backend failure): the chart still renders."""
    def boom(*_a, **_kw):
        raise RuntimeError("price provider exploded")

    monkeypatch.setattr(fss.market_data_service, "get_price_series", boom)
    resp = _series(client, tickers=[SEEDED], metrics=["revenue", "pe_ttm"])
    assert resp.status_code == 200, resp.text
    out = SeriesResponse.model_validate(resp.json())
    by = {s.metric: s for s in out.series}
    assert all(p.value is not None for p in by["revenue"].points)
    assert all(p.value is None and p.reason == "no_price" for p in by["pe_ttm"].points)
    assert any("price history unavailable" in w and SEEDED in w for w in out.warnings)


def test_provider_error_inside_the_chain_is_no_price_too(client, seeded, monkeypatch):
    """The normal case: `DataService._try_chain` swallows a provider's
    exception and the chain returns nothing — the same `no_price`."""
    provider = get_data_service()._test_provider
    assert provider is not None

    def boom(*_a, **_kw):
        raise ConnectionError("provider down")

    monkeypatch.setattr(type(provider), "get_price_history", boom)
    resp = _series(client, tickers=[SEEDED], metrics=["pe_ttm"])
    assert resp.status_code == 200, resp.text
    out = SeriesResponse.model_validate(resp.json())
    assert all(p.value is None and p.reason == "no_price" for p in out.series[0].points)


# ---------------------------------------------------------------------------
# Empty history
# ---------------------------------------------------------------------------

def test_known_company_without_rows_is_not_backfilled_and_commentary_is_free(client, seeded, monkeypatch):
    monkeypatch.setattr(settings, "fundamentals_anon_commentary", True)
    resp = _series(client, tickers=[EMPTY], metrics=["revenue"])
    assert resp.status_code == 200, resp.text
    out = SeriesResponse.model_validate(resp.json())
    assert [(u.ticker, u.reason) for u in out.unavailable] == [(EMPTY, "not_backfilled")]
    assert "Run research" in out.unavailable[0].remedy
    assert out.periods == [] and out.series[0].points == []

    with patch.object(llm, "chat_json") as call:
        resp = _commentary(client, tickers=[EMPTY], metrics=["revenue"], fingerprint=out.fingerprint)
    call.assert_not_called()
    body = CommentaryOut.model_validate(resp.json())
    assert body.degraded is True and body.degraded_reason == cc.REASON_NO_DATA
    assert any(EMPTY in o.text and "not_backfilled" in o.text for o in body.observed)
    assert any("Run research" in c for c in body.caveats)
    with SessionLocal() as db:
        assert db.query(ChartCommentary).count() == 0


# ---------------------------------------------------------------------------
# Usage meter outage
# ---------------------------------------------------------------------------

def test_meter_db_error_is_503_for_commentary_while_series_still_works(auth_on, client, seeded, memo, llm_available, monkeypatch):
    _sub, tok = free_user(auth_on)
    uid = user_id_for(client, tok)

    def db_down(*_a, **_kw):
        raise OperationalError("UPDATE usage_counters", {}, Exception("connection reset"))

    monkeypatch.setattr(usage, "reserve", db_down)
    fp = _fp([SEEDED], ["revenue"], years=5)
    with patch.object(llm, "chat_json", return_value=CANNED) as call:
        resp = _commentary(client, tok, tickers=[SEEDED], metrics=["revenue"], fingerprint=fp)
    call.assert_not_called()
    detail = assert_structured(resp, code="usage_unavailable", status=503)
    assert detail["feature"] == "chart_commentary" and resp.headers["retry-after"] == "10"
    assert "Nothing was charged" in detail["message"]
    with SessionLocal() as db:
        assert usage.counters_for(db, uid, usage.period_key(NOW)) == {}
        assert db.query(ChartCommentary).count() == 0
        # Known limitation of `authorize()` (auth/entitlements.py, outside
        # this slice): the lease taken before the failed reserve is not
        # released and heals by its 120 s TTL. Pinned deliberately — when
        # that is fixed this count becomes 0 and the assertion (and the
        # `_clean` cleanup above) should be updated, not the fix reverted.
        leaked = db.query(ratelimit.ActiveAction).filter(
            ratelimit.ActiveAction.user_id == uid, ratelimit.ActiveAction.feature == "chart_commentary",
        ).count()
        assert leaked == 1
    # The series read never touches the meter.
    resp = _series(client, tok, tickers=[SEEDED], metrics=["revenue"])
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# LLM breaker
# ---------------------------------------------------------------------------

def test_breaker_open_degrades_commentary_in_agreement_with_provider_status(client, seeded, memo, llm_available, monkeypatch):
    monkeypatch.setattr(settings, "fundamentals_anon_commentary", True)
    for _ in range(llm._BREAKER_THRESHOLD):
        llm._record_failure("openai")
    status = client.get("/api/providers/status").json()["llm"]
    assert status["degraded"] is True and any("circuit breaker is open" in r for r in status["degradation_reasons"])

    fp = _fp([SEEDED], ["revenue"])
    resp = _commentary(client, tickers=[SEEDED], metrics=["revenue"], fingerprint=fp)   # real chat_json: short-circuits
    assert resp.status_code == 200, resp.text
    body = CommentaryOut.model_validate(resp.json())
    assert body.degraded is True and body.degraded_reason == cc.REASON_BREAKER
    assert body.observed and body.memo_view == []
    with SessionLocal() as db:
        assert db.query(ChartCommentary).count() == 0     # never attempted, nothing stored

    llm.reset_circuit_breaker()
    status = client.get("/api/providers/status").json()["llm"]
    assert status["degraded"] is False
    with patch.object(llm, "chat_json", return_value=CANNED) as call:
        resp = _commentary(client, tickers=[SEEDED], metrics=["revenue"], fingerprint=fp)
    assert call.call_count == 1
    body = CommentaryOut.model_validate(resp.json())
    assert body.degraded is False and [m.ticker for m in body.memo_view] == [SEEDED]
