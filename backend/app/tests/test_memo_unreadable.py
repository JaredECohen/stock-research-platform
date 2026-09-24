"""FIX-004 residual: a stored memo that no longer validates is refused cleanly.

The owner's ruling (2026-09-21) is that ambiguous legacy bull/bear shapes stay
rejected and the stored payload is never rewritten or coerced. Before this fix
the refusal escaped `GET /memo` as a raw pydantic ValidationError, i.e. an ASGI
500 on every read path. These tests pin the structured 422 `memo_unreadable`,
the single ERROR log line naming the row (never quoting it), the released
grant, and a byte-for-byte unchanged stored row. They also pin that the two
unambiguous legacy shapes (ABBV v7) and a normal memo read exactly as before.
"""
from __future__ import annotations

import logging
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.database import SessionLocal
from app.main import app
from app.models import MemoSnapshot
from app.services import memo_store
from app.tests.test_memo_store import _stub_memo

ROUTE_LOGGER = "app.api.routes_stocks"

AMBIGUOUS = [
    [{"key_point": None}],
    ["SENTINEL-A", {"key_point": "SENTINEL-B"}],
    [{"other": "SENTINEL-C"}],
]

# Trimmed from docs/reviews/2026-09-13-ABBV-v7-compatibility.json (values only).
LINZESS = (
    "Label expansion of Linzess to pediatric functional constipation unlocks new patient "
    "population and revenue runway; demonstrates continued commercial momentum in GI portfolio."
)
ABBV_BEAR = {
    "headline": "Bear case: Drug Manufacturers - General thesis breaks on execution.",
    "key_points": [
        "Valuation in the top cohort quartile — any execution slip re-rates the multiple.",
        "DCF driver — Generic sector — downside scenario: Deterministic generic fallback: "
        "growth -400bp, margin -300bp, terminal growth -50bp, WACC +100bp.",
        "DCF bear case implies $191.95 (-8%).",
    ],
}


class _Grant:
    def __init__(self) -> None:
        self.committed = False
        self.released = False

    def commit(self, db):
        self.committed = True

    def release(self, db):
        self.released = True


def _forbidden(*a, **k):
    raise AssertionError("reading a stored memo must neither generate nor backfill")


def _route_setup(monkeypatch, path: str) -> list[_Grant]:
    from app.api import routes_stocks
    from app.config import settings

    grants: list[_Grant] = []

    def authorize(*a, **k):
        grants.append(_Grant())
        return grants[-1]

    # Account middleware stays in developer mode; the handler branch is chosen below.
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(routes_stocks, "customer_wall_on", lambda: False)
    monkeypatch.setattr(routes_stocks, "authorize", authorize)
    monkeypatch.setattr(routes_stocks, "run_stock_memo", _forbidden)
    monkeypatch.setattr(routes_stocks, "_ensure_lazy_universe", _forbidden)
    monkeypatch.setattr(memo_store, "memo_freshness",
                        lambda *a, **k: {"stale": False, "reason": "", "trigger": None})
    # `latest_inline` is the legacy wall-off branch; `store_only` is the path
    # the login wall (or MEMO_INLINE_GENERATION=false) takes.
    monkeypatch.setattr(settings, "memo_inline_generation", path != "store_only")
    return grants


def _store(ticker: str, *, bull_case=None, bear_case=None, version: int | None = None) -> tuple[int, int, dict]:
    """Insert a snapshot directly, as a legacy writer left it (test fixture setup)."""
    payload = _stub_memo(ticker).model_dump(mode="json")
    if bull_case is not None:
        payload["bull_case"] = bull_case
    if bear_case is not None:
        payload["bear_case"] = bear_case
    with SessionLocal() as db:
        row = MemoSnapshot(ticker=ticker, version=version or 1, trigger="incremental_patch",
                           memo_json=payload)
        db.add(row)
        db.commit()
        return row.id, row.version, deepcopy(payload)


def _rows(ticker: str) -> list[MemoSnapshot]:
    with SessionLocal() as db:
        return list(db.scalars(select(MemoSnapshot).where(MemoSnapshot.ticker == ticker)))


@pytest.mark.parametrize("value", AMBIGUOUS, ids=["null_key_point", "mixed", "unknown_key"])
@pytest.mark.parametrize("path", ["version", "latest_inline", "store_only"])
def test_unreadable_stored_memo_is_structured_422_on_every_read_path(monkeypatch, caplog, path, value):
    grants = _route_setup(monkeypatch, path)
    ticker = f"ZZUNR{path[:3].upper()}{AMBIGUOUS.index(value)}"
    snap_id, version, original = _store(ticker, bull_case=value)
    url = f"/api/stocks/{ticker}/memo" + (f"?version={version}" if path == "version" else "")
    caplog.set_level(logging.ERROR, logger=ROUTE_LOGGER)
    with TestClient(app) as client:
        r = client.get(url)
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "memo_unreadable"
    assert detail["feature"] == "memo_view"
    assert detail["extra"] == {"ticker": ticker, "version": version, "fields": ["bull_case"]}
    assert "cannot be displayed" in detail["message"]
    assert "SENTINEL" not in r.text
    # The view is not charged: every grant taken is given back.
    assert all(g.released and not g.committed for g in grants)
    if path != "latest_inline":
        assert len(grants) == 1
    errors = [rec for rec in caplog.records if rec.name == ROUTE_LOGGER and rec.levelno == logging.ERROR]
    assert len(errors) == 1
    line = errors[0].getMessage()
    for part in (f"ticker={ticker}", f"version={version}", f"snapshot_id={snap_id}", "fields=bull_case"):
        assert part in line
    assert "SENTINEL" not in caplog.text
    rows = _rows(ticker)
    assert [row.id for row in rows] == [snap_id]
    assert rows[0].memo_json == original


def test_readable_dict_memo_unaffected(monkeypatch, caplog):
    _route_setup(monkeypatch, "store_only")
    memo = _stub_memo("ZZUNROK")
    memo_store.save_memo(memo)
    caplog.set_level(logging.ERROR, logger=ROUTE_LOGGER)
    with TestClient(app) as client:
        r = client.get("/api/stocks/ZZUNROK/memo")
    assert r.status_code == 200, r.text
    assert r.json()["bull_case"] == {"headline": "bull", "key_points": []}
    assert r.json()["degradation_events"] == []
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]


def test_abbv_v7_legacy_shape_still_served_exactly(monkeypatch, caplog):
    # Behaviour-preservation pin (owner requirement: ABBV v7 byte-identical).
    # Passes before and after FIX-004's residual by design.
    grants = _route_setup(monkeypatch, "version")
    ticker = "ZZABBV"
    legacy = [{"key_point": LINZESS}]
    snap_id, version, original = _store(ticker, bull_case=legacy, bear_case=ABBV_BEAR, version=7)
    caplog.set_level(logging.ERROR, logger=ROUTE_LOGGER)
    with TestClient(app) as client:
        r = client.get(f"/api/stocks/{ticker}/memo?version=7")
    assert r.status_code == 200, r.text
    assert r.headers["X-Memo-Version"] == "7"
    assert r.headers["X-Memo-Source"] == "cache"
    body = r.json()
    assert body["bull_case"] == {"headline": "", "key_points": [LINZESS]}
    assert body["bear_case"] == ABBV_BEAR
    assert body["degraded_agents"] == ["Stored memo compatibility"]
    assert body["degradation_events"] == [{
        "agent": "Stored memo compatibility", "error_type": "LegacyCaseShape",
        "message": "bull_case was a legacy list; exact points retained, headline unavailable.",
        "field": "bull_case", "source_snapshot_id": snap_id, "source_snapshot_version": 7,
        "source_snapshot_ticker": ticker, "original_value": legacy,
    }]
    assert len(grants) == 1 and grants[0].committed and not grants[0].released
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    rows = _rows(ticker)
    assert [row.id for row in rows] == [snap_id]
    assert rows[0].memo_json == original
