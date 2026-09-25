"""W7: the operator's learning surface is token-guarded, bounded and honest.

Seven routes under `/api/admin/*`: none is browser-called, so none may be
exempt from the admin token. Mode changes answer 409 with the gates when a
promotion is premature; the per-item kill switch appends history; preview
writes nothing (not even a render row).
"""
from __future__ import annotations

from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event

from app.api import admin_auth
from app.config import settings
from app.learning import control, ledger
from app.main import app
from app.models import LearningControlEvent, LearningItem, LearningRender
from app.services import corpus_inventory
from app.tests.learning_helpers import add_company, learning_db

TOKEN = "test-admin-token-learning"
ROUTES = [
    ("GET", "/api/admin/learning/status"),
    ("POST", "/api/admin/learning/mode"),
    ("GET", "/api/admin/learning/items"),
    ("POST", "/api/admin/learning/items/{item_id}/status"),
    ("GET", "/api/admin/learning/renders"),
    ("GET", "/api/admin/learning/preview"),
    ("GET", "/api/admin/corpus-inventory"),
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    sessions, engine = learning_db(tmp_path, monkeypatch, corpus_inventory)
    monkeypatch.setattr(settings, "admin_api_token", TOKEN)
    monkeypatch.setattr(settings, "learning_mode_max", "inject")
    client = TestClient(app)   # bare: no lifespan seed (see test_admin_auth)
    yield sessions, engine, client
    control._cache_clear()
    engine.dispose()


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def _lesson(sessions, ticker: str = "ADMA", **kw) -> int:
    with sessions() as s:
        item = ledger._new_item(
            s, kind="lesson", scope_type="company", scope_key=ticker,
            text="When margins expand, expect the stock to outperform the benchmark over 90 days.",
            condition="margins expand", observable="outperform", origin_kind="postmortem",
            origin_ref=kw.pop("origin_ref", "1"), origin_ticker=ticker, source_date=date(2026, 6, 1),
            now=datetime(2026, 6, 1), **kw,
        )
        s.commit()
        return item.id


def test_all_seven_require_token_in_production(env):
    _, _, client = env
    spec = app.openapi()["paths"]
    for method, path in ROUTES:
        assert method.lower() in spec.get(path, {}), f"{method} {path} is not mounted"
        assert admin_auth.is_protected(method, path) and not admin_auth.is_exempt(method, path)
    # Read-only probes over HTTP: without the bearer they are refused.
    for path in ("/api/admin/learning/status", "/api/admin/learning/items", "/api/admin/learning/renders",
                 "/api/admin/learning/preview?ticker=ADMA", "/api/admin/corpus-inventory"):
        assert client.get(path).status_code == 401, path
    assert client.post("/api/admin/learning/mode", json={"mode": "off", "reason": "x"}).status_code == 401


def test_status_shape(env):
    sessions, _, client = env
    _lesson(sessions)
    body = client.get("/api/admin/learning/status", headers=_auth()).json()
    assert (body["ceiling"], body["db_mode"], body["effective_mode"]) == ("inject", "shadow", "shadow")
    assert set(body["gates"]) == {"G1", "G2", "G3", "G4", "promotable"}
    assert body["counts"]["items"]["kind"] == {"lesson": 1}
    assert set(body["counts"]) == {"items", "evidence", "renders_30d", "considered_30d"}
    assert body["recent_events"] == [] and body["ledger_epoch"] is None


def test_mode_post_409_then_force(env):
    sessions, _, client = env
    r = client.post("/api/admin/learning/mode", headers=_auth(), json={"mode": "inject", "reason": "early"})
    assert r.status_code == 409 and r.json()["error"] == "gates_not_met"
    assert r.json()["gates"]["promotable"] is False
    assert client.post("/api/admin/learning/mode", headers=_auth(),
                       json={"mode": "inject", "reason": ""}).status_code == 422
    assert client.post("/api/admin/learning/mode", headers=_auth(),
                       json={"mode": "loud", "reason": "x"}).status_code == 422
    r = client.post("/api/admin/learning/mode", headers=_auth(),
                    json={"mode": "inject", "reason": "owner override", "force": True})
    assert r.status_code == 200 and r.json()["effective_mode"] == "inject"
    with sessions() as s:
        [row] = s.query(LearningControlEvent).all()
        assert (row.mode, row.forced, row.actor) == ("inject", True, "admin")
    r = client.post("/api/admin/learning/mode", headers=_auth(), json={"mode": "shadow", "reason": "back"})
    assert r.status_code == 200 and r.json()["effective_mode"] == "shadow"


def test_item_status_transitions_append_history(env):
    sessions, _, client = env
    item = _lesson(sessions)
    url = f"/api/admin/learning/items/{item}/status"
    r = client.post(url, headers=_auth(), json={"status": "suppressed", "reason": "weak backfill"})
    assert r.status_code == 200 and r.json()["status"] == "suppressed"
    client.post(url, headers=_auth(), json={"status": "active", "reason": "reviewed again"})
    with sessions() as s:
        row = s.get(LearningItem, item)
        assert [h["to"] for h in row.status_history] == ["active", "suppressed", "active"]
        assert row.status_history[1] | {"at": None} == {"at": None, "from": "active", "to": "suppressed",
                                                        "reason": "weak backfill", "actor": "admin"}
    assert client.post("/api/admin/learning/items/99999/status", headers=_auth(),
                       json={"status": "retired", "reason": "x"}).status_code == 404
    assert client.post(url, headers=_auth(), json={"status": "superseded", "reason": "x"}).status_code == 422
    with sessions() as s:
        s.get(LearningItem, item).status = "superseded"
        s.commit()
    r = client.post(url, headers=_auth(), json={"status": "active", "reason": "x"})
    assert r.status_code == 409 and r.json()["error"] == "item_superseded"


def test_items_listing_carries_posterior_and_evidence(env):
    sessions, _, client = env
    item = _lesson(sessions)
    with sessions() as s:
        s.add(ledger.LearningEvidence(item_id=item, verdict="held", applies="yes", ticker="ADMB",
                                      horizon_days=90, independence_key="ADMA:90:1", alpha=0.07,
                                      rationale="margins expanded", observed_at=datetime(2026, 9, 1)))
        s.commit()
    body = client.get("/api/admin/learning/items?kind=lesson&scope_type=company&scope_key=adma",
                      headers=_auth()).json()
    [row] = body["items"]
    assert row["posterior"]["judged"] == 1 and row["evidence"][0]["verdict"] == "held"
    assert client.get("/api/admin/learning/items?limit=201", headers=_auth()).status_code == 422


def test_renders_filter_by_snapshot(env):
    sessions, _, client = env
    with sessions() as s:
        for sid in (7, 8):
            s.add(LearningRender(run_id=f"r{sid}", consumer="pm_memo", ticker="ADMA", mode="shadow",
                                 memo_snapshot_id=sid, chars=100, items=[], dropped=[]))
        s.commit()
    body = client.get("/api/admin/learning/renders?memo_snapshot_id=8", headers=_auth()).json()
    assert [r["run_id"] for r in body["renders"]] == ["r8"]


def test_preview_writes_nothing(env):
    sessions, engine, client = env
    with sessions() as s:
        add_company(s, "ADMA")
    _lesson(sessions)
    writes: list[str] = []

    def spy(conn, cursor, statement, *args):
        if statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            writes.append(statement)
    event.listen(engine, "before_cursor_execute", spy)
    try:
        r = client.get("/api/admin/learning/preview?ticker=adma&consumer=pm_memo", headers=_auth())
    finally:
        event.remove(engine, "before_cursor_execute", spy)
    assert r.status_code == 200 and writes == []
    body = r.json()
    assert body["ticker"] == "ADMA" and body["effective_mode"] == "shadow"
    assert [lesson["posterior"]["stance"] for lesson in body["lessons"]] == ["untested"]
    assert {"scope_type": "sector", "scope_key": "information_technology"} in body["scopes"]
    assert client.get("/api/admin/learning/preview?ticker=ADMA&consumer=chat", headers=_auth()).status_code == 422


def test_corpus_inventory_route_shape(env):
    _, _, client = env
    r = client.get("/api/admin/corpus-inventory", headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert {"repair_plan", "storage", "openai_configured", "pgvector"} <= set(body)
