"""The scoped financial repair is an authenticated plan-then-apply operation."""
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

BASE = "/api/admin/market-data/bk-repair"
DIGEST = "a" * 64


@pytest.fixture
def surface(monkeypatch):
    from app.services import bk_fundamental_repair as repair
    monkeypatch.setattr(settings, "admin_api_token", "test-bk-repair-token")
    return TestClient(app), repair, {"Authorization": "Bearer test-bk-repair-token"}


@pytest.mark.parametrize("method,path,body", [
    ("post", "/plan", {"bad_fetched_at": "2026-09-13T04:29:39.595667Z"}),
    ("get", "/example", None),
    ("post", "/example/apply", {"digest": DIGEST}),
])
def test_repair_requires_admin_before_calling_any_service(surface, monkeypatch, method, path, body):
    client, repair, _ = surface
    def forbidden(*args, **kwargs):
        raise AssertionError("Unauthorized repair reached the service")
    for name in ("prepare_bk_repair", "read_bk_repair_plan", "apply_bk_repair"):
        monkeypatch.setattr(repair, name, forbidden)
    assert client.request(method, BASE + path, json=body).status_code == 401


def test_authenticated_repair_preserves_plan_and_digest_contract(surface, monkeypatch):
    client, repair, headers = surface
    calls = []
    plan = {"plan_id": "example", "digest": DIGEST, "actions": [{"id": 17, "action": "quarantine"}]}
    def prepare(stamp):
        calls.append(("prepare", stamp))
        return plan
    def apply(plan_id, digest):
        calls.append(("apply", plan_id, digest))
        return {"plan_id": plan_id, "status": "applied"}
    monkeypatch.setattr(repair, "prepare_bk_repair", prepare)
    monkeypatch.setattr(repair, "read_bk_repair_plan", lambda plan_id: plan if plan_id == "example" else None)
    monkeypatch.setattr(repair, "apply_bk_repair", apply)
    response = client.post(BASE + "/plan", headers=headers, json={"bad_fetched_at": "2026-09-13T04:29:39.595667Z"})
    assert response.status_code == 200 and response.json() == plan
    assert calls == [("prepare", datetime.fromisoformat("2026-09-13T04:29:39.595667+00:00"))]
    assert client.get(BASE + "/example", headers=headers).json() == plan
    assert client.get(BASE + "/missing", headers=headers).status_code == 404
    assert client.post(BASE + "/example/apply", headers=headers, json={"digest": "wrong"}).status_code == 422
    assert len(calls) == 1
    assert client.post(BASE + "/example/apply", headers=headers, json={"digest": DIGEST}).json()["status"] == "applied"
    assert calls[-1] == ("apply", "example", DIGEST)


@pytest.mark.parametrize("error,status", [(LookupError, 404), (ValueError, 400), (RuntimeError, 409)])
def test_repair_errors_have_safe_actionable_http_status(surface, monkeypatch, error, status):
    client, repair, headers = surface
    def fail(*args):
        raise error("private diagnostic must not be exposed")
    monkeypatch.setattr(repair, "apply_bk_repair", fail)
    response = client.post(BASE + "/example/apply", headers=headers, json={"digest": DIGEST})
    assert response.status_code == status
    assert "private diagnostic" not in response.text
