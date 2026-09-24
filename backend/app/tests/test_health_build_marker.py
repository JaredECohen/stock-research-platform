"""`/health` carries the deploy-visible canary: the commit Render built."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def test_health_reports_the_deployed_commit(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(settings, "render_git_commit", "")
    assert client.get("/health").json()["build"] == {"git_commit": None}
    monkeypatch.setattr(settings, "render_git_commit", "0123456789abcdef0123456789abcdef01234567")
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["build"]["git_commit"] == "0123456789abcdef0123456789abcdef01234567"


def test_health_reports_the_routing_value_this_process_loaded(monkeypatch):
    """render.yaml turns routing on, but a Blueprint sync may not apply a new
    key; the post-deploy check reads what the web process actually loaded."""
    client = TestClient(app)
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    assert client.get("/health").json()["industry_analyst_routing"] is True
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", False)
    assert client.get("/health").json()["industry_analyst_routing"] is False
