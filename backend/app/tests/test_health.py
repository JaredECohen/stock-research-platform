"""`/health` and the startup LLM lines report the 2026-09-25 program's
switches as THIS process loaded them (slice B8-A2a).

Agents verify each production wave without the admin token: `/health` on
web, and the `LLM routing:` / `model_access` startup lines on both services
(the worker has no HTTP port).
"""
from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

NEW_FLAGS = {"debate_mode", "reviewer_mode", "llm_research_model", "chat_sdk", "attribution_mode"}


def test_health_reports_new_flags(monkeypatch):
    client = TestClient(app)
    body = client.get("/health").json()
    assert NEW_FLAGS <= set(body)
    # Code defaults: every wave switch off, research tier blank (legacy).
    assert body["debate_mode"] == "off" and body["reviewer_mode"] == "legacy"
    assert body["llm_research_model"] is None and body["chat_sdk"] is False
    assert body["attribution_mode"] in ("warn", "strict", "off")

    monkeypatch.setattr(settings, "debate_mode", "on")
    monkeypatch.setattr(settings, "reviewer_mode", "full")
    monkeypatch.setattr(settings, "llm_research_model", "claude-opus-5-5")
    monkeypatch.setattr(settings, "chat_agents_sdk", True)
    monkeypatch.setattr(settings, "llm_attribution_mode", "strict")
    monkeypatch.setattr(settings, "app_env", "production")
    from app.agents import llm
    monkeypatch.setattr(llm, "_PROD_WARN_NOTED", False)  # restored after the test
    body = client.get("/health").json()
    assert (body["debate_mode"], body["reviewer_mode"], body["llm_research_model"], body["chat_sdk"]) == (
        "on", "full", "claude-opus-5-5", True)
    # The EFFECTIVE mode: production always runs the guard in warn.
    assert body["attribution_mode"] == "warn"
    # Mode names and a model name only — never key material.
    monkeypatch.setattr(settings, "openai_api_key", "sk-SENTINEL-health-openai")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-SENTINEL-health")
    assert "SENTINEL" not in client.get("/health").text


def test_web_startup_logs_routing_summary_and_starts_model_access(monkeypatch, caplog):
    """M1 handoff: the routing line is `model_summary()` (tiers, Gemini, chat,
    failover map, attribution mode), not only the role table, and the
    `models.list` check is started off the boot path."""
    from app import llm_startup

    started: list[bool] = []
    monkeypatch.setattr(llm_startup, "start_model_access_check", lambda: started.append(True))
    caplog.set_level(logging.INFO, logger="marketmosaic")
    with TestClient(app):
        pass
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("LLM routing: ")]
    assert len(lines) == 1, lines
    line = lines[0]
    for key in ("provider=", "attribution=", "reviewer=", "debate=", "chat=", "gemini.news=",
                "tier.research=", "tier.research_pm=", "failover_map=", "role.pm="):
        assert key in line, key
    assert started == [True]
