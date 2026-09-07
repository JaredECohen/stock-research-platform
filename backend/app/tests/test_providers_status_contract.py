"""`GET /api/providers/status` — the `llm` block is a frontend contract.

The status page is built against this exact key set, so the test pins
it (adding a key is a deliberate change here first). `degraded` is the
one derived field: true when nothing is configured while data is live,
when the active provider's breaker is open, or when a failover happened
inside the cooldown window — each with a sentence a human can act on.
Nothing in the payload may carry key material.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.agents import llm
from app.config import settings
from app.main import app

LLM_KEYS = {
    "configured", "provider_choice", "active_provider",
    "openai_configured", "anthropic_configured", "gemini_configured",
    "openai_strong_model", "openai_cheap_model",
    "anthropic_strong_model", "anthropic_cheap_model",
    "role_models", "breakers", "failover", "degraded", "degradation_reasons",
}
ROLES = {"pm", "sector", "tool", "macro", "critic", "strong", "cheap"}
BREAKER_KEYS = {"failure_count", "is_open", "seconds_since_last_failure", "cooldown_seconds"}
FAILOVER_KEYS = {"enabled", "count", "last_from", "last_to", "last_at", "last_reason"}
TOP_LEVEL_KEYS = {"mode", "providers", "missing_api_keys", "llm_configured", "llm", "feature_flags"}

# Stub credentials that look real enough for the leak assertion to mean
# something. They never reach a client: no test here makes an LLM call.
FAKE_OPENAI = "sk-proj-status-contract-test-0123456789abcdef"
FAKE_ANTHROPIC = "sk-ant-api03-status-contract-test-0123456789"


@pytest.fixture(autouse=True)
def _clean_breakers():
    llm.reset_circuit_breaker()
    llm.reset_failover_state()
    yield
    llm.reset_circuit_breaker()
    llm.reset_failover_state()


def _status() -> dict:
    r = TestClient(app).get("/api/providers/status")
    assert r.status_code == 200
    return r.json()


def test_llm_block_has_exactly_the_contract_keys():
    body = _status()
    assert TOP_LEVEL_KEYS <= set(body), "existing top-level keys must survive"
    block = body["llm"]
    assert set(block) == LLM_KEYS
    assert set(block["role_models"]) == ROLES
    assert all(isinstance(m, str) and m for m in block["role_models"].values())
    assert set(block["breakers"]) == {"openai", "anthropic", "gemini"}
    for state in block["breakers"].values():
        assert set(state) == BREAKER_KEYS
    assert set(block["failover"]) == FAILOVER_KEYS
    assert isinstance(block["failover"]["enabled"], bool)
    assert isinstance(block["degraded"], bool)
    assert isinstance(block["degradation_reasons"], list)


def test_blank_keys_in_live_mode_report_no_provider_as_degraded(monkeypatch):
    """The suite runs without keys; the data path is live-only — so the
    default payload is honestly degraded, with the reason spelled out."""
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    block = _status()["llm"]
    assert block["configured"] is False
    assert block["degraded"] is True
    assert "No LLM provider configured" in block["degradation_reasons"]


def test_healthy_configured_default_is_not_degraded(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", FAKE_OPENAI)
    block = _status()["llm"]
    assert block["configured"] is True
    assert block["active_provider"] == "openai"
    assert block["degraded"] is False
    assert block["degradation_reasons"] == []
    assert block["failover"]["count"] == 0
    assert block["failover"]["last_at"] is None


def test_open_breaker_on_the_active_provider_is_degraded(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", FAKE_OPENAI)
    for _ in range(llm._BREAKER_THRESHOLD):
        llm._record_failure("openai")
    block = _status()["llm"]
    assert block["breakers"]["openai"]["is_open"] is True
    assert block["degraded"] is True
    assert (
        "Openai circuit breaker is open after 3 consecutive failures"
        in block["degradation_reasons"]
    )


def test_open_breaker_on_an_inactive_provider_is_not_degraded(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", FAKE_OPENAI)
    for _ in range(llm._BREAKER_THRESHOLD):
        llm._record_failure("gemini")
    block = _status()["llm"]
    assert block["breakers"]["gemini"]["is_open"] is True
    assert block["degraded"] is False


def test_recent_failover_is_degraded_until_the_cooldown_passes(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", FAKE_OPENAI)
    monkeypatch.setattr(settings, "anthropic_api_key", FAKE_ANTHROPIC)
    llm._record_failover("openai", "anthropic", "call_failed")

    block = _status()["llm"]
    assert block["degraded"] is True
    reason = next(r for r in block["degradation_reasons"] if r.startswith("Failed over"))
    assert reason.startswith("Failed over from openai to anthropic")
    assert reason.endswith("ago")
    fo = block["failover"]
    assert fo["count"] == 1 and fo["last_from"] == "openai" and fo["last_to"] == "anthropic"
    assert fo["last_reason"] == "call_failed"
    datetime.fromisoformat(fo["last_at"])  # ISO-8601, parseable

    # Age the failover past the window: still reported, no longer degrading.
    llm._FAILOVER_STATE["last_at"] = time.time() - settings.llm_failover_cooldown_seconds - 1
    block = _status()["llm"]
    assert block["failover"]["count"] == 1
    assert block["degraded"] is False


def test_status_payload_never_carries_key_material(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", FAKE_OPENAI)
    monkeypatch.setattr(settings, "anthropic_api_key", FAKE_ANTHROPIC)
    monkeypatch.setattr(settings, "fmp_api_key", "fmp-status-contract-secret")
    for _ in range(llm._BREAKER_THRESHOLD):
        llm._record_failure("openai")
    llm._record_failover("openai", "anthropic", "breaker_open")

    text = json.dumps(_status())
    assert "sk-" not in text
    assert "Bearer" not in text
    for secret in (FAKE_OPENAI, FAKE_ANTHROPIC, "fmp-status-contract-secret"):
        assert secret not in text
