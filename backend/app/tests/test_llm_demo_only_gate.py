"""Demo-only mode constructs no LLM client, whatever keys the environment
carries — the LLM analogue of the provider chain's gate, so a developer
.env with live keys cannot turn the suite into paid traffic."""
from __future__ import annotations

import pytest

from app.agents import llm
from app.config import settings


@pytest.fixture()
def keys_present(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-not-a-real-key-000000")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test-not-a-real-key-000000")
    monkeypatch.setattr(settings, "gemini_api_key", "AIza-test-not-a-real-key")
    monkeypatch.setattr(settings, "llm_provider", "auto")


def _boom(*_a, **_k):
    raise AssertionError("an LLM client was constructed under demo-only mode")


def test_demo_only_builds_no_client_even_with_keys(keys_present, monkeypatch):
    assert settings.use_demo_data_only, "the suite runs demo-only"
    assert settings.has_llm and settings.has_gemini, "keys are present; the gate, not the keys, must refuse"
    monkeypatch.setattr(llm, "OpenAI", _boom, raising=False)
    monkeypatch.setattr(llm, "Anthropic", _boom, raising=False)
    if getattr(llm, "_genai", None) is not None:
        monkeypatch.setattr(llm._genai, "Client", _boom)
    assert llm._openai_client() is None
    assert llm._anthropic_client() is None
    assert llm._gemini_client() is None
    llm.reset_circuit_breaker()
    assert llm.chat_json("say hi", route="cheap") is None
    assert llm.chat_text("say hi", route="cheap") is None
    assert not llm._breaker_open("openai") and not llm._breaker_open("anthropic"), "a skipped call is not a failure"


def test_live_pair_lets_the_factories_run(keys_present, monkeypatch):
    """Flip the pair and the factories construct again (with the SDK
    constructor stubbed so nothing leaves the process)."""
    monkeypatch.setattr(settings, "enable_live_data", True)
    assert not settings.use_demo_data_only
    monkeypatch.setattr(llm, "OpenAI", lambda **_k: object(), raising=False)
    assert llm._openai_client() is not None
