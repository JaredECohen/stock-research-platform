"""Phase 4 — provider wrapper tests.

Verifies:
- Demo fallback paths return populated payloads when no API keys are present.
- Circuit breaker trips after 3 consecutive failures and short-circuits.
- Critic agent still produces output when Anthropic is missing.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agents import llm as llm_mod
from app.agents.critic_agent import run_critic
from app.agents import news_agent, social_agent
from app.cache import cache_get


def setup_function(_fn):
    llm_mod.reset_circuit_breaker()


def test_news_agent_demo_fallback_returns_alerts():
    alerts = news_agent.run("NVDA", force_refresh=True)
    # Demo news_service has at least one item for NVDA in the seed set; if not,
    # we still want a non-error empty list — exercise the cache write path.
    snap = cache_get("news_hot:NVDA", "news_hot")
    assert snap is not None
    assert isinstance(alerts, list)


def test_social_agent_demo_fallback_returns_scalar():
    payload = social_agent.run("MSFT", force_refresh=True)
    assert "sentiment_extremity" in payload
    assert payload["contrarian_flag"] in ("bullish_setup", "bearish_setup", "neutral")


def test_critic_runs_without_anthropic_key():
    review = run_critic({
        "ticker": "NVDA", "rating_label": "Bullish",
        "sources_used": ["filing:0001"], "key_risks": [], "dcf_summary": {},
    })
    assert review is not None
    assert review.overall_assessment


def test_circuit_breaker_trips_after_three_failures():
    # Simulate three OpenAI failures by patching the chat helper to return None.
    # The breaker is keyed off "openai" because that's the active provider.
    llm_mod.reset_circuit_breaker()
    with patch.object(llm_mod, "_openai_client", return_value=object()), \
         patch.object(llm_mod, "_openai_chat_json", return_value=None):
        # Force-route to openai so we don't depend on settings.
        for _ in range(3):
            assert llm_mod.chat_json("hi", provider_override="openai") is None
        # Now the breaker is open; even if the function would succeed we get None.
        with patch.object(llm_mod, "_openai_chat_json", return_value={"ok": True}):
            assert llm_mod.chat_json("hi", provider_override="openai") is None
    llm_mod.reset_circuit_breaker()


def test_provider_override_routes_to_anthropic_branch_with_no_key(monkeypatch):
    """With ANTHROPIC_API_KEY blank, an explicit override still safely returns None.

    Force-clear the key via monkeypatch so this stays deterministic regardless
    of the developer's `.env` (which may legitimately have a real key set)."""
    from app.config import settings
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    out = llm_mod.chat_json("hello", provider_override="anthropic")
    assert out is None
