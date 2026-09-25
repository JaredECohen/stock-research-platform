"""FIX-016 / L7: outside demo mode, social sentiment is an explicit "none".

We have no social data source. The pre-fix code asked an ungrounded Gemini to
"aggregate" X / Reddit / StockTwits (the model has no access to them, so the
number was invented) and fell back to a hash of the ticker's characters.
These tests pin the replacement contract:

- live mode makes no LLM call and returns no numbers, even with a Gemini key
  configured and a pre-fix payload sitting in today's cache bucket;
- the hash stub is reachable only in demo mode;
- the daily loop records "no social data source" and does no work, while
  staying registered (the KNOWN_LOOPS pins elsewhere stay at their count).
"""
from __future__ import annotations

from datetime import date

import pytest

from app.agents import llm, social_agent, tools
from app.cache import cache_put
from app.config import settings
from app.monitoring import KNOWN_LOOPS, social_loop

NUMERIC_KEYS = ("sentiment_extremity", "contrarian_flag")


@pytest.fixture
def live_mode(monkeypatch):
    """Production's shape: live data on, and a Gemini key configured, which is
    exactly the condition under which the old code made its ungrounded call."""
    monkeypatch.setattr(settings, "use_demo_data", False)
    monkeypatch.setattr(settings, "enable_live_data", True)
    monkeypatch.setattr(settings, "gemini_api_key", "test-not-a-real-key")
    assert not settings.use_demo_data_only and settings.has_gemini


@pytest.fixture
def no_llm(monkeypatch):
    """Record any LLM entry point the social path reaches; each one raises."""
    calls: list[str] = []

    def _boom(name):
        def _f(*a, **k):
            calls.append(name)
            raise AssertionError(f"social path called llm.{name}")
        return _f

    for name in ("gemini_chat_json", "gemini_chat_text", "chat_json", "chat_text"):
        if hasattr(llm, name):
            monkeypatch.setattr(llm, name, _boom(name))
    return calls


def _assert_unavailable(payload: dict, ticker: str) -> None:
    assert payload.get("ticker") == ticker
    assert payload.get("source") == "none"
    assert payload.get("status") == "unavailable"
    for key in NUMERIC_KEYS:
        assert key not in payload, f"live social payload carries {key!r}: {payload}"
    assert not any(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in payload.values()), payload


def test_social_live_mode_no_llm_no_numbers(live_mode, no_llm):
    payload = social_agent.run("MSFT", force_refresh=True)
    _assert_unavailable(payload, "MSFT")
    assert no_llm == []

    # A pre-fix row in today's bucket (the Gemini guess or the hash stub,
    # written before deploy) must not be served back as data.
    cache_put(
        f"social_hot:NVDA:{date.today().isoformat()}", "social_hot",
        payload={"ticker": "NVDA", "sentiment_extremity": 71.0,
                 "contrarian_flag": "neutral", "source": "gemini"},
        sources_used=["social:NVDA"], generated_by="social_agent",
        cost_tokens=20, ttl_seconds=24 * 3600,
    )
    _assert_unavailable(social_agent.run("NVDA"), "NVDA")
    assert no_llm == []


def test_stub_only_in_demo(monkeypatch, live_mode):
    # Live: the hash stub is unreachable, whether through the agent or the
    # tool the news/risk scopes name.
    _assert_unavailable(tools.get_social_sentiment("AAPL"), "AAPL")

    # Demo: the stub still gives the offline demo and the suite a stable shape.
    monkeypatch.setattr(settings, "use_demo_data", True)
    monkeypatch.setattr(settings, "enable_live_data", False)
    demo = tools.get_social_sentiment("AAPL")
    assert isinstance(demo["sentiment_extremity"], float)
    assert demo["contrarian_flag"] in ("bullish_setup", "bearish_setup", "neutral")
    assert social_agent.run("AAPL", force_refresh=True)["source"] == "stub"


def test_social_loop_note_no_source(monkeypatch, live_mode, no_llm):
    recorded: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        social_loop, "record_run",
        lambda name, **kw: recorded.append((name, kw)),
    )

    def _must_not_run(*a, **k):
        raise AssertionError("live social_loop did work despite having no source")

    monkeypatch.setattr(social_loop, "select_focus", _must_not_run)
    monkeypatch.setattr(social_loop.social_agent, "run", _must_not_run)

    assert social_loop.run_once() == []
    assert social_loop.run_once(["NVDA"]) == []

    assert [name for name, _ in recorded] == ["social_loop", "social_loop"]
    for _, kw in recorded:
        assert "no social data source" in kw["note"]
        assert kw.get("success", True) is True
    assert no_llm == []
    # Still registered: cron-health keeps reporting it and KNOWN_LOOPS keeps
    # its count (pinned in test_worker_service / test_cron_health_cross_process).
    assert "social_loop" in KNOWN_LOOPS
