"""Bounded openai <-> anthropic failover in `llm.chat_json` / `chat_text`.

"Bounded" is the property under test: one hop per call, never a loop,
never to an unconfigured provider, never past an open breaker, and the
partner runs on its own route default rather than the failed provider's
model name. Everything is monkeypatched — no client is ever built.
"""
from __future__ import annotations

import contextvars
import logging
import time
from typing import Any
from unittest.mock import patch

import pytest

from app.agents import llm
from app.agents.safe_runner import DegradationLog
from app.config import settings


@pytest.fixture(autouse=True)
def _clean_llm_state(monkeypatch):
    """Both providers 'configured' (stub keys, never used — the client
    factories are patched), openai active, clean breakers/failover."""
    monkeypatch.setattr(settings, "openai_api_key", "stub-openai")
    monkeypatch.setattr(settings, "anthropic_api_key", "stub-anthropic")
    monkeypatch.setattr(settings, "llm_failover_enabled", True)
    monkeypatch.setattr(
        type(settings), "active_llm_provider", property(lambda self: "openai"),
    )
    llm.reset_circuit_breaker()
    llm.reset_failover_state()
    yield
    llm.reset_circuit_breaker()
    llm.reset_failover_state()


class _Calls:
    """Fake provider wrappers that record the model they were handed."""

    def __init__(self, openai_result: Any, anthropic_result: Any):
        self.openai_calls: list[dict[str, Any]] = []
        self.anthropic_calls: list[dict[str, Any]] = []
        self._openai_result = openai_result
        self._anthropic_result = anthropic_result

    def openai_json(self, client, *, model, system, user, max_tokens):
        self.openai_calls.append({"model": model, "max_tokens": max_tokens})
        return self._openai_result

    def openai_text(self, client, *, model, system, user, max_tokens):
        self.openai_calls.append({"model": model, "max_tokens": max_tokens})
        return self._openai_result

    def anthropic(self, client, *, model, system, user, max_tokens):
        self.anthropic_calls.append({"model": model, "max_tokens": max_tokens, "system": system})
        return self._anthropic_result


def _patched(calls: _Calls):
    return [
        patch.object(llm, "_openai_client", return_value=object()),
        patch.object(llm, "_anthropic_client", return_value=object()),
        patch.object(llm, "_openai_chat_json", side_effect=calls.openai_json),
        patch.object(llm, "_openai_chat_text", side_effect=calls.openai_text),
        patch.object(llm, "_anthropic_chat", side_effect=calls.anthropic),
    ]


def _open_breaker(provider: str) -> None:
    llm._FAILURE_COUNTERS[provider] = llm._BREAKER_THRESHOLD
    llm._FAILURE_LAST_AT[provider] = time.time()


def _run(calls: _Calls, fn=llm.chat_json, **kw):
    patches = _patched(calls)
    for p in patches:
        p.start()
    try:
        return fn("hi", **kw)
    finally:
        for p in patches:
            p.stop()


# ---------------------------------------------------------------------------
# The hop happens, once, on the partner's own model
# ---------------------------------------------------------------------------

def test_primary_failure_fails_over_once_to_secondary_default_model():
    calls = _Calls(openai_result=None, anthropic_result='{"ok": true}')
    # A provider-shaped override for the primary must not leak to the partner.
    out = _run(calls, route="cheap", model="gpt-4.1-mini")

    assert out == {"ok": True}
    assert len(calls.openai_calls) == 1 and calls.openai_calls[0]["model"] == "gpt-4.1-mini"
    assert len(calls.anthropic_calls) == 1
    assert calls.anthropic_calls[0]["model"] == settings.anthropic_cheap_model
    assert "Return ONLY valid JSON" in calls.anthropic_calls[0]["system"], (
        "the partner path still gets the JSON-mode system suffix"
    )

    state = llm.get_failover_state()
    assert state["count"] == 1
    assert state["last_from"] == "openai" and state["last_to"] == "anthropic"
    assert state["last_reason"] == "call_failed"
    assert state["last_at"] is not None and time.time() - state["last_at"] < 5

    # The primary's failure counted against it; the partner's success reset it.
    assert llm._FAILURE_COUNTERS["openai"] == 1
    assert llm._FAILURE_COUNTERS["anthropic"] == 0


def test_strong_route_fails_over_to_the_partner_strong_default():
    calls = _Calls(openai_result=None, anthropic_result='{"ok": true}')
    _run(calls, route="strong", model="gpt-5.5-pro")
    assert calls.anthropic_calls[0]["model"] == settings.anthropic_strong_model


def test_failover_event_feeds_the_degradation_log():
    calls = _Calls(openai_result=None, anthropic_result='{"ok": true}')
    _run(calls)

    events = llm.consume_failover_events()
    assert events == [{"from": "openai", "to": "anthropic", "reason": "call_failed"}]
    assert llm.consume_failover_events() == [], "consume clears"

    # Mirror what graph.run_stock_memo does with the events.
    dlog = DegradationLog()
    for ev in events:
        dlog.record_soft(
            "LLM provider",
            f"failed over from {ev['from']} to {ev['to']}: {ev['reason']}",
            kind="ProviderFailover",
        )
    assert dlog.degraded_agents() == ["LLM provider"]
    assert dlog.failures[0]["error_type"] == "ProviderFailover"
    assert "openai to anthropic" in dlog.failures[0]["message"]


def test_failover_events_are_context_local():
    calls = _Calls(openai_result=None, anthropic_result='{"ok": true}')
    _run(calls)
    # A fresh context (what a different worker thread sees) has no events…
    assert contextvars.Context().run(llm.consume_failover_events) == []
    # …and ours are still here.
    assert len(llm.consume_failover_events()) == 1


def test_primary_breaker_open_skips_primary_and_hops():
    _open_breaker("openai")
    calls = _Calls(openai_result={"never": "called"}, anthropic_result='{"ok": true}')
    out = _run(calls)

    assert out == {"ok": True}
    assert calls.openai_calls == [], "an open breaker must not spend a call on the primary"
    assert len(calls.anthropic_calls) == 1
    assert llm.get_failover_state()["last_reason"] == "breaker_open"


def test_chat_text_fails_over_the_same_way():
    calls = _Calls(openai_result=None, anthropic_result="plain answer")
    out = _run(calls, fn=llm.chat_text, route="cheap")
    assert out == "plain answer"
    assert len(calls.openai_calls) == 1 and len(calls.anthropic_calls) == 1
    assert calls.anthropic_calls[0]["model"] == settings.anthropic_cheap_model
    assert llm.get_failover_state()["count"] == 1


def test_anthropic_primary_fails_over_to_openai(monkeypatch):
    monkeypatch.setattr(
        type(settings), "active_llm_provider", property(lambda self: "anthropic"),
    )
    calls = _Calls(openai_result={"ok": True}, anthropic_result=None)
    out = _run(calls, route="cheap", model="claude-haiku-4-5")
    assert out == {"ok": True}
    assert len(calls.anthropic_calls) == 1 and len(calls.openai_calls) == 1
    assert calls.openai_calls[0]["model"] == settings.openai_cheap_model
    state = llm.get_failover_state()
    assert (state["last_from"], state["last_to"]) == ("anthropic", "openai")


# ---------------------------------------------------------------------------
# The hop does NOT happen
# ---------------------------------------------------------------------------

def test_secondary_breaker_open_returns_none_without_recording():
    _open_breaker("anthropic")
    calls = _Calls(openai_result=None, anthropic_result='{"ok": true}')
    assert _run(calls) is None
    assert calls.anthropic_calls == []
    assert llm.get_failover_state()["count"] == 0
    assert llm.consume_failover_events() == []


def test_failover_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "llm_failover_enabled", False)
    calls = _Calls(openai_result=None, anthropic_result='{"ok": true}')
    assert _run(calls) is None
    assert calls.anthropic_calls == []
    assert llm.get_failover_state()["count"] == 0


def test_single_configured_provider_never_fails_over(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    calls = _Calls(openai_result=None, anthropic_result='{"ok": true}')
    assert _run(calls) is None
    assert calls.anthropic_calls == []
    assert llm.get_failover_state()["count"] == 0
    assert llm.consume_failover_events() == []


def test_primary_success_does_not_touch_the_partner():
    calls = _Calls(openai_result={"ok": True}, anthropic_result='{"no": true}')
    assert _run(calls) == {"ok": True}
    assert calls.anthropic_calls == []
    assert llm.get_failover_state()["count"] == 0


def test_gemini_is_not_a_failover_participant():
    calls = _Calls(openai_result={"ok": True}, anthropic_result='{"ok": true}')
    with patch.object(llm, "_gemini_client", return_value=None):
        assert _run(calls, provider_override="gemini") is None
    assert calls.openai_calls == [] and calls.anthropic_calls == []
    assert llm.get_failover_state()["count"] == 0


# ---------------------------------------------------------------------------
# Bounded: the partner's failure counts against the partner, and stops there
# ---------------------------------------------------------------------------

def test_failover_failure_is_recorded_on_the_secondary_and_does_not_retry():
    calls = _Calls(openai_result=None, anthropic_result=None)
    assert _run(calls) is None
    assert len(calls.openai_calls) == 1, "no retry on the primary"
    assert len(calls.anthropic_calls) == 1, "exactly one hop, no loop"
    assert llm._FAILURE_COUNTERS["openai"] == 1
    assert llm._FAILURE_COUNTERS["anthropic"] == 1
    # The hop was attempted, so it is recorded even though it failed.
    assert llm.get_failover_state()["count"] == 1


def test_repeated_failover_failures_trip_the_secondary_breaker():
    """Three failed hops open the partner's breaker like any other call would;
    the fourth call then returns None without touching the partner."""
    calls = _Calls(openai_result=None, anthropic_result=None)
    for _ in range(3):
        _run(calls)
    assert llm._breaker_open("anthropic")
    assert llm._breaker_open("openai")
    before = len(calls.anthropic_calls)
    assert _run(calls) is None
    assert len(calls.anthropic_calls) == before


def test_breaker_state_carries_failover_unless_told_otherwise():
    state = llm.get_breaker_state()
    assert set(state) == {"openai", "anthropic", "gemini", "failover"}
    assert set(state["failover"]) == {"count", "last_from", "last_to", "last_at", "last_reason"}
    # The status contract renders `breakers` as one row per provider.
    assert set(llm.get_breaker_state(include_failover=False)) == {"openai", "anthropic", "gemini"}


def test_failover_is_logged_at_warning_through_log_safety(monkeypatch, caplog):
    seen: list[tuple] = []
    real = llm.log_safely

    def _spy(log, msg, exc, **kw):
        seen.append((msg, exc))
        real(log, msg, exc, **kw)

    monkeypatch.setattr(llm, "log_safely", _spy)
    calls = _Calls(openai_result=None, anthropic_result='{"ok": true}')
    with caplog.at_level(logging.WARNING, logger="app.agents.llm"):
        _run(calls)
    assert seen == [("LLM failover from openai to anthropic (call_failed)", None)]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warnings] == ["LLM failover from openai to anthropic (call_failed)"]
