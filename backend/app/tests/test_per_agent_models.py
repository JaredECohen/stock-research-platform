"""Per-agent model wiring tests.

Each agent now passes its dedicated `OPENAI_*_MODEL` (or
`ANTHROPIC_CRITIC_MODEL`) when it calls `llm.chat_json` / `chat_text`. This
test patches the underlying `_openai_chat_json` / `_anthropic_chat` helpers
and asserts the model name actually reaches the provider call — flipping
the env reroutes that one agent without touching code.
"""
from __future__ import annotations

from unittest.mock import patch

from app.agents import (
    critic_agent,
    earnings_agent,
    filing_agent,
    sector_agents,
    valuation_agent,
)
from app.agents import (
    llm as llm_mod,
)
from app.config import settings


def _capture_openai_calls():
    """Return (patcher, calls list). Each call appended as kwargs dict."""
    calls: list[dict] = []

    def _fake(client, *, model, system, user, max_tokens):
        calls.append({"model": model, "system": system, "user_len": len(user)})
        return {"headline": "h", "summary": "s", "key_points": [], "confidence": 0.8}

    p_client = patch.object(llm_mod, "_openai_client", return_value=object())
    p_call = patch.object(llm_mod, "_openai_chat_json", side_effect=_fake)
    p_active = patch.object(
        type(settings), "active_llm_provider",
        new=property(lambda self: "openai"),
    )
    p_breaker = patch.object(llm_mod, "_breaker_open", return_value=False)
    return calls, [p_client, p_call, p_active, p_breaker]


def _enter(patches):
    started = [p.start() for p in patches]
    return started


def _exit(patches):
    for p in patches:
        p.stop()


# Per-agent override model names must be provider-shaped: `chat_json`/`chat_text`
# drop any model that isn't named for the active provider family (see
# `llm._model_matches_provider`) so an OpenAI-shaped env can't 404 an
# Anthropic run. Synthetic names like "TEST-PM-MODEL" would be silently
# dropped to the route default, so use `gpt-*` / `claude-*` shapes that the
# guard passes through — this still validates that the per-agent env reaches
# the provider call.
def test_pm_synthesis_uses_openai_pm_model(monkeypatch):
    monkeypatch.setattr(settings, "openai_pm_model", "gpt-test-pm-model")
    calls, patches = _capture_openai_calls()
    _enter(patches)
    try:
        from app.agents.graph import _pm_synthesis
        _pm_synthesis(profile={"ticker": "T"}, findings={}, dcf=None)
    finally:
        _exit(patches)
    assert calls, "pm_synthesis should have called the OpenAI helper"
    assert calls[0]["model"] == "gpt-test-pm-model"


def test_sector_agent_uses_openai_sector_model(monkeypatch):
    monkeypatch.setattr(settings, "openai_sector_model", "gpt-test-sector-model")
    calls, patches = _capture_openai_calls()
    _enter(patches)
    try:
        from app.services.fundamentals_service import get_full_financials
        fin = get_full_financials("NVDA")
        sector_agents.run_sector_agent(fin["profile"], fin["ratios"])
    finally:
        _exit(patches)
    assert calls and calls[0]["model"] == "gpt-test-sector-model"


def test_earnings_agent_uses_openai_tool_model(monkeypatch):
    monkeypatch.setattr(settings, "openai_tool_model", "gpt-test-tool-model")
    calls, patches = _capture_openai_calls()
    _enter(patches)
    try:
        earnings_agent.run_earnings_agent(
            profile={"ticker": "T"},
            transcript={"period": "2024Q4", "management_tone": "constructive",
                        "prepared_remarks": "remarks", "qa": "qa"},
            earnings={"next_earnings_date": "2025-01-01"},
        )
    finally:
        _exit(patches)
    assert calls and calls[0]["model"] == "gpt-test-tool-model"


def test_filing_agent_uses_openai_tool_model(monkeypatch):
    monkeypatch.setattr(settings, "openai_tool_model", "gpt-test-tool-model")
    calls, patches = _capture_openai_calls()
    _enter(patches)
    try:
        filing_agent.run_filing_agent(
            profile={"ticker": "T"},
            filings=[{"type": "10-K", "accession_number": "0001-DEMO-10K",
                      "period_end": "2024-12-31", "mda": "discussion",
                      "risk_factors": ["r1"], "business_description": "biz",
                      "segments": []}],
        )
    finally:
        _exit(patches)
    assert calls and calls[0]["model"] == "gpt-test-tool-model"


def test_valuation_agent_uses_openai_tool_model(monkeypatch):
    monkeypatch.setattr(settings, "openai_tool_model", "gpt-test-tool-model")
    calls, patches = _capture_openai_calls()
    _enter(patches)
    try:
        valuation_agent.run_valuation_agent(
            profile={"ticker": "T", "last_price": 100},
            ratios={"PE": 25, "EV_EBITDA": 18, "FCF_yield": 0.04},
            dcf=None,
        )
    finally:
        _exit(patches)
    assert calls and calls[0]["model"] == "gpt-test-tool-model"


def test_critic_uses_anthropic_critic_model_when_anthropic_configured(monkeypatch):
    """Critic flips to Anthropic when ANTHROPIC_API_KEY is set, and uses the
    `anthropic_critic_model` env explicitly — not the strong-route default."""
    monkeypatch.setattr(settings, "anthropic_api_key", "stub-key")
    monkeypatch.setattr(settings, "anthropic_critic_model", "claude-test-critic-model")

    captured: list[dict] = []

    def _fake_anthropic(client, *, model, system, user, max_tokens, json_mode=False):
        captured.append({"model": model})
        return {"overall_assessment": "ok"} if json_mode else '{"overall_assessment": "ok"}'

    with patch.object(llm_mod, "_anthropic_client", return_value=object()), \
         patch.object(llm_mod, "_anthropic_chat", side_effect=_fake_anthropic), \
         patch.object(llm_mod, "_breaker_open", return_value=False):
        critic_agent.run_critic({
            "ticker": "T", "rating_label": "Bullish",
            "sources_used": ["filing:0001"], "key_risks": [],
            "dcf_summary": {"summary": "x"},
        })
    assert captured and captured[0]["model"] == "claude-test-critic-model"


# ---------------------------------------------------------------------------
# resolve_role_model / model_summary — an unset per-role env is "" and must
# resolve to the route default *before* it reaches the Agents SDK, which
# rejects an empty model name outright (chat_json already tolerated it).
# ---------------------------------------------------------------------------

def _openai_active():
    return patch.object(
        type(settings), "active_llm_provider", new=property(lambda self: "openai"),
    )


def test_blank_sector_model_resolves_to_cheap_default(monkeypatch):
    monkeypatch.setattr(settings, "openai_sector_model", "")
    with _openai_active():
        assert llm_mod.resolve_role_model("sector") == settings.openai_cheap_model


def test_blank_pm_model_resolves_to_strong_default(monkeypatch):
    monkeypatch.setattr(settings, "openai_pm_model", "   ")
    with _openai_active():
        assert llm_mod.resolve_role_model("pm") == settings.openai_strong_model


def test_provider_foreign_sector_model_resolves_to_active_default(monkeypatch):
    monkeypatch.setattr(settings, "openai_sector_model", "claude-haiku-4-5")
    with _openai_active():
        assert llm_mod.resolve_role_model("sector") == settings.openai_cheap_model


def test_matching_sector_model_passes_through(monkeypatch):
    monkeypatch.setattr(settings, "openai_sector_model", "gpt-test-sector")
    with _openai_active():
        assert llm_mod.resolve_role_model("sector") == "gpt-test-sector"


def test_critic_role_follows_the_anthropic_env_when_anthropic_is_active(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_critic_model", "claude-test-critic")
    with patch.object(type(settings), "active_llm_provider",
                      new=property(lambda self: "anthropic")):
        assert llm_mod.resolve_role_model("critic") == "claude-test-critic"
        monkeypatch.setattr(settings, "anthropic_critic_model", "")
        assert llm_mod.resolve_role_model("critic") == settings.anthropic_strong_model


def test_critic_role_mirrors_the_forced_anthropic_route(monkeypatch):
    """critic_agent force-routes to Anthropic whenever a key is present, so
    the role table must report the Anthropic model under an OpenAI-active
    deployment — otherwise the ops page names a model the critic never ran."""
    monkeypatch.setattr(settings, "anthropic_api_key", "stub-key")
    monkeypatch.setattr(settings, "anthropic_critic_model", "claude-test-critic")
    with _openai_active():
        assert llm_mod.resolve_role_model("critic") == "claude-test-critic"
        assert llm_mod.model_summary()["role_models"]["critic"] == "claude-test-critic"
        monkeypatch.setattr(settings, "anthropic_critic_model", "")
        assert llm_mod.resolve_role_model("critic") == settings.anthropic_strong_model
        # Every other role still follows the active provider.
        assert llm_mod.resolve_role_model("pm") == settings.openai_strong_model


def test_critic_role_falls_back_to_the_active_provider_without_anthropic(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "anthropic_critic_model", "claude-test-critic")
    with _openai_active():
        assert llm_mod.resolve_role_model("critic") == settings.openai_strong_model


def test_chat_sdk_agent_never_carries_a_blank_pm_model(monkeypatch):
    """The chat surface builds a *real* `agents.Agent`; an unset
    OPENAI_PM_MODEL used to reach it as "" and silently drop chat to the
    non-SDK path. The SDK module is patched so nothing is ever run."""
    import agents as real_sdk

    from app.agents import chat_sdk

    captured: dict = {}

    class _FakeAgent:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(real_sdk, "Agent", _FakeAgent)
    monkeypatch.setattr(real_sdk, "function_tool", lambda fn: fn)
    monkeypatch.setattr(settings, "use_agents_sdk", True)
    monkeypatch.setattr(settings, "openai_api_key", "stub-key")
    monkeypatch.setattr(settings, "openai_pm_model", "")
    with _openai_active():
        assert chat_sdk._build_chat_agent() is not None
    assert captured["model"] == settings.openai_strong_model

    captured.clear()
    monkeypatch.setattr(settings, "openai_pm_model", "gpt-test-chat-pm")
    with _openai_active():
        chat_sdk._build_chat_agent()
    assert captured["model"] == "gpt-test-chat-pm"


def test_explicit_provider_overrides_the_active_one(monkeypatch):
    """The Agents SDK only speaks OpenAI, so sdk_runtime resolves against
    OpenAI even when Anthropic is the active provider."""
    monkeypatch.setattr(settings, "openai_sector_model", "")
    with patch.object(type(settings), "active_llm_provider",
                      new=property(lambda self: "anthropic")):
        assert llm_mod.resolve_role_model("sector", provider="openai") == settings.openai_cheap_model
        assert llm_mod.resolve_role_model("sector") == settings.anthropic_cheap_model


def test_unknown_role_is_an_error():
    import pytest
    with pytest.raises(ValueError):
        llm_mod.resolve_role_model("janitor")


def test_sdk_runtime_agents_never_carry_a_blank_model(monkeypatch):
    from app.agents import sdk_runtime
    monkeypatch.setattr(settings, "openai_sector_model", "")
    monkeypatch.setattr(settings, "openai_tool_model", "")
    monkeypatch.setattr(settings, "openai_pm_model", "")
    with _openai_active():
        assert sdk_runtime._build_sector_agent("Technology").model == settings.openai_cheap_model
        assert sdk_runtime._build_tool_agent("earnings").model == settings.openai_cheap_model
        assert sdk_runtime._sdk_model("pm") == settings.openai_strong_model


def test_model_summary_has_every_role_and_no_key_material(monkeypatch):
    import json
    monkeypatch.setattr(settings, "openai_api_key", "sk-summary-test-key-0123456789")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-summary-test-key-0123456789")
    summary = llm_mod.model_summary()
    assert set(summary) == {"active_provider", "provider_choice", "role_models", "configured"}
    assert set(summary["role_models"]) == {"pm", "sector", "tool", "macro", "critic", "strong", "cheap"}
    assert all(summary["role_models"].values()), "no role may resolve to a blank model"
    assert summary["configured"] == {"openai": True, "anthropic": True, "gemini": settings.has_gemini}
    text = json.dumps(summary)
    assert "sk-" not in text and "summary-test-key" not in text


def test_empty_model_string_falls_back_to_route_default():
    """An empty string for `model` should NOT override; it should resolve to
    `_model_for(provider, route)` — important for ergonomic env handling
    (an unset env var becomes "" not None, and we don't want that to break)."""
    calls: list[dict] = []

    def _fake(client, *, model, system, user, max_tokens):
        calls.append({"model": model})
        return {"ok": True}

    with patch.object(llm_mod, "_openai_client", return_value=object()), \
         patch.object(llm_mod, "_openai_chat_json", side_effect=_fake), \
         patch.object(type(settings), "active_llm_provider",
                      new=property(lambda self: "openai")), \
         patch.object(llm_mod, "_breaker_open", return_value=False):
        llm_mod.chat_json("hi", route="strong", model="")
    assert calls
    # Should have used the strong-route default, not the empty string.
    assert calls[0]["model"] == settings.openai_strong_model
