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
    llm as llm_mod,
    sector_agents,
    valuation_agent,
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


def test_pm_synthesis_uses_openai_pm_model(monkeypatch):
    monkeypatch.setattr(settings, "openai_pm_model", "TEST-PM-MODEL")
    calls, patches = _capture_openai_calls()
    _enter(patches)
    try:
        from app.agents.graph import _pm_synthesis
        _pm_synthesis(profile={"ticker": "T"}, findings={}, dcf=None)
    finally:
        _exit(patches)
    assert calls, "pm_synthesis should have called the OpenAI helper"
    assert calls[0]["model"] == "TEST-PM-MODEL"


def test_sector_agent_uses_openai_sector_model(monkeypatch):
    monkeypatch.setattr(settings, "openai_sector_model", "TEST-SECTOR-MODEL")
    calls, patches = _capture_openai_calls()
    _enter(patches)
    try:
        from app.services.fundamentals_service import get_full_financials
        fin = get_full_financials("NVDA")
        sector_agents.run_sector_agent(fin["profile"], fin["ratios"])
    finally:
        _exit(patches)
    assert calls and calls[0]["model"] == "TEST-SECTOR-MODEL"


def test_earnings_agent_uses_openai_tool_model(monkeypatch):
    monkeypatch.setattr(settings, "openai_tool_model", "TEST-TOOL-MODEL")
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
    assert calls and calls[0]["model"] == "TEST-TOOL-MODEL"


def test_filing_agent_uses_openai_tool_model(monkeypatch):
    monkeypatch.setattr(settings, "openai_tool_model", "TEST-TOOL-MODEL")
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
    assert calls and calls[0]["model"] == "TEST-TOOL-MODEL"


def test_valuation_agent_uses_openai_tool_model(monkeypatch):
    monkeypatch.setattr(settings, "openai_tool_model", "TEST-TOOL-MODEL")
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
    assert calls and calls[0]["model"] == "TEST-TOOL-MODEL"


def test_critic_uses_anthropic_critic_model_when_anthropic_configured(monkeypatch):
    """Critic flips to Anthropic when ANTHROPIC_API_KEY is set, and uses the
    `anthropic_critic_model` env explicitly — not the strong-route default."""
    monkeypatch.setattr(settings, "anthropic_api_key", "stub-key")
    monkeypatch.setattr(settings, "anthropic_critic_model", "TEST-CRITIC-MODEL")

    captured: list[dict] = []

    def _fake_anthropic(client, *, model, system, user, max_tokens):
        captured.append({"model": model})
        return '{"overall_assessment": "ok"}'

    with patch.object(llm_mod, "_anthropic_client", return_value=object()), \
         patch.object(llm_mod, "_anthropic_chat", side_effect=_fake_anthropic), \
         patch.object(llm_mod, "_breaker_open", return_value=False):
        critic_agent.run_critic({
            "ticker": "T", "rating_label": "Bullish",
            "sources_used": ["filing:0001"], "key_risks": [],
            "dcf_summary": {"summary": "x"},
        })
    assert captured and captured[0]["model"] == "TEST-CRITIC-MODEL"


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
