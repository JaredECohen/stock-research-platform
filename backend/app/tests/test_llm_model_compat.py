"""Request compatibility for the 2026-09-25 model fleet (slice B7-M1;
bull/bear design §10.3 and its critique #4, #5, #11, #12; model research
2026-09-25).

Each test captures the exact kwargs sent to a fake provider client. The
load-bearing guarantee is `test_today_models_request_kwargs_unchanged`:
the models production runs today (claude-opus-4-8, claude-haiku-4-5,
gpt-5.5, gpt-4.1-mini) get byte-identical requests, so this slice can
deploy (wave G1) without changing a single model call. The one deliberate
exception, gpt-5.x as a FAILOVER target, is pinned separately.
"""
from __future__ import annotations

import uuid

import pytest

from app.agents import llm
from app.config import settings
from app.tests import llm_fakes
from app.tests.llm_fakes import FakeClient, anthropic_response, openai_response


@pytest.fixture(autouse=True)
def _reset():
    llm.reset_circuit_breaker()
    llm.reset_failover_state()
    yield
    llm.reset_circuit_breaker()
    llm.reset_failover_state()


def _anthropic(monkeypatch, *responses, partner=None):
    client = FakeClient(*(responses or (anthropic_response(),)))
    llm_fakes.live(monkeypatch, anthropic=client, openai=partner, active="anthropic")
    if partner is None:
        monkeypatch.setattr(settings, "openai_api_key", "")
    return client


def _openai(monkeypatch, *responses, partner=None):
    client = FakeClient(*(responses or (openai_response(),)))
    llm_fakes.live(monkeypatch, openai=client, anthropic=partner, active="openai")
    if partner is None:
        monkeypatch.setattr(settings, "anthropic_api_key", "")
    return client


# ---------------------------------------------------------------------------
# Sampling parameters, thinking, effort
# ---------------------------------------------------------------------------

def test_opus_5_5_sends_no_temperature(monkeypatch):
    client = _anthropic(monkeypatch)
    llm.chat_json("p", model="claude-opus-5-5", action="pm.synthesis")
    (req,) = client.requests
    assert "temperature" not in req and "top_p" not in req and "top_k" not in req
    assert "thinking" not in req, "Opus 5.5 cannot disable thinking; the param is never sent"
    # Its API default is "medium"; sent explicitly so the row says what ran.
    assert req["output_config"] == {"effort": "medium"}
    assert req["max_tokens"] == 16000


@pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-fable-5-1", "claude-opus-5"])
def test_newer_claude_families_send_no_temperature(monkeypatch, model):
    client = _anthropic(monkeypatch)
    llm.chat_text("p", model=model, action="analyst.sector")
    assert "temperature" not in client.requests[0]


def test_claude_3_still_sends_temperature(monkeypatch):
    client = _anthropic(monkeypatch)
    llm.chat_text("p", model="claude-3-5-haiku-20241022", action="analyst.sector")
    (req,) = client.requests
    assert req["temperature"] == 0.3
    assert "output_config" not in req


def test_gpt6_uses_max_completion_tokens_no_temperature(monkeypatch):
    client = _openai(monkeypatch)
    llm.chat_json("p", model="gpt-6-sol", action="pm.synthesis", effort="high")
    (req,) = client.requests
    assert "temperature" not in req and "max_tokens" not in req
    assert req["max_completion_tokens"] == 25000, "GPT-6 reserves room for reasoning"
    assert req["reasoning_effort"] == "high"


def test_today_models_request_kwargs_unchanged(monkeypatch):
    """What production sends today, byte for byte (the G1 deploy gate)."""
    system_block = [{"type": "text", "text": "sys\n\nReturn ONLY valid JSON, no prose.",
                     "cache_control": {"type": "ephemeral"}}]
    for model in ("claude-opus-4-8", "claude-haiku-4-5"):
        client = _anthropic(monkeypatch)
        llm.chat_json("p", system="sys", model=model, action="analyst.sector")
        assert client.requests == [{
            "model": model, "max_tokens": 1600, "system": system_block,
            "messages": [{"role": "user", "content": "p"}],
        }], model
    client = _openai(monkeypatch)
    llm.chat_json("p", system="sys", model="gpt-5.5", action="analyst.sector")
    assert client.requests == [{
        "model": "gpt-5.5",
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": "p\n\nReturn ONLY valid JSON."}],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 1600,
    }]
    client = _openai(monkeypatch)
    llm.chat_text("p", model="gpt-4.1-mini", action="analyst.sector")
    assert client.requests == [{
        "model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "p"}],
        "max_tokens": 600, "temperature": 0.3,
    }]


def test_gpt5_failover_hop_gets_the_reasoning_floor(monkeypatch):
    """The documented change to today's requests: gpt-5.5 as a FAILOVER
    target no longer gets the critic's 1,600 tokens, which reasoning used
    up before any output (bullish-skew F5)."""
    partner = FakeClient(openai_response())
    primary = _anthropic(monkeypatch, anthropic_response("not json"), partner=partner)
    llm.chat_json("p", route="strong", action="risk.committee_review")
    assert primary.requests[0]["max_tokens"] == 1600
    (hop,) = partner.requests
    assert hop["model"] == settings.openai_strong_model == "gpt-5.5"
    assert hop["max_completion_tokens"] == 25000


@pytest.mark.parametrize("model,expected", [
    ("claude-opus-5-5", 16000), ("claude-opus-5", 16000), ("claude-fable-5-1", 16000),
    ("claude-sonnet-5", 16000),                       # critique #12
    ("claude-opus-4-8", 1600), ("claude-haiku-4-5", 1600), ("claude-sonnet-4-6", 1600),
])
def test_thinking_floor_only_on_always_thinking_models(monkeypatch, model, expected):
    client = _anthropic(monkeypatch)
    llm.chat_json("p", model=model, action="analyst.sector")
    assert client.requests[0]["max_tokens"] == expected


def test_anthropic_nonstream_clamp_including_failover_hop(monkeypatch):
    """The SDK refuses a non-streaming request above ~21.3k max_tokens;
    a 25k reviewer budget used to be forwarded verbatim to the Anthropic
    failover hop and fail there (critique #5)."""
    client = _anthropic(monkeypatch)
    llm.chat_json("p", model="claude-opus-4-8", max_tokens=25000, action="risk.review")
    assert client.requests[0]["max_tokens"] == 16000

    partner = FakeClient(anthropic_response())
    primary = _openai(monkeypatch, openai_response("not json"), partner=partner)
    llm.chat_json("p", model="gpt-6-astra", max_tokens=25000, action="risk.review")
    assert primary.requests[0]["max_completion_tokens"] == 25000
    assert partner.requests[0]["max_tokens"] == 16000


def test_effort_never_sent_to_haiku(monkeypatch):
    client = _anthropic(monkeypatch)
    llm.chat_json("p", model="claude-haiku-4-5", effort="high", action="chart.commentary")
    assert "output_config" not in client.requests[0]
    client = _anthropic(monkeypatch)
    llm.chat_json("p", model="claude-opus-4-8", effort="high", action="pm.synthesis")
    assert client.requests[0]["output_config"] == {"effort": "high"}
    client = _openai(monkeypatch)
    llm.chat_json("p", model="gpt-4.1-mini", effort="high", action="chart.commentary")
    assert "reasoning_effort" not in client.requests[0]


def test_effort_is_recorded_as_sent(monkeypatch):
    _openai(monkeypatch)
    run_id = f"eff-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(run_id=run_id):
        llm.chat_json("p", model="gpt-5.5", effort="medium", action="analyst.sector")
        llm.chat_json("p", model="gpt-4.1-mini", effort="medium", action="analyst.sector")
    first, second = llm_fakes.rows_for(run_id)
    assert (first.effort, second.effort) == ("medium", None)
    assert first.max_tokens == 1600


def test_astra_effort_none_rejected_at_request_time(monkeypatch):
    _openai(monkeypatch)
    with pytest.raises(ValueError, match="gpt-6-astra"):
        llm.chat_json("p", model="gpt-6-astra", effort="none", action="risk.review")


def test_schema_sends_strict_json_schema_on_openai_only(monkeypatch):
    schema = {"title": "Review", "type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    client = _openai(monkeypatch)
    llm.chat_json("p", model="gpt-6-astra", schema=schema, action="risk.review")
    assert client.requests[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Review", "schema": schema, "strict": True},
    }
    client = _anthropic(monkeypatch)
    llm.chat_json("p", model="claude-opus-4-8", schema=schema, action="risk.review")
    assert "response_format" not in client.requests[0]
    assert "output_config" not in client.requests[0]


# ---------------------------------------------------------------------------
# failover=False, breaker isolation, refusals
# ---------------------------------------------------------------------------

def test_failover_false_returns_none_without_hop_or_breaker_count(monkeypatch):
    partner = FakeClient(openai_response())
    _anthropic(monkeypatch, anthropic_response("not json"), partner=partner)
    run_id = f"nofo-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(run_id=run_id):
        out = llm.chat_json("p", model="claude-opus-5-5", failover=False,
                            action="debate.bull_open")
    assert out is None
    assert partner.requests == [], "no provider hop"
    assert llm._FAILURE_COUNTERS["anthropic"] == 0, "a content failure is not provider health"
    assert llm.get_failover_state()["count"] == 0
    (row,) = llm_fakes.rows_for(run_id)
    assert row.error_type == "invalid_json_response"


def test_failover_false_still_counts_transport_errors(monkeypatch):
    _anthropic(monkeypatch, RuntimeError("connection reset"))
    assert llm.chat_json("p", failover=False, action="debate.bull_open") is None
    assert llm._FAILURE_COUNTERS["anthropic"] == 1


def test_debate_failures_do_not_open_breaker_for_pm(monkeypatch):
    """Two failed openings plus a same-route retry used to open the shared
    breaker just before PM synthesis, sending the PM to failover or the
    keyword path (critique #4)."""
    client = _anthropic(monkeypatch, anthropic_response("truncated {", stop_reason="max_tokens"),
                        anthropic_response("truncated {", stop_reason="max_tokens"),
                        anthropic_response("truncated {", stop_reason="max_tokens"),
                        anthropic_response('{"rating": "Neutral"}'))
    for action in ("debate.bull_open", "debate.bear_open", "debate.bull_open"):
        assert llm.chat_json("p", failover=False, action=action) is None
    assert not llm.breaker_open("anthropic")
    assert llm.chat_json("p", route="strong", action="pm.synthesis") == {"rating": "Neutral"}
    assert len(client.requests) == 4


def test_refusal_classified_no_breaker_count_when_failover_false(monkeypatch):
    partner = FakeClient(openai_response())
    _anthropic(monkeypatch,
               anthropic_response("", stop_reason="refusal", refusal_category="bio"),
               partner=partner)
    run_id = f"ref-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(run_id=run_id):
        out = llm.chat_json("p", model="claude-opus-5-5", failover=False,
                            action="debate.bear_open", ticker="LLY")
    assert out is None
    usage = llm.last_usage()
    assert usage["refused"] is True
    (row,) = llm_fakes.rows_for(run_id)
    assert row.error_type == "refusal:bio" and row.finish_reason == "refusal"
    assert llm._FAILURE_COUNTERS["anthropic"] == 0
    assert partner.requests == []


def test_refusal_fails_over_when_allowed(monkeypatch):
    """Normal calls fail over on a refusal (plan P16); the reason says why."""
    partner = FakeClient(openai_response())
    _anthropic(monkeypatch,
               anthropic_response("", stop_reason="refusal", refusal_category="unknown-new"),
               partner=partner)
    run_id = f"reffo-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(run_id=run_id):
        assert llm.chat_json("p", action="analyst.sector") == {"ok": True}
    first, second = llm_fakes.rows_for(run_id)
    assert first.error_type == "refusal:other"
    assert second.failover_reason == "refused"
    assert llm._FAILURE_COUNTERS["anthropic"] == 0


def test_gpt6_long_context_request_warns(monkeypatch, caplog):
    """GPT-6 bills a request over 272K input at 2x/1.5x, which the estimate
    does not price: never silent."""
    import logging
    _openai(monkeypatch, openai_response(prompt_tokens=300_000))
    with caplog.at_level(logging.WARNING, logger="app.agents.llm"):
        llm.chat_json("p", model="gpt-6-sol", action="pm.synthesis")
        llm.chat_json("p", model="gpt-5.5", action="pm.synthesis")
    hits = [r.getMessage() for r in caplog.records if "long-context" in r.getMessage()]
    assert len(hits) == 1 and "300000" in hits[0] and "gpt-6-sol" in hits[0]


def test_public_failover_partner_and_breaker_open(monkeypatch):
    llm_fakes.live(monkeypatch, openai=FakeClient(openai_response()),
                   anthropic=FakeClient(anthropic_response()))
    assert llm.failover_partner("anthropic") == "openai"
    assert llm.failover_partner("gemini") is None
    assert llm.breaker_open("anthropic") is False
    llm._FAILURE_COUNTERS["anthropic"] = llm._BREAKER_THRESHOLD
    import time
    llm._FAILURE_LAST_AT["anthropic"] = time.time()
    assert llm.breaker_open("anthropic") is True
