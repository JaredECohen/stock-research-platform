"""Real accounting/breaker paths with offline provider responses, never API calls."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import cache, database
from app.agents import llm
from app.config import settings
from app.models import LLMCallLog
from app.services import llm_metrics

PROVIDERS = ("anthropic", "openai", "gemini")


@pytest.fixture
def accounting(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'accounting.db'}")
    LLMCallLog.__table__.create(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(database, "SessionLocal", sessions)
    monkeypatch.setattr(llm_metrics, "SessionLocal", sessions)
    monkeypatch.setattr(settings, "openai_api_key", "offline-openai")
    monkeypatch.setattr(settings, "anthropic_api_key", "offline-anthropic")
    monkeypatch.setattr(settings, "gemini_api_key", "offline-gemini")
    monkeypatch.setattr(settings, "vertex_project_id", "")
    monkeypatch.setattr(settings, "llm_failover_enabled", False)
    monkeypatch.setattr(cache, "log_cost", lambda *args, **kwargs: None)
    # Offline clients stand in for live ones, so this runs as a live
    # deployment would: demo-only mode has no failover partner (critique #2).
    monkeypatch.setattr(llm, "_demo_only", lambda: False)
    llm.reset_circuit_breaker()
    llm.reset_failover_state()
    llm.last_usage()
    yield sessions
    llm.reset_circuit_breaker()
    llm.reset_failover_state()
    llm.last_usage()
    engine.dispose()


def _wire(monkeypatch, provider, text, *, input_tokens=121, output_tokens=37, response=None, stop_reason=None):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if isinstance(text, Exception):
            raise text
        if response is not None:
            return response
        if provider == "anthropic":
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
                content=[SimpleNamespace(text=text)],
                stop_reason=stop_reason,
            )
        if provider == "openai":
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
                choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=stop_reason)],
            )
        return SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=input_tokens, candidates_token_count=output_tokens),
            text=text,
            candidates=[SimpleNamespace(finish_reason=stop_reason)],
        )

    client = SimpleNamespace(
        messages=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        models=SimpleNamespace(generate_content=create),
    )
    monkeypatch.setattr(llm, f"_{provider}_client", lambda: client)
    return calls


def _rows(sessions):
    with sessions() as db:
        return list(db.execute(select(LLMCallLog).order_by(LLMCallLog.id)).scalars())


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("text,error", [
    ('sensitive response: {"unfinished":', "invalid_json_response"),
    ("", "empty_response"),
    ("null", "invalid_json_response"),
])
def test_unusable_json_records_one_failure_and_real_usage(accounting, monkeypatch, caplog, provider, text, error):
    calls = _wire(monkeypatch, provider, text)
    with caplog.at_level(logging.DEBUG), llm.llm_call_context(agent_name="Accounting", run_id="bad-json"):
        assert llm.chat_json("private prompt", provider_override=provider) is None
    rows = _rows(accounting)
    assert len(calls) == len(rows) == 1
    assert rows[0].success is False
    assert rows[0].error == error
    assert (rows[0].tokens_in, rows[0].tokens_out) == (121, 37)
    assert rows[0].agent_name == "Accounting" and rows[0].run_id == "bad-json"
    assert llm._FAILURE_COUNTERS[provider] == 1
    info = llm_metrics.cost_per_run("bad-json")
    assert info["n_calls"] == info["n_failures"] == 1
    assert info["tokens_total"] == 158
    assert info["cost_usd_total"] > 0
    assert llm.last_usage()["total_tokens"] == 158
    assert llm.last_usage() is None
    assert "sensitive response" not in caplog.text
    assert "private prompt" not in caplog.text


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("text", ['{}', '[]', 'false', '0'])
def test_existing_falsey_json_outputs_remain_successes(accounting, monkeypatch, provider, text):
    calls = _wire(monkeypatch, provider, text)
    llm._FAILURE_COUNTERS[provider] = 1
    out = llm.chat_json("unchanged prompt", provider_override=provider)
    assert out == json.loads(text)
    assert type(out) is type(json.loads(text))
    assert len(calls) == 1
    assert _rows(accounting)[0].success is True
    assert _rows(accounting)[0].error == ""
    assert llm._FAILURE_COUNTERS[provider] == 0


@pytest.mark.parametrize("provider", ("anthropic", "gemini"))
@pytest.mark.parametrize("text,expected", [
    ('```json\n{"ok": true}\n```', {"ok": True}),
    ('{"items":[{"value":1},{"value":', {"items": [{"value": 1}]}),
])
def test_existing_json_recovery_runs_once_before_accounting(accounting, monkeypatch, provider, text, expected):
    calls = _wire(monkeypatch, provider, text)
    original = llm._extract_json
    parse_calls = []

    def parse_once(value):
        parse_calls.append(value)
        return original(value)

    monkeypatch.setattr(llm, "_extract_json", parse_once)
    assert llm.chat_json("recover with existing rules", provider_override=provider) == expected
    assert len(calls) == len(parse_calls) == len(_rows(accounting)) == 1
    assert _rows(accounting)[0].success is True
    assert llm._FAILURE_COUNTERS[provider] == 0


@pytest.mark.parametrize("provider", PROVIDERS)
def test_repeated_unusable_json_trips_existing_breaker_without_extra_calls(accounting, monkeypatch, provider):
    calls = _wire(monkeypatch, provider, 'no JSON object')
    with llm.llm_call_context(run_id="breaker-json"):
        for _ in range(4):
            assert llm.chat_json("same request", provider_override=provider) is None
    assert len(calls) == 3
    # The fourth call made no request: it is written as a skip row (design
    # gap G11), which every cost aggregate excludes.
    rows = _rows(accounting)
    assert len(rows) == 4
    assert [r.error_type for r in rows][-1] == "skipped:breaker_open"
    assert (rows[-1].tokens_in, rows[-1].tokens_out) == (0, 0)
    assert llm._FAILURE_COUNTERS[provider] == 3
    assert llm._breaker_open(provider)
    info = llm_metrics.cost_per_run("breaker-json")
    assert info["n_calls"] == info["n_failures"] == 3
    assert info["tokens_total"] == 3 * 158


@pytest.mark.parametrize("backup_text", ['{"backup":true}', 'also malformed'])
def test_anthropic_parse_failure_retains_usage_and_fails_over_exactly_once(accounting, monkeypatch, backup_text):
    monkeypatch.setattr(settings, "llm_failover_enabled", True)
    primary = _wire(monkeypatch, "anthropic", "malformed primary", input_tokens=101, output_tokens=11)
    backup = _wire(monkeypatch, "openai", backup_text, input_tokens=202, output_tokens=22)
    with llm.llm_call_context(agent_name="META-shaped probe", run_id="failover-json"):
        out = llm.chat_json("same request", provider_override="anthropic", model="claude-offline-model", route="strong")
    assert out == ({"backup": True} if backup_text.startswith('{') else None)
    assert len(primary) == len(backup) == 1
    assert primary[0]["model"] == "claude-offline-model"
    assert backup[0]["model"] == settings.openai_strong_model
    rows = _rows(accounting)
    assert len(rows) == 2
    assert [(row.provider, row.tokens_in, row.tokens_out) for row in rows] == [
        ("anthropic", 101, 11), ("openai", 202, 22),
    ]
    assert rows[0].success is False
    assert rows[1].success is (out is not None)
    info = llm_metrics.cost_per_run("failover-json")
    assert info["n_calls"] == 2 and info["tokens_total"] == 336
    assert info["n_failures"] == (1 if out is not None else 2)
    assert llm.consume_failover_events() == [{"from": "anthropic", "to": "openai", "reason": "call_failed"}]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_http_failure_is_distinct_from_parse_failure_without_body_logging(accounting, monkeypatch, caplog, provider):
    request = httpx.Request("POST", "https://offline.invalid/test")
    failure = httpx.HTTPStatusError(
        "private response body and key material", request=request,
        response=httpx.Response(429, request=request),
    )
    calls = _wire(monkeypatch, provider, failure)
    with caplog.at_level(logging.DEBUG):
        assert llm.chat_json("private request", provider_override=provider) is None
    row = _rows(accounting)[0]
    assert len(calls) == 1
    assert row.success is False and row.error == "provider_error:HTTPStatusError"
    assert row.tokens_in == row.tokens_out == 0  # no response usage was available
    assert llm._FAILURE_COUNTERS[provider] == 1
    assert "private response body" not in caplog.text
    assert "key material" not in caplog.text
    assert "private request" not in caplog.text


@pytest.mark.parametrize("provider", PROVIDERS)
def test_response_processing_failure_preserves_usage_already_received(accounting, monkeypatch, caplog, provider):
    class BadResponse:
        usage = SimpleNamespace(input_tokens=121, output_tokens=37, prompt_tokens=121, completion_tokens=37)
        usage_metadata = SimpleNamespace(prompt_token_count=121, candidates_token_count=37)

        def __getattr__(self, name):
            raise ValueError("private response detail")

    calls = _wire(monkeypatch, provider, None, response=BadResponse())
    with caplog.at_level(logging.DEBUG):
        assert llm.chat_json("request", provider_override=provider) is None
    row = _rows(accounting)[0]
    assert len(calls) == 1 and row.success is False
    assert row.error == "response_error:ValueError"
    assert (row.tokens_in, row.tokens_out) == (121, 37)
    assert "private response detail" not in caplog.text


@pytest.mark.parametrize("provider,reason,diagnostic", [
    ("anthropic", "max_tokens", "stop_reason=max_tokens"),
    ("openai", "length", "finish_reason=length"),
    ("gemini", SimpleNamespace(name="MAX_TOKENS"), "finish_reason=max_tokens"),
])
def test_parse_failure_keeps_existing_safe_stop_metadata(accounting, monkeypatch, provider, reason, diagnostic):
    calls = _wire(monkeypatch, provider, "unfinished JSON", stop_reason=reason)
    assert llm.chat_json("request", provider_override=provider) is None
    row = _rows(accounting)[0]
    assert len(calls) == 1
    assert row.error == f"invalid_json_response;{diagnostic}"
    assert (row.tokens_in, row.tokens_out) == (121, 37)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_unknown_stop_metadata_cannot_leak_response_content(accounting, monkeypatch, caplog, provider):
    _wire(monkeypatch, provider, "unfinished JSON", stop_reason="private response and key material")
    with caplog.at_level(logging.DEBUG):
        assert llm.chat_json("request", provider_override=provider) is None
    row = _rows(accounting)[0]
    assert row.error.endswith("=unrecognized")
    assert "private response" not in row.error + caplog.text
    assert "key material" not in row.error + caplog.text


@pytest.mark.parametrize("provider", ("anthropic", "gemini"))
def test_text_mode_keeps_non_json_responses_successful(accounting, monkeypatch, provider):
    calls = _wire(monkeypatch, provider, "ordinary text without JSON")
    llm._FAILURE_COUNTERS[provider] = 1
    assert llm.chat_text("plain request", provider_override=provider) == "ordinary text without JSON"
    assert len(calls) == 1
    row = _rows(accounting)[0]
    assert row.success is True and row.error == ""
    assert (row.tokens_in, row.tokens_out) == (121, 37)
    assert llm._FAILURE_COUNTERS[provider] == 0
