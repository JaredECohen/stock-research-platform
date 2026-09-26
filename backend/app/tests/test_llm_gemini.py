"""Gemini request shape, usage accounting, client bound and the grounding
cap (slice B7-M1; news trace N3(b,c,e) and its critique; model research
2026-09-25; DEVPLAN item 7).

Base note: production's new Gemini key cannot call gemini-2.5-flash, so the
news/social default is gemini-3.5-flash-lite (hotfix ca08521); the 2.5-flash
`thinking_budget=0` branch survives only for an explicit override.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from app.agents import llm
from app.config import Settings, settings
from app.database import SessionLocal
from app.models import LLMCallLog
from app.tests import llm_fakes
from app.tests.llm_fakes import FakeClient, gemini_response


@pytest.fixture(autouse=True)
def _reset():
    llm.reset_circuit_breaker()
    yield
    llm.reset_circuit_breaker()


@pytest.mark.parametrize("model,config", [
    ("gemini-3.5-flash-lite", {"thinking_level": "MINIMAL"}),
    ("gemini-3.8-flash", {"thinking_level": "LOW"}),       # never "minimal" on 3.7/3.8
    ("gemini-3.7-flash", {"thinking_level": "LOW"}),
    ("gemini-3.1-pro-preview", {"thinking_level": "LOW"}),
    ("gemini-2.5-flash", {"thinking_budget": 0}),          # explicit override only
    ("gemini-2.5-pro", None),                              # Pro cannot disable thinking
])
def test_gemini_thinking_level_per_model(monkeypatch, model, config):
    client = FakeClient(gemini_response())
    llm_fakes.live(monkeypatch, gemini=client)
    run_id = f"gem-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(run_id=run_id):
        llm.gemini_chat_text("p", model=model, action="news.search")
    (req,) = client.requests
    assert req["config"].get("thinking_config") == config
    (row,) = llm_fakes.rows_for(run_id)
    assert row.effort == {"MINIMAL": "minimal", "LOW": "low"}.get(
        (config or {}).get("thinking_level"), "budget0" if config else None)


def test_thinking_config_is_valid_for_the_pinned_sdk():
    """The dict form must validate against google-genai 2.22's config type."""
    types = pytest.importorskip("google.genai.types")
    for model in ("gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash"):
        cfg = types.GenerateContentConfig(thinking_config=llm._gemini_thinking(model)[1])
        assert cfg.thinking_config is not None


def test_the_news_default_sends_minimal_thinking(monkeypatch):
    client = FakeClient(gemini_response())
    llm_fakes.live(monkeypatch, gemini=client)
    monkeypatch.setattr(settings, "vertex_project_id", "")
    llm.gemini_chat_json("p", enable_search_grounding=True, action="news.search")
    assert Settings.model_fields["gemini_news_model"].default == "gemini-3.5-flash-lite"
    assert client.requests[0]["model"] == settings.gemini_news_model
    assert client.requests[0]["config"]["thinking_config"] == {"thinking_level": "MINIMAL"}


def test_gemini_usage_counts_thoughts_and_tool_tokens(monkeypatch):
    """prompt 100 + tool-use prompt 900 is input; candidates 200 + thoughts
    300 is output (thoughts bill as output and candidates leave them out)."""
    llm_fakes.live(monkeypatch, gemini=FakeClient(gemini_response(
        prompt=100, candidates=200, thoughts=300, tool_use=900, cached=400)))
    run_id = f"gem-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(run_id=run_id):
        llm.gemini_chat_text("p", model="gemini-3.5-flash-lite", action="news.search")
    (row,) = llm_fakes.rows_for(run_id)
    assert (row.tokens_in, row.tokens_out) == (1000, 500)
    assert row.reasoning_tokens == 300
    # Cached prompt tokens (inside the prompt count) reach the row and are
    # re-priced at the model's cache-read rate.
    assert row.cache_read_tokens == 400
    from app.services import llm_metrics
    expected = llm_metrics.estimate_cost_usd("gemini", "gemini-3.5-flash-lite", 1000, 500,
                                             cache_read_tokens=400)
    assert row.cost_usd == pytest.approx(expected)
    assert expected < llm_metrics.estimate_cost_usd("gemini", "gemini-3.5-flash-lite", 1000, 500)


def test_gemini_client_timeout_30000ms(monkeypatch):
    """google-genai 2.22 `HttpOptions.timeout` is MILLISECONDS; 30 would
    have been a 30 ms timeout (news trace critique)."""
    monkeypatch.setattr(settings, "enable_live_data", True)
    monkeypatch.setattr(settings, "use_demo_data", False)
    monkeypatch.setattr(settings, "vertex_project_id", "")
    monkeypatch.setattr(settings, "gemini_api_key", "stub-key")
    with patch.object(llm, "_genai") as fake_genai:
        llm._gemini_client()
    kwargs = fake_genai.Client.call_args.kwargs
    assert kwargs["api_key"] == "stub-key"
    options = kwargs["http_options"]
    timeout = options["timeout"] if isinstance(options, dict) else options.timeout
    assert timeout == 30_000


def _seed_grounded(n: int, *, success: bool = True, when: datetime | None = None,
                   tokens: tuple[int, int] = (0, 0), error: str | None = None) -> None:
    with SessionLocal() as db:
        LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
        for _ in range(n):
            db.add(LLMCallLog(agent_name="News Agent", provider="gemini",
                              model="gemini-3.5-flash-lite", grounded=True, success=success,
                              tokens_in=tokens[0], tokens_out=tokens[1], error_type=error,
                              generated_at=when or datetime.utcnow()))
        db.commit()


# Mid-afternoon UTC, so no test here straddles the midnight it computes.
_NOON = datetime(2026, 9, 25, 14, 0, 0)


@pytest.fixture
def _no_grounded_rows():
    def wipe():
        with SessionLocal() as db:
            LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
            db.query(LLMCallLog).filter(LLMCallLog.grounded.is_(True)).delete()
            db.commit()
    wipe()
    yield
    wipe()


def test_grounding_cap_counts_db_rows_and_skips(monkeypatch, _no_grounded_rows):
    """The cap is read from llm_call_logs (grounded responses, today UTC) so
    the web and worker processes share it; over the cap the call is a skip
    row and never reaches the client."""
    monkeypatch.setattr(settings, "gemini_grounded_max_per_day", 3)
    monkeypatch.setattr(llm, "_utcnow", lambda: _NOON)
    client = FakeClient(gemini_response())
    llm_fakes.live(monkeypatch, gemini=client)
    _seed_grounded(2, when=_NOON - timedelta(hours=1))
    # No response, nothing billed (transport errors, skip rows): not counted.
    _seed_grounded(5, success=False, when=_NOON - timedelta(hours=1),
                   error="provider_error:APIConnectionError")
    _seed_grounded(5, when=_NOON - timedelta(days=2))           # nor earlier days
    run_id = f"cap-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(run_id=run_id):
        assert llm.gemini_chat_json("p", enable_search_grounding=True,
                                    action="news.search") == {"ok": True}
        assert llm.gemini_chat_json("p", enable_search_grounding=True,
                                    action="news.search") is None
        # An ungrounded call is not capped.
        assert llm.gemini_chat_json("p", action="news.search") == {"ok": True}
    assert len(client.requests) == 2
    first, capped, plain = llm_fakes.rows_for(run_id)
    assert first.grounded is True and first.success is True
    assert capped.error_type == "skipped:grounding_cap" and capped.tokens_in == 0
    assert plain.grounded is None


def test_grounding_cap_counts_billed_responses_that_failed_to_parse(monkeypatch,
                                                                    _no_grounded_rows):
    """Google bills a grounded search whether or not the JSON parses; the
    cap bounds that spend (owner decision A11), so a parse-failure regime
    must trip it rather than run on until only the breaker stops it."""
    monkeypatch.setattr(settings, "gemini_grounded_max_per_day", 2)
    monkeypatch.setattr(llm, "_utcnow", lambda: _NOON)
    _seed_grounded(2, success=False, when=_NOON - timedelta(hours=1), tokens=(1200, 900),
                   error="invalid_json_response;finish_reason=max_tokens")
    assert llm._grounding_cap_reached() is True


def test_grounding_cap_is_the_utc_calendar_day(monkeypatch, _no_grounded_rows):
    """"Today UTC", not the last 24 hours: at 00:30 UTC, calls made at
    23:59 the previous day belong to yesterday's budget."""
    just_after_midnight = datetime(2026, 9, 25, 0, 30, 0)
    monkeypatch.setattr(settings, "gemini_grounded_max_per_day", 2)
    monkeypatch.setattr(llm, "_utcnow", lambda: just_after_midnight)
    _seed_grounded(2, when=datetime(2026, 9, 24, 23, 59, 0))
    assert llm._grounding_cap_reached() is False
    _seed_grounded(2, when=datetime(2026, 9, 25, 0, 10, 0))
    assert llm._grounding_cap_reached() is True


def test_grounding_cap_zero_skips_every_grounded_call(monkeypatch):
    monkeypatch.setattr(settings, "gemini_grounded_max_per_day", 0)
    client = FakeClient(gemini_response())
    llm_fakes.live(monkeypatch, gemini=client)
    run_id = f"cap0-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(run_id=run_id):
        assert llm.gemini_chat_json("p", enable_search_grounding=True,
                                    action="news.search") is None
    assert client.requests == []
    (row,) = llm_fakes.rows_for(run_id)
    assert row.error_type == "skipped:grounding_cap"


def test_grounding_cap_fails_open_when_the_count_fails(monkeypatch, caplog):
    """Losing the news pass is worse than a few over-cap grounded calls."""
    import logging

    import app.database as database

    def _broken():
        raise RuntimeError("db down")

    monkeypatch.setattr(settings, "gemini_grounded_max_per_day", 1)
    monkeypatch.setattr(database, "SessionLocal", _broken)
    with caplog.at_level(logging.WARNING, logger="app.agents.llm"):
        assert llm._grounding_cap_reached() is False
    assert any("grounding-cap count failed" in r.getMessage() for r in caplog.records)


def test_grounding_cap_default_is_250():
    assert Settings.model_fields["gemini_grounded_max_per_day"].default == 250


# The plan's `test_longdoc_default_is_preview_id` is already pinned on the
# base by hotfix ca08521 (`test_llm_prices.
# test_gemini_defaults_name_models_a_new_key_can_call`, which rejects
# "gemini-3.1-pro"); a copy here could not fail on the base, so it is not
# repeated.
