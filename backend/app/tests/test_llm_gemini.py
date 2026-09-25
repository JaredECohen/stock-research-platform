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
        prompt=100, candidates=200, thoughts=300, tool_use=900)))
    run_id = f"gem-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(run_id=run_id):
        llm.gemini_chat_text("p", model="gemini-3.5-flash-lite", action="news.search")
    (row,) = llm_fakes.rows_for(run_id)
    assert (row.tokens_in, row.tokens_out) == (1000, 500)
    assert row.reasoning_tokens == 300
    from app.services import llm_metrics
    assert row.cost_usd == pytest.approx(
        llm_metrics.estimate_cost_usd("gemini", "gemini-3.5-flash-lite", 1000, 500))


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


def _seed_grounded(n: int, *, success: bool = True, when: datetime | None = None) -> None:
    with SessionLocal() as db:
        LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
        for _ in range(n):
            db.add(LLMCallLog(agent_name="News Agent", provider="gemini",
                              model="gemini-3.5-flash-lite", grounded=True, success=success,
                              generated_at=when or datetime.utcnow()))
        db.commit()


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
    """The cap is read from llm_call_logs (successful grounded rows, today
    UTC) so the web and worker processes share it; over the cap the call is
    a skip row and never reaches the client."""
    monkeypatch.setattr(settings, "gemini_grounded_max_per_day", 3)
    client = FakeClient(gemini_response())
    llm_fakes.live(monkeypatch, gemini=client)
    _seed_grounded(2)
    _seed_grounded(5, success=False)                          # failures do not count
    _seed_grounded(5, when=datetime.utcnow() - timedelta(days=2))  # nor earlier days
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


def test_grounding_cap_default_is_250():
    assert Settings.model_fields["gemini_grounded_max_per_day"].default == 250


# The plan's `test_longdoc_default_is_preview_id` is already pinned on the
# base by hotfix ca08521 (`test_llm_prices.
# test_gemini_defaults_name_models_a_new_key_can_call`, which rejects
# "gemini-3.1-pro"); a copy here could not fail on the base, so it is not
# repeated.
