"""Prompt caching (2026-09): request shape, cache accounting, cost math.

Offline — provider clients are recording fakes; nothing reaches the network.
The invariant under test: caching changes how the request is *packaged*
(system block, split user turn, usage fields), never the prompt bytes.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import cache, database
from app.agents import llm, prompts
from app.config import settings
from app.models import LLMCallLog
from app.services import llm_metrics

JSON_SUFFIX = "\n\nReturn ONLY valid JSON, no prose."


@pytest.fixture
def offline(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'caching.db'}")
    LLMCallLog.__table__.create(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(database, "SessionLocal", sessions)
    monkeypatch.setattr(llm_metrics, "SessionLocal", sessions)
    monkeypatch.setattr(settings, "openai_api_key", "offline-openai")
    monkeypatch.setattr(settings, "anthropic_api_key", "offline-anthropic")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "vertex_project_id", "")
    monkeypatch.setattr(settings, "llm_failover_enabled", False)
    monkeypatch.setattr(settings, "llm_prompt_caching_enabled", True)
    monkeypatch.setattr(cache, "log_cost", lambda *args, **kwargs: None)
    llm.reset_circuit_breaker()
    llm.reset_failover_state()
    llm.last_usage()
    yield sessions
    llm.reset_circuit_breaker()
    llm.reset_failover_state()
    llm.last_usage()
    engine.dispose()


def _anthropic_fake(monkeypatch, *, text='{"ok": true}', cache_write=0, cache_read=0):
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=121, output_tokens=37,
                cache_creation_input_tokens=cache_write,
                cache_read_input_tokens=cache_read,
            ),
            content=[SimpleNamespace(text=text)],
            stop_reason="end_turn",
        )

    monkeypatch.setattr(
        llm, "_anthropic_client",
        lambda: SimpleNamespace(messages=SimpleNamespace(create=create)),
    )
    return calls


def _one_row(sessions) -> LLMCallLog:
    with sessions() as db:
        return db.execute(select(LLMCallLog)).scalars().one()


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------

def test_system_prompt_becomes_one_cached_block_with_identical_text(offline, monkeypatch):
    calls = _anthropic_fake(monkeypatch)
    assert llm.chat_json("private prompt", system="You are the PM.",
                         provider_override="anthropic") == {"ok": True}
    system = calls[0]["system"]
    assert isinstance(system, list) and len(system) == 1
    assert system[0] == {
        "type": "text",
        "text": "You are the PM." + JSON_SUFFIX,
        "cache_control": {"type": "ephemeral"},
    }
    # No static prefix declared → the user turn is untouched.
    assert calls[0]["messages"] == [{"role": "user", "content": "private prompt"}]


def test_text_route_caches_the_bare_system_prompt(offline, monkeypatch):
    calls = _anthropic_fake(monkeypatch, text="prose")
    assert llm.chat_text("q", system="sys", provider_override="anthropic") == "prose"
    assert calls[0]["system"][0]["text"] == "sys"
    assert calls[0]["messages"][0]["content"] == "q"


def test_flag_off_restores_the_legacy_request_shape(offline, monkeypatch):
    monkeypatch.setattr(settings, "llm_prompt_caching_enabled", False)
    calls = _anthropic_fake(monkeypatch)
    with llm.llm_call_context(static_prefix_chars=5):
        llm.chat_json("AAA\n\nBBB", system="sys", provider_override="anthropic")
    assert calls[0]["system"] == "sys" + JSON_SUFFIX
    assert calls[0]["messages"][0]["content"] == "AAA\n\nBBB"


@pytest.mark.parametrize("n,expect_split", [
    (5, True),    # lands right after the "\n\n" join
    (3, False),   # mid-text: refuse rather than split on a non-newline
    (8, False),   # whole prompt: nothing volatile left to separate
    (0, False),   # unset
])
def test_static_prefix_splits_only_on_a_newline_boundary(offline, monkeypatch, n, expect_split):
    prompt = "AAA\n\nBBB"
    calls = _anthropic_fake(monkeypatch)
    with llm.llm_call_context(static_prefix_chars=n):
        llm.chat_json(prompt, system="sys", provider_override="anthropic")
    content = calls[0]["messages"][0]["content"]
    if expect_split:
        assert content == [
            {"type": "text", "text": "AAA\n\n", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "BBB"},
        ]
        assert "".join(block["text"] for block in content) == prompt
    else:
        assert content == prompt


def test_static_prefix_is_scoped_to_its_context(offline, monkeypatch):
    calls = _anthropic_fake(monkeypatch)
    with llm.llm_call_context(agent_name="PM", static_prefix_chars=5):
        llm.chat_json("AAA\n\nBBB", provider_override="anthropic")
    llm.chat_json("AAA\n\nBBB", provider_override="anthropic")
    assert isinstance(calls[0]["messages"][0]["content"], list)
    assert calls[1]["messages"][0]["content"] == "AAA\n\nBBB"
    assert "static_prefix_chars" not in llm.current_call_context()


def test_pm_synthesis_marks_its_template_as_the_cached_prefix(offline, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")  # Anthropic is the only provider
    calls = _anthropic_fake(monkeypatch, text='{"rating_label": "Neutral"}')
    from app.agents.graph import _pm_synthesis
    for _ in range(2):
        _pm_synthesis(profile={"ticker": "T"}, findings={}, dcf=None)
    assert len(calls) == 2
    heads = []
    for call in calls:
        content = call["messages"][0]["content"]
        assert isinstance(content, list) and len(content) == 2
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        assert content[0]["text"] == prompts.PM_SYNTHESIS_PROMPT + "\n\n"
        assert "cache_control" not in content[1]
        # The volatile tail carries the PM context (when any) and the findings.
        assert "\n\nFindings:\n" in content[1]["text"] or content[1]["text"].startswith("Findings:\n")
        heads.append(content[0]["text"])
        assert call["system"][0]["text"] == prompts.PM_SYSTEM + JSON_SUFFIX
    assert heads[0] == heads[1]


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------

def test_anthropic_cache_usage_is_persisted_and_priced(offline, monkeypatch):
    _anthropic_fake(monkeypatch, cache_write=300, cache_read=1200)
    with llm.llm_call_context(agent_name="Caching", run_id="cache-run"):
        assert llm.chat_json("p", provider_override="anthropic") == {"ok": True}
    usage = llm.last_usage()
    assert usage["cache_write_tokens"] == 300 and usage["cache_read_tokens"] == 1200
    assert usage["total_tokens"] == 158  # input + output; cached tokens are priced, not summed
    row = _one_row(offline)
    assert (row.tokens_in, row.cache_write_tokens, row.cache_read_tokens) == (121, 300, 1200)
    info = llm_metrics.cost_per_run("cache-run")
    assert info["cache_read_tokens"] == 1200 and info["cache_write_tokens"] == 300
    call = info["calls"][0]
    assert (call["cache_read_tokens"], call["cache_write_tokens"]) == (1200, 300)
    expected = llm_metrics.estimate_cost_usd(
        "anthropic", row.model, 121, 37, cache_read_tokens=1200, cache_write_tokens=300,
    )
    assert abs(call["cost_usd"] - expected) < 1e-9
    assert llm_metrics.cost_per_agent()["Caching"]["cache_read_tokens"] == 1200
    assert llm_metrics.cost_per_provider()["anthropic"]["cache_write_tokens"] == 300


def test_openai_cached_prompt_tokens_are_recorded(offline, monkeypatch):
    def create(**kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1000, completion_tokens=10,
                prompt_tokens_details=SimpleNamespace(cached_tokens=640),
            ),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'),
                                     finish_reason="stop")],
        )

    monkeypatch.setattr(
        llm, "_openai_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    with llm.llm_call_context(run_id="openai-cache"):
        assert llm.chat_json("p", provider_override="openai") == {"ok": True}
    assert llm.last_usage()["cache_read_tokens"] == 640
    row = _one_row(offline)
    assert (row.cache_read_tokens, row.cache_write_tokens) == (640, 0)


def test_cost_math_prices_cache_reads_and_writes_per_provider():
    p_in, _ = llm_metrics.MODEL_PRICES_PER_MTOK["claude-haiku-4-5"]
    base = llm_metrics.estimate_cost_usd("anthropic", "claude-haiku-4-5", 1_000_000, 0)
    assert abs(base - p_in) < 1e-6
    read = llm_metrics.estimate_cost_usd(
        "anthropic", "claude-haiku-4-5", 0, 0, cache_read_tokens=1_000_000)
    assert abs(read - 0.1 * p_in) < 1e-6
    write = llm_metrics.estimate_cost_usd(
        "anthropic", "claude-haiku-4-5", 0, 0, cache_write_tokens=1_000_000)
    assert abs(write - 1.25 * p_in) < 1e-6
    # OpenAI reports cached tokens inside prompt_tokens and bills them at half price.
    o_in, _ = llm_metrics.MODEL_PRICES_PER_MTOK["gpt-5"]
    openai = llm_metrics.estimate_cost_usd(
        "openai", "gpt-5", 1_000_000, 0, cache_read_tokens=400_000)
    assert abs(openai - (600_000 * o_in + 400_000 * 0.5 * o_in) / 1e6) < 1e-6
    # Malformed inputs never raise.
    assert llm_metrics.estimate_cost_usd(
        "openai", "gpt-5", None, None, cache_read_tokens=None, cache_write_tokens=-5) == 0.0
