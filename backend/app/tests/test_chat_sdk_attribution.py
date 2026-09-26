"""Ask-the-PM on the OpenAI Agents SDK: attribution and privacy (slice
B8-C1; FIX-020; attribution critique #1, #4, #5, #18; plan P14).

The real `openai-agents` package runs every turn here; only the transport
is fake. A real `AsyncOpenAI` client is installed as the SDK's default
client with `responses.create` replaced by a script, so the request the
SDK builds for GPT-6 (no temperature, reasoning effort) is the one that
would go out, and the RunHooks usage rows come from the SDK's own
lifecycle. The hook signature and the usage field names differ between
SDK versions, so these tests run only on the pinned version (CI installs
it exactly); run them in the CI-version venv (CLAUDE.md).

Nothing reaches a network: the fake client answers in-process, trace
export is disabled, and the legacy provider clients are `llm_fakes`.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from app.agents import chat_sdk, llm
from app.agents import orchestrator as orch_mod
from app.config import settings
from app.schemas import ChatMessage
from app.services import memo_store
from app.tests import llm_fakes
from app.tests.factories import make_memo
from app.tests.gating_helpers import purge_memos
from app.tests.llm_fakes import FakeClient, anthropic_response, openai_response

BACKEND = Path(__file__).resolve().parents[2]
OUTPUT_SENTINEL = "OUTPUT-SENTINEL-7c1f"
ARG_SENTINEL = "ZZARGSENT"
REFUSAL_SENTINEL = "REFUSAL-SENTINEL-93aa"
TICKER = "ZZSDK"


def _pinned_sdk_version() -> str:
    text = (BACKEND / "requirements.txt").read_text()
    m = re.search(r"^openai-agents==([^\s#]+)", text, re.MULTILINE)
    assert m, "openai-agents is no longer pinned in requirements.txt"
    return m.group(1)


@pytest.fixture(autouse=True)
def _pinned_sdk():
    pytest.importorskip("agents")
    installed = metadata.version("openai-agents")
    pinned = _pinned_sdk_version()
    if installed != pinned:
        pytest.skip(f"openai-agents {installed} installed, {pinned} pinned: run in the CI venv")


# ---------------------------------------------------------------------------
# Scripted Responses API transport
# ---------------------------------------------------------------------------

def _usage(inp: int, out: int, cached: int = 0, reasoning: int = 0) -> Any:
    from openai.types.responses.response_usage import (
        InputTokensDetails,
        OutputTokensDetails,
        ResponseUsage,
    )
    return ResponseUsage(
        input_tokens=inp, output_tokens=out, total_tokens=inp + out,
        input_tokens_details=InputTokensDetails(cached_tokens=cached, cache_write_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning),
    )


def _response(output: list[Any], usage: Any) -> Any:
    from openai.types.responses import Response
    # The served model differs from the one sent: the row must not claim it.
    return Response(
        id=f"resp_{uuid.uuid4().hex[:8]}", created_at=0, model="gpt-6-sol-2026-09-01",
        object="response", output=output, parallel_tool_calls=True, tool_choice="auto",
        tools=[], usage=usage, status="completed",
    )


def tool_call(name: str, arguments: dict[str, Any], usage: Any) -> Any:
    from openai.types.responses import ResponseFunctionToolCall
    return _response([ResponseFunctionToolCall(
        type="function_call", call_id=f"call_{uuid.uuid4().hex[:6]}", name=name,
        arguments=json.dumps(arguments), id=f"fc_{uuid.uuid4().hex[:6]}", status="completed",
    )], usage)


def message(text: str, usage: Any) -> Any:
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText
    return _response([ResponseOutputMessage(
        id=f"msg_{uuid.uuid4().hex[:6]}", type="message", role="assistant", status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )], usage)


def refusal(text: str, usage: Any) -> Any:
    from openai.types.responses import ResponseOutputMessage, ResponseOutputRefusal
    return _response([ResponseOutputMessage(
        id=f"msg_{uuid.uuid4().hex[:6]}", type="message", role="assistant", status="completed",
        content=[ResponseOutputRefusal(type="refusal", refusal=text)],
    )], usage)


class ScriptedResponses:
    """The SDK's default AsyncOpenAI client, answering from a script."""

    def __init__(self, monkeypatch, *steps: Any) -> None:
        from agents.models import _openai_shared
        from openai import AsyncOpenAI
        self.requests: list[dict[str, Any]] = []
        self._steps = list(steps)
        # Loopback base URL: nothing is sent, and the netguard would refuse
        # anything else anyway.
        self.client = AsyncOpenAI(api_key="stub-openai", base_url="http://127.0.0.1:9/v1")

        async def create(**kwargs: Any) -> Any:
            self.requests.append(kwargs)
            step = self._steps.pop(0)
            if isinstance(step, BaseException):
                raise step
            return step

        self.client.responses.create = create  # type: ignore[method-assign]
        monkeypatch.setattr(_openai_shared, "_default_openai_client", self.client)


@pytest.fixture
def live_chat(monkeypatch):
    """A live deployment with the chat flag on: stub keys, fake legacy
    clients (installed per test), the SDK gate open, GPT-6 chat settings."""
    def _setup(*, anthropic: Any = None, openai: Any = None, active: str = "anthropic",
               chat_model: str = "gpt-6-sol", chat_effort: str = "medium") -> None:
        llm_fakes.live(monkeypatch, openai=openai, anthropic=anthropic, active=active)
        monkeypatch.setattr(settings, "chat_agents_sdk", True)
        monkeypatch.setattr(settings, "use_agents_sdk", False)
        monkeypatch.setattr(settings, "chat_model", chat_model)
        monkeypatch.setattr(settings, "chat_effort", chat_effort)
    return _setup


@pytest.fixture
def stored_memo():
    purge_memos(TICKER)
    memo_store.save_memo(make_memo(ticker=TICKER, company_name="ZZ SDK Corp"), trigger="first_run")
    yield TICKER
    purge_memos(TICKER)


def _run_id() -> str:
    return f"chat:{uuid.uuid4().hex}"


def _sdk_rows(run_id: str) -> list[Any]:
    return [r for r in llm_fakes.rows_for(run_id) if r.action == "chat.sdk_turn"]


def _trace_rows(run_id: str) -> list[Any]:
    from app.database import SessionLocal
    from app.models import SDKTrace
    with SessionLocal() as db:
        return list(db.query(SDKTrace).filter(SDKTrace.run_id == run_id).order_by(SDKTrace.id))


# ---------------------------------------------------------------------------
# Model and settings (model research 2026-09-25: GPT-6 rejects temperature)
# ---------------------------------------------------------------------------

def test_gpt6_sdk_model_settings_no_temperature(live_chat, monkeypatch):
    live_chat()
    agent = chat_sdk._build_chat_agent()
    assert agent is not None and agent.model == "gpt-6-sol"
    assert agent.model_settings.temperature is None
    assert agent.model_settings.reasoning.effort == "medium"

    transport = ScriptedResponses(monkeypatch, message("An answer.", _usage(50, 5)))
    answer = chat_sdk.answer_via_sdk(message="What is the macro backdrop?", history=[],
                                     run_id=_run_id())
    assert answer and answer.startswith("An answer.")
    (sent,) = transport.requests
    # What the pinned SDK actually puts on the wire.
    assert sent["model"] == "gpt-6-sol"
    assert not isinstance(sent.get("temperature"), (int, float))
    assert not isinstance(sent.get("top_p"), (int, float))
    assert getattr(sent["reasoning"], "effort", None) == "medium"


def test_blank_chat_settings_keep_todays_model(live_chat):
    live_chat(chat_model="", chat_effort="")
    agent = chat_sdk._build_chat_agent()
    assert agent is not None
    assert agent.model == llm.resolve_role_model("pm", provider="openai")
    assert agent.model_settings.temperature is None


def test_non_openai_chat_model_is_refused_not_substituted(live_chat):
    live_chat(chat_model="claude-opus-5-5")
    with pytest.raises(ValueError, match="not an OpenAI model"):
        chat_sdk.chat_model_and_settings()
    # The turn is not attempted; the orchestrator answers on the legacy path.
    assert chat_sdk.run_chat_turn(message="hi", history=[]) == (None, False)


# ---------------------------------------------------------------------------
# Usage rows via RunHooks (critique #4)
# ---------------------------------------------------------------------------

def test_sdk_usage_rows_via_hooks(live_chat, monkeypatch):
    live_chat()
    ScriptedResponses(
        monkeypatch,
        tool_call("get_macro_snapshot", {}, _usage(1200, 40, cached=200, reasoning=30)),
        message(f"Rates are the risk. {OUTPUT_SENTINEL}", _usage(1800, 300, cached=1000, reasoning=120)),
    )
    sentinel_usage = {"provider": "anthropic", "total_tokens": 7}
    llm._USAGE_STATE.last = dict(sentinel_usage)
    run_id = _run_id()
    answer = chat_sdk.answer_via_sdk(message="Why are rates a risk?", history=[], run_id=run_id)
    assert answer and OUTPUT_SENTINEL in answer          # the user still gets the answer

    rows = _sdk_rows(run_id)
    assert len(rows) == 2
    assert {r.call_id for r in rows} == {rows[0].call_id} and len(rows[0].call_id) == 32
    assert [r.attempt for r in rows] == [1, 2]
    for r in rows:
        assert r.agent_name == "chat-pm" and r.role == "chat"
        assert r.provider == "openai" and r.model == "gpt-6-sol" and r.requested_model == "gpt-6-sol"
        assert r.model_resolution == "sdk_agent_model"
        assert r.served_model is None       # 0.22.0 does not expose it
        assert r.effort == "medium" and r.success is True and r.error_type is None
        assert r.cost_usd and r.cost_usd > 0
    first, second = rows
    assert (first.tokens_in, first.tokens_out, first.cache_read_tokens, first.reasoning_tokens) == (
        1200, 40, 200, 30)
    assert (second.tokens_in, second.tokens_out, second.cache_read_tokens, second.reasoning_tokens) == (
        1800, 300, 1000, 120)
    # Priced by the model sent, cached reads at gpt-6-sol's own rate.
    from app.services.llm_metrics import estimate_cost_usd
    assert second.cost_usd == pytest.approx(
        estimate_cost_usd("openai", "gpt-6-sol", 1800, 300, cache_read_tokens=1000))
    # SDK rows never overwrite the snapshot cache's view of the last call.
    assert llm.last_usage() == sentinel_usage


def test_sdk_rows_never_carry_model_output(live_chat, monkeypatch, caplog):
    live_chat()
    ScriptedResponses(
        monkeypatch,
        tool_call("get_company_lite", {"ticker": ARG_SENTINEL}, _usage(900, 20)),
        message(f"Answer {OUTPUT_SENTINEL}", _usage(1000, 50)),
    )
    run_id = _run_id()
    with caplog.at_level(logging.DEBUG):
        chat_sdk.answer_via_sdk(message="Tell me about it", history=[], run_id=run_id)
    rows = _sdk_rows(run_id)
    assert len(rows) == 2, "no rows written: the loop below would check nothing"
    for row in rows:
        values = " ".join(str(v) for v in vars(row).values())
        assert OUTPUT_SENTINEL not in values and ARG_SENTINEL not in values
    assert OUTPUT_SENTINEL not in caplog.text and ARG_SENTINEL not in caplog.text


# ---------------------------------------------------------------------------
# SDKTrace keeps what ran, never what the model wrote (FIX-020)
# ---------------------------------------------------------------------------

def test_sdk_trace_has_no_output_text(live_chat, monkeypatch, caplog):
    live_chat()
    ScriptedResponses(
        monkeypatch,
        tool_call("get_company_lite", {"ticker": ARG_SENTINEL}, _usage(900, 20)),
        message(f"Answer {OUTPUT_SENTINEL}", _usage(1000, 50)),
    )
    run_id = _run_id()
    with caplog.at_level(logging.DEBUG):
        answer = chat_sdk.answer_via_sdk(message="Tell me about it", history=[], run_id=run_id)
    assert answer and OUTPUT_SENTINEL in answer
    (trace,) = _trace_rows(run_id)
    assert trace.surface == "chat" and trace.final_output == ""
    assert trace.new_items, "the item stream (what ran) is still recorded"
    assert all(set(item) <= {"type", "agent", "tool"} for item in trace.new_items)
    assert {"type": "tool_call_item", "agent": "chat-pm", "tool": "get_company_lite"} in trace.new_items
    dumped = json.dumps(trace.new_items)
    assert OUTPUT_SENTINEL not in dumped and ARG_SENTINEL not in dumped
    assert OUTPUT_SENTINEL not in caplog.text and ARG_SENTINEL not in caplog.text


def test_refused_turn_keeps_no_refusal_text(live_chat, monkeypatch, caplog):
    live_chat()
    ScriptedResponses(monkeypatch, refusal(REFUSAL_SENTINEL, _usage(700, 12)))
    run_id = _run_id()
    with caplog.at_level(logging.DEBUG):
        out = chat_sdk.run_chat_turn(message="Say something", history=[], run_id=run_id)
    assert out == (None, True)
    (row,) = _sdk_rows(run_id)
    assert row.success is False and row.error_type == "refusal:unspecified"
    assert row.tokens_in == 700            # the refused request still cost tokens
    (trace,) = _trace_rows(run_id)
    assert trace.final_output == "" and trace.error == "ModelRefusalError"
    assert REFUSAL_SENTINEL not in json.dumps(trace.new_items)
    assert REFUSAL_SENTINEL not in caplog.text


def test_run_config_disables_tracing_per_run():
    config = chat_sdk.sdk_run_config()
    assert config.tracing_disabled is True
    assert config.trace_include_sensitive_data is False


def test_chat_turn_passes_the_no_tracing_run_config(live_chat, monkeypatch):
    """The factory being right is not enough: the chat turn's own Runner
    call must carry it (#18), as defence in depth under the global switch."""
    import agents
    live_chat()
    ScriptedResponses(monkeypatch, message("An answer.", _usage(50, 5)))
    seen: list[Any] = []
    real_run_sync = agents.Runner.run_sync

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("run_config"))
        return real_run_sync(*args, **kwargs)

    monkeypatch.setattr(agents.Runner, "run_sync", spy)
    answer, attempted = chat_sdk.run_chat_turn(message="Macro?", history=[], run_id=_run_id())
    assert answer and attempted
    (config,) = seen
    assert config is not None
    assert config.tracing_disabled is True
    assert config.trace_include_sensitive_data is False


def test_demo_only_mode_closes_both_sdk_gates(live_chat, monkeypatch):
    """The SDK builds its own OpenAI client, so it must ask the question
    `llm._demo_only` answers for every other client: with a key present
    and demo-only mode on, neither the chat agent nor the legacy memo
    exchange may run (a developer key would otherwise spend money)."""
    from app.agents import sdk_runtime
    live_chat()
    monkeypatch.setattr(llm, "_demo_only", lambda: True)
    transport = ScriptedResponses(monkeypatch)
    assert settings.openai_api_key and settings.chat_agents_sdk
    assert chat_sdk._can_use_sdk() is False
    assert chat_sdk.run_chat_turn(message="Macro?", history=[]) == (None, False)
    assert sdk_runtime._can_use_real_sdk() is False
    assert sdk_runtime._run_via_real_sdk(TICKER, run_id=_run_id()) is None
    assert transport.requests == []


def test_tier_resolving_chat_to_a_non_openai_model_is_refused(live_chat, monkeypatch):
    """Not only CHAT_MODEL: a tier override can resolve `chat.sdk_turn` to
    Claude, which the OpenAI-only Agents SDK must not be sent."""
    live_chat(chat_model="")
    monkeypatch.setattr(settings, "llm_research_model", "claude-opus-5-5")
    monkeypatch.setattr(settings, "llm_action_tier_overrides", "chat.sdk_turn:research")
    transport = ScriptedResponses(monkeypatch)
    with pytest.raises(ValueError, match="speaks only OpenAI"):
        chat_sdk.chat_model_and_settings()
    assert chat_sdk._build_chat_agent() is None
    assert chat_sdk.run_chat_turn(message="hi", history=[]) == (None, False)
    assert transport.requests == []


@pytest.mark.parametrize("module", ["app.agents.chat_sdk", "app.agents.sdk_runtime"])
def test_tracing_disabled_globally(module, tmp_path):
    """Importing either SDK module turns trace export off for the whole
    process, whatever OPENAI_AGENTS_DISABLE_TRACING says (critique #18).
    A fresh interpreter, so no other test's import has done it already."""
    probe = (
        "import os\n"
        "os.environ.pop('OPENAI_AGENTS_DISABLE_TRACING', None)\n"
        "from agents.tracing import get_trace_provider\n"
        "before = type(get_trace_provider().create_trace('probe')).__name__\n"
        f"import {module}\n"
        "after = type(get_trace_provider().create_trace('probe')).__name__\n"
        "print(before, after)\n"
    )
    env = {
        "PATH": "/usr/bin:/bin", "ENABLE_LIVE_DATA": "false", "USE_DEMO_DATA": "true",
        "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": "",
        "DATABASE_URL": f"sqlite:///{tmp_path}/probe.db",
    }
    proc = subprocess.run([sys.executable, "-c", probe], cwd=BACKEND, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    before, after = proc.stdout.split()[-2:]
    assert before != "NoOpTrace", "the probe is meaningless if tracing starts disabled"
    assert after == "NoOpTrace"


# ---------------------------------------------------------------------------
# Failover and the turn's run id (critique #5; plan C1)
# ---------------------------------------------------------------------------

def _follow_up(monkeypatch) -> None:
    monkeypatch.setattr(orch_mod, "classify_intent", lambda _m: ("general_research_chat", [TICKER], None))
    monkeypatch.setattr(orch_mod, "_extract_tickers", lambda _text: [TICKER])


# (active provider, LLM_RESEARCH_MODEL, LLM_ACTION_TIER_OVERRIDES, the model
# the one legacy attempt must be sent). The last three route `chat.answer`
# to OpenAI, which a configured tier used to enforce over the fallback's
# provider_override: a second OpenAI attempt and no Anthropic one at all.
_FALLBACK_ROUTES = {
    "blank_tier": ("anthropic", "", "", None),
    "active_openai": ("openai", "", "", None),
    "openai_research_tier": ("anthropic", "gpt-6-sol", "", "claude-opus-5-5"),
    "chat_answer_on_chat_tier": ("anthropic", "", "chat.answer:chat", "claude-opus-5-5"),
}


@pytest.mark.parametrize("config", sorted(_FALLBACK_ROUTES))
@pytest.mark.parametrize("sdk_outcome", ["error", "refusal"])
@pytest.mark.parametrize("legacy_ok", [True, False])
def test_chat_failover_to_legacy_single_attempt_per_provider(
        live_chat, monkeypatch, stored_memo, sdk_outcome, legacy_ok, config):
    """An SDK failure or refusal falls back to ONE single-shot `chat.answer`
    on a provider the turn has not used: the SDK is not run a second time
    (the old duplicate `answer_via_sdk` call), a failed Anthropic answer
    does not hop back to OpenAI, and a `chat.answer` route that itself
    lands on OpenAI (active provider or configured tier) moves to Anthropic."""
    active, research_model, overrides, expected_model = _FALLBACK_ROUTES[config]
    claude = FakeClient(anthropic_response("Legacy answer." if legacy_ok else "",
                                           stop_reason="end_turn"))
    gpt = FakeClient(openai_response("must not be reached"))
    live_chat(anthropic=claude, openai=gpt, active=active)
    monkeypatch.setattr(settings, "llm_research_model", research_model)
    monkeypatch.setattr(settings, "llm_action_tier_overrides", overrides)
    step = (RuntimeError("SDK transport exploded") if sdk_outcome == "error"
            else refusal(REFUSAL_SENTINEL, _usage(300, 5)))
    transport = ScriptedResponses(monkeypatch, step)
    _follow_up(monkeypatch)
    run_id = _run_id()
    with llm.llm_call_context(run_id=run_id):
        resp = orch_mod.Orchestrator().chat(
            f"Why is {TICKER} cheap?", [ChatMessage(role="user", content=f"Tell me about {TICKER}")])

    assert len(transport.requests) == 1          # one SDK attempt, not two
    assert len(claude.requests) == 1             # one legacy attempt
    assert gpt.requests == []                    # no hop back to OpenAI
    if legacy_ok:
        assert resp.answer.startswith("Legacy answer.")
    else:
        assert "Legacy answer." not in resp.answer
    rows = llm_fakes.rows_for(run_id)
    assert [r.action for r in rows] == ["chat.sdk_turn", "chat.answer"]
    assert rows[0].provider == "openai" and rows[0].success is False
    assert rows[1].provider == "anthropic" and rows[1].attempt == 1
    if expected_model:
        # The tier's own mapped failover (gpt-6-sol -> Opus 5.5), not a guess.
        assert rows[1].model == expected_model


@pytest.mark.parametrize("sdk_state", ["flag_off", "gate_closed"])
def test_legacy_chat_keeps_its_failover_when_the_sdk_did_not_run(
        live_chat, monkeypatch, stored_memo, sdk_state):
    """Every production chat turn until wave H: with no SDK attempt this
    turn, the legacy `chat.answer` is today's call, failover included. A
    failed Anthropic answer hops to OpenAI exactly once."""
    claude = FakeClient(anthropic_response("", stop_reason="end_turn"))
    gpt = FakeClient(openai_response("OpenAI partner answer."))
    live_chat(anthropic=claude, openai=gpt, active="anthropic")
    if sdk_state == "flag_off":
        monkeypatch.setattr(settings, "chat_agents_sdk", False)
    else:
        # Flag on, but the gate refuses the turn before any request.
        monkeypatch.setattr(settings, "chat_model", "claude-opus-5-5")
    transport = ScriptedResponses(monkeypatch)
    _follow_up(monkeypatch)
    run_id = _run_id()
    with llm.llm_call_context(run_id=run_id):
        resp = orch_mod.Orchestrator().chat(
            f"Why is {TICKER} cheap?", [ChatMessage(role="user", content=f"Tell me about {TICKER}")])

    assert transport.requests == []
    assert len(claude.requests) == 1
    assert len(gpt.requests) == 1                 # today's failover hop
    assert resp.answer.startswith("OpenAI partner answer.")
    rows = llm_fakes.rows_for(run_id)
    assert [(r.action, r.provider, r.attempt) for r in rows] == [
        ("chat.answer", "anthropic", 1), ("chat.answer", "openai", 2)]


def test_a_raised_sdk_turn_counts_as_the_openai_attempt(live_chat, monkeypatch, stored_memo):
    """`run_chat_turn` raising after it reached the model: the attempt is
    spent. The memo-context answer must not run the SDK again, and the
    legacy answer must not go to OpenAI."""
    claude = FakeClient(anthropic_response("Legacy answer.", stop_reason="end_turn"))
    gpt = FakeClient(openai_response("must not be reached"))
    live_chat(anthropic=claude, openai=gpt, active="openai")
    transport = ScriptedResponses(monkeypatch, message("An answer.", _usage(50, 5)),
                                  message("A second SDK answer.", _usage(50, 5)))
    real_turn = chat_sdk.run_chat_turn

    def turn_then_raise(**kwargs: Any) -> Any:
        real_turn(**kwargs)
        raise RuntimeError("post-processing blew up")

    monkeypatch.setattr(chat_sdk, "run_chat_turn", turn_then_raise)
    _follow_up(monkeypatch)
    resp = orch_mod.Orchestrator().chat(
        f"Why is {TICKER} cheap?", [ChatMessage(role="user", content=f"Tell me about {TICKER}")])

    assert len(transport.requests) == 1          # the SDK ran once
    assert gpt.requests == []                    # and the legacy answer avoided OpenAI
    assert len(claude.requests) == 1
    assert resp.answer.startswith("Legacy answer.")


def test_chat_run_id_links_sdk_and_legacy_rows(live_chat, monkeypatch, stored_memo):
    """Through the route: the turn's `chat:<hex>` id is on the SDK usage
    rows, the SDKTrace row and the legacy fallback's rows alike."""
    from fastapi.testclient import TestClient

    from app.database import SessionLocal
    from app.main import app
    from app.models import LLMCallLog, SDKTrace

    claude = FakeClient(anthropic_response("Legacy answer.", stop_reason="end_turn"))
    live_chat(anthropic=claude, openai=FakeClient(openai_response("unused")), active="anthropic")
    ScriptedResponses(
        monkeypatch,
        tool_call("get_macro_snapshot", {}, _usage(400, 10)),
        RuntimeError("SDK transport exploded"),
    )
    _follow_up(monkeypatch)
    with SessionLocal() as db:
        LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
        floor = db.query(LLMCallLog.id).order_by(LLMCallLog.id.desc()).limit(1).scalar() or 0
        trace_floor = db.query(SDKTrace.id).order_by(SDKTrace.id.desc()).limit(1).scalar() or 0

    r = TestClient(app).post("/api/chat", json={
        "message": f"Why is {TICKER} cheap?",
        "history": [{"role": "user", "content": f"Tell me about {TICKER}"}],
    })
    assert r.status_code == 200, r.text
    assert r.json()["answer"].startswith("Legacy answer.")

    with SessionLocal() as db:
        rows = list(db.query(LLMCallLog).filter(LLMCallLog.id > floor).order_by(LLMCallLog.id))
        traces = list(db.query(SDKTrace).filter(SDKTrace.id > trace_floor))
    assert [r.action for r in rows] == ["chat.sdk_turn", "chat.sdk_turn", "chat.answer"]
    run_ids = {r.run_id for r in rows} | {t.run_id for t in traces}
    assert len(run_ids) == 1
    (run_id,) = run_ids
    assert re.fullmatch(r"chat:[0-9a-f]{32}", run_id)
    assert rows[1].error_type == "response_error:RuntimeError"   # the failed second request
    assert {r.origin for r in rows} == {"api:/api/chat"}
    assert {r.feature for r in rows} == {"pm_chat"}
    assert rows[2].agent_name == "PM Chat"          # the umbrella named no agent
