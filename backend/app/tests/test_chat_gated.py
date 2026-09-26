"""Chat under the login wall never generates a memo in-request (FEAT-002, S2).

`Orchestrator.chat(allow_inline_memo=False)` answers from stored memos
and names what is missing in `needs_analysis`; both inline entry points
(the legacy graph call and the SDK runtime, resolved at call time) are
replaced with sentinels that fail the test if reached. With the flag at
its default the historical path is pinned, so `test_routes.py`'s chat
tests keep meaning what they meant.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agents import llm, sdk_runtime
from app.agents import orchestrator as orch_mod
from app.api import routes_chat
from app.config import settings
from app.main import app
from app.schemas import ChatResponse
from app.services import memo_store
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth
from app.tests.factories import make_memo
from app.tests.gating_helpers import assert_structured, free_user, seed_demo_universe, usage_events, user_id_for


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture()
def auth_on(monkeypatch, clerk):
    yield from enable_auth(monkeypatch, clerk)


@pytest.fixture()
def client():
    seed_demo_universe()
    return TestClient(app)


@pytest.fixture()
def sentinels(monkeypatch):
    """Both inline memo entry points raise; a deterministic intent."""
    reached: list[str] = []

    def memo_ran(ticker, *_a, **_kw):
        reached.append(ticker)
        raise AssertionError(f"inline memo generation started for {ticker}")

    monkeypatch.setattr(orch_mod, "run_stock_memo", memo_ran)
    monkeypatch.setattr(sdk_runtime, "run_stock_memo_via_sdk", memo_ran)
    monkeypatch.setattr(orch_mod, "_is_conceptual_followup", lambda _m, _h: False)
    return reached


class _Snap:
    def __init__(self, memo):
        self.memo_json = memo.model_dump(mode="json")
        self.version = 1


def _stored(*tickers: str):
    memos = {t: _Snap(make_memo(ticker=t, company_name=f"{t} Inc")) for t in tickers}

    def latest_memo(ticker, **_kw):
        return memos.get(ticker.upper())

    return latest_memo


# ---------------------------------------------------------------------------
# Orchestrator unit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_sdk", [False, True])
def test_single_stock_without_memo_needs_analysis(monkeypatch, sentinels, use_sdk):
    monkeypatch.setattr(settings, "use_agents_sdk", use_sdk)
    monkeypatch.setattr(orch_mod, "classify_intent", lambda _m: ("single_stock_analysis", ["NVDA"], None))
    monkeypatch.setattr(memo_store, "latest_memo", _stored())
    resp = orch_mod.Orchestrator().chat("Analyze NVDA", [], allow_inline_memo=False)
    assert resp.intent == "single_stock_analysis"
    assert resp.memo is None
    assert resp.needs_analysis == ["NVDA"]
    assert "NVDA" in resp.answer and "research" in resp.answer.lower()
    assert sentinels == []


def test_single_stock_with_stored_memo_answers_from_it(monkeypatch, sentinels):
    monkeypatch.setattr(orch_mod, "classify_intent", lambda _m: ("single_stock_analysis", ["NVDA"], None))
    monkeypatch.setattr(memo_store, "latest_memo", _stored("NVDA"))
    resp = orch_mod.Orchestrator().chat("Analyze NVDA", [], allow_inline_memo=False)
    assert resp.memo is not None and resp.memo.ticker == "NVDA"
    assert resp.needs_analysis == []
    assert "NVDA Inc" in resp.answer
    assert sentinels == []


def test_comparison_mixes_stored_and_missing(monkeypatch, sentinels):
    monkeypatch.setattr(orch_mod, "classify_intent", lambda _m: ("stock_comparison", ["NVDA", "MSFT"], None))
    monkeypatch.setattr(memo_store, "latest_memo", _stored("NVDA"))
    resp = orch_mod.Orchestrator().chat("Compare NVDA and MSFT", [], allow_inline_memo=False)
    assert resp.intent == "stock_comparison"
    assert resp.memo is not None and resp.memo.ticker == "NVDA"
    assert resp.needs_analysis == ["MSFT"]
    assert "memo not yet generated: MSFT" in resp.sources
    assert "MSFT" in resp.answer
    assert sentinels == []


def test_comparison_with_nothing_stored(monkeypatch, sentinels):
    monkeypatch.setattr(orch_mod, "classify_intent", lambda _m: ("stock_comparison", ["NVDA", "MSFT"], None))
    monkeypatch.setattr(memo_store, "latest_memo", _stored())
    resp = orch_mod.Orchestrator().chat("Compare NVDA and MSFT", [], allow_inline_memo=False)
    assert resp.memo is None
    assert resp.needs_analysis == ["NVDA", "MSFT"]
    assert sentinels == []


def test_default_flag_still_generates_inline(monkeypatch):
    """Behaviour-preserving: the historical path is the default."""
    monkeypatch.setattr(settings, "use_agents_sdk", False)
    monkeypatch.setattr(orch_mod, "classify_intent", lambda _m: ("single_stock_analysis", ["NVDA"], None))
    monkeypatch.setattr(orch_mod, "_is_conceptual_followup", lambda _m, _h: False)
    calls: list[str] = []

    def fake_run(ticker, **_kw):
        calls.append(ticker)
        return make_memo(ticker=ticker, company_name="NVIDIA")

    monkeypatch.setattr(orch_mod, "run_stock_memo", fake_run)
    resp = orch_mod.Orchestrator().chat("Analyze NVDA", [])
    assert calls == ["NVDA"]
    assert resp.memo is not None and resp.needs_analysis == []


def test_chat_flag_does_not_route_inline_memo_through_sdk(monkeypatch):
    """CHAT_AGENTS_SDK (plan P14) moves the chat AGENT to the Agents SDK
    and nothing else: the turn tries the SDK chat agent, and when that
    yields no answer, "analyze X" still runs the graph's `run_stock_memo`,
    never `sdk_runtime.run_stock_memo_via_sdk` (which would run the memo
    twice under a run id nothing links to). Only the legacy
    USE_AGENTS_SDK routes the inline memo through the SDK runtime."""
    from app.agents import chat_sdk

    monkeypatch.setattr(settings, "chat_agents_sdk", True)
    monkeypatch.setattr(settings, "use_agents_sdk", False)
    monkeypatch.setattr(orch_mod, "classify_intent", lambda _m: ("single_stock_analysis", ["NVDA"], None))
    monkeypatch.setattr(orch_mod, "_is_conceptual_followup", lambda _m, _h: True)
    turns: list[str] = []

    def sdk_turn(*, message, history, run_id=None):
        turns.append(message)
        return None, True          # the SDK ran and produced no answer

    monkeypatch.setattr(chat_sdk, "run_chat_turn", sdk_turn, raising=False)
    graph_runs: list[str] = []

    def fake_run(ticker, **_kw):
        graph_runs.append(ticker)
        return make_memo(ticker=ticker, company_name="NVIDIA")

    def via_sdk(ticker, *_a, **_kw):
        raise AssertionError("the chat flag routed the inline memo through sdk_runtime")

    monkeypatch.setattr(orch_mod, "run_stock_memo", fake_run)
    monkeypatch.setattr(sdk_runtime, "run_stock_memo_via_sdk", via_sdk)
    resp = orch_mod.Orchestrator().chat("Analyze NVDA", [])
    assert turns == ["Analyze NVDA"]        # the chat agent is on the SDK
    assert graph_runs == ["NVDA"]           # the memo is not
    assert resp.memo is not None and resp.memo.ticker == "NVDA"

    # The legacy flag alone no longer turns the SDK chat agent on.
    turns.clear()
    monkeypatch.setattr(settings, "chat_agents_sdk", False)
    monkeypatch.setattr(settings, "use_agents_sdk", True)
    monkeypatch.setattr(sdk_runtime, "run_stock_memo_via_sdk",
                        lambda t, *a, **k: make_memo(ticker=t, company_name="NVIDIA"))
    orch_mod.Orchestrator().chat("Analyze NVDA", [])
    assert turns == []


def test_response_schema_defaults_needs_analysis_empty():
    assert ChatResponse(intent="general_research_chat", answer="hi").needs_analysis == []


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

def test_route_under_the_wall_disables_inline_memo_and_meters_pm_chat(auth_on, client, sentinels, monkeypatch):
    monkeypatch.setattr(memo_store, "latest_memo", _stored())
    _sub, tok = free_user(auth_on)
    resp = client.post("/api/chat", json={"message": "Analyze NVDA as a long-term investment.", "history": []},
                       headers=bearer(tok))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["intent"] == "single_stock_analysis"
    assert body["memo"] is None
    assert body["needs_analysis"] == ["NVDA"]
    assert sentinels == []
    uid = user_id_for(client, tok)
    events = usage_events(uid, "pm_chat")
    assert [e.status for e in events] == ["committed"]
    assert events[0].plan_at_charge == "free"


def test_route_passes_the_flag_and_the_call_context(auth_on, client, monkeypatch):
    seen: dict = {}

    def fake_chat(message, history, *, allow_inline_memo=True):
        seen["flag"] = allow_inline_memo
        seen["ctx"] = dict(llm.current_call_context())
        return ChatResponse(intent="general_research_chat", answer="ok")

    monkeypatch.setattr(routes_chat._orch, "chat", fake_chat)
    _sub, tok = free_user(auth_on)
    uid = user_id_for(client, tok)
    assert client.post("/api/chat", json={"message": "hello", "history": []}, headers=bearer(tok)).status_code == 200
    assert seen["flag"] is False
    assert seen["ctx"]["user_id"] == uid and seen["ctx"]["feature"] == "pm_chat"


def test_inline_override_cannot_reenable_memo_runs_under_the_wall(auth_on, client, sentinels, monkeypatch):
    """Regression: the flag followed `memo_inline_generation_effective`
    alone, so `MEMO_INLINE_GENERATION=true` with the wall on had chat
    generate memos in-request with no `research_run` charge. The wall
    always wins; the override only matters with the wall off."""
    monkeypatch.setattr(settings, "memo_inline_generation", True)
    monkeypatch.setattr(memo_store, "latest_memo", _stored())
    _sub, tok = free_user(auth_on)
    resp = client.post("/api/chat", json={"message": "Analyze NVDA as a long-term investment.", "history": []},
                       headers=bearer(tok))
    assert resp.status_code == 200, resp.text
    assert resp.json()["needs_analysis"] == ["NVDA"]
    assert sentinels == []


def test_inline_off_with_the_wall_off_disables_memo_runs(client, monkeypatch):
    seen: dict = {}

    def fake_chat(message, history, *, allow_inline_memo=True):
        seen["flag"] = allow_inline_memo
        return ChatResponse(intent="general_research_chat", answer="ok")

    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "memo_inline_generation", False)
    monkeypatch.setattr(routes_chat._orch, "chat", fake_chat)
    assert client.post("/api/chat", json={"message": "hello", "history": []}).status_code == 200
    assert seen["flag"] is False


def test_route_with_the_wall_off_allows_inline_memo(client, monkeypatch):
    seen: dict = {}

    def fake_chat(message, history, *, allow_inline_memo=True):
        seen["flag"] = allow_inline_memo
        seen["ctx"] = dict(llm.current_call_context())
        return ChatResponse(intent="general_research_chat", answer="ok")

    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(routes_chat._orch, "chat", fake_chat)
    assert client.post("/api/chat", json={"message": "hello", "history": []}).status_code == 200
    assert seen["flag"] is True
    assert seen["ctx"]["user_id"] is None and seen["ctx"]["feature"] == "pm_chat"


def test_free_chat_allowance_is_402_when_spent(auth_on, client, monkeypatch):
    monkeypatch.setattr(settings, "entitlement_overrides_json", '{"pm_chat": {"free": 1}}')
    monkeypatch.setattr(routes_chat._orch, "chat",
                        lambda *_a, **_kw: ChatResponse(intent="general_research_chat", answer="ok"))
    _sub, tok = free_user(auth_on)
    assert client.post("/api/chat", json={"message": "hello", "history": []}, headers=bearer(tok)).status_code == 200
    second = client.post("/api/chat", json={"message": "hello", "history": []}, headers=bearer(tok))
    detail = assert_structured(second, code="quota_exceeded", status=402)
    assert detail["feature"] == "pm_chat" and detail["used"] == 1 and detail["limit"] == 1


def test_anonymous_chat_is_401(auth_on, client):
    assert_structured(client.post("/api/chat", json={"message": "hello", "history": []}),
                      code="auth_required", status=401)
