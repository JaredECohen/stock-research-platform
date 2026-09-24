"""W2a in chat: every chat exit serves the presented memo.

Chat is a customer exit like the memo page (integration plan S11): the
rendered answer, the LLM context, the SDK `get_memo` tool, both inline
generation paths, the stored-memo path and the legacy follow-up path all
read the presented memo; and the `ask_*` specialist tools refuse to relay a
fallback stand-in as the specialist's answer (critique delta 4).
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from app.agents import chat_sdk, llm, sdk_runtime
from app.agents import orchestrator as orch_mod
from app.config import settings
from app.schemas import AgentFinding, ChatMessage, StockMemoOut
from app.services import memo_sections, memo_store
from app.services.memo_sections import UNAVAILABLE_TEXT
from app.tests.factories import make_finding
from app.tests.gating_helpers import purge_memos

FIXTURES = Path(__file__).parent / "fixtures" / "memo_sections"
PM_TAIL = memo_sections.SIG["pm_view_tail"].text
KD = memo_sections.SIG["bb_key_disagreement"].text


def _fixture(name: str, ticker: str = "ZZCHAT") -> StockMemoOut:
    memo = StockMemoOut.model_validate(json.loads((FIXTURES / f"{name}.json").read_text()))
    return memo.model_copy(update={"ticker": ticker})


@pytest.fixture(autouse=True)
def _purge():
    purge_memos("ZZCHAT", "ZZCHAT2")
    yield
    purge_memos("ZZCHAT", "ZZCHAT2")


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: None)
    monkeypatch.setattr(orch_mod, "classify_intent",
                        lambda _m: ("single_stock_analysis", ["ZZCHAT"], None))
    monkeypatch.setattr(orch_mod, "_is_conceptual_followup", lambda _m, _h: False)


def _assert_presented_googl(memo: StockMemoOut | None) -> None:
    assert memo is not None and memo.section_availability
    assert memo.one_sentence_thesis == UNAVAILABLE_TEXT and memo.final_pm_view == UNAVAILABLE_TEXT


def test_chat_context_omits_hidden_sections():
    shown = memo_sections.present_memo(_fixture("googl_live_prepflag"))
    ctx = orch_mod._memo_for_chat_context(shown)
    for key in ("thesis", "confidence", "mispricing_thesis"):
        assert key not in ctx, key
    assert "one_sentence_thesis" in ctx["sections_unavailable"]
    # The hidden sector analyst's influence is not offered; risk always is.
    assert "sector" not in ctx["agent_influence"]
    blob = json.dumps(ctx)
    assert PM_TAIL not in blob and KD not in blob
    # A clean memo keeps every key.
    clean = orch_mod._memo_for_chat_context(memo_sections.present_memo(_fixture("meta_v1")))
    assert {"thesis", "confidence", "mispricing_thesis", "valuation_summary"} <= set(clean)

    answer = orch_mod._render_memo_answer(shown)
    assert UNAVAILABLE_TEXT in answer
    assert "confidence unavailable in this version" in answer
    assert "rating reflects the quantitative factor blend" in answer
    assert PM_TAIL not in answer and KD not in answer
    comparison = orch_mod._render_comparison_answer([shown])
    assert "confidence unavailable in this version" in comparison


def test_orchestrator_inline_paths_are_presented(monkeypatch):
    raw = _fixture("googl_live_prepflag")
    monkeypatch.setattr(orch_mod, "run_stock_memo", lambda t, *a, **k: raw)
    monkeypatch.setattr(settings, "use_agents_sdk", False)
    resp = orch_mod.Orchestrator().chat("Analyze ZZCHAT", [])
    _assert_presented_googl(resp.memo)
    assert PM_TAIL not in resp.answer

    monkeypatch.setattr(settings, "use_agents_sdk", True)
    monkeypatch.setattr(sdk_runtime, "run_stock_memo_via_sdk", lambda t, *a, **k: raw)
    resp = orch_mod.Orchestrator().chat("Analyze ZZCHAT", [])
    _assert_presented_googl(resp.memo)

    monkeypatch.setattr(settings, "use_agents_sdk", False)
    monkeypatch.setattr(orch_mod, "classify_intent",
                        lambda _m: ("stock_comparison", ["ZZCHAT", "ZZCHAT2"], None))
    resp = orch_mod.Orchestrator().chat("Compare ZZCHAT and ZZCHAT2", [])
    _assert_presented_googl(resp.memo)
    assert PM_TAIL not in resp.answer
    assert raw.section_availability == {}  # the generated memo is not mutated


def test_orchestrator_stored_memo_path_is_presented(monkeypatch):
    memo_store.save_memo(_fixture("googl_live_prepflag"))
    resp = orch_mod.Orchestrator().chat("Analyze ZZCHAT", [], allow_inline_memo=False)
    _assert_presented_googl(resp.memo)
    assert PM_TAIL not in resp.answer


def test_legacy_follow_up_context_is_presented(monkeypatch):
    memo_store.save_memo(_fixture("googl_live_prepflag"))
    monkeypatch.setattr(settings, "use_agents_sdk", False)
    monkeypatch.setattr(orch_mod, "_extract_tickers", lambda _text: ["ZZCHAT"])
    prompts: list[str] = []

    def chat_text(prompt, **_k):
        prompts.append(prompt)
        return "An answer."

    monkeypatch.setattr(llm, "chat_text", chat_text)
    out = orch_mod.Orchestrator()._answer_with_memo_context(
        "Why is ZZCHAT cheap?", [ChatMessage(role="user", content="Tell me about ZZCHAT")])
    assert out is not None and prompts
    assert '"ticker": "ZZCHAT"' in prompts[0]
    assert PM_TAIL not in prompts[0] and '"thesis"' not in prompts[0]


class _FakeAgent:
    def __init__(self, **kwargs: Any) -> None:
        self.name = kwargs["name"]
        self.tools = kwargs["tools"]


@pytest.fixture
def tools(monkeypatch) -> dict[str, Any]:
    fake = types.ModuleType("agents")
    fake.Agent = _FakeAgent  # type: ignore[attr-defined]
    fake.function_tool = lambda fn: fn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agents", fake)
    monkeypatch.setattr(chat_sdk, "_can_use_sdk", lambda: True)
    agent = chat_sdk._build_chat_agent()
    assert agent is not None
    return {t.__name__: t for t in agent.tools}


def test_sdk_get_memo_is_presented(tools):
    memo_store.save_memo(_fixture("googl_live_prepflag"))
    out = tools["get_memo"]("zzchat")
    assert out["ticker"] == "ZZCHAT"
    assert "thesis" not in out and "confidence" not in out
    assert PM_TAIL not in json.dumps(out)


def _template(agent: str) -> AgentFinding:
    return make_finding(agent, headline="h", summary="s",
                        data={"deterministic_fallback": "LLM returned no usable output"})


def test_ask_tools_refuse_template_findings(tools, monkeypatch):
    from app.agents import earnings_agent, filing_agent, macro_agent, sector_agents, valuation_agent
    from app.services import fundamentals_service, valuation_service
    monkeypatch.setattr(fundamentals_service, "get_full_financials",
                        lambda t, **_: {"ticker": t, "profile": {"ticker": t}, "ratios": {}, "earnings": {}})
    monkeypatch.setattr(valuation_service, "build_dcf", lambda t, **_: None)
    cases = {
        "ask_sector": (sector_agents, "run_sector_agent", "Sector Analyst"),
        "ask_earnings": (earnings_agent, "run_earnings_agent", "Earnings Analyst"),
        "ask_filings": (filing_agent, "run_filing_agent", "Filing Analyst"),
        "ask_valuation": (valuation_agent, "run_valuation_agent", "Valuation Analyst"),
        "ask_macro": (macro_agent, "run_macro_agent", "Macro Analyst"),
    }
    def call(tool: str) -> dict[str, Any]:
        if tool == "ask_macro":
            return tools[tool]("q", ticker="zzchat")
        return tools[tool]("zzchat", "q")

    for tool, (module, fn, agent) in cases.items():
        monkeypatch.setattr(module, fn, lambda *a, _agent=agent, **k: _template(_agent))
        out = call(tool)
        assert out["error"] == chat_sdk.SPECIALIST_UNAVAILABLE, tool
        assert "headline" not in out and "summary" not in out
        # A real read is relayed as before.
        monkeypatch.setattr(module, fn, lambda *a, _agent=agent, **k: make_finding(
            _agent, headline="A real read", summary="Grounded."))
        out = call(tool)
        assert out["headline"] == "A real read", tool
    # No-input stubs are stand-ins too.
    monkeypatch.setattr(earnings_agent, "run_earnings_agent", lambda *a, **k: make_finding(
        "Earnings Analyst", headline="Earnings transcript unavailable."))
    assert tools["ask_earnings"]("zzchat", "q")["error"] == chat_sdk.SPECIALIST_UNAVAILABLE
