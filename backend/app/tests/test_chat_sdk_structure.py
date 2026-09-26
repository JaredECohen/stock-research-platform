"""Structural tests for the chat-SDK tool registry (`agents/chat_sdk.py`).

The tools are closures created inside `_build_chat_agent`, which is
gated on an OpenAI key. To exercise them without one, the registry is
built against a stand-in `agents` module whose `function_tool` is the
identity, so every tool stays a plain callable and can be invoked with
a demo ticker. A separate test builds against the real `openai-agents`
package (no key needed to construct an `Agent`) to pin the tool list.

Nothing here talks to a provider or an LLM: the DemoProvider is wired by
`conftest.py`, an autouse guard makes any socket connect raise (the
keyless SEC / BLS providers in `data_service` would otherwise call out
regardless of ENABLE_LIVE_DATA), a second autouse guard pins both LLM
seams to their no-answer path instead of trusting the environment's
keys to be blank (`comps_service` and the sector specialist both call
`llm.chat_json` when a key is configured), and the `ask_*` tools are
tested by capturing the profile they hand to the (monkeypatched)
specialist — which is exactly where the 2026-08-12 unscoped-scan OOM
originated.
"""
from __future__ import annotations

import socket
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from app.agents import chat_sdk, llm
from app.config import settings
from app.schemas import AgentFinding, BullBearCase, CriticReview, StockMemoOut
from app.services import memo_store
from app.tests.fixtures.seed_demo_data import run_full_seed

DEMO = "MSFT"
UNKNOWN = "ZZZUNKNOWN"
EXPECTED_TOOLS = [
    "get_memo", "get_dcf_summary", "get_comps", "get_macro_snapshot",
    "get_company_lite", "list_universe", "screener_query", "custom_screen",
    "get_industry_context",
    "ask_sector", "ask_earnings", "ask_filings", "ask_valuation", "ask_macro",
]
FINDING_KEYS = {"agent", "ticker", "headline", "summary", "key_points", "confidence"}


@pytest.fixture(scope="module", autouse=True)
def _seeded():
    run_full_seed()


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Every provider on the chat surface catches its own transport
    errors, so refusing the connect keeps the tests green while proving
    no test here depends on sec.gov / bls.gov being reachable."""
    def _refuse(*_a, **_k):
        raise RuntimeError("network access attempted during an offline structural test")
    monkeypatch.setattr(socket.socket, "connect", _refuse)


@pytest.fixture(autouse=True)
def no_llm(monkeypatch) -> list[dict[str, Any]]:
    """What a blank key yields from `llm.chat_json` / `chat_text` is None;
    pin that so every tool takes its deterministic branch even under a
    developer `.env` with a live key. The key itself is blanked too:
    `embeddings.embed` (reached by `get_comps` via doc-chunk ingest) has
    no seam through `llm` and calls OpenAI directly whenever a key is
    set. Returns the recorded calls so a test can prove the seam was
    reached."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: calls.append(k) or None)
    monkeypatch.setattr(llm, "chat_text", lambda *a, **k: calls.append(k) or None)
    return calls


def _stub_memo(ticker: str) -> StockMemoOut:
    finding = AgentFinding(agent="x", headline="h", summary="s", confidence=0.5)
    return StockMemoOut(
        ticker=ticker, company_name=ticker, sector="Technology",
        final_pm_view="pm view", rating_label="Neutral", confidence_score=50,
        one_sentence_thesis="thesis", business_summary="bd",
        sector_agent_view=finding, earnings_agent_view=finding,
        filing_agent_view=finding, valuation_agent_view=finding,
        comps_agent_view=finding, macro_sensitivity=finding,
        bull_case=BullBearCase(headline="bull", key_points=["b1"]),
        bear_case=BullBearCase(headline="bear", key_points=["r1"]),
        catalysts=[], key_risks=[], thesis_breakers=[],
        dcf_summary={"wacc": 0.09, "base_upside": 0.1}, portfolio_fit="",
        risk_committee_challenge=CriticReview(overall_assessment="ok"),
        final_verdict="verdict",
    )


class _FakeAgent:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.name = kwargs["name"]
        self.tools = kwargs["tools"]


@pytest.fixture
def tools(monkeypatch) -> dict[str, Any]:
    fake = types.ModuleType("agents")
    fake.Agent = _FakeAgent
    fake.function_tool = lambda fn: fn
    monkeypatch.setitem(sys.modules, "agents", fake)
    monkeypatch.setattr(chat_sdk, "_can_use_sdk", lambda: True)
    agent = chat_sdk._build_chat_agent()
    assert agent is not None
    return {t.__name__: t for t in agent.tools}


def _finding(**overrides: Any) -> SimpleNamespace:
    base = dict(headline="hl", summary="sum", key_points=["k"], confidence=0.6)
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Gate + registry
# ---------------------------------------------------------------------------

def test_sdk_gate_is_closed_without_a_key(monkeypatch):
    # Blank the key ourselves rather than asserting the environment did:
    # a developer `.env` carries a live key and must not fail this test.
    monkeypatch.setattr(settings, "openai_api_key", "")
    assert chat_sdk._can_use_sdk() is False
    assert chat_sdk._build_chat_agent() is None
    assert chat_sdk.answer_via_sdk(message="hi", history=[]) is None


def test_registry_builds_against_the_real_sdk_without_a_key(monkeypatch):
    pytest.importorskip("agents")
    monkeypatch.setattr(chat_sdk, "_can_use_sdk", lambda: True)
    agent = chat_sdk._build_chat_agent()
    assert agent is not None and agent.name == "chat-pm"
    assert [t.name for t in agent.tools] == EXPECTED_TOOLS
    assert all(t.description for t in agent.tools)


def test_fake_registry_exposes_the_documented_tools(tools):
    assert list(tools) == EXPECTED_TOOLS


def test_instructions_ask_for_cited_figures_not_visible_reasoning(monkeypatch):
    """Model research 2026-09-25: reasoning models are asked to cite the
    figures they relied on, not to "reason out loud" / "show the working"
    (their reasoning is internal and billed), and the specialist-cost line
    no longer quotes the old ~$0.05."""
    fake = types.ModuleType("agents")
    fake.Agent = _FakeAgent
    fake.function_tool = lambda fn: fn
    monkeypatch.setitem(sys.modules, "agents", fake)
    monkeypatch.setattr(chat_sdk, "_can_use_sdk", lambda: True)
    agent = chat_sdk._build_chat_agent()
    assert agent is not None
    text = " ".join(agent.kwargs["instructions"].split())
    assert "cite the specific figures you relied on" in text
    assert "Cite the specific figures you relied on" in text
    assert "reason out loud" not in text and "Show the working" not in text
    assert "~$0.05" not in text


# ---------------------------------------------------------------------------
# Memo / DCF tools
# ---------------------------------------------------------------------------

def test_memo_tools_on_unknown_ticker_return_error_dicts(tools):
    for ticker in (UNKNOWN, "", None):
        out = tools["get_memo"](ticker)
        assert set(out) == {"error"} and "memo" in out["error"].lower()
        out = tools["get_dcf_summary"](ticker)
        assert set(out) == {"error"}


def test_memo_tools_on_a_saved_memo(tools):
    memo_store.save_memo(_stub_memo("TSTCHAT"), trigger="first_run")
    memo = tools["get_memo"]("tstchat")
    assert "error" not in memo
    assert {
        "ticker", "name", "sector", "rating", "stock_score", "confidence",
        "thesis", "factor_scores", "dcf", "valuation_summary", "key_risks",
        "thesis_breakers", "bull_case", "bear_case", "mispricing_thesis",
        "forward_catalysts", "agent_influence", "macro_regime_at_memo",
        "price_at_memo",
    } <= set(memo)
    assert memo["ticker"] == "TSTCHAT" and memo["rating"] == "Neutral"
    assert memo["dcf"]["wacc"] == 0.09
    assert memo["bull_case"] == ["b1"] and memo["bear_case"] == ["r1"]

    dcf = tools["get_dcf_summary"]("TSTCHAT")
    assert set(dcf) == {
        "ticker", "dcf_summary_pm_adjusted", "dcf_summary_initial",
        "pm_adjustments", "pm_adjustment_headline",
    }
    assert dcf["dcf_summary_pm_adjusted"]["wacc"] == 0.09
    assert dcf["pm_adjustments"] == [] and dcf["pm_adjustment_headline"] == ""


def test_memo_tool_reports_unloadable_snapshot_as_error(tools, monkeypatch):
    memo_store.save_memo(_stub_memo("TSTBADMEMO"), trigger="first_run")

    def _boom(_snap):
        raise ValueError("schema drift")
    monkeypatch.setattr(memo_store, "memo_to_pydantic", _boom)
    assert "schema drift" in tools["get_memo"]("TSTBADMEMO")["error"]
    assert "schema drift" in tools["get_dcf_summary"]("TSTBADMEMO")["error"]


# ---------------------------------------------------------------------------
# Data tools
# ---------------------------------------------------------------------------

def test_get_comps_on_demo_ticker(tools):
    out = tools["get_comps"](DEMO.lower())
    assert set(out) == {
        "ticker", "peers", "target", "peer_median", "premium_discount",
        "interpretation", "history",
    }
    assert out["ticker"] == DEMO
    assert out["peers"] and all(isinstance(p, str) for p in out["peers"])
    assert isinstance(out["target"], dict) and isinstance(out["peer_median"], dict)
    assert set(tools["get_comps"](UNKNOWN)) == {"error"}


def test_get_macro_snapshot(tools):
    out = tools["get_macro_snapshot"]()
    assert set(out) == {"fred_snapshot", "regime_broadcast"}
    assert isinstance(out["fred_snapshot"], dict) and out["fred_snapshot"]
    assert out["regime_broadcast"] is None or isinstance(out["regime_broadcast"], dict)


def test_get_company_lite(tools):
    out = tools["get_company_lite"](DEMO.lower())
    assert set(out) == {
        "ticker", "name", "sector", "industry", "market_cap", "business",
        "last_price", "last_price_as_of", "last_price_source", "metrics", "screener_scores",
    }
    assert out["ticker"] == DEMO and out["market_cap"] > 0
    assert out["screener_scores"] is not None       # seeded by run_full_seed
    assert set(tools["get_company_lite"](UNKNOWN)) == {"error"}


def test_list_universe(tools):
    everything = tools["list_universe"]()
    assert set(everything) == {"count", "tickers"}
    assert everything["count"] == len(everything["tickers"]) > 0
    assert all(set(r) == {"ticker", "company_name", "sector"} for r in everything["tickers"])
    tech = tools["list_universe"](sector="technology")   # case-insensitive
    assert 0 < tech["count"] < everything["count"]
    assert all(r["sector"] == "Technology" for r in tech["tickers"])
    assert tools["list_universe"](sector="No Such Sector")["count"] == 0


def test_screener_query(tools):
    out = tools["screener_query"](sort_by="quality", limit=3)
    assert set(out) == {"sort_by", "sector_filter", "theme", "count", "rows"}
    assert out["sort_by"] == "quality" and len(out["rows"]) <= 3
    assert out["count"] >= len(out["rows"])
    scores = [r["quality"] or 0 for r in out["rows"]]
    assert scores == sorted(scores, reverse=True)
    assert all(set(r) == {
        "ticker", "name", "sector", "pm_score", "quality", "growth",
        "valuation", "risk", "thesis",
    } for r in out["rows"])
    assert tools["screener_query"](sort_by="bogus")["sort_by"] == "pm_score"
    tech = tools["screener_query"](sector="tech", limit=50)
    assert tech["rows"] and all("tech" in r["sector"].lower() for r in tech["rows"])


def test_custom_screen(tools):
    out = tools["custom_screen"]('[{"metric": "market_cap", "op": ">", "value": 0}]', limit=5)
    assert set(out) == {"matched", "rule_count", "rows"}
    assert out["rule_count"] == 1
    assert out["matched"] == len(out["rows"]) <= 5
    for row in out["rows"]:
        assert {"ticker", "company_name", "sector", "metrics"} <= set(row)
    assert "error" in tools["custom_screen"]("not json at all")
    assert "error" in tools["custom_screen"]('[{"metric": "nope", "op": ">", "value": 1}]')


# ---------------------------------------------------------------------------
# Specialist tools — profile scoping
# ---------------------------------------------------------------------------

def test_profile_for_guarantees_a_ticker():
    assert chat_sdk._profile_for({}, "nvda") == {"ticker": "NVDA"}
    assert chat_sdk._profile_for({"profile": None}, " amd ") == {"ticker": "AMD"}
    assert chat_sdk._profile_for({"profile": {"ticker": "AAPL", "sector": "Tech"}}, "msft") == {
        "ticker": "AAPL", "sector": "Tech",
    }
    assert chat_sdk._profile_for({"profile": {"ticker": ""}}, "goog")["ticker"] == "GOOG"
    # With nothing to seed from, the helper does not invent a ticker.
    assert "ticker" not in chat_sdk._profile_for({}, "   ")


@pytest.fixture
def provider_miss(monkeypatch):
    """A fundamentals lookup that misses — the shape that produced the
    unscoped `doc_chunks` scan before `_profile_for` existed."""
    from app.services import fundamentals_service
    monkeypatch.setattr(
        fundamentals_service, "get_full_financials",
        lambda ticker, **_: {"ticker": ticker, "profile": {}, "ratios": {}, "earnings": {}},
    )


def _capture(monkeypatch, module_path: str, fn_name: str, profile_kw: str = "profile"):
    import importlib
    module = importlib.import_module(module_path)
    seen: dict[str, Any] = {}

    def fake(*args: Any, **kwargs: Any):
        seen["profile"] = kwargs.get(profile_kw, args[0] if args else None)
        return _finding()
    monkeypatch.setattr(module, fn_name, fake)
    return seen


def test_ask_sector_scopes_profile_on_provider_miss(tools, provider_miss, monkeypatch):
    seen = _capture(monkeypatch, "app.agents.sector_agents", "run_sector_agent")
    out = tools["ask_sector"]("zzzoff", "what changes if rates fall?")
    assert set(out) == FINDING_KEYS and out["agent"] == "sector" and out["ticker"] == "ZZZOFF"
    assert seen["profile"]["ticker"] == "ZZZOFF"


def test_ask_earnings_scopes_profile_on_provider_miss(tools, provider_miss, monkeypatch):
    seen = _capture(monkeypatch, "app.agents.earnings_agent", "run_earnings_agent")
    out = tools["ask_earnings"]("zzzoff", "tone?")
    assert out["agent"] == "earnings" and seen["profile"]["ticker"] == "ZZZOFF"


def test_ask_filings_scopes_profile_on_provider_miss(tools, provider_miss, monkeypatch):
    seen = _capture(monkeypatch, "app.agents.filing_agent", "run_filing_agent")
    out = tools["ask_filings"]("zzzoff", "new risk factor?")
    assert out["agent"] == "filings" and seen["profile"]["ticker"] == "ZZZOFF"


def test_ask_valuation_scopes_profile_on_provider_miss(tools, provider_miss, monkeypatch):
    from app.services import valuation_service
    monkeypatch.setattr(valuation_service, "build_dcf", lambda ticker, **_: None)
    seen = _capture(monkeypatch, "app.agents.valuation_agent", "run_valuation_agent")
    out = tools["ask_valuation"]("zzzoff", "why 15x terminal?")
    assert out["agent"] == "valuation" and seen["profile"]["ticker"] == "ZZZOFF"


def test_ask_macro_with_ticker_scopes_profile(tools, provider_miss, monkeypatch):
    seen = _capture(monkeypatch, "app.agents.macro_agent", "run_macro_agent")
    out = tools["ask_macro"]("rates up 100bps", ticker="zzzoff")
    assert set(out) == FINDING_KEYS and out["ticker"] == "ZZZOFF"
    assert seen["profile"]["ticker"] == "ZZZOFF"


def test_ask_macro_without_ticker_runs_scenario(tools, monkeypatch):
    from app.agents import macro_agent
    monkeypatch.setattr(macro_agent, "run_macro_scenario", lambda q: SimpleNamespace(
        scenario=q, narrative="n", favored_sectors=["Energy"],
        pressured_sectors=["Utilities"], risks=["r"],
    ))
    out = tools["ask_macro"]("oil to 120")
    assert set(out) == {
        "agent", "scenario", "narrative", "favored_sectors", "pressured_sectors", "risks",
    }
    assert out["scenario"] == "oil to 120"


def test_specialist_failure_is_an_error_dict_not_an_exception(tools, provider_miss, monkeypatch):
    from app.agents import sector_agents

    def _boom(*_a, **_k):
        raise RuntimeError("specialist exploded")
    monkeypatch.setattr(sector_agents, "run_sector_agent", _boom)
    out = tools["ask_sector"]("zzzoff", "q")
    assert set(out) == {"error"} and "specialist exploded" in out["error"]


def test_no_specialist_tool_ever_hands_vector_search_a_falsy_ticker(tools, provider_miss, no_llm, monkeypatch):
    """Belt-and-braces on the OOM guard: run the real sector specialist
    on its deterministic branch and assert every retrieval call it
    makes is scoped. `vector_store.search` refuses a falsy ticker by
    design, so this pins the *caller* side of that contract.

    The deterministic branch is forced by the autouse `no_llm` pin
    rather than assumed from blank keys — with a developer `.env` in
    place this test used to attempt a real, billable sector-agent
    completion."""
    from app.services import vector_store
    real_search = vector_store.search
    calls = []
    no_llm.clear()

    def guarded(query, **kwargs):
        calls.append(kwargs.get("ticker"))
        assert kwargs.get("ticker"), "unscoped vector search from the chat surface"
        return real_search(query, **kwargs)
    monkeypatch.setattr(vector_store, "search", guarded)
    out = tools["ask_sector"]("zzzoff", "anything")
    assert "error" not in out, out
    assert all(calls), calls
    # The specialist consulted the LLM exactly through the pinned seam,
    # so the run above is the same code path production takes minus
    # the completion — not a short-circuit that never reached it.
    assert no_llm, "sector specialist never reached llm.chat_json"
