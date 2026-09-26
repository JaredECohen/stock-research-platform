"""Phase 3 — OpenAI Agents SDK runtime tests.

Verifies that:
- The agent topology builds and includes PM, sector, and tool agents.
- Handoff plumbing is wired (PM lists sectors as handoffs; sectors list
  tool agents).
- `run_stock_memo_via_sdk` returns a populated `StockMemoOut` even with no
  LLM keys (falls through to the legacy graph as the deterministic backstop).
- The orchestrator dispatches through the SDK when `USE_AGENTS_SDK=true`.
- Cache-backed tools resolve by reading the snapshot store.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.agents.sdk_runtime import (
    SECTOR_NAMES,
    TOOL_NAMES,
    Runner,
    get_agents,
    get_cached_company_cold,
    run_stock_memo_via_sdk,
)
from app.cache import cache_put
from app.config import settings


def test_sdk_topology_has_pm_sectors_and_tools():
    agents_map = get_agents()
    assert "pm" in agents_map
    assert all(f"sector:{s}" in agents_map for s in SECTOR_NAMES)
    assert all(f"tool:{t}" in agents_map for t in TOOL_NAMES)
    pm = agents_map["pm"]
    sector_handoff_names = {a.name for a in pm.handoffs}
    for s in SECTOR_NAMES:
        assert f"sector-{s.lower()}" in sector_handoff_names
    sector = agents_map["sector:Technology"]
    tool_handoff_names = {a.name for a in sector.handoffs}
    for t in TOOL_NAMES:
        assert f"{t}-tool" in tool_handoff_names


def test_runner_max_iterations_is_capped():
    """Sanity-check the depth guard so peer-sector recursion can't blow the stack."""
    pm = get_agents()["pm"]
    result = Runner.run(pm, {"ticker": "NVDA"}, max_iterations=2)
    assert result.iterations <= 2
    assert result.final_output is not None


def test_run_stock_memo_via_sdk_returns_populated_memo():
    memo = run_stock_memo_via_sdk("NVDA")
    assert memo is not None
    assert memo.ticker == "NVDA"
    assert memo.rating_label in ("Bullish", "Mixed Positive", "Neutral", "Mixed Negative", "Bearish")
    assert memo.final_pm_view


def test_cache_backed_tool_resolves_company_cold():
    cache_put(
        "TESTSDK", "company_cold",
        payload={"profile": {"ticker": "TESTSDK", "company_name": "Test Co"}},
        sources_used=["filing:TESTSDK:000001"],
    )
    payload = get_cached_company_cold.fn("TESTSDK")
    assert payload is not None
    assert payload["profile"]["ticker"] == "TESTSDK"


def test_orchestrator_routes_via_sdk_when_flag_is_on(monkeypatch):
    monkeypatch.setattr(settings, "use_agents_sdk", True)
    client = TestClient(__import__("app.main", fromlist=["app"]).app)
    r = client.post("/api/chat", json={"message": "Analyze NVDA as a long-term investment.", "history": []})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "single_stock_analysis"
    assert data["memo"] is not None
    assert data["memo"]["ticker"] == "NVDA"


# ---------------------------------------------------------------------------
# Real-SDK memo exchange: one run id, one memo run, no output kept (B8-C1)
# ---------------------------------------------------------------------------

EXCHANGE_SENTINEL = "EXCHANGE-OUTPUT-SENTINEL-51d0"


def test_sdk_exchange_links_run_id_runs_memo_once_and_keeps_no_output(monkeypatch, caplog):
    """Attribution critique #5 and FIX-020. The exchange's
    `produce_legacy_memo` tool runs the memo under the exchange's run id,
    that memo is the one returned (the shim used to run a second full
    memo), and neither the INFO line nor the SDKTrace row carries the
    model's text. Nothing is sent: the SDK's Agent/Runner are replaced and
    the fake Runner calls the tool the way the model would."""
    import logging
    from types import SimpleNamespace

    import pytest
    real_sdk = pytest.importorskip("agents")

    from app.agents import graph, sdk_runtime
    from app.database import SessionLocal
    from app.models import SDKTrace
    from app.tests.factories import make_memo

    class _FakeAgent:
        def __init__(self, **kw):
            self.name = kw["name"]
            self.model = kw["model"]
            self.tools = kw.get("tools", [])

    seen: dict = {}

    class _FakeRunner:
        @staticmethod
        def run_sync(agent, prompt, **kw):
            seen["run_config"] = kw.get("run_config")
            seen["hooks"] = kw.get("hooks")
            (tool,) = agent.tools
            tool("nvda")
            return SimpleNamespace(
                final_output=f"Bullish. {EXCHANGE_SENTINEL}",
                new_items=[SimpleNamespace(
                    type="tool_call_item", agent=SimpleNamespace(name="pm"),
                    raw_item=SimpleNamespace(name="produce_legacy_memo",
                                             arguments=f'{{"note": "{EXCHANGE_SENTINEL}"}}'),
                )],
            )

    memo_runs: list[dict] = []

    def fake_memo(ticker, **kw):
        memo_runs.append({"ticker": ticker, **kw})
        return make_memo(ticker=ticker.upper(), company_name="NVIDIA")

    monkeypatch.setattr(real_sdk, "Agent", _FakeAgent)
    monkeypatch.setattr(real_sdk, "Runner", _FakeRunner)
    monkeypatch.setattr(real_sdk, "function_tool", lambda fn: fn)
    monkeypatch.setattr(sdk_runtime, "_can_use_real_sdk", lambda: True)
    monkeypatch.setattr(graph, "run_stock_memo", fake_memo)

    with caplog.at_level(logging.DEBUG, logger=sdk_runtime.__name__):
        memo = run_stock_memo_via_sdk("NVDA")

    assert memo.ticker == "NVDA"
    assert len(memo_runs) == 1, "the memo ran twice (tool + shim)"
    run_id = memo_runs[0].get("run_id")
    assert run_id, "produce_legacy_memo ran the memo without the exchange's run id"
    assert seen["run_config"].tracing_disabled is True
    assert seen["run_config"].trace_include_sensitive_data is False
    assert seen["hooks"] is not None
    assert f"Agents SDK exchange for NVDA (run {run_id}): items=1 chars=" in caplog.text
    assert EXCHANGE_SENTINEL not in caplog.text
    with SessionLocal() as db:
        (trace,) = db.query(SDKTrace).filter(SDKTrace.run_id == run_id).all()
    assert trace.surface == "memo" and trace.final_output == ""
    assert trace.new_items == [{"type": "tool_call_item", "agent": "pm", "tool": "produce_legacy_memo"}]


def _install_fake_exchange(monkeypatch, run_sync):
    """The real SDK's Agent/Runner replaced by fakes (nothing is sent), and
    `graph.run_stock_memo` counted. Returns the list of memo runs."""
    import pytest
    real_sdk = pytest.importorskip("agents")

    from app.agents import graph, sdk_runtime
    from app.tests.factories import make_memo

    class _FakeAgent:
        def __init__(self, **kw):
            self.name = kw["name"]
            self.model = kw["model"]
            self.tools = kw.get("tools", [])

    class _FakeRunner:
        pass

    _FakeRunner.run_sync = staticmethod(run_sync)
    memo_runs: list[dict] = []

    def fake_memo(ticker, **kw):
        memo_runs.append({"ticker": ticker, **kw})
        return make_memo(ticker=ticker.upper(), company_name="NVIDIA")

    monkeypatch.setattr(real_sdk, "Agent", _FakeAgent)
    monkeypatch.setattr(real_sdk, "Runner", _FakeRunner)
    monkeypatch.setattr(real_sdk, "function_tool", lambda fn: fn)
    monkeypatch.setattr(sdk_runtime, "_can_use_real_sdk", lambda: True)
    monkeypatch.setattr(graph, "run_stock_memo", fake_memo)
    return memo_runs


def _memo_trace(run_id):
    from app.database import SessionLocal
    from app.models import SDKTrace
    with SessionLocal() as db:
        return db.query(SDKTrace).filter(SDKTrace.run_id == run_id).all()


def test_sdk_exchange_failing_after_its_tool_ran_reuses_that_memo(monkeypatch):
    """Critique #5: the run can fail after `produce_legacy_memo` built the
    memo (max turns, a refusal on the summary turn). That memo is returned;
    the shim must not run a second memo under the same run id (PM synthesis
    billed twice, a second memo version saved)."""
    def run_sync(agent, prompt, **kw):
        (tool,) = agent.tools
        tool("nvda")
        raise RuntimeError("final summary turn failed")

    memo_runs = _install_fake_exchange(monkeypatch, run_sync)
    memo = run_stock_memo_via_sdk("NVDA")

    assert memo.ticker == "NVDA"
    assert len(memo_runs) == 1, "the memo ran twice (tool, then the shim)"
    run_id = memo_runs[0]["run_id"]
    (trace,) = _memo_trace(run_id)
    assert trace.error == "RuntimeError" and trace.final_output == ""


def test_sdk_exchange_without_a_tool_call_runs_the_shim_memo_under_its_run_id(monkeypatch):
    """When the model never calls the tool, the shim's PM handler runs the
    memo once, under the exchange's run id, so the SDKTrace row and the
    memo's rows join in the admin timeline."""
    from types import SimpleNamespace

    def run_sync(agent, prompt, **kw):
        return SimpleNamespace(final_output="No tool today.", new_items=[])

    memo_runs = _install_fake_exchange(monkeypatch, run_sync)
    memo = run_stock_memo_via_sdk("NVDA")

    assert memo.ticker == "NVDA"
    assert len(memo_runs) == 1
    run_id = memo_runs[0].get("run_id")
    assert run_id, "the shim ran the memo under a fresh run id"
    (trace,) = _memo_trace(run_id)
    assert trace.surface == "memo"
