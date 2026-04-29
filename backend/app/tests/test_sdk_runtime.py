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
