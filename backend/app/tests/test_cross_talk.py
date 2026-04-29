"""Phase 6 — sector-to-sector cross-talk + macro broadcast subscription tests."""
from __future__ import annotations

from app.agents.graph import run_stock_memo
from app.agents.sector_agents import run_sector_agent
from app.agents.sdk_runtime import query_peer_sector
from app.cache import cache_put
from app.monitoring import macro_loop
from app.services.fundamentals_service import get_full_financials


def test_sector_finding_includes_cross_sector_relevance():
    fin = get_full_financials("NVDA")
    finding = run_sector_agent(fin["profile"], fin["ratios"])
    cross = (finding.data or {}).get("cross_sector_relevance") or []
    # The Technology adjacency map includes NEE (Utilities), which is in the
    # demo universe — so the call should yield at least one peer ticker.
    assert isinstance(cross, list)
    assert any(t in cross for t in ("NEE", "CAT"))


def test_pm_memo_surfaces_cross_sector_relevance_for_nvda():
    memo = run_stock_memo("NVDA", force_refresh=True)
    sector_data = memo.sector_agent_view.data or {}
    cross = sector_data.get("cross_sector_relevance") or []
    assert cross, "Expected sector view to populate cross_sector_relevance"
    # The verdict should mention pull-through tickers.
    assert "Cross-sector pull-through" in memo.final_verdict
    assert any(t in memo.final_verdict for t in cross)


def test_sector_agent_subscribes_to_macro_broadcast():
    macro_loop.run_once()  # ensure a macro_broadcast snapshot exists
    fin = get_full_financials("NVDA")
    finding = run_sector_agent(fin["profile"], fin["ratios"])
    macro_block = (finding.data or {}).get("macro_broadcast") or {}
    assert "regime" in macro_block


def test_query_peer_sector_returns_cached_payload():
    cache_put(
        "Utilities:Electric Utilities:NEE", "sector_warm",
        payload={"target_ticker": "NEE", "sector": "Utilities", "regime": "soft_landing"},
        sources_used=["test"], generated_by="test", cost_tokens=0,
    )
    result = query_peer_sector.fn("Utilities", "Will load growth from data centers help NEE?")
    assert result["sector"] == "Utilities"
    assert result["snapshot"] is not None
    assert result["snapshot"]["target_ticker"] == "NEE"
