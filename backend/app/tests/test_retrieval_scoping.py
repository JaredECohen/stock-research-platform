"""Caller-side regression tests for the 2026-08-12 Render OOM-kill.

`test_vector_store_memory_guard.py` covers the choke point itself. This
file covers the paths that *reached* it with a missing ticker: the chat
specialist tools, whose `fin.get("profile") or {}` produced a profile
with no ticker whenever the fundamentals lookup missed, and the two
agents that forwarded that ticker into `vector_store.search`.
"""
from __future__ import annotations

from app.agents.chat_sdk import _profile_for


# ---------------------------------------------------------------------------
# chat_sdk._profile_for — the actual entry point of the production bug
# ---------------------------------------------------------------------------

def test_profile_for_seeds_ticker_when_lookup_missed():
    """The bug: `fin.get("profile") or {}` yielded a profile with no
    ticker, which the specialists forwarded into an unscoped search."""
    assert _profile_for({}, "nvda")["ticker"] == "NVDA"
    assert _profile_for({"profile": None}, "nvda")["ticker"] == "NVDA"
    assert _profile_for({"profile": {}}, "nvda")["ticker"] == "NVDA"


def test_profile_for_preserves_a_real_profile():
    """A populated profile must pass through untouched — the fix is a
    backfill, not an override."""
    fin = {"profile": {"ticker": "MSFT", "sector": "Technology", "name": "Microsoft"}}
    out = _profile_for(fin, "msft")
    assert out == {"ticker": "MSFT", "sector": "Technology", "name": "Microsoft"}


def test_profile_for_does_not_mutate_the_source():
    """`get_full_financials` results are cached; mutating them in place
    would leak a synthesized ticker into other callers."""
    original = {"sector": "Energy"}
    fin = {"profile": original}
    _profile_for(fin, "xom")
    assert original == {"sector": "Energy"}


def test_profile_for_with_no_ticker_available():
    """Nothing to seed from → return the profile as-is rather than
    inventing a key. `vector_store` refuses the unscoped search."""
    assert _profile_for({}, "") == {}
    assert "ticker" not in _profile_for({"profile": {"sector": "X"}}, "  ")


# ---------------------------------------------------------------------------
# Agent call sites skip retrieval instead of scanning globally
# ---------------------------------------------------------------------------

def test_filing_agent_with_no_ticker_does_not_search(monkeypatch):
    from app.agents import filing_agent
    from app.services import vector_store

    called = []
    monkeypatch.setattr(
        vector_store, "search",
        lambda *a, **k: called.append(k) or [],
    )
    finding = filing_agent.run_filing_agent(
        {"sector": "Technology"},  # no ticker
        [{"type": "10-K", "url": "", "risk_factors": ["r1"], "mda": "m"}],
    )
    assert called == [], "unscoped vector search was issued"
    assert finding.agent  # still produced a finding via the BM25 fallback


def test_earnings_agent_with_no_ticker_does_not_search(monkeypatch):
    from app.agents import earnings_agent
    from app.services import vector_store

    called = []
    monkeypatch.setattr(
        vector_store, "search",
        lambda *a, **k: called.append(k) or [],
    )
    finding = earnings_agent.run_earnings_agent(
        {"sector": "Technology"},  # no ticker
        {"period": "2025Q4", "prepared_remarks": "we grew", "qa": "q and a"},
        {},
    )
    assert called == [], "unscoped vector search was issued"
    assert finding.agent


def test_filing_agent_with_a_ticker_still_searches(monkeypatch):
    """Guard must not disable retrieval on the normal path."""
    from app.agents import filing_agent
    from app.services import vector_store

    called = []
    monkeypatch.setattr(
        vector_store, "search",
        lambda *a, **k: called.append(k) or [],
    )
    filing_agent.run_filing_agent(
        {"ticker": "AAPL", "sector": "Technology"},
        [{"type": "10-K", "url": "", "risk_factors": ["r1"], "mda": "m"}],
    )
    assert len(called) == 1
    assert called[0]["ticker"] == "AAPL"
