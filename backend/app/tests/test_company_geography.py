"""Tests for the company_geography service."""
from __future__ import annotations

from app.services import company_geography


def test_seeded_tickers_present():
    seeded = set(company_geography.list_seeded_tickers())
    for ticker in ["PLD", "AMT", "AVB", "EQR", "WMT", "COST", "HD", "MCD",
                   "JPM", "BAC", "WFC", "NEE", "DUK", "XOM", "CVX"]:
        assert ticker in seeded, f"{ticker} missing from geography seed"


def test_seed_returns_consistent_shape():
    geo = company_geography.get_geography("PLD", allow_llm_fallback=False)
    assert geo is not None
    assert geo["source"] == "seed"
    assert geo["ticker"] == "PLD"
    assert geo["type"] == "REIT"
    assert isinstance(geo["metros"], dict) and geo["metros"]
    assert isinstance(geo["states"], dict) and geo["states"]


def test_unknown_ticker_without_llm_returns_none():
    assert company_geography.get_geography(
        "ZZZZ_UNKNOWN", allow_llm_fallback=False,
    ) is None


def test_metro_weights_within_unit_range():
    """Weights are relative concentration hints in [0, 1.5]."""
    for ticker in company_geography.list_seeded_tickers():
        geo = company_geography.get_geography(ticker, allow_llm_fallback=False)
        if geo is None:
            continue
        for code, weight in (geo.get("metros") or {}).items():
            assert 0.0 <= weight <= 1.5, f"{ticker} metro {code} weight {weight} out of range"
        for code, weight in (geo.get("states") or {}).items():
            assert 0.0 <= weight <= 1.5, f"{ticker} state {code} weight {weight} out of range"


def test_reit_seeds_have_metro_concentration():
    """REIT seed entries should expose at least one metro weight."""
    reit_tickers = ["PLD", "EQIX", "AVB", "EQR", "VICI"]
    for ticker in reit_tickers:
        geo = company_geography.get_geography(ticker, allow_llm_fallback=False)
        assert geo and geo.get("metros"), f"{ticker} should expose metro weights"
