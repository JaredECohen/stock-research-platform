"""Research labels and comparisons must describe the data actually used."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from app import cache
from app.services import sector_research_service as service


def _wire(monkeypatch, profiles, ratios):
    monkeypatch.setattr(service, "get_data_service", lambda: SimpleNamespace(
        get_company_profile=lambda ticker: profiles[ticker],
        list_tickers=lambda: list(profiles),
    ))
    monkeypatch.setattr(service, "get_full_financials", lambda ticker: {
        "profile": profiles[ticker], "ratios": ratios.get(ticker, {}),
        "income": [], "cash": [],
    })
    monkeypatch.setattr(service, "get_filings", lambda ticker: [])
    monkeypatch.setattr(cache, "cache_get", lambda *args, **kwargs: None)
    monkeypatch.setattr(cache, "cache_put", lambda *args, **kwargs: None)


@pytest.mark.parametrize("basis", ["sub_industry", "industry", "sector"])
def test_cohort_reports_actual_fallback_without_changing_peer_membership(monkeypatch, basis):
    target = {"sector": "Technology", "industry": "Software", "sub_industry": "Application"}
    profiles = {"TARGET": target}
    for i in range(3):
        profiles[f"PEER{i}"] = {
            "sector": "Technology",
            "industry": "Software" if basis != "sector" else f"Other{i}",
            "sub_industry": "Application" if basis == "sub_industry" else f"Other{i}",
        }
    _wire(monkeypatch, profiles, {ticker: {"EV_EBITDA": 15} for ticker in profiles})
    peers = service.build_cohort("TARGET")
    payload = service.run_sector_research("TARGET", force_refresh=True)
    assert peers == ["PEER0", "PEER1", "PEER2"]
    assert payload["cohort"]["peers"] == peers
    assert payload["cohort"]["selection_basis"] == basis


@pytest.mark.parametrize("profile", [{}, {"ticker": "TARGET"}])
def test_missing_taxonomy_does_not_claim_a_sector_cohort(monkeypatch, profile):
    _wire(monkeypatch, {"TARGET": profile}, {"TARGET": {"EV_EBITDA": 10}})
    payload = service.run_sector_research("TARGET", force_refresh=True)
    assert payload["cohort"] == {"peers": [], "size": 0, "selection_basis": "unavailable"}


def test_invalid_ebitda_multiples_are_not_cheap_and_every_exclusion_is_named(monkeypatch, caplog):
    values = {"TARGET": -2, "NEGATIVE": -10, "ZERO": 0, "NAN": float("nan"),
              "INFINITY": float("inf"), "MISSING": None, "NO_RATIOS": None, "VALID1": 10, "VALID2": 20}
    profiles = {ticker: {"sector": "Technology", "industry": "Software"} for ticker in values}
    ratios = {ticker: {"EV_EBITDA": value, "revenue_growth": .1} for ticker, value in values.items()}
    ratios["NO_RATIOS"] = {}
    _wire(monkeypatch, profiles, ratios)
    with caplog.at_level(logging.INFO):
        payload = service.run_sector_research("TARGET", force_refresh=True)
    placement = payload["kpi_placements"]["EV_EBITDA"]
    assert placement["target"] is None
    assert "quartile" not in placement
    assert placement["distribution"]["n"] == 2
    assert placement["distribution"]["median"] == 15
    assert payload["outliers"]["valuation_cheapest"] == "VALID1"
    assert payload["valuation_exclusions"]["count"] == 7
    assert set(payload["valuation_exclusions"]["tickers"]) == set(values) - {"VALID1", "VALID2"}
    for ticker in payload["valuation_exclusions"]["tickers"]:
        assert ticker in caplog.text
    assert ratios["TARGET"]["EV_EBITDA"] == -2  # source factor inputs stay intact


def test_theme_mentions_per_peer_are_not_labelled_as_company_share(monkeypatch):
    monkeypatch.setattr(service, "get_filings", lambda ticker: (
        [{"risk_factors": ["competition"], "mda": ""}] * 3 if ticker == "ACTIVE" else []
    ))
    themes = service.aggregate_cohort_filing_themes(["ACTIVE", "QUIET"])
    assert themes == [{"theme": "Competitive intensity", "cohort_mentions": 3, "mentions_per_peer": 1.5}]


def test_no_valid_ebitda_multiple_has_no_cheapest_company_or_distribution(monkeypatch):
    profiles = {ticker: {"sector": "Technology"} for ticker in ["TARGET", "NEGATIVE", "ZERO"]}
    _wire(monkeypatch, profiles, {"TARGET": {}, "NEGATIVE": {"EV_EBITDA": -10}, "ZERO": {"EV_EBITDA": 0}})
    payload = service.run_sector_research("TARGET", force_refresh=True)
    assert payload["outliers"]["valuation_cheapest"] is None
    assert "EV_EBITDA" not in payload["kpi_placements"]
    assert payload["valuation_exclusions"]["count"] == 3
    assert set(payload["valuation_exclusions"]["tickers"]) == set(profiles)


def test_theme_aggregation_does_not_silently_drop_the_tail(monkeypatch):
    monkeypatch.setattr(service, "get_filings", lambda ticker: [{
        "risk_factors": [needle for needle, _ in service._RISK_KEYWORDS], "mda": "",
    }])
    themes = service.aggregate_cohort_filing_themes(["ALL"])
    assert {theme["theme"] for theme in themes} == {label for _, label in service._RISK_KEYWORDS}


def test_legacy_cached_research_is_rebuilt_with_corrected_semantics(monkeypatch):
    profiles = {"TARGET": {"sector": "Technology", "industry": "Software"}}
    _wire(monkeypatch, profiles, {"TARGET": {"EV_EBITDA": 10}})
    legacy = SimpleNamespace(id=1, payload={"cohort": {"selection_basis": "sub_industry"}})
    monkeypatch.setattr(cache, "cache_get", lambda *args, **kwargs: legacy)
    payload = service.run_sector_research("TARGET")
    assert payload["cohort"]["selection_basis"] == "sector"
    assert "valuation_exclusions" in payload
