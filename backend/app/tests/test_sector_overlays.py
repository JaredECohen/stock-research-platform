"""Tests for the dynamic sector data tools + overlay computations."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from app.providers.base import ProviderStatus
from app.services import data_catalog_service, sector_overlays
from app.services.data_service import get_data_service


class _StubMacroProvider:
    """Minimal provider returning the same shape FRED/EIA do.

    Substitutes for the real network-bound providers in unit tests. The
    `responses` dict maps series_id -> {points: [...]}; series_ids not
    present return None so the catalog service marks them as errored.
    """
    name = "stub-macro"

    def __init__(self, responses: Dict[str, Dict[str, Any]]) -> None:
        self._responses = responses

    def status(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, configured=True, healthy=True)

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        return self._responses.get(series_id)

    # The data_catalog_service walks _live_chain('macro') looking for any
    # provider with `get_macro_series`. We don't need to implement the
    # other BaseProvider methods.
    def get_company_profile(self, *_a, **_k): return None
    def get_price_history(self, *_a, **_k): return None
    def get_financial_statements(self, *_a, **_k): return None
    def get_ratios(self, *_a, **_k): return None
    def get_key_metrics(self, *_a, **_k): return None
    def get_earnings(self, *_a, **_k): return None
    def get_earnings_transcripts(self, *_a, **_k): return None
    def get_filings(self, *_a, **_k): return None
    def get_news(self, *_a, **_k): return None
    def get_estimates(self, *_a, **_k): return None
    def list_tickers(self) -> List[str]: return []
    def list_macro_series(self) -> List[Dict[str, Any]]: return []


def _monthly_points(start: float, deltas: List[float]) -> List[Dict[str, Any]]:
    """Build a list of monthly {date, value} points starting from `start`."""
    out: List[Dict[str, Any]] = []
    value = start
    year, month = 2021, 1
    for d in deltas:
        value += d
        out.append({
            "date": f"{year:04d}-{month:02d}-28",
            "value": round(value, 4),
        })
        month += 1
        if month > 12:
            year += 1
            month = 1
    return out


@pytest.fixture
def stub_provider_responses() -> Dict[str, Dict[str, Any]]:
    """Hand-rolled responses for a representative slice of the catalog."""
    return {
        "CSUSHPISA": {
            "series_id": "CSUSHPISA",
            "name": "Case-Shiller National HPI",
            "units": "index",
            "points": _monthly_points(280.0, [1.0] * 36),
        },
        "MORTGAGE30US": {
            "series_id": "MORTGAGE30US",
            "name": "30-Year Mortgage",
            "units": "%",
            "points": _monthly_points(3.0, [0.1] * 36),
        },
        "HOUST": {
            "series_id": "HOUST",
            "name": "Housing Starts",
            "units": "thousands",
            "points": _monthly_points(1400.0, [10.0] * 36),
        },
        "PERMIT": {
            "series_id": "PERMIT",
            "name": "Building Permits",
            "units": "thousands",
            "points": _monthly_points(1500.0, [5.0] * 36),
        },
        "EXHOSLUSM495S": {
            "series_id": "EXHOSLUSM495S",
            "name": "Existing Home Sales",
            "units": "thousands",
            "points": _monthly_points(5000.0, [-10.0] * 36),
        },
        "MSPUS": {
            "series_id": "MSPUS",
            "name": "Median Sales Price",
            "units": "$",
            "points": _monthly_points(420000.0, [500.0] * 36),
        },
        "RRVRUSQ156N": {
            "series_id": "RRVRUSQ156N",
            "name": "Rental Vacancy",
            "units": "%",
            "points": _monthly_points(6.0, [0.05] * 36),
        },
        "CUUR0000SEHA": {
            "series_id": "CUUR0000SEHA",
            "name": "Rent CPI",
            "units": "index",
            "points": _monthly_points(330.0, [0.5] * 36),
        },
        "LXXRSA": {
            "series_id": "LXXRSA",
            "name": "Case-Shiller LA",
            "units": "index",
            "points": _monthly_points(350.0, [1.5] * 36),
        },
        "NYXRSA": {
            "series_id": "NYXRSA",
            "name": "Case-Shiller NYC",
            "units": "index",
            "points": _monthly_points(280.0, [0.8] * 36),
        },
        "ATXRSA": {
            "series_id": "ATXRSA",
            "name": "Case-Shiller Atlanta",
            "units": "index",
            "points": _monthly_points(250.0, [1.0] * 36),
        },
        # Inflation / credit for the broader prepare_sector_context fan-out
        "BAMLH0A0HYM2": {
            "series_id": "BAMLH0A0HYM2",
            "name": "High-Yield Spread",
            "units": "%",
            "points": _monthly_points(3.5, [0.01] * 36),
        },
        "T10Y2Y": {
            "series_id": "T10Y2Y",
            "name": "10Y-2Y Term Spread",
            "units": "%",
            "points": _monthly_points(-0.2, [0.01] * 36),
        },
        "TOTALSL": {
            "series_id": "TOTALSL",
            "name": "Consumer Credit",
            "units": "billions",
            "points": _monthly_points(4800.0, [10.0] * 36),
        },
        "DRCCLACBS": {
            "series_id": "DRCCLACBS",
            "name": "CC Delinquency",
            "units": "%",
            "points": _monthly_points(2.5, [0.02] * 36),
        },
        "CPIAUCSL": {
            "series_id": "CPIAUCSL",
            "name": "Headline CPI",
            "units": "index",
            "points": _monthly_points(290.0, [0.7] * 36),
        },
        "PCEPI": {
            "series_id": "PCEPI",
            "name": "PCE",
            "units": "index",
            "points": _monthly_points(120.0, [0.3] * 36),
        },
        "CORESTICKM159SFRBATL": {
            "series_id": "CORESTICKM159SFRBATL",
            "name": "Sticky Core CPI YoY",
            "units": "%",
            "points": _monthly_points(4.0, [-0.02] * 36),
        },
    }


@pytest.fixture
def install_stub_provider(stub_provider_responses):
    ds = get_data_service()
    stub = _StubMacroProvider(stub_provider_responses)
    ds.register_test_provider(stub)
    yield stub
    # Reinstall the conftest demo provider
    from app.tests.fixtures.demo_provider import DemoProvider
    ds.register_test_provider(DemoProvider())


def test_fetch_series_returns_snapshot_with_derived_stats(install_stub_provider):
    snap = data_catalog_service.fetch_series("CSUSHPISA")
    assert snap is not None and snap.error is None
    assert snap.latest is not None
    assert snap.yoy_pct is not None
    assert snap.z_score_5y is not None
    assert len(snap.sample_points) > 0


def test_discover_for_ticker_routes_by_sector_and_geography(install_stub_provider):
    profile = {"sector": "Real Estate", "sub_industry": "Industrial REITs"}
    discovered = data_catalog_service.discover_for_ticker("PLD", profile=profile)
    sector_ids = {s["series_id"] for s in discovered["sector_relevant"]}
    geo_ids = {s["series_id"] for s in discovered["geography_relevant"]}
    # Real Estate sector should pull in mortgage rates + Case-Shiller national
    assert "MORTGAGE30US" in sector_ids
    assert "CSUSHPISA" in sector_ids
    # Geography should surface metro HPI series for PLD's biggest metros (LA, NYC, ATL).
    assert any(sid in geo_ids for sid in ("LXXRSA", "NYXRSA", "ATXRSA"))


def test_real_estate_overlay_computes_footprint_weighted_yoy(install_stub_provider):
    profile = {"sector": "Real Estate", "sub_industry": "Industrial REITs"}
    overlay = sector_overlays.compute_real_estate_overlay("PLD", profile=profile)
    assert overlay["available"] is True
    assert overlay["geography"] is not None
    assert overlay["metro_overlay"], "PLD has metro weights in the seed"
    weighted = overlay["footprint_weighted_home_price_yoy"]
    assert weighted is not None
    # All monthly deltas are positive in the fixture, so the weighted YoY
    # must come out positive too.
    assert weighted > 0
    assert overlay["narrative_hints"]


def test_dispatch_picks_overlay_set_by_sector(install_stub_provider):
    profile = {"sector": "Real Estate", "sub_industry": "Industrial REITs"}
    bundles = sector_overlays.compute_sector_overlays("PLD", profile=profile)
    assert "real_estate" in bundles["bundles"]
    assert bundles["overlays_run"][:1] == ["real_estate"]


def test_prepare_sector_context_assembles_full_payload(install_stub_provider):
    from app.agents.sector_tools import prepare_sector_context, render_prompt_block
    profile = {"sector": "Real Estate", "sub_industry": "Industrial REITs"}
    ctx = prepare_sector_context("PLD", profile=profile)
    assert ctx["ticker"] == "PLD"
    assert ctx["discovered_catalog"]["sector_relevant"], "sector axis must surface entries"
    assert ctx["prefetched_snapshots"], "should pre-fetch at least one snapshot"
    assert ctx["overlays"]["bundles"]
    block = render_prompt_block(ctx)
    assert "Sector data context" in block
    assert "Pre-fetched readings" in block
