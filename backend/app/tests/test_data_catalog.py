"""Tests for the curated data catalog + registry lookups."""
from __future__ import annotations

from app.data_catalog import (
    SERIES_REGISTRY,
    by_category,
    by_id,
    by_sector_tag,
    list_categories,
    list_regions,
    search,
)


def test_registry_is_non_empty_and_unique():
    assert len(SERIES_REGISTRY) >= 50
    ids = [s.series_id for s in SERIES_REGISTRY]
    assert len(ids) == len(set(ids)), "series_ids must be globally unique"


def test_required_top_series_present():
    must_have = [
        "DGS10", "FEDFUNDS", "CPIAUCSL",                # rates / inflation
        "CSUSHPISA", "MORTGAGE30US", "HOUST",           # real estate
        "RSAFS", "PSAVERT",                              # consumer
        "PET.RWTC.D", "NG.RNGWHHD.D",                    # energy prices
        "PET.WCESTUS1.W", "NG.NW2_EPG0_SWO_R48_BCF.W",   # energy storage
        "BAMLH0A0HYM2", "T10Y2Y",                        # credit
        "MARTS_44X72",                                   # census retail
    ]
    for sid in must_have:
        assert by_id(sid) is not None, f"missing required series {sid}"


def test_categories_cover_all_axes():
    cats = set(list_categories())
    assert {"rates", "inflation", "real_estate", "energy", "retail",
            "credit", "labor"}.issubset(cats)


def test_metro_regions_present():
    regions = list_regions()
    metro_regions = [r for r in regions if r.startswith("metro:")]
    assert len(metro_regions) >= 10, "expected at least 10 metro-tagged series"


def test_sector_tag_routing():
    re_specs = by_sector_tag("Real Estate")
    energy_specs = by_sector_tag("Energy")
    fin_specs = by_sector_tag("Financials")
    assert any(s.series_id == "MORTGAGE30US" for s in re_specs)
    assert any(s.series_id == "PET.RWTC.D" for s in energy_specs)
    assert any(s.series_id == "BAMLH0A0HYM2" for s in fin_specs)


def test_search_by_keyword_and_source():
    results = search(keywords=["mortgage"])
    assert any(s.series_id == "MORTGAGE30US" for s in results)

    eia_only = search(sources=["EIA"])
    assert eia_only and all(s.source == "EIA" for s in eia_only)


def test_search_by_region_metro():
    sf_only = search(region="metro:SF")
    assert sf_only and all(s.region == "metro:SF" for s in sf_only)


def test_by_category_returns_consistent_source():
    real_estate = by_category("real_estate")
    assert real_estate
    # All real_estate entries should be FRED or Census in this registry.
    assert all(s.source in {"FRED", "Census"} for s in real_estate)
