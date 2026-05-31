"""Curated data catalog for sector-relevant time series.

This package is the single source of truth for *what data exists* across
all macro / energy / regional providers. The sector analyst agent
browses this catalog to discover series relevant to a ticker, then
fetches them through `services.data_catalog_service`.

Layout:
    registry.py       - Curated SeriesSpec list across FRED / EIA / BLS /
                        Census. Pure data structure. Each entry carries
                        the metadata an LLM needs to decide relevance:
                        category, region, sector_tags, frequency, source,
                        description.
"""
from .registry import (
    SERIES_REGISTRY,
    SeriesSpec,
    by_category,
    by_id,
    by_sector_tag,
    by_sub_industry_tag,
    list_categories,
    list_regions,
    list_sector_tags,
    search,
)

__all__ = [
    "SERIES_REGISTRY",
    "SeriesSpec",
    "by_category",
    "by_id",
    "by_sector_tag",
    "by_sub_industry_tag",
    "list_categories",
    "list_regions",
    "list_sector_tags",
    "search",
]
