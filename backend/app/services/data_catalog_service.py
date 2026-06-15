"""Runtime layer over the curated `data_catalog` registry.

This is what the sector agent (and other consumers) actually call. It
takes a SeriesSpec from the registry, dispatches to the right provider,
caches the result, and computes the small derived statistics that make
a time-series snapshot useful in a memo (level, MoM/YoY %, z-score
versus the trailing 5y).

Discovery surface:
  - `discover_for_ticker(ticker, profile)` — returns the catalog entries
    most relevant to a ticker based on its GICS sector / sub-industry,
    plus any extras the caller passes in. Pure metadata; no fetch.
  - `discover_by_query(...)` — direct passthrough to `data_catalog.search`.

Fetch surface:
  - `fetch_series(series_id)` — read-through cached SeriesSnapshot.
  - `fetch_snapshots(series_ids)` — bulk variant; small parallelism.

All fetches go through `provider_cache` with capability=`macro` and
the default TTL.
"""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..data_catalog import (
    SERIES_REGISTRY,
    SeriesSpec,
    by_id,
    by_sector_tag,
    by_sub_industry_tag,
    search as catalog_search,
)
from . import provider_cache

log = logging.getLogger(__name__)


@dataclass
class SeriesSnapshot:
    """Latest reading + trailing changes + spec metadata, ready to display."""
    series_id: str
    name: str
    source: str
    category: str
    units: str
    frequency: str
    region: str
    description: str
    latest: Optional[Dict[str, Any]] = None
    prior: Optional[Dict[str, Any]] = None
    change_1m: Optional[float] = None
    change_3m: Optional[float] = None
    change_12m: Optional[float] = None
    yoy_pct: Optional[float] = None
    z_score_5y: Optional[float] = None
    sample_points: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def discover_for_ticker(
    ticker: str,
    *,
    profile: Optional[Dict[str, Any]] = None,
    extra_categories: Optional[Iterable[str]] = None,
    max_per_axis: int = 8,
) -> Dict[str, List[Dict[str, Any]]]:
    """Suggest catalog entries for a ticker.

    Routes by:
      - GICS sector  -> registry entries with that `sector_tag`
      - sub-industry -> registry entries with that `sub_industry_tag`
      - extra categories the caller wants to surface regardless
      - the ticker's geographic footprint (if known) -> any region-
        specific entries that match its metros/states

    Output groups the results by axis so the agent can see WHY each
    series was surfaced. Each entry is a `SeriesSpec.to_dict()` payload.
    """
    sector = (profile or {}).get("sector") or ""
    sub_industry = (profile or {}).get("sub_industry") or ""
    groups: Dict[str, List[Dict[str, Any]]] = {
        "sector_relevant": [],
        "sub_industry_relevant": [],
        "geography_relevant": [],
        "category_relevant": [],
    }
    seen: set[str] = set()

    if sector:
        for spec in by_sector_tag(sector)[:max_per_axis]:
            if spec.series_id in seen:
                continue
            groups["sector_relevant"].append(spec.to_dict())
            seen.add(spec.series_id)

    if sub_industry:
        for spec in by_sub_industry_tag(sub_industry)[:max_per_axis]:
            if spec.series_id in seen:
                continue
            groups["sub_industry_relevant"].append(spec.to_dict())
            seen.add(spec.series_id)

    # Geography-aware suggestions
    try:
        from .company_geography import get_geography
        geo = get_geography(ticker, allow_llm_fallback=False)  # seed/cache only — no LLM here
    except Exception:
        geo = None
    if geo:
        metros = list((geo.get("metros") or {}).keys())
        for metro in metros[:6]:
            region_key = f"metro:{metro}"
            for spec in SERIES_REGISTRY:
                if spec.region == region_key and spec.series_id not in seen:
                    groups["geography_relevant"].append(spec.to_dict())
                    seen.add(spec.series_id)

    if extra_categories:
        for cat in extra_categories:
            for spec in SERIES_REGISTRY:
                if spec.category == cat.lower() and spec.series_id not in seen:
                    groups["category_relevant"].append(spec.to_dict())
                    seen.add(spec.series_id)
                    if len(groups["category_relevant"]) >= max_per_axis:
                        break

    return groups


def discover_by_query(**kwargs: Any) -> List[Dict[str, Any]]:
    """Thin passthrough to `data_catalog.search`."""
    return [s.to_dict() for s in catalog_search(**kwargs)]


def fetch_series(
    series_id: str, *, force_refresh: bool = False,
) -> Optional[SeriesSnapshot]:
    """Read-through cached snapshot for a single series.

    Resolves the provider chain by the spec's `source` field, fetches
    once, computes derived stats, and returns the SeriesSnapshot. Never
    raises — failures return a snapshot with `.error` set.
    """
    spec = by_id(series_id)
    if spec is None:
        return None

    def _fetcher() -> Optional[Dict[str, Any]]:
        # Try the named provider first; on miss fall through every other
        # provider on the macro chain (a test fixture, FRED, EIA, BLS,
        # Census all expose `get_macro_series`). First non-empty wins.
        candidates: List[Any] = []
        named = _resolve_provider(spec.source)
        if named is not None:
            candidates.append(named)
        candidates.extend(_macro_chain_providers())
        seen_ids: set[int] = set()
        for provider in candidates:
            if id(provider) in seen_ids:
                continue
            seen_ids.add(id(provider))
            try:
                result = provider.get_macro_series(series_id)
            except Exception as exc:  # pragma: no cover
                log.warning(
                    "Provider %s raised for %s: %s",
                    getattr(provider, "name", "?"), series_id, exc,
                )
                continue
            if result:
                return result
        return None

    payload = provider_cache.cached_call(
        "macro", f"catalog:{series_id}", _fetcher, force_refresh=force_refresh,
    )

    snap = SeriesSnapshot(
        series_id=spec.series_id,
        name=spec.name,
        source=spec.source,
        category=spec.category,
        units=spec.units,
        frequency=spec.frequency,
        region=spec.region,
        description=spec.description,
    )
    if not payload:
        snap.error = f"No data returned by {spec.source}."
        return snap

    points = payload.get("points") or []
    if not points:
        snap.error = "Series returned empty."
        return snap

    snap.sample_points = points[-24:]
    snap.latest = points[-1]
    snap.prior = points[-2] if len(points) >= 2 else None
    snap.change_1m = _delta(points, lag=1)
    snap.change_3m = _delta(points, lag=3)
    snap.change_12m = _delta(points, lag=12)
    snap.yoy_pct = _yoy_pct(points)
    snap.z_score_5y = _z_score(points)
    return snap


def fetch_snapshots(
    series_ids: Iterable[str], *, force_refresh: bool = False,
) -> List[SeriesSnapshot]:
    return [
        snap for sid in series_ids
        if (snap := fetch_series(sid, force_refresh=force_refresh)) is not None
    ]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _resolve_provider(source: str) -> Optional[Any]:
    """Return the singleton provider matching `source`, if any.

    Walks data_service's live chain so registered providers (including
    a test fixture installed via `register_test_provider`) are visible.
    """
    try:
        from .data_service import get_data_service
        ds = get_data_service()
    except Exception:  # pragma: no cover
        return None
    source_u = source.upper()
    for cap in ("macro", "energy", "construction", "retail", "inflation", "labor"):
        chain = ds._live_chain(cap) if hasattr(ds, "_live_chain") else []
        for provider in chain:
            if str(getattr(provider, "name", "")).upper() == source_u:
                return provider
    # Direct attribute lookups for the static-attached providers.
    if source_u == "FRED":
        return getattr(ds, "fred", None)
    if source_u == "EIA":
        return getattr(ds, "eia", None)
    if source_u == "BLS":
        return getattr(ds, "bls", None)
    if source_u == "CENSUS":
        return getattr(ds, "census", None)
    return None


def _macro_chain_providers() -> List[Any]:
    """Every provider on data_service's macro/energy/inflation/labor chains.

    Used as the fallback fan-out when the named provider for a series
    misses (e.g. test fixtures, or a series the named provider doesn't
    cover yet).
    """
    try:
        from .data_service import get_data_service
        ds = get_data_service()
    except Exception:  # pragma: no cover
        return []
    seen: set[int] = set()
    out: List[Any] = []
    for cap in ("macro", "energy", "inflation", "labor", "retail", "construction"):
        chain = ds._live_chain(cap) if hasattr(ds, "_live_chain") else []
        for p in chain:
            if id(p) in seen:
                continue
            seen.add(id(p))
            out.append(p)
    return out


def _delta(points: List[Dict[str, Any]], *, lag: int) -> Optional[float]:
    if len(points) <= lag:
        return None
    a = points[-1].get("value")
    b = points[-1 - lag].get("value")
    if a is None or b is None:
        return None
    try:
        return float(a) - float(b)
    except (TypeError, ValueError):
        return None


def _yoy_pct(points: List[Dict[str, Any]]) -> Optional[float]:
    if len(points) < 13:
        return None
    a = points[-1].get("value")
    b = points[-13].get("value")
    if a is None or b is None:
        return None
    try:
        b_f = float(b)
        if b_f == 0:
            return None
        return (float(a) - b_f) / abs(b_f)
    except (TypeError, ValueError):
        return None


def _z_score(points: List[Dict[str, Any]]) -> Optional[float]:
    """Z-score of the latest reading vs the trailing 5 years (60 monthly /
    260 daily). Best-effort; returns None when sample is too small."""
    values = [p.get("value") for p in points if p.get("value") is not None]
    if len(values) < 24:
        return None
    sample = values[-260:] if len(values) > 260 else values
    try:
        mean = statistics.fmean(sample)
        stdev = statistics.pstdev(sample)
    except statistics.StatisticsError:
        return None
    latest = sample[-1]
    if stdev == 0 or math.isnan(stdev):
        return None
    return (latest - mean) / stdev
