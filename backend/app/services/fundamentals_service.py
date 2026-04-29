"""Fundamentals service: shape statements + ratios for downstream consumers.

Phase 2 made `get_full_financials` cache-aware. Each ticker fetch produces a
"company_cold" snapshot whose `sources_used` includes filing accession numbers
and earnings-transcript period IDs. When a new 10-K/10-Q/8-K lands, the
EDGAR poller calls `cache.invalidate(ticker, kind="company_cold")` and any
warm snapshots that depend on it (sector_warm, company_warm:dcf, …) auto-stale.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _gather_sources(ticker: str) -> List[str]:
    """Identifiers a 'cold' snapshot is keyed against — used for cache fingerprint.

    Filing accession numbers and transcript period IDs are the primary tripwires
    for invalidation: when EDGAR returns a new accession we'll see the hash
    change, and the `EdgarPoller` (Phase 5) explicitly invalidates.
    """
    # Lazy imports avoid a cold-start import cycle (filings/transcripts service
    # internally calls back into the data service which calls fundamentals).
    from .filings_service import get_filings
    from .transcripts_service import latest_transcript

    sources: List[str] = []
    try:
        for f in get_filings(ticker) or []:
            acc = f.get("accession_number") or f.get("type") or ""
            if acc:
                sources.append(f"filing:{ticker}:{acc}")
    except Exception:
        pass
    try:
        tr = latest_transcript(ticker)
        if tr:
            sources.append(f"transcript:{ticker}:{tr.get('period', '')}")
    except Exception:
        pass
    return sources


def _build_full_financials(ticker: str) -> Dict[str, Any]:
    """Raw provider call — kept private so the public `get_full_financials`
    can decide whether to consult the cache."""
    from .data_service import get_data_service  # avoid import-time cycle
    ds = get_data_service()
    statements = ds.get_financial_statements(ticker) or {}
    ratios = ds.get_ratios(ticker) or {}
    profile = ds.get_company_profile(ticker) or {}
    earnings = ds.get_earnings(ticker) or {}
    return dict(
        ticker=ticker,
        profile=profile,
        income=statements.get("income", []),
        balance=statements.get("balance", []),
        cash=statements.get("cash", []),
        ratios=ratios,
        earnings=earnings,
    )


def get_full_financials(ticker: str, *, force_refresh: bool = False) -> Dict:
    """Cache-aware fundamentals fetcher.

    Returns the same dict shape as before. Quarterly TTL (90d) bounds the worst
    case if invalidation signals are missed; in practice the EDGAR poller is
    the primary source of staleness.
    """
    from ..cache import cache_get, cache_put  # lazy: cache → models → us

    if not force_refresh:
        cached = cache_get(ticker, "company_cold", max_age_seconds=90 * 24 * 3600)
        if cached and isinstance(cached.payload, dict):
            payload = dict(cached.payload)
            payload.pop("schema_version", None)
            # Demo/in-memory data is JSON-safe so this is a clean round-trip.
            return payload

    full = _build_full_financials(ticker)
    sources_used = _gather_sources(ticker)

    cache_put(
        ticker, "company_cold",
        payload=full, sources_used=sources_used,
        generated_by="fundamentals_service",
        cost_tokens=120,  # provider calls + ratio derivation ~ rough
        ttl_seconds=90 * 24 * 3600,
    )
    return full


def get_latest_year(records: List[Dict]) -> Optional[Dict]:
    if not records:
        return None
    return sorted(records, key=lambda r: r.get("period", ""))[-1]
