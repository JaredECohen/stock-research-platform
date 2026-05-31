"""News + catalysts service.

Two surfaces:
  - `get_news(ticker)` — back-compat: returns whatever the highest-
    priority news provider (Alpha Vantage, then Polygon) returns for a
    ticker. Cached + clipped by `data_service`.
  - `get_news_combined(ticker, ...)` — pulls from Alpha Vantage / Polygon
    AND GDELT, deduplicates by URL + normalized title, and returns a
    single ranked list. Used by callers that want broader international
    coverage (geopolitical / supply-chain stories that US-centric finance
    feeds miss).
  - `search_news_broad(query, ...)` — free-form GDELT query without a
    specific ticker, for theme-level searches.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .data_service import get_data_service


def get_news(ticker: str) -> List[Dict[str, Any]]:
    return get_data_service().get_news(ticker) or []


def get_news_combined(
    ticker: str,
    *,
    include_gdelt: bool = True,
    max_results: int = 30,
) -> List[Dict[str, Any]]:
    """Return news from every configured source, deduplicated + ranked.

    Each article carries `_sources: [provider_name, ...]` so the caller
    can show source attribution. Most-recent-first ordering; falls back
    to title-alpha when timestamps are missing.
    """
    ds = get_data_service()
    streams: List[List[Dict[str, Any]]] = []

    primary = ds.get_news(ticker) or []
    if primary:
        for art in primary:
            art.setdefault("_sources", []).append("primary")
        streams.append(primary)

    if include_gdelt:
        try:
            gdelt = ds.gdelt.get_news(ticker) or []
            for art in gdelt:
                art.setdefault("_sources", []).append("gdelt")
            streams.append(gdelt)
        except Exception:
            pass

    merged: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for stream in streams:
        for art in stream:
            url = (art.get("url") or "").strip()
            title_key = _title_key(art.get("title") or "")
            if url and url in seen_urls:
                continue
            if title_key and title_key in seen_titles:
                continue
            if url:
                seen_urls.add(url)
            if title_key:
                seen_titles.add(title_key)
            merged.append(art)

    merged.sort(key=_article_sort_key, reverse=True)
    return merged[:max_results]


def search_news_broad(
    query: str,
    *,
    tickers: Optional[List[str]] = None,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Free-form GDELT search, no ticker resolution."""
    try:
        return get_data_service().gdelt.search_news(
            query=query, tickers=tickers, limit=limit,
        )
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _title_key(title: str) -> str:
    """Normalize a headline for dedup: lowercase + strip punctuation + collapse spaces."""
    if not title:
        return ""
    s = re.sub(r"[^\w\s]+", "", title.lower())
    s = re.sub(r"\s+", " ", s).strip()
    # First 12 words is enough to catch near-duplicates that differ in trailing fluff.
    return " ".join(s.split()[:12])


def _article_sort_key(article: Dict[str, Any]) -> tuple:
    """Sort articles newest-first, falling back to title."""
    ts = article.get("published_at") or article.get("publishedAt") or ""
    return (str(ts), str(article.get("title", "")))
