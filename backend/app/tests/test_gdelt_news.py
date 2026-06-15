"""Tests for the GDELT provider + combined news service."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from app.providers.gdelt_provider import GDELTProvider, _parse_gdelt_ts
from app.services import news_service


def _make_response(payload: Any, status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    return resp


def test_parse_gdelt_timestamp():
    ts = _parse_gdelt_ts("20251201T143055Z")
    assert ts == datetime(2025, 12, 1, 14, 30, 55, tzinfo=timezone.utc)
    assert _parse_gdelt_ts(None) is None
    assert _parse_gdelt_ts("garbage") is None


def test_gdelt_search_normalizes_articles():
    now = datetime.now(timezone.utc)
    fixture = {
        "articles": [
            {
                "title": "Apple announces new iPhone",
                "url": "https://example.com/a",
                "domain": "example.com",
                "seendate": now.strftime("%Y%m%dT%H%M%SZ"),
                "snippet": "Apple unveiled the new iPhone today...",
                "language": "English",
                "sourcecountry": "United States",
                "tone": "-2.3",
            },
            {
                "title": "Old story",
                "url": "https://example.com/old",
                "domain": "example.com",
                "seendate": (now - timedelta(days=90)).strftime("%Y%m%dT%H%M%SZ"),
                "snippet": "...",
            },
            {
                "title": "Duplicate by URL",
                "url": "https://example.com/a",  # same as the first
                "domain": "example.com",
                "seendate": now.strftime("%Y%m%dT%H%M%SZ"),
            },
        ],
    }

    provider = GDELTProvider()
    with patch("app.providers.gdelt_provider.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.return_value = _make_response(fixture)
        results = provider.search_news(query="AAPL", tickers=["AAPL"], max_age_days=30)

    # 3 articles in -> 1 after dedup + age filter (one is too old, one is dup)
    assert len(results) == 1
    art = results[0]
    assert art["title"] == "Apple announces new iPhone"
    assert art["url"] == "https://example.com/a"
    assert art["source"] == "example.com"
    assert art["tickers"] == ["AAPL"]
    assert art["tone"] == -2.3
    assert art["published_at"].startswith(now.strftime("%Y-%m-%d"))


def test_gdelt_returns_empty_on_non_200():
    provider = GDELTProvider()
    with patch("app.providers.gdelt_provider.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.return_value = _make_response({}, status=503)
        results = provider.search_news(query="X")
    assert results == []


def test_gdelt_returns_empty_on_unparseable_json():
    provider = GDELTProvider()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("not json")
    with patch("app.providers.gdelt_provider.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.return_value = resp
        results = provider.search_news(query="X")
    assert results == []


def test_news_service_combined_dedups_across_sources():
    """get_news_combined should merge primary + GDELT and dedup by URL/title."""
    primary = [
        {"title": "Apple beats earnings", "url": "https://p.com/1", "published_at": "2025-12-01T10:00:00Z"},
        {"title": "Shared headline", "url": "https://p.com/2", "published_at": "2025-12-01T09:00:00Z"},
    ]
    gdelt = [
        {"title": "Apple beats earnings",  # dup title
         "url": "https://g.com/different", "published_at": "2025-12-01T10:05:00Z"},
        {"title": "GDELT-only international story",
         "url": "https://g.com/intl", "published_at": "2025-12-01T11:00:00Z"},
    ]
    mock_ds = MagicMock()
    mock_ds.get_news.return_value = primary
    mock_ds.gdelt.get_news.return_value = gdelt
    with patch("app.services.news_service.get_data_service", return_value=mock_ds):
        merged = news_service.get_news_combined("AAPL")
    titles = [m["title"] for m in merged]
    # 3 distinct articles (one dedup)
    assert "Apple beats earnings" in titles
    assert "Shared headline" in titles
    assert "GDELT-only international story" in titles
    assert len(merged) == 3
    # Should be sorted newest-first
    assert merged[0]["published_at"] >= merged[-1]["published_at"]


def test_gdelt_failure_does_not_break_combined():
    primary = [{"title": "x", "url": "https://p.com/x", "published_at": "2025-12-01T10:00:00Z"}]
    mock_ds = MagicMock()
    mock_ds.get_news.return_value = primary
    mock_ds.gdelt.get_news.side_effect = RuntimeError("boom")
    with patch("app.services.news_service.get_data_service", return_value=mock_ds):
        merged = news_service.get_news_combined("AAPL")
    assert len(merged) == 1
    assert merged[0]["title"] == "x"
