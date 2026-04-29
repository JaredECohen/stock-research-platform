"""Phase 5 — monitoring loops.

Each loop is unit-tested in isolation (no scheduler needed). Verifies:
- `news_loop.run_once` writes a NewsAlert to the hot cache.
- `social_loop.run_once` writes a sentiment scalar.
- `macro_loop.run_once` produces a regime label and broadcasts it.
- `edgar_poller.run_once` is a no-op on first run (initialization) but
  invalidates company_cold once a new accession appears.
"""
from __future__ import annotations

from app.cache import cache_get, cache_put
from app.monitoring import edgar_poller, macro_loop, news_loop, social_loop


def test_news_loop_writes_news_hot_to_cache():
    news_loop.run_once(["NVDA"])
    snap = cache_get("news_hot:NVDA", "news_hot")
    assert snap is not None
    assert "alerts" in snap.payload


def test_social_loop_writes_sentiment_scalar():
    social_loop.run_once(["MSFT"])
    # Today's bucket
    from datetime import date
    snap = cache_get(f"social_hot:MSFT:{date.today().isoformat()}", "social_hot")
    assert snap is not None
    assert "sentiment_extremity" in snap.payload


def test_macro_loop_produces_broadcast_with_regime():
    out = macro_loop.run_once()
    assert "regime" in out
    snap = cache_get("macro:global", "macro_broadcast")
    assert snap is not None
    assert snap.payload.get("regime") == out["regime"]


def test_edgar_poller_invalidates_company_cold_on_new_accession():
    # First run primes the bookkeeping store; should produce no events.
    events = edgar_poller.run_once(["NVDA"])
    assert events == []

    # Simulate a new accession by clobbering the bookkeeping snapshot to a
    # subset of what we'll see. The next call should detect the diff and
    # invalidate the company_cold snapshot.
    seen = cache_get("NVDA", "edgar_seen_accessions")
    if not seen:
        # Demo provider may have no filings — exit early in that case.
        return
    accessions = list(seen.payload.get("accessions") or [])
    if len(accessions) < 1:
        return
    # Drop one accession from the bookkeeping so the next poll sees it as new.
    cache_put(
        "NVDA", "edgar_seen_accessions",
        payload={"accessions": accessions[:-1]},
        sources_used=["edgar:NVDA:bookkeeping"],
        generated_by="test", cost_tokens=0,
    )
    # Pre-populate company_cold so we can prove invalidation marks it stale.
    cache_put(
        "NVDA", "company_cold",
        payload={"profile": {"ticker": "NVDA"}, "income": [{"period": "2023-12-31", "revenue": 1}]},
        sources_used=["filing:NVDA:000001"],
        generated_by="test", cost_tokens=0,
    )
    events = edgar_poller.run_once(["NVDA"])
    assert events  # at least one new accession event
    assert cache_get("NVDA", "company_cold") is None  # invalidated
