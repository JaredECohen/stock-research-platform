"""W7 critique (accepted): a bounded automatic re-index retry inside the
existing nightly `history_backfill` loop.

With strict embeddings an OpenAI outage during a post-pass raises instead of
writing hash vectors, which leaves the new filing with zero chunks — and
`run_ingest_post_passes` never retries. So each night, after the ingest and
fundamentals passes, the loop re-indexes the newest zero-chunk in-scope
sources from the last 14 days: at most 10 sources, $0.10 and 20 MB, never a
post-pass. The note gains `reindexed=` / `reindex_deferred=` and names every
source; an outage is named without failing the loop; no new loop exists.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.models import DocChunk
from app.monitoring import KNOWN_LOOPS, history_backfill
from app.services import corpus_repair, filing_memory
from app.services import fundamental_refresh as fr
from app.tests.corpus_fixtures import corpus_db, live_openai  # noqa: F401


@pytest.fixture
def loop(monkeypatch):
    runs: list[dict] = []
    monkeypatch.setattr(history_backfill, "_tier1_tickers", lambda: [])
    monkeypatch.setattr(fr, "nightly", lambda: fr._empty_result())
    monkeypatch.setattr(history_backfill, "record_run",
                        lambda *a, **k: runs.append({"success": k.get("success"), "note": k.get("note", "")}))
    monkeypatch.setattr(filing_memory, "post_pass", lambda *a, **k: pytest.fail("the retry ran a post-pass"))
    return runs


def _seed_recent(db):  # noqa: F811
    """12 recent zero-chunk filings, one older than the window, one indexed."""
    # The window is relative to the loop's clock, so the fixture is too.
    today = datetime.utcnow().date()
    db.company("AAA")
    recent = [db.filing(ticker="AAA", accession=f"AAA-R{n}", filed=today - timedelta(days=n), words=50)
              for n in range(1, 13)]
    old = db.filing(ticker="AAA", accession="AAA-OLD", filed=today - timedelta(days=30), words=50)
    indexed = db.filing(ticker="AAA", accession="AAA-IDX", filed=today - timedelta(days=1), words=50)
    db.chunk(source_id=indexed)
    return recent, old


def test_history_backfill_retries_recent_zero_chunk_sources_bounded(corpus_db, live_openai, loop):  # noqa: F811
    recent, old = _seed_recent(corpus_db)
    totals = history_backfill.run_once(day=0)
    run = loop[-1]
    assert run["success"] is True
    assert "reindexed=10 reindex_deferred=2" in run["note"]
    # Newest first; the two oldest in the window are deferred, by identity.
    assert "reindexed sources: " + ", ".join(f"AAA:filing:{i}" for i in recent[:10]) in run["note"]
    assert f"reindex deferred: AAA:filing:{recent[10]}, AAA:filing:{recent[11]}" in run["note"]
    assert (totals["reindexed"], totals["reindex_deferred"]) == (10, 2)
    with corpus_db.Session() as db:
        indexed = {r.source_id for r in db.query(DocChunk).filter_by(source_type="filing")}
    assert set(recent[:10]) <= indexed and old not in indexed and not set(recent[10:]) & indexed
    assert f"AAA:filing:{old}" not in run["note"]  # outside the 14-day window: not a candidate
    # The next night drains the remainder.
    history_backfill.run_once(day=1)
    assert "reindexed=2 reindex_deferred=0" in loop[-1]["note"]


def test_retry_runs_with_the_nightly_caps(corpus_db, live_openai, loop, monkeypatch):  # noqa: F811
    seen = []
    monkeypatch.setattr(corpus_repair, "index_missing", lambda **kw: seen.append(kw) or {
        "sources_indexed": 0, "deferred": [], "indexed": []})
    history_backfill.run_once(day=0)
    assert len(seen) == 1
    caps = {k: seen[0][k] for k in ("recent_days", "max_sources", "max_usd", "max_added_mb")}
    assert caps == {"recent_days": 14, "max_sources": 10, "max_usd": 0.10, "max_added_mb": 20}


def test_openai_outage_is_named_and_does_not_fail_the_loop(corpus_db, live_openai, loop):  # noqa: F811
    recent, _ = _seed_recent(corpus_db)
    live_openai.fail = TimeoutError("sk-live-SECRET timed out")
    history_backfill.run_once(day=0)
    run = loop[-1]
    assert run["success"] is True
    assert "reindexed=0 reindex_deferred=12" in run["note"]
    assert "reindex stopped: embedding_unavailable" in run["note"]
    assert "SECRET" not in run["note"]


def test_crashed_retry_fails_the_loop(corpus_db, live_openai, loop, monkeypatch):  # noqa: F811
    monkeypatch.setattr(corpus_repair, "index_missing",
                        lambda **kw: (_ for _ in ()).throw(ValueError("bug")))
    history_backfill.run_once(day=0)
    assert loop[-1]["success"] is False and "reindex crashed: ValueError" in loop[-1]["note"]


def test_demo_or_keyless_process_does_not_retry(corpus_db, loop, monkeypatch):  # noqa: F811
    # CI's own configuration: no key, demo data. Its vectors would be hash.
    monkeypatch.setattr(corpus_repair, "index_missing", lambda **kw: pytest.fail("retried without OpenAI"))
    history_backfill.run_once(day=0)
    assert "reindex" not in loop[-1]["note"] and loop[-1]["success"] is True


def test_single_ticker_admin_run_does_not_retry(corpus_db, live_openai, loop, monkeypatch):  # noqa: F811
    monkeypatch.setattr(corpus_repair, "index_missing", lambda **kw: pytest.fail("single-ticker run retried"))
    monkeypatch.setattr(history_backfill, "backfill_ticker",
                        lambda t, **k: {"financial_periods": 0, "filings": 0, "transcripts": 0})
    totals = history_backfill.run_once("AAA")
    assert "reindexed" not in totals and "reindex" not in loop[-1]["note"]


def test_no_new_loop():
    assert "history_backfill" in KNOWN_LOOPS
    assert not [name for name in KNOWN_LOOPS if "reindex" in name or "corpus" in name]
