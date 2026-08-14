"""Regression tests for the 2026-08-12 Render OOM-kill.

Root cause: `vector_store.search` applied its ticker predicate only
`if ticker:`, so a falsy ticker silently became a full-table scan of
`doc_chunks`. Every matched row rehydrated its JSON embedding into a list
of 1536 boxed Python floats (~48 KB each), so a call whose contract is
"return 4 passages" allocated multiple GB across the corpus. The SIGKILL
bypassed the caller's `try/except`, leaving no traceback.

These tests pin the three properties that failure violated:
  1. a missing ticker never scans (and never even embeds the query);
  2. an explicit opt-in still works, and stays bounded by the cap;
  3. scoped search returns the same ranking it always did — the memory
     fix must not change retrieval semantics.

Runs network-free: with no `OPENAI_API_KEY` the embeddings service falls
back to deterministic hash vectors, which is also how CI runs.
"""
from __future__ import annotations

import pytest

from app.database import SessionLocal
from app.models import DocChunk
from app.services import embeddings as emb_svc
from app.services import vector_store

# Namespaced so these rows can't collide with fixture/demo data or with a
# concurrently-running test session against the shared sqlite file.
T1 = "ZZVSA"
T2 = "ZZVSB"


def _chunk(text: str, section: str = "risk_factors") -> dict:
    return {"text": text, "section": section, "meta": {"probe": True}}


@pytest.fixture()
def seeded():
    """Insert a small, known corpus for two tickers; clean up after."""
    _purge()
    vector_store.upsert_source(
        ticker=T1, source_type="filing", source_id=9001,
        chunks=[
            _chunk("alpha supply chain concentration risk"),
            _chunk("beta currency translation exposure"),
            _chunk("gamma segment margin compression", section="mda"),
        ],
    )
    vector_store.upsert_source(
        ticker=T2, source_type="filing", source_id=9002,
        chunks=[_chunk("delta regulatory inquiry disclosure")],
    )
    vector_store.upsert_source(
        ticker=T1, source_type="transcript", source_id=9003,
        chunks=[_chunk("epsilon guidance raised on demand", section="qa")],
    )
    yield
    _purge()


def _purge() -> None:
    with SessionLocal() as db:
        db.query(DocChunk).filter(DocChunk.ticker.in_([T1, T2])).delete(
            synchronize_session=False
        )
        db.commit()


# ---------------------------------------------------------------------------
# 1. The guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_ticker", [None, "", "   "])
def test_missing_ticker_refuses_to_scan(seeded, bad_ticker, caplog):
    """A falsy ticker returns nothing instead of scanning the corpus."""
    with caplog.at_level("WARNING"):
        hits = vector_store.search("supply chain risk", ticker=bad_ticker)
    assert hits == []
    assert "refusing corpus-wide scan" in caplog.text


def test_missing_ticker_does_no_work_at_all(seeded, monkeypatch):
    """The guard fires before the query is embedded.

    Bailing out early matters beyond memory: the old path paid for an
    embedding round-trip and a full table read before discovering it had
    no filter to apply.
    """
    calls = []
    monkeypatch.setattr(
        vector_store.emb_svc, "embed_one",
        lambda text: calls.append(text) or [0.0] * emb_svc.FALLBACK_DIM,
    )
    assert vector_store.search("anything", ticker=None) == []
    assert calls == [], "query was embedded despite the global-scan guard"


def test_blank_query_still_short_circuits(seeded):
    assert vector_store.search("   ", ticker=T1) == []


# ---------------------------------------------------------------------------
# 2. The opt-in, and the cap
# ---------------------------------------------------------------------------

def test_allow_global_scan_opt_in_still_searches(seeded):
    """The escape hatch preserves the old capability for a deliberate caller."""
    hits = vector_store.search(
        "regulatory inquiry", source_types=["filing"],
        top_k=10, allow_global_scan=True,
    )
    assert {h["ticker"] for h in hits} == {T1, T2}


def test_candidate_cap_bounds_the_scan(seeded, monkeypatch, caplog):
    """With the cap below the corpus size, only the newest chunks are considered."""
    with SessionLocal() as db:
        ids = sorted(
            r[0] for r in db.query(DocChunk.id).filter(
                DocChunk.ticker.in_([T1, T2])
            ).all()
        )
    newest_two = set(ids[-2:])

    monkeypatch.setattr(vector_store, "MAX_SCAN_CANDIDATES", 2)
    monkeypatch.setattr(vector_store, "SCORE_BATCH_SIZE", 1)
    with caplog.at_level("WARNING"):
        hits = vector_store.search(
            "anything at all", top_k=10, allow_global_scan=True,
        )
    assert hits, "cap should truncate, not empty, the result"
    assert {h["id"] for h in hits} <= newest_two
    assert "candidate cap" in caplog.text


# ---------------------------------------------------------------------------
# 3. Retrieval semantics must be unchanged
# ---------------------------------------------------------------------------

def test_scoped_search_returns_only_that_ticker(seeded):
    hits = vector_store.search("currency exposure", ticker=T1, top_k=10)
    assert hits
    assert {h["ticker"] for h in hits} == {T1}


def test_exact_text_match_ranks_first(seeded):
    """An identical string is its own nearest neighbour — the cheapest
    end-to-end assertion that ranking still works."""
    query = "beta currency translation exposure"
    hits = vector_store.search(query, ticker=T1, source_types=["filing"], top_k=3)
    assert hits[0]["text"] == query
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-6)


def test_scores_match_the_reference_cosine(seeded):
    """The numpy batch product must agree with the scalar implementation.

    This is the parity check on the optimisation: same numbers, less
    memory. A float32 matmul is compared against float64 scalar math, so
    the tolerance is loose by design.
    """
    query = "gamma segment margin compression"
    q_vec = emb_svc.embed_one(query)
    hits = vector_store.search(query, ticker=T1, top_k=10)
    assert hits
    with SessionLocal() as db:
        by_id = {
            r.id: r.embedding
            for r in db.query(DocChunk).filter(DocChunk.ticker == T1).all()
        }
    for hit in hits:
        expected = emb_svc.cosine(q_vec, by_id[hit["id"]])
        assert hit["score"] == pytest.approx(expected, abs=1e-5)


def test_results_are_sorted_descending(seeded):
    hits = vector_store.search("margin", ticker=T1, top_k=10)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_respected(seeded):
    assert len(vector_store.search("risk", ticker=T1, top_k=2)) == 2


def test_source_type_filter(seeded):
    hits = vector_store.search("guidance", ticker=T1, source_types=["transcript"])
    assert hits
    assert {h["source_type"] for h in hits} == {"transcript"}


def test_section_filter(seeded):
    hits = vector_store.search("margin", ticker=T1, sections=["mda"])
    assert hits
    assert {h["section"] for h in hits} == {"mda"}


def test_hit_payload_shape_is_unchanged(seeded):
    """Callers index `text`, `section`, `meta` — the two-query hydration
    must still populate every key the old single-query path returned."""
    hit = vector_store.search("supply chain", ticker=T1, top_k=1)[0]
    assert set(hit) == {
        "id", "ticker", "source_type", "source_id", "section",
        "period_end", "text", "score", "meta",
    }
    assert hit["meta"] == {"probe": True}
    assert hit["text"]


def test_embedding_dim_mismatch_rows_are_skipped(seeded):
    """A stale row from a previous embedding model must not be scored."""
    with SessionLocal() as db:
        db.add(DocChunk(
            ticker=T1, source_type="filing", source_id=9099,
            section="risk_factors", text="wrong-dimension row",
            embedding=[0.5, 0.5, 0.5], embedding_dim=3,
            embedding_model="stale-model", meta={},
        ))
        db.commit()
    hits = vector_store.search("risk", ticker=T1, top_k=50)
    assert "wrong-dimension row" not in [h["text"] for h in hits]


def test_search_survives_a_corpus_larger_than_one_batch(seeded, monkeypatch):
    """Force multiple partitions so the streaming loop's batch boundaries
    are exercised, and confirm the ranking is identical to a single-batch run."""
    single = vector_store.search("margin compression", ticker=T1, top_k=3)
    monkeypatch.setattr(vector_store, "SCORE_BATCH_SIZE", 1)
    batched = vector_store.search("margin compression", ticker=T1, top_k=3)
    assert [h["id"] for h in batched] == [h["id"] for h in single]


# ---------------------------------------------------------------------------
# Top-k heap
# ---------------------------------------------------------------------------

def test_heap_push_keeps_the_highest_scores():
    heap: list = []
    for i, score in enumerate([0.1, 0.9, 0.5, 0.95, 0.2, 0.7]):
        vector_store._heap_push(heap, score, i, 3)
    assert sorted((s for s, _ in heap), reverse=True) == [0.95, 0.9, 0.7]


# ---------------------------------------------------------------------------
# pgvector path is a pure optimisation
# ---------------------------------------------------------------------------

def test_pgvector_disabled_on_sqlite(seeded):
    """The probe must resolve False off Postgres so search uses the
    streaming path rather than emitting invalid SQL."""
    vector_store.reset_pgvector_probe()
    assert vector_store.pgvector_available() is False
    assert vector_store.sync_pgvector_column() == 0


def test_search_falls_back_when_pgvector_returns_none(seeded, monkeypatch):
    """A pgvector failure degrades to streaming instead of losing results."""
    monkeypatch.setattr(vector_store, "pgvector_available", lambda: True)
    monkeypatch.setattr(
        vector_store, "_search_pgvector",
        lambda *a, **k: None,  # simulates the exception path
    )
    hits = vector_store.search("currency exposure", ticker=T1, top_k=3)
    assert hits, "fallback path returned nothing"
    assert {h["ticker"] for h in hits} == {T1}


# ---------------------------------------------------------------------------
# pgvector backfill (runs unattended on worker boot — must be safe everywhere)
# ---------------------------------------------------------------------------

def test_backfill_and_index_noops_off_postgres(seeded):
    """SQLite (local, CI) must skip cleanly rather than emit invalid SQL."""
    vector_store.reset_pgvector_probe()
    out = vector_store.backfill_and_index()
    assert out == {"skipped": True, "populated": 0, "indexed": False}


def test_ensure_hnsw_index_noops_off_postgres(seeded):
    vector_store.reset_pgvector_probe()
    assert vector_store.ensure_hnsw_index() is False


def test_backfill_loop_terminates_and_is_bounded(monkeypatch):
    """The loop runs unattended on every worker boot. A backfill that never
    returns 0 must still stop, rather than spin the boot thread forever."""
    calls = []
    monkeypatch.setattr(vector_store, "pgvector_available", lambda: True)
    monkeypatch.setattr(
        vector_store, "sync_pgvector_column",
        lambda **kw: (calls.append(1), 100)[1],  # never drains
    )
    monkeypatch.setattr(vector_store, "ensure_hnsw_index", lambda **kw: True)
    out = vector_store.backfill_and_index()
    assert len(calls) == 50, "loop is not bounded"
    assert out["populated"] == 5000


def test_backfill_stops_as_soon_as_it_drains(monkeypatch):
    calls = []
    monkeypatch.setattr(vector_store, "pgvector_available", lambda: True)
    monkeypatch.setattr(
        vector_store, "sync_pgvector_column",
        lambda **kw: (calls.append(1), 7 if len(calls) == 1 else 0)[1],
    )
    monkeypatch.setattr(vector_store, "ensure_hnsw_index", lambda **kw: True)
    out = vector_store.backfill_and_index()
    assert len(calls) == 2  # one productive pass, one that returns 0
    assert out == {"skipped": False, "populated": 7, "indexed": True}
