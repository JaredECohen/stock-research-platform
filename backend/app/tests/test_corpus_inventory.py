"""W7 §8.2 — the corpus census is exact, and never reads a vector.

Nothing had ever counted which `doc_chunks` rows carry a usable vector, and
a repair sized from a guess is sized wrong. The census must count every
class exactly, split zero-chunk sources by why they have none, price the
repair in tokens, dollars and bytes, and do all of it without selecting a
single embedding value (a 1536-dim JSON vector is ~31 KB; the worker has
512 MiB).
"""
from __future__ import annotations

import math
from datetime import date, datetime

import pytest

from app.models import FilingDoc
from app.services import corpus_inventory as ci
from app.services import embeddings as emb_svc
from app.services import filing_memory
from app.tests.corpus_fixtures import EIGHT_K, NOW, corpus_db, embedding_value_refs, ingest_sec_filings  # noqa: F401

MAY, JUNE = datetime(2026, 5, 10), datetime(2026, 6, 10)


def seed(db):
    """A corpus with every chunk class and every source category."""
    db.company("AAA")
    db.company("BBB", tier="screener_only")
    db.company("CCC", tier="screener_only")
    db.memo("CCC")
    f1 = db.filing(ticker="AAA", accession="AAA-1")
    f2 = db.filing(ticker="AAA", accession="AAA-DEMO-1")
    t1 = db.transcript(ticker="AAA", period="2026Q1")
    ids = {
        "ok": [db.chunk(source_id=f1, token_count=7) for _ in range(2)],
        "hash": [db.chunk(source_id=f1, dim=256, token_count=5) for _ in range(3)],
        "legacy": [db.chunk(source_type="transcript", source_id=t1, dim=3072, token_count=11,
                            model="text-embedding-3-large", created_at=JUNE)],
        "other": [db.chunk(source_id=f1, dim=768, token_count=13, model="x-768")],
        "no_emb": [db.chunk(source_id=f1, dim=None, token_count=17, embedding=None),
                   db.chunk(source_type="transcript", source_id=t1, token_count=19, embedding=None,
                            created_at=JUNE)],
        "orphan": [db.chunk(source_id=9999, dim=256, token_count=23)],
        "demo_ticker": [db.chunk(ticker="ZZZ", source_id=f1, dim=256, token_count=29)],
        "demo_accession": [db.chunk(source_id=f2, dim=256, token_count=31)],
    }
    # Sources with zero chunks, one per reason.
    sources = {
        "f_indexable": db.filing(ticker="AAA", accession="AAA-3", filed=date(2026, 8, 20), words=400),
        "f_memo_scope": db.filing(ticker="CCC", accession="CCC-1", filed=date(2026, 7, 1), words=900),
        "f_old": db.filing(ticker="AAA", accession="AAA-4", filed=date(2022, 1, 1)),
        "f_fetch_error": db.filing(ticker="AAA", accession="AAA-5", fetch_error="HTTP 503"),
        "f_empty": db.filing(ticker="AAA", accession="AAA-6", words=0, sections={}),
        "f_tier": db.filing(ticker="BBB", accession="BBB-1"),
        "f_not_company": db.filing(ticker="ZZZ", accession="ZZZ-1"),
        "f_demo": db.filing(ticker="AAA", accession="0000-DEMO-7"),
        "t_indexable": db.transcript(ticker="AAA", period="2026Q2", called=date(2026, 8, 1), words=50),
        "t_old": db.transcript(ticker="AAA", period="2023Q1", called=date(2023, 1, 1)),
    }
    return ids, sources


def test_every_chunk_class_counted_exactly(corpus_db):  # noqa: F811
    seed(corpus_db)
    inv = ci.build_inventory(now=NOW)
    classes = inv["chunks"]["classes"]
    assert {c: classes[c]["rows"] for c in ci.CHUNK_CLASSES} == {
        "no_embedding": 2, "hash_fallback": 6, "legacy_dim": 1, "other_dim": 1,
        "awaiting_sync": 0, "ok": 2,
    }
    assert {c: classes[c]["tokens"] for c in ci.CHUNK_CLASSES} == {
        "no_embedding": 17 + 19, "hash_fallback": 3 * 5 + 23 + 29 + 31, "legacy_dim": 11,
        "other_dim": 13, "awaiting_sync": 0, "ok": 14,
    }
    assert classes["no_embedding"]["by_source_type"] == {"filing": 1, "transcript": 1}
    assert inv["chunks"]["total"] == 12 and inv["chunks"]["unusable"] == 10
    assert inv["unusable_by_month"] == {"2026-05": 8, "2026-06": 2}
    assert inv["unusable_by_ticker_top20"] == [{"ticker": "AAA", "rows": 9}, {"ticker": "ZZZ", "rows": 1}]
    assert inv["orphans"] == {"filing": 1, "transcript": 0, "total": 1}
    assert inv["demo_source_rows"] == {"ticker_not_in_companies": 1, "demo_accession": 1, "total": 2}
    assert inv["pgvector"] is False and inv["storage"] == {"database_bytes": None, "doc_chunks_bytes": None}


def test_awaiting_sync_is_pgvector_only_and_not_unusable():
    dim = emb_svc.EMBEDDING_DIM
    assert ci.classify_chunk(dim, False, True, pgvector=True) == "awaiting_sync"
    assert ci.classify_chunk(dim, False, True, pgvector=False) == "ok"
    assert ci.classify_chunk(dim, True, True, pgvector=True) == "no_embedding"
    assert "awaiting_sync" not in ci.UNUSABLE_CLASSES


def test_sources_without_chunks_split(corpus_db):  # noqa: F811
    _, sources = seed(corpus_db)
    inv = ci.build_inventory(now=NOW)
    split = inv["sources_without_chunks"]
    assert split["filing"] == {"indexable": 2, "fetch_error": 1, "empty_body": 1, "out_of_scope": 2, "demo": 2}
    assert split["transcript"] == {"indexable": 1, "fetch_error": 0, "empty_body": 0, "out_of_scope": 1, "demo": 0}
    with corpus_db.Session() as db:
        by_id = {(s.kind, s.id): s.category for s in ci.missing_sources(db, now=NOW)}
    assert by_id[("filing", sources["f_memo_scope"])] == "indexable"  # a memo puts a non-tier ticker in scope
    assert by_id[("filing", sources["f_tier"])] == "out_of_scope"
    assert by_id[("filing", sources["f_not_company"])] == "demo"
    assert by_id[("filing", sources["f_fetch_error"])] == "fetch_error"
    # The recent window used by the nightly retry excludes older sources.
    with corpus_db.Session() as db:
        recent = {s.id for s in ci.missing_sources(db, now=NOW, since=date(2026, 8, 10))
                  if s.category == "indexable"}
    assert recent == {sources["f_indexable"]}


def test_a_fetched_8k_is_empty_body_not_indexable(corpus_db):  # noqa: F811
    """An 8-K's items map to no section key, so the real ingest path stores
    `sections={"_source_metadata": ...}` and the whole body in `raw_text`:
    a positive `word_count` and nothing `index_filing` would ever chunk.
    Classed `indexable`, each one took a nightly retry slot forever."""
    corpus_db.company("AAA")
    (eight_k,) = ingest_sec_filings(corpus_db, "AAA", [("AAA-8K-1", date(2026, 9, 20), "8-K", EIGHT_K)])
    blank = corpus_db.filing(ticker="AAA", accession="AAA-BLANK", words=300,
                             sections={"mda": "   ", "risk_factors": [], "segments": None})
    listed = corpus_db.filing(ticker="AAA", accession="AAA-LIST", words=300,
                              sections={"risk_factors": ["Supply concentration."]})
    with corpus_db.Session() as db:
        row = db.get(FilingDoc, eight_k)
        assert set(row.sections) == {"_source_metadata"} and row.word_count > 0  # the real shape
        db.expunge(row)
        by_id = {s.id: s.category for s in ci.missing_sources(db, now=NOW)}
    assert list(filing_memory._iter_filing_chunks(row)) == []  # nothing to embed, ever
    assert by_id == {eight_k: "empty_body", blank: "empty_body", listed: "indexable"}
    plan = ci.build_inventory(now=NOW)["repair_plan"]["index_missing"]
    assert plan["sources"] == 1  # the 8-K no longer inflates tokens, USD or bytes


def test_cost_and_bytes_arithmetic(corpus_db):  # noqa: F811
    seed(corpus_db)
    plan = ci.build_inventory(now=NOW)["repair_plan"]
    # Re-embed: unusable, non-orphan, non-demo rows only (3 hash, legacy,
    # other, 2 no-embedding); the orphan and both demo rows are excluded.
    tokens = 3 * 5 + 11 + 13 + 17 + 19
    current = (3 * 256 + 3072 + 768) * ci.SQLITE_JSON_BYTES_PER_DIM  # null embeddings store nothing
    assert plan["reembed"] == {
        "rows": 7, "tokens": tokens, "usd": round(tokens / 1e6 * emb_svc.EMBEDDING_USD_PER_MTOK, 8),
        # Gross, never net of the superseded values: an in-place UPDATE does
        # not shrink the database until VACUUM (and then only for reuse).
        "added_bytes": 7 * (ci.SQLITE_OK_EMBEDDING_BYTES + ci.VECTOR_COLUMN_BYTES),
        "superseded_bytes": current,
    }
    words = (400, 900, 50)
    est = [math.ceil(w * 1.3) for w in words]
    chunks = sum(max(1, math.ceil(t / 500)) for t in est)
    assert plan["index_missing"] == {
        "sources": 3, "est_tokens": sum(est), "usd": round(sum(est) / 1e6 * 0.02, 8),
        "added_bytes": chunks * (ci.SQLITE_OK_EMBEDDING_BYTES + ci.VECTOR_COLUMN_BYTES + ci.CHUNK_TEXT_BYTES),
    }
    assert plan["total_usd"] == round(plan["reembed"]["usd"] + plan["index_missing"]["usd"], 8)


def test_census_never_selects_embedding_values(corpus_db):  # noqa: F811
    seed(corpus_db)
    corpus_db.statements.clear()
    ci.build_inventory(now=NOW)
    assert corpus_db.statements, "the SQL hook saw nothing"
    offenders = [s for s in corpus_db.statements if embedding_value_refs(s)]
    assert offenders == []
    # And `filing_docs.sections` bodies are only ever projected by JSON path.
    assert not [s for s in corpus_db.statements if "filing_docs.sections" in s and "JSON_EXTRACT" not in s]


@pytest.mark.parametrize("sql", [
    "SELECT doc_chunks.embedding FROM doc_chunks",
    "SELECT id, embedding, text FROM doc_chunks WHERE embedding IS NOT NULL",
    "SELECT length(doc_chunks.embedding) FROM doc_chunks",
])
def test_the_hook_would_catch_a_value_read(sql):
    assert embedding_value_refs(sql)


def test_census_writes_nothing(corpus_db):  # noqa: F811
    seed(corpus_db)
    corpus_db.statements.clear()
    ci.build_inventory(now=NOW)
    writes = [s for s in corpus_db.statements if s.lstrip().split()[0].upper() in ("INSERT", "UPDATE", "DELETE")]
    assert writes == []
