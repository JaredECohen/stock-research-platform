"""W7 §8.1 — no new unusable vectors can form.

`embeddings.embed` used to turn *any* OpenAI failure into 256-dim hash
vectors when a key was configured. Those rows can never sync into the
pgvector column and are skipped by every 1536-dim query, so an outage minted
permanently unreachable chunks, and a failed query embedding "matched" hash
rows by byte pattern. Now: hash vectors only outside production, and only
with no key or in demo-only mode; everything else raises
`EmbeddingUnavailable`, which rolls back a re-index and empties a search.
"""
from __future__ import annotations

import pytest

from app.config import settings
from app.services import embeddings as emb_svc
from app.services import vector_store
from app.services.embeddings import EmbeddingUnavailable
from app.tests.corpus_fixtures import corpus_db, live_openai  # noqa: F401


def test_key_set_client_raises_embedding_unavailable_type_only_message(live_openai, monkeypatch):  # noqa: F811
    live_openai.fail = RuntimeError("POST /v1/embeddings key=sk-live-SECRET body='confidential text'")
    monkeypatch.setattr(emb_svc, "_hash_embed", lambda *a: pytest.fail("fell back to a hash vector"))
    with pytest.raises(EmbeddingUnavailable) as err:
        emb_svc.embed(["confidential text"])
    assert str(err.value) == "openai embeddings failed: RuntimeError"
    # No chained cause either: a logged traceback would carry the secret.
    assert err.value.__cause__ is None and err.value.__suppress_context__


@pytest.mark.parametrize("dim, count", [(3, 1), (emb_svc.EMBEDDING_DIM, 0)])
def test_wrong_dimension_raises(live_openai, monkeypatch, dim, count):  # noqa: F811
    live_openai.dim = dim
    real_create = live_openai.create
    monkeypatch.setattr(live_openai, "create",
                        lambda **kw: _truncate(real_create(**kw), count))
    with pytest.raises(EmbeddingUnavailable, match="unexpected shape"):
        emb_svc.embed(["a"])


def _truncate(resp, count):
    if count == 0:
        resp.data = []
    return resp


def test_no_key_non_production_hash_256(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "app_env", "development")
    vecs = emb_svc.embed(["alpha", "beta"])
    assert [len(v) for v in vecs] == [emb_svc.FALLBACK_DIM] * 2
    assert vecs == emb_svc.embed(["alpha", "beta"])  # deterministic for CI


def test_no_key_production_raises(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "app_env", "production")
    with pytest.raises(EmbeddingUnavailable, match="no embedding provider configured in production"):
        emb_svc.embed(["alpha"])


def test_demo_mode_with_key_uses_hash_outside_production(monkeypatch):
    import openai

    calls = []
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: calls.append(kw) or pytest.fail("demo mode spent money"))
    monkeypatch.setattr(settings, "openai_api_key", "test-not-a-secret")
    monkeypatch.setattr(settings, "use_demo_data", True)
    monkeypatch.setattr(settings, "enable_live_data", False)
    monkeypatch.setattr(settings, "app_env", "development")
    assert [len(v) for v in emb_svc.embed(["demo filing"])] == [emb_svc.FALLBACK_DIM]
    assert not emb_svc.semantic_available() and calls == []

    # Production never writes a hash vector, demo flags or not.
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: calls.append(kw) or (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(EmbeddingUnavailable):
        emb_svc.embed(["demo filing"])
    assert len(calls) == 1


def test_usage_meter_records_provider_tokens_else_estimates(live_openai):  # noqa: F811
    live_openai.tokens_per_text = 123
    with emb_svc.usage_meter() as meter:
        emb_svc.embed(["one", "two"])
    assert (meter.tokens, meter.estimated_tokens, meter.calls) == (246, 0, 1)
    live_openai.tokens_per_text = None  # a response without `usage`
    with emb_svc.usage_meter() as meter:
        emb_svc.embed(["one", "two"])
    assert meter.tokens == 0 and meter.estimated_tokens == emb_svc.count_tokens("one") + emb_svc.count_tokens("two")


def _seed_source(corpus_db):  # noqa: F811
    corpus_db.company("AAA")
    fid = corpus_db.filing(ticker="AAA")
    written = vector_store.upsert_source(
        ticker="AAA", source_type="filing", source_id=fid,
        chunks=[{"text": "Services revenue grew."}, {"text": "Margins expanded."}],
    )
    assert written == 2
    return fid


def test_upsert_failure_preserves_existing_chunks(corpus_db, live_openai):  # noqa: F811
    # Good chunks written while embeddings were healthy.
    fid = _seed_source(corpus_db)
    before = {i: (r.text, r.embedding_dim) for i, r in corpus_db.rows().items()}
    assert {d for _, d in before.values()} == {emb_svc.EMBEDDING_DIM}

    # Then OpenAI fails mid re-index: the delete rolls back with the inserts.
    live_openai.fail = TimeoutError("read timed out")
    with pytest.raises(EmbeddingUnavailable):
        vector_store.upsert_source(ticker="AAA", source_type="filing", source_id=fid,
                                   chunks=[{"text": "Replacement text."}], raise_on_error=True)
    assert vector_store.upsert_source(ticker="AAA", source_type="filing", source_id=fid,
                                      chunks=[{"text": "Replacement text."}]) == 0
    after = {i: (r.text, r.embedding_dim) for i, r in corpus_db.rows().items()}
    assert after == before
    assert all(r.embedding_model != "hash-fallback" for r in corpus_db.rows().values())


def test_search_returns_empty_when_query_embedding_unavailable(corpus_db, monkeypatch):  # noqa: F811
    # Hash rows exist (written in dev); production has no provider. The old
    # code embedded the query as a hash too and returned byte-pattern
    # "matches"; now the query cannot be embedded, so search yields [] and
    # the analysts fall back to BM25.
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "app_env", "development")
    _seed_source(corpus_db)
    assert vector_store.search("Services revenue grew.", ticker="AAA")  # dev: plumbing works
    monkeypatch.setattr(settings, "app_env", "production")
    assert vector_store.search("Services revenue grew.", ticker="AAA") == []
