"""Embeddings in the LLM call log (slice B8-E1; attribution design §4.7,
gaps G10 and G22).

Embedding requests used to leave no `llm_call_logs` row at all, so an
OpenAI outage showed up only as analysts quietly falling back to BM25, and
the corpus spend was invisible to every cost report. Each request now
writes one attributed row, a failure writes its row before raising, and
neither touches `last_usage()`, which callers read straight after a CHAT
call to bill it. The same slice makes the daily GC honour `SDKTrace`'s
documented 90-day retention.

`live_openai` serves `embeddings.embed` from a fake client: no network, no
key, no spend.
"""
from __future__ import annotations

import ast
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import sessionmaker

from app import database
from app.agents import llm, llm_attribution
from app.database import Base, SessionLocal
from app.models import CronLoopRun, LLMCallLog, SDKTrace
from app.monitoring import llm_log_gc
from app.services import corpus_repair, filing_memory, vector_store
from app.services import embeddings as emb_svc
from app.services.embeddings import EmbeddingUnavailable
from app.tests.corpus_fixtures import NOW, corpus_db, live_openai  # noqa: F401
from app.tests.test_corpus_inventory import seed

_APP = Path(llm.__file__).resolve().parents[1]


def _rows(run_id: str) -> list[LLMCallLog]:
    with SessionLocal() as db:
        rows = db.query(LLMCallLog).filter(LLMCallLog.run_id == run_id).order_by(LLMCallLog.id).all()
        for r in rows:
            db.expunge(r)
    return rows


@pytest.fixture
def run_id():
    rid = f"e1-{uuid.uuid4().hex[:12]}"
    with llm.llm_call_context(run_id=rid):
        yield rid


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def test_embed_rows_share_call_id_and_leave_last_usage(live_openai, run_id):  # noqa: F811
    live_openai.tokens_per_text = 20_000  # large enough to price above the 6-dp rounding
    # The provider names the model that served the request; the row keeps it
    # beside the one that was asked for.
    real_create = live_openai.create
    live_openai.create = lambda **kw: SimpleNamespace(
        **vars(real_create(**kw)), model="text-embedding-3-small-v2")
    sentinel = {"provider": "anthropic", "model": "chat-call-being-billed", "total_tokens": 999}
    llm._USAGE_STATE.last = dict(sentinel)

    emb_svc.embed_one("gross margin guidance", action="embed.query", ticker="NVDA")
    emb_svc.embed(["a", "b", "c"], action="embed.index", ticker="NVDA")

    rows = _rows(run_id)
    # `embed_one` -> `embed` is ONE call: the nested entry reuses the outer
    # scope, so it is one row with one call_id, not two.
    assert len(rows) == 2
    query, index = rows
    assert (query.action, query.agent_name, query.role) == ("embed.query", "Embedder", "embed")
    assert (index.action, index.tokens_in, index.tokens_out) == ("embed.index", 60_000, 0)
    for r in rows:
        assert (r.provider, r.model, r.ticker, r.success, r.attempt) == (
            "openai", emb_svc.EMBEDDING_MODEL, "NVDA", True, 1)
        assert r.requested_model == emb_svc.EMBEDDING_MODEL and r.error_type is None
        assert r.served_model == "text-embedding-3-small-v2"
        assert r.call_id and len(r.call_id) == 32
        assert r.cost_usd == pytest.approx(r.tokens_in * 0.02 / 1_000_000)
    assert query.call_id != index.call_id
    # The chat call's usage is still there for its caller to bill.
    assert llm.last_usage() == sentinel


def test_embed_error_row_before_raise(live_openai, run_id, monkeypatch):  # noqa: F811
    live_openai.fail = RuntimeError("POST /v1/embeddings key=sk-live-SECRET body='private text'")
    # Error rows leave the chat call's usage alone too: a failed retrieval
    # between chat calls must not be billed as the chat call.
    sentinel = {"provider": "anthropic", "model": "chat-call-being-billed", "total_tokens": 999}
    llm._USAGE_STATE.last = dict(sentinel)
    order: list[str] = []
    real_record = llm._record_usage
    monkeypatch.setattr(llm, "_record_usage",
                        lambda *a, **kw: order.append("row") or real_record(*a, **kw))

    with pytest.raises(EmbeddingUnavailable):
        try:
            emb_svc.embed(["private text"], action="embed.index", ticker="AAA")
        finally:
            order.append("raised")
    assert order == ["row", "raised"]
    assert llm.last_usage() == sentinel

    # A wrong-shape response is billed and fails the same way.
    live_openai.fail = None
    live_openai.dim = 3
    llm._USAGE_STATE.last = dict(sentinel)  # `last_usage()` consumes it
    with pytest.raises(EmbeddingUnavailable, match="unexpected shape"):
        emb_svc.embed(["x"], action="embed.repair")
    assert llm.last_usage() == sentinel

    failed, shape = _rows(run_id)
    assert (failed.success, failed.error_type, failed.action) == (False, "provider_error:RuntimeError", "embed.index")
    assert "SECRET" not in (failed.error or "") and "private" not in (failed.error or "")
    assert (shape.success, shape.error_type, shape.tokens_in) == (False, "unexpected_shape", 10)


def test_embed_unavailable_client_writes_a_skip_row(run_id, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "app_env", "production")
    with pytest.raises(EmbeddingUnavailable):
        emb_svc.embed(["x"], action="embed.index")
    (row,) = _rows(run_id)
    assert (row.error_type, row.tokens_in, row.success) == ("skipped:client_unavailable", 0, False)


def test_hash_vectors_write_no_row(run_id, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "app_env", "development")
    assert emb_svc.embed(["x"], action="embed.index")
    assert _rows(run_id) == []


def test_guard_runs_before_the_hash_path(monkeypatch):
    """Demo and CI embed with hash vectors, so the guard must run before that
    return or an unattributed caller is never found where tests run."""
    from app.config import settings

    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "llm_attribution_mode", "strict")
    monkeypatch.setattr(llm_attribution, "caller_site", lambda: ("app/services/_probe.py:probe", False))
    llm_attribution.reset_violations()
    try:
        with pytest.raises(llm_attribution.LLMAttributionError):
            emb_svc.embed(["x"])
        with pytest.raises(llm_attribution.LLMAttributionError):
            emb_svc.embed_one("x")
        assert llm_attribution.VIOLATIONS == {"app/services/_probe.py:probe": 2}
    finally:
        llm_attribution.reset_violations()


# ---------------------------------------------------------------------------
# Call sites
# ---------------------------------------------------------------------------

def test_every_embed_call_site_names_a_registered_action():
    """The same static assertion M1 makes for the chat entries, for
    `EMBED_ENTRY_POINTS`: every production call passes a literal, registered
    `action=`."""
    missing: list[str] = []
    found = 0
    for path in sorted(_APP.rglob("*.py")):
        rel = path.relative_to(_APP.parent).as_posix()
        if rel.startswith("app/tests/") or rel == "app/services/embeddings.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=rel)):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name not in llm_attribution.EMBED_ENTRY_POINTS:
                continue
            found += 1
            action = next((k.value for k in node.keywords if k.arg == "action"), None)
            spec = llm_attribution.spec_for(action.value) if isinstance(action, ast.Constant) else None
            if spec is None or spec.kind != "embed":
                missing.append(f"{rel}:{node.lineno}")
    assert found >= 3, "the sweep found no embed call sites; is it looking in the right place?"
    assert missing == []


def test_call_sites_pass_their_actions(corpus_db, monkeypatch):  # noqa: F811
    seen: list[tuple[str | None, str | None]] = []
    monkeypatch.setattr(emb_svc, "embed_one", lambda text, **kw: seen.append(
        (kw.get("action"), kw.get("ticker"))) or [0.0] * emb_svc.FALLBACK_DIM)
    monkeypatch.setattr(emb_svc, "embed", lambda texts, **kw: seen.append(
        (kw.get("action"), kw.get("ticker"))) or [[0.0] * emb_svc.FALLBACK_DIM for _ in texts])

    vector_store.search("margins", ticker="AAA")
    vector_store.upsert_source(ticker="AAA", source_type="filing", source_id=1,
                               chunks=[{"text": "Revenue rose."}])
    assert seen == [("embed.query", "AAA"), ("embed.index", "AAA")]


def test_query_embed_failure_logs_type_only(monkeypatch, caplog):
    def boom(text, **kw):
        raise RuntimeError("key=sk-live-SECRET query='private question'")

    monkeypatch.setattr(emb_svc, "embed_one", boom)
    with caplog.at_level("WARNING", logger=vector_store.log.name):
        assert vector_store.search("private question", ticker="AAA") == []
    assert "embed query failed: RuntimeError" in caplog.text
    assert "SECRET" not in caplog.text and "private question" not in caplog.text


def _index_missing_rows(corpus_db, run_id):  # noqa: F811
    seed(corpus_db)
    res = corpus_repair.index_missing(max_sources=10, max_usd=1.0, max_added_mb=100, now=NOW)
    assert res["sources_indexed"] >= 1
    rows = _rows(run_id)
    assert rows and {r.action for r in rows} == {"embed.index"}
    return rows


def test_reindex_rows_carry_the_repair_origin(corpus_db, live_openai, run_id):  # noqa: F811
    assert {r.origin for r in _index_missing_rows(corpus_db, run_id)} == {"repair:corpus"}


def test_reindex_rows_carry_the_repair_origin_inside_a_loop(corpus_db, live_openai, run_id):  # noqa: F811
    # The nightly retry runs inside `history_backfill`, whose own ingest
    # writes `embed.index` rows under the loop origin. The repair's rows
    # must still be told apart from those.
    with llm.llm_call_context(origin="loop:history_backfill"):
        rows = _index_missing_rows(corpus_db, run_id)
        assert llm.current_call_context()["origin"] == "loop:history_backfill"
    assert {r.origin for r in rows} == {"repair:corpus"}


def test_reembed_rows_are_embed_repair(corpus_db, live_openai, run_id):  # noqa: F811
    seed(corpus_db)
    res = corpus_repair.reembed(max_usd=5.0, max_rows=100_000, max_added_mb=500.0, now=NOW)
    assert res["rows_repaired"] > 0
    rows = _rows(run_id)
    assert rows and {(r.action, r.origin) for r in rows} == {("embed.repair", "repair:corpus")}


# ---------------------------------------------------------------------------
# Rows written while the indexer holds its transaction
# ---------------------------------------------------------------------------

@pytest.fixture
def one_sqlite_file(tmp_path, monkeypatch):
    """`vector_store` and the call-log writer on ONE SQLite file, as in local
    dev (the corpus fixture gives vector_store a file of its own, which is
    why the suite never saw this). A short busy timeout keeps a regression
    quick to fail rather than 5 s per batch."""
    engine = create_engine(f"sqlite:///{tmp_path / 'shared.sqlite'}", future=True,
                           connect_args={"check_same_thread": False, "timeout": 0.5})
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
    monkeypatch.setattr(database, "SessionLocal", maker)
    for module in (vector_store, filing_memory):
        monkeypatch.setattr(module, "SessionLocal", maker)
    yield maker
    engine.dispose()


def _shared_rows(maker, run_id):
    with maker() as db:
        return db.query(LLMCallLog).filter(LLMCallLog.run_id == run_id).order_by(LLMCallLog.id).all()


def test_index_rows_survive_the_open_transaction(one_sqlite_file, live_openai, run_id, caplog):  # noqa: F811
    chunks = [{"text": f"Revenue rose in segment {i}."} for i in range(6)]
    started = time.monotonic()
    with caplog.at_level(logging.DEBUG, logger="app.agents.llm"):
        written = vector_store.upsert_source(ticker="AAA", source_type="filing", source_id=4242,
                                             chunks=iter(chunks), batch_size=2)
    elapsed = time.monotonic() - started
    assert written == 6 and len(live_openai.calls) == 3
    rows = _shared_rows(one_sqlite_file, run_id)
    # One row per batch request, all attributed and none lost to the lock.
    assert [(r.action, r.ticker, r.success) for r in rows] == [("embed.index", "AAA", True)] * 3
    assert len({r.call_id for r in rows}) == 3
    assert "persist failed" not in caplog.text and "locked" not in caplog.text
    assert elapsed < 1.0


def test_index_error_row_survives_the_rollback(one_sqlite_file, live_openai, run_id, caplog):  # noqa: F811
    live_openai.fail = RuntimeError("upstream 503")
    with caplog.at_level(logging.DEBUG, logger="app.agents.llm"):
        with pytest.raises(EmbeddingUnavailable):
            vector_store.upsert_source(ticker="AAA", source_type="filing", source_id=4243,
                                       chunks=[{"text": "Margins expanded."}], raise_on_error=True)
    (row,) = _shared_rows(one_sqlite_file, run_id)
    assert (row.action, row.success, row.error_type) == ("embed.index", False, "provider_error:RuntimeError")
    assert "persist failed" not in caplog.text


# ---------------------------------------------------------------------------
# SDK trace retention (G22)
# ---------------------------------------------------------------------------

def test_sdk_traces_gc_90d():
    now = datetime.utcnow()
    old_run, fresh_run = f"e1-old-{uuid.uuid4().hex[:8]}", f"e1-fresh-{uuid.uuid4().hex[:8]}"
    with SessionLocal() as db:
        SDKTrace.__table__.create(bind=db.get_bind(), checkfirst=True)
        db.add(SDKTrace(run_id=old_run, surface="chat", new_items=[{"x": 1}],
                        generated_at=now - timedelta(days=91)))
        db.add(SDKTrace(run_id=fresh_run, surface="chat", new_items=[],
                        generated_at=now - timedelta(days=89)))
        db.commit()

    result = llm_log_gc.run_once(max_age_days=90)

    assert result["deleted_sdk_traces"] >= 1
    with SessionLocal() as db:
        left = {r[0] for r in db.query(SDKTrace.run_id).filter(
            SDKTrace.run_id.in_((old_run, fresh_run))).all()}
        assert left == {fresh_run}
        note = db.query(CronLoopRun).filter(CronLoopRun.loop_name == "llm_log_gc").one().note
        assert "sdk_traces" in note
        db.query(SDKTrace).filter(SDKTrace.run_id == fresh_run).delete(synchronize_session=False)
        db.commit()


def test_sdk_traces_gc_without_the_table(tmp_path, monkeypatch):
    """A database init_db() never touched: the sweep returns 0 without
    creating the table, and the rest of the GC run (and its cron row) still
    happens."""
    engine = create_engine(f"sqlite:///{tmp_path / 'bare.sqlite'}", future=True)
    bare = sessionmaker(bind=engine, future=True)
    with bare() as db:
        assert llm_log_gc.gc_sdk_traces(db=db) == 0
    assert not sa_inspect(engine).has_table(SDKTrace.__tablename__)

    real_gc = llm_log_gc.gc_sdk_traces

    def on_bare(**kw):
        with bare() as db:
            return real_gc(db=db, **kw)

    monkeypatch.setattr(llm_log_gc, "gc_sdk_traces", on_bare)
    result = llm_log_gc.run_once(max_age_days=90)
    assert result["deleted_sdk_traces"] == 0
    with SessionLocal() as db:
        run = db.query(CronLoopRun).filter(CronLoopRun.loop_name == "llm_log_gc").one()
        assert "0 sdk_traces rows" in run.note
    engine.dispose()
