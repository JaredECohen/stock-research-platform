"""Shared fixtures for the W7 corpus tests (strict embeddings, census, repair).

The census counts the *whole* `doc_chunks` table, and the suite's shared
database holds whatever other tests left behind, so every corpus test runs
against its own sqlite file: `corpus_db` points each module that opens
sessions at it. `live_openai` puts `embeddings.embed` on its OpenAI path with
a fake client — no network, no key, deterministic 1536-dim vectors and a
reported `usage.total_tokens` the test controls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import Base
from app.models import Company, DocChunk, EarningsTranscript, FilingDoc, MemoSnapshot
from app.services import corpus_inventory, corpus_repair, filing_memory, vector_store
from app.services import embeddings as emb_svc

NOW = datetime(2026, 9, 24, 12, 0, 0)


@dataclass
class CorpusDB:
    engine: Any
    Session: Any
    statements: list[str] = field(default_factory=list)

    def add(self, *rows):
        with self.Session() as db:
            db.add_all(rows)
            db.commit()
            ids = [getattr(r, "id", None) for r in rows]
        return ids[0] if len(ids) == 1 else ids

    def chunk(self, *, ticker="AAA", source_type="filing", source_id=None, dim=emb_svc.EMBEDDING_DIM,
              text="Revenue rose on services demand.", token_count=7, model=None, meta=None,
              created_at=datetime(2026, 5, 10), embedding="auto"):
        vec: list[float] | None
        if embedding == "auto":
            vec = [0.5] * dim if dim else None
        else:
            vec = embedding
        row = DocChunk(
            ticker=ticker, source_type=source_type, source_id=source_id, section="mda",
            text=text, token_count=token_count, embedding_dim=dim,
            embedding_model=model or (emb_svc.EMBEDDING_MODEL if dim == emb_svc.EMBEDDING_DIM else "hash-fallback"),
            meta=meta or {"k": "v"}, created_at=created_at,
        )
        if vec is not None:
            row.embedding = vec
        return self.add(row)

    def company(self, ticker="AAA", tier="auto_analysis"):
        return self.add(Company(ticker=ticker, company_name=ticker, sector="Tech", industry="Software",
                                universe_tier=tier))

    def filing(self, *, ticker="AAA", accession=None, filed=date(2026, 9, 1), words=400,
               sections=None, fetch_error=None):
        secs = sections if sections is not None else {"mda": "Revenue rose. " * max(1, words // 2)}
        if fetch_error:
            secs = {**secs, "_source_metadata": {"text_fetch_error": fetch_error}}
        return self.add(FilingDoc(
            ticker=ticker, accession_number=accession or f"{ticker}-{filed.isoformat()}-{words}",
            filing_type="10-Q", filing_date=filed, sections=secs, word_count=words,
        ))

    def transcript(self, *, ticker="AAA", period="2026Q2", called=date(2026, 8, 1), words=50):
        return self.add(EarningsTranscript(
            ticker=ticker, period=period, call_date=called, word_count=words,
            blocks=[{"speaker": "CFO", "segment": "prepared", "text": "Margins expanded. " * 20}],
            full_text="Margins expanded. " * 20,
        ))

    def memo(self, ticker):
        return self.add(MemoSnapshot(ticker=ticker, version=1, memo_json={}))

    def rows(self):
        with self.Session() as db:
            return {r.id: r for r in db.query(DocChunk).all()}


@pytest.fixture
def corpus_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'corpus.sqlite'}", future=True,
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
    for module in (corpus_inventory, corpus_repair, vector_store, filing_memory):
        monkeypatch.setattr(module, "SessionLocal", maker)
    db = CorpusDB(engine=engine, Session=maker)

    @event.listens_for(engine, "before_cursor_execute")
    def _capture(conn, cursor, statement, parameters, context, executemany):
        db.statements.append(statement)

    yield db
    engine.dispose()


class FakeEmbeddings:
    """`client.embeddings` with controllable usage, failure and shape."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.tokens_per_text: int | None = None  # None → omit `usage`
        self.fail: Exception | None = None
        self.dim = emb_svc.EMBEDDING_DIM

    def create(self, *, model, input):
        self.calls.append(list(input))
        if self.fail is not None:
            raise self.fail
        data = [SimpleNamespace(embedding=[0.01 * (i + 1)] * self.dim) for i in range(len(input))]
        usage = (SimpleNamespace(total_tokens=self.tokens_per_text * len(input))
                 if self.tokens_per_text is not None else None)
        return SimpleNamespace(data=data, usage=usage)


@pytest.fixture
def live_openai(monkeypatch):
    """A live (non-demo) process with a key, served by `FakeEmbeddings`."""
    import openai

    fake = FakeEmbeddings()
    fake.tokens_per_text = 10
    monkeypatch.setattr(settings, "openai_api_key", "test-not-a-secret")
    monkeypatch.setattr(settings, "use_demo_data", False)
    monkeypatch.setattr(settings, "enable_live_data", True)
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: SimpleNamespace(embeddings=fake))
    assert emb_svc.semantic_available()
    return fake


_ALLOWED_EMBEDDING_REFS = re.compile(
    r"(?:doc_chunks\.)?embedding\s+IS\s+(?:NOT\s+)?NULL"
    r"|pg_column_size\((?:doc_chunks\.)?embedding\)",
    re.IGNORECASE,
)


def embedding_value_refs(statement: str) -> list[str]:
    """Bare references to the `embedding` column once the allowed forms
    (`IS [NOT] NULL`, `pg_column_size(...)`) are removed. `embedding_dim`,
    `embedding_model`, `embedding_vec` and `embedding_repair` are other
    identifiers and never match the word boundary."""
    stripped = _ALLOWED_EMBEDDING_REFS.sub("", statement)
    return re.findall(r"\bembedding\b", stripped, flags=re.IGNORECASE)
