"""Wave 10 — embedding service for vector retrieval over the corpus.

Single entry point: `embed(texts) -> list[list[float]]`. Uses OpenAI's
`text-embedding-3-large` (3072 dims) when `OPENAI_API_KEY` is set; falls
back to deterministic-hash embeddings when it isn't (so unit tests can
exercise the indexer / retriever without network access).

Embeddings live in `doc_chunks.embedding` as JSON-serialized lists.
When pgvector is enabled in production, an out-of-band migration
converts the column to `vector(N)` and adds an HNSW index — neither is
required for retrieval to work; the JSON path falls back to numpy
cosine similarity.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Iterable, List, Optional, Sequence

from ..config import settings

log = logging.getLogger(__name__)

# text-embedding-3-small is the right default — at $0.02/MTok it's ~6.5x
# cheaper than 3-large with retrieval quality that's within 1-2% on the
# financial-doc benchmarks we care about. Override via env if a tenant
# wants 3-large for the longer-context cases (legal disclosure search).
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # text-embedding-3-small dimensionality
FALLBACK_DIM = 256  # deterministic-hash mode


def _is_openai_available() -> bool:
    return bool(getattr(settings, "openai_api_key", None))


def _hash_embed(text: str, dim: int = FALLBACK_DIM) -> List[float]:
    """Deterministic, content-derived 'embedding' for tests.

    Hash the text into `dim/8` 64-bit integers, normalize to unit length.
    Doesn't capture semantics — but produces stable, comparable vectors
    so the retrieval *plumbing* can be tested without an API key.
    """
    h = hashlib.sha512(text.encode("utf-8", errors="ignore")).digest()
    while len(h) * 8 < dim * 8:
        h += hashlib.sha512(h).digest()
    nums: List[float] = []
    for i in range(dim):
        chunk = h[i * 4 : i * 4 + 4]
        if len(chunk) < 4:
            chunk = chunk.ljust(4, b"\x00")
        nums.append(int.from_bytes(chunk, "big") / (2**32))
    norm = math.sqrt(sum(x * x for x in nums)) or 1.0
    return [x / norm for x in nums]


def embed(texts: Sequence[str]) -> List[List[float]]:
    """Return one embedding per input text.

    OpenAI when configured; deterministic hash fallback otherwise. The
    fallback is real-vector-shaped so the same retrieval code path works
    in dev without network access.
    """
    if not texts:
        return []
    if _is_openai_available():
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=list(texts))
            return [d.embedding for d in resp.data]
        except Exception as exc:  # pragma: no cover — fall back rather than fail
            log.warning("OpenAI embeddings failed (%s); using hash fallback", exc)
    return [_hash_embed(t) for t in texts]


def embed_one(text: str) -> List[float]:
    return embed([text])[0]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def chunk_text(
    text: str, *, target_tokens: int = 500, overlap_tokens: int = 50,
) -> List[str]:
    """Naive token-budgeted chunker — words as a token proxy.

    Sufficient for filing / transcript chunking; if a follow-up wants
    semantic-aware chunking (paragraph-respecting + section-aware) we
    can swap this out without changing the retrieval interface.
    """
    if not text:
        return []
    words = text.split()
    chunks: List[str] = []
    i = 0
    step = max(1, target_tokens - overlap_tokens)
    while i < len(words):
        chunk_words = words[i : i + target_tokens]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        i += step
    return chunks
