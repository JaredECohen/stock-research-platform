"""Retrieval service: chunks + simple BM25-ish keyword search.

Keyword-scores chunks built from filings, transcripts and news. It never
delegates to an embeddings index. In the memo pipeline the filing analyst
calls `search` only as a fallback, when `vector_store.search` returns no
passages, fails, or is skipped for lack of a ticker; the earnings analyst
does not call it (`agents.tools.
retrieve` also wraps it). No setting chooses between the two paths —
`settings.enable_vector_search` is reported on `/api/providers/status`
but read by no retrieval code.

`search_many` is the bull/bear debate's BM25 fallback (design §5.3). It
builds ONE index per debate instead of one per query, and it indexes long
documents as overlapping windows with stable ids (accession + section +
offset hash). `search` keeps whole documents on purpose: the filing
analyst's prompt and its `chunk:<accession>` refs must not change while
the debate is off (bull/bear critique #9 names only the fallback).
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .filings_service import get_filings
from .news_service import get_news
from .transcripts_service import get_transcripts

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _chunks_for_ticker(ticker: str) -> list[dict]:
    """Build retrievable chunks across filings, transcripts, and news."""
    chunks: list[dict] = []

    for f in get_filings(ticker):
        if "business_description" in f and f.get("business_description"):
            chunks.append(dict(
                ticker=ticker, source_type="filing", source_id=f.get("accession_number", "10K"),
                section="business_description",
                title=f"{f.get('type', '10-K')} business description",
                url=f.get("url", ""),
                text=f.get("business_description", ""),
            ))
        for risk in f.get("risk_factors", []) or []:
            chunks.append(dict(
                ticker=ticker, source_type="filing", source_id=f.get("accession_number", ""),
                section="risk_factors",
                title=f"{f.get('type', '10-K')} risk factor",
                url=f.get("url", ""),
                text=risk,
            ))
        if f.get("mda"):
            chunks.append(dict(
                ticker=ticker, source_type="filing", source_id=f.get("accession_number", ""),
                section="mda",
                title=f"{f.get('type', '10-K')} MD&A",
                url=f.get("url", ""),
                text=f["mda"],
            ))

    for t in get_transcripts(ticker):
        if t.get("prepared_remarks"):
            chunks.append(dict(
                ticker=ticker, source_type="transcript", source_id=t.get("period", ""),
                section="prepared_remarks",
                title=f"Earnings call {t.get('period', '')}",
                url="",
                text=t["prepared_remarks"],
            ))
        if t.get("qa"):
            chunks.append(dict(
                ticker=ticker, source_type="transcript", source_id=t.get("period", ""),
                section="qa",
                title=f"Earnings call Q&A {t.get('period', '')}",
                url="",
                text=t["qa"],
            ))

    for n in get_news(ticker):
        chunks.append(dict(
            ticker=ticker, source_type="news", source_id=n.get("url", ""),
            section="article",
            title=n.get("title", ""),
            url=n.get("url", ""),
            text=(n.get("title", "") + ". " + (n.get("summary") or "")),
        ))

    return chunks


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], df: dict[str, int], n_docs: int,
                avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    score = 0.0
    tf = Counter(doc_tokens)
    dl = len(doc_tokens) or 1
    for q in query_tokens:
        if q not in tf:
            continue
        idf = math.log(1 + (n_docs - df.get(q, 0) + 0.5) / (df.get(q, 0) + 0.5))
        f = tf[q]
        score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
    return score


def search(ticker: str, query: str, *, limit: int = 4) -> list[dict]:
    chunks = _chunks_for_ticker(ticker)
    if not chunks:
        return []
    docs = [(_tokens(c["text"]), c) for c in chunks]
    df: dict[str, int] = defaultdict(int)
    for tokens, _ in docs:
        for t in set(tokens):
            df[t] += 1
    n_docs = len(docs)
    avgdl = sum(len(t) for t, _ in docs) / max(1, n_docs)
    q_tokens = _tokens(query)
    scored: list[tuple[float, dict]] = []
    for tokens, c in docs:
        s = _bm25_score(q_tokens, tokens, df, n_docs, avgdl)
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda r: r[0], reverse=True)
    out: list[dict] = []
    for s, c in scored[:limit]:
        item = dict(c)
        item["score"] = round(s, 3)
        # Truncate text for prompt-friendly chunks
        if len(item["text"]) > 800:
            item["text"] = item["text"][:800] + "…"
        out.append(item)
    return out


def list_chunks(ticker: str) -> list[dict]:
    return _chunks_for_ticker(ticker)


# ---------------------------------------------------------------------------
# search_many: one index per debate, long documents in windows
# ---------------------------------------------------------------------------

# A window is about one passage of an MD&A or a call; the debate pool then
# clips each passage to 650 chars around it. Before this, an MD&A was ONE
# chunk truncated to its first 800 chars, so the "passage" a query found was
# the start of the document rather than the span that matched.
WINDOW_CHARS = 1200
WINDOW_OVERLAP = 200


def _window_bounds(text: str) -> list[tuple[int, int]]:
    """[(start, end)] covering `text` in overlapping windows, each ending on
    whitespace where one is near, so a window rarely splits a word."""
    n = len(text)
    if n <= WINDOW_CHARS:
        return [(0, n)]
    out: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(n, start + WINDOW_CHARS)
        if end < n:
            cut = text.rfind(" ", start + WINDOW_CHARS - WINDOW_OVERLAP, end)
            if cut > start:
                end = cut
        out.append((start, end))
        if end >= n:
            break
        start = max(end - WINDOW_OVERLAP, start + 1)
    return out


def chunk_id(source_type: str, source_id: Any, section: str, offset: int) -> str:
    """Stable id for one window: readable accession/section, plus a hash of
    the full identity so two windows (or two risk factors, which share one
    accession and one section name) never collide."""
    sid = str(source_id or source_type or "doc")
    digest = hashlib.sha1(f"{source_type}|{sid}|{section}|{offset}".encode()).hexdigest()[:10]
    return f"{sid}:{section}:{digest}"


def split_long_documents(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each document as windows carrying `id`, `offset` and `section_key`.

    `section_key` numbers repeated sections within one document (every risk
    factor of a 10-K shares `source_id` and `section`), so ids stay unique
    and stable across processes for the same filings. A chunk that already
    has an `id` (a news-pack item) keeps it."""
    seen: Counter[tuple[str, str, str]] = Counter()
    out: list[dict[str, Any]] = []
    for c in chunks:
        text = str(c.get("text") or "")
        if not text.strip():
            continue
        st = str(c.get("source_type") or "")
        sid = str(c.get("source_id") or "")
        section = str(c.get("section") or "")
        n = seen[(st, sid, section)]
        seen[(st, sid, section)] += 1
        section_key = f"{section}.{n}" if n or section == "risk_factors" else section
        if c.get("id"):
            out.append({**c, "offset": 0, "section_key": section_key})
            continue
        for start, end in _window_bounds(text):
            piece = dict(c)
            piece["text"] = text[start:end]
            piece["offset"] = start
            piece["section_key"] = section_key
            piece["id"] = chunk_id(st, sid, section_key, start)
            out.append(piece)
    return out


@dataclass
class _Index:
    docs: list[tuple[list[str], dict[str, Any]]]
    df: dict[str, int]
    avgdl: float


def _build_index(chunks: list[dict[str, Any]]) -> _Index:
    docs = [(_tokens(c["text"]), c) for c in chunks]
    df: dict[str, int] = defaultdict(int)
    for tokens, _ in docs:
        for t in set(tokens):
            df[t] += 1
    avgdl = sum(len(t) for t, _ in docs) / max(1, len(docs))
    return _Index(docs=docs, df=df, avgdl=avgdl)


def search_many(
    ticker: str,
    queries: Sequence[str],
    *,
    limit: int = 3,
    source_types: Sequence[str | None] | None = None,
    extra_chunks: Sequence[dict[str, Any]] = (),
    include_ticker_news: bool = True,
) -> list[list[dict[str, Any]]]:
    """BM25 over the ticker's filings, transcripts and news (plus
    `extra_chunks`, e.g. the debate's news pack), building the index ONCE for
    every query. Returns one result list per query, in query order.

    `source_types[i]` restricts query i to one corpus ("filing",
    "transcript", "news"); None searches all. Ties break on the stable
    chunk id, so the same corpus always gives the same order.

    `include_ticker_news=False` leaves the ticker's own news rows out of
    the index, so `extra_chunks` is the only news searched. The debate
    passes its date-windowed news pack that way: a get_news chunk carries
    no publication date, so it would slip past the pack's window and as-of
    rule and duplicate a pack story under another id. Raises
    ValueError when `source_types` does not line up with `queries`, and on a
    blank ticker (an unscoped search is never what a caller means).
    """
    if not (ticker or "").strip():
        raise ValueError("search_many needs a ticker")
    if source_types is not None and len(source_types) != len(queries):
        raise ValueError("source_types must have one entry per query")
    if not queries:
        return []
    own = _chunks_for_ticker(ticker)
    if not include_ticker_news:
        own = [c for c in own if c.get("source_type") != "news"]
    chunks = split_long_documents([*own, *extra_chunks])
    if not chunks:
        return [[] for _ in queries]
    index = _build_index(chunks)
    n_docs = len(index.docs)
    out: list[list[dict[str, Any]]] = []
    for i, query in enumerate(queries):
        wanted = source_types[i] if source_types is not None else None
        q_tokens = _tokens(query)
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for tokens, c in index.docs:
            if wanted and c.get("source_type") != wanted:
                continue
            s = _bm25_score(q_tokens, tokens, index.df, n_docs, index.avgdl)
            if s > 0:
                scored.append((s, str(c.get("id") or ""), c))
        scored.sort(key=lambda r: (-r[0], r[1]))
        results = []
        for s, _, c in scored[:limit]:
            item = dict(c)
            item["score"] = round(s, 3)
            results.append(item)
        out.append(results)
    return out
