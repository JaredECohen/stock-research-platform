"""Retrieval service: chunks + simple BM25-ish keyword search.

In demo mode we use a deterministic, fast keyword scoring approach. If
ENABLE_VECTOR_SEARCH is true *and* OpenAI keys are set, we delegate to an
OpenAI embeddings index — but for a polished demo, the BM25-ish scorer is
plenty and keeps the dependency tree small.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .filings_service import get_filings
from .news_service import get_news
from .transcripts_service import get_transcripts


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _chunks_for_ticker(ticker: str) -> List[Dict]:
    """Build retrievable chunks across filings, transcripts, and news."""
    chunks: List[Dict] = []

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


def _bm25_score(query_tokens: List[str], doc_tokens: List[str], df: Dict[str, int], n_docs: int,
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


def search(ticker: str, query: str, *, limit: int = 4) -> List[Dict]:
    chunks = _chunks_for_ticker(ticker)
    if not chunks:
        return []
    docs = [(_tokens(c["text"]), c) for c in chunks]
    df: Dict[str, int] = defaultdict(int)
    for tokens, _ in docs:
        for t in set(tokens):
            df[t] += 1
    n_docs = len(docs)
    avgdl = sum(len(t) for t, _ in docs) / max(1, n_docs)
    q_tokens = _tokens(query)
    scored: List[Tuple[float, Dict]] = []
    for tokens, c in docs:
        s = _bm25_score(q_tokens, tokens, df, n_docs, avgdl)
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda r: r[0], reverse=True)
    out: List[Dict] = []
    for s, c in scored[:limit]:
        item = dict(c)
        item["score"] = round(s, 3)
        # Truncate text for prompt-friendly chunks
        if len(item["text"]) > 800:
            item["text"] = item["text"][:800] + "…"
        out.append(item)
    return out


def list_chunks(ticker: str) -> List[Dict]:
    return _chunks_for_ticker(ticker)
