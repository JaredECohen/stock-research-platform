"""Wave 10 — embedding service for vector retrieval over the corpus.

Single entry point: `embed(texts) -> list[list[float]]`. Uses OpenAI's
`text-embedding-3-small` (1536 dims) when `OPENAI_API_KEY` is set and the
process is live. Deterministic-hash embeddings (256 dims) are produced only
outside production, and only when there is no key or the process is in
demo-only mode — so unit tests and the demo loop can exercise the indexer
and retriever without network access or spend.

Strict by design (W7 §8.1). Any other state — a configured key whose
request fails, a response of the wrong shape, or production with no key —
raises `EmbeddingUnavailable` instead of quietly substituting a hash
vector. A 256-dim hash row can never sync into the `vector(1536)` column
and is skipped by every 1536-dim query, so the old silent fallback turned
an OpenAI blip into a permanently unreachable chunk. Raising lets
`upsert_source` roll its delete back (the source keeps its good chunks),
the post-pass report name the failure, and `vector_store.search` return []
so the analysts fall back to BM25.

Embeddings live in `doc_chunks.embedding` as JSON-serialized lists (the
source of truth). On Postgres with pgvector, `embedding_vec vector(1536)`
is a derived index column; the JSON path falls back to numpy cosine
similarity.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import settings

log = logging.getLogger(__name__)

# text-embedding-3-small is the right default — at $0.02/MTok it's ~6.5x
# cheaper than 3-large with retrieval quality that's within 1-2% on the
# financial-doc benchmarks we care about. Override via env if a tenant
# wants 3-large for the longer-context cases (legal disclosure search).
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # text-embedding-3-small dimensionality
FALLBACK_DIM = 256  # deterministic-hash mode

# OpenAI list price for `EMBEDDING_MODEL`, USD per million input tokens.
# The corpus repair's caps and receipts are denominated in it.
EMBEDDING_USD_PER_MTOK = 0.02


class EmbeddingUnavailable(RuntimeError):
    """No usable semantic embedding.

    Callers fall back (search → [] → BM25) or record a failure; never a hash
    vector in production, and never when a live key is configured. The
    message carries the exception *type* only: provider exceptions can echo
    request text or credentials, and this message reaches loop notes.
    """


def _is_openai_available() -> bool:
    return bool(getattr(settings, "openai_api_key", None))


def _hash_allowed() -> bool:
    """True when deterministic hash vectors are an acceptable answer.

    Never in production, whatever else is configured. Elsewhere, when there
    is no key (CI, a fresh checkout) or the process is demo-only: a
    developer `.env` carries a live key, and demo-mode indexing of demo
    filings would otherwise spend real money embedding fixtures.
    """
    if (getattr(settings, "app_env", "") or "").lower() == "production":
        return False
    return not _is_openai_available() or bool(settings.use_demo_data_only)


def semantic_available() -> bool:
    """True when `embed` returns real `EMBEDDING_DIM` vectors (or raises).

    The gate for anything that *writes repairs*: a repair run in a process
    where hash vectors are allowed would replace unusable rows with more
    unusable rows.
    """
    return _is_openai_available() and not _hash_allowed()


def tokens_to_usd(tokens: int) -> float:
    return tokens / 1_000_000 * EMBEDDING_USD_PER_MTOK


@dataclass
class UsageMeter:
    """Embedding tokens billed inside one `usage_meter()` block.

    `tokens` sums the provider's `usage.total_tokens` per request — the
    number OpenAI invoices. `estimated_tokens` covers a response that omits
    usage, counted with our tokenizer, so a spend cap can never be bypassed
    by a missing field. Hash vectors cost nothing and add nothing.
    """
    tokens: int = 0
    estimated_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.tokens + self.estimated_tokens

    @property
    def usd(self) -> float:
        return tokens_to_usd(self.total_tokens)


_usage_meter: ContextVar[UsageMeter | None] = ContextVar("embedding_usage_meter", default=None)


@contextmanager
def usage_meter() -> Iterator[UsageMeter]:
    """Meter every OpenAI embedding request made inside the block.

    A context variable rather than a return value because the spend happens
    several calls down (`index_filing` → `upsert_source` → `embed`) behind
    interfaces that return chunk counts, and the repair caps must see what
    was billed, not what was estimated beforehand.
    """
    meter = UsageMeter()
    token = _usage_meter.set(meter)
    try:
        yield meter
    finally:
        _usage_meter.reset(token)


# (attempt scope, call context, args, kwargs) of each row held back by
# `deferred_call_rows`.
_PendingRow = tuple[dict[str, Any] | None, dict[str, Any], tuple[Any, ...], dict[str, Any]]
_deferred_rows: ContextVar[list[_PendingRow] | None] = ContextVar(
    "embedding_deferred_call_rows", default=None)


@contextmanager
def deferred_call_rows() -> Iterator[None]:
    """Hold the `llm_call_logs` rows of embeddings made inside the block and
    write them when it exits, whether it exits cleanly or by raising.

    For a caller that embeds while holding an open write transaction.
    `vector_store.upsert_source` keeps one transaction across every batch
    (its idempotency depends on it), and the row writer opens its own
    connection: on one SQLite file that INSERT waited out the 5 s busy
    timeout per batch and was then dropped as "database is locked", so the
    index spend vanished from the call log and every batch stalled. Rows
    keep the attempt scope and call context they were made under, so a
    deferred row is identical to an immediate one. Nested blocks defer to
    the outermost, which writes after the outermost caller has closed its
    session.
    """
    if _deferred_rows.get() is not None:
        yield
        return
    pending: list[_PendingRow] = []
    token = _deferred_rows.set(pending)
    try:
        yield
    finally:
        _deferred_rows.reset(token)
        for att, ctx, args, kwargs in pending:
            _write_row(att, ctx, args, kwargs)


def _write_row(att: dict[str, Any] | None, ctx: dict[str, Any],
               args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    from ..agents import llm

    a_tok = llm._ATTEMPT.set(att)
    c_tok = llm._CALL_CONTEXT.set(ctx)
    try:
        llm._record_usage(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - the writer already swallows DB errors
        # A lost audit row must not replace the caller's own exception.
        log.warning("deferred embedding call row failed: %s", type(exc).__name__)
    finally:
        llm._CALL_CONTEXT.reset(c_tok)
        llm._ATTEMPT.reset(a_tok)


def _record_row(*args: Any, **kwargs: Any) -> None:
    """`llm._record_usage` now, or at the end of `deferred_call_rows`."""
    from ..agents import llm

    pending = _deferred_rows.get()
    if pending is None:
        llm._record_usage(*args, **kwargs)
        return
    pending.append((llm._ATTEMPT.get(), llm._CALL_CONTEXT.get(), args, kwargs))


def _hash_embed(text: str, dim: int = FALLBACK_DIM) -> list[float]:
    """Deterministic, content-derived 'embedding' for tests.

    Hash the text into `dim/8` 64-bit integers, normalize to unit length.
    Doesn't capture semantics — but produces stable, comparable vectors
    so the retrieval *plumbing* can be tested without an API key.
    """
    h = hashlib.sha512(text.encode("utf-8", errors="ignore")).digest()
    while len(h) * 8 < dim * 8:
        h += hashlib.sha512(h).digest()
    nums: list[float] = []
    for i in range(dim):
        chunk = h[i * 4 : i * 4 + 4]
        if len(chunk) < 4:
            chunk = chunk.ljust(4, b"\x00")
        nums.append(int.from_bytes(chunk, "big") / (2**32))
    norm = math.sqrt(sum(x * x for x in nums)) or 1.0
    return [x / norm for x in nums]


def embed(texts: Sequence[str], *, action: str | None = None,
          ticker: str | None = None) -> list[list[float]]:
    """Return one embedding per input text, or raise `EmbeddingUnavailable`.

    OpenAI when configured and live; deterministic hash vectors only where
    `_hash_allowed()`. See the module docstring for why there is no silent
    fallback.

    `action` names the caller in the attribution registry (`embed.index`,
    `embed.query`, `embed.repair`). Every OpenAI request writes one
    `llm_call_logs` row, and every raise after the guard writes its row
    first, so an embedding outage is visible in the call log rather than
    only as an emptied search (design gap G10). Hash vectors make no
    request and write nothing. Inside `deferred_call_rows` the rows are
    written when that block exits instead, still before its caller sees
    the raise.
    """
    if not texts:
        return []
    from ..agents import llm

    # The guard runs first, as in every other public LLM entry (attribution
    # critique #9): a caller that names no action is found in demo and CI,
    # where the hash path below would otherwise hide it. `embed_one` opens
    # the scope as the outer entry and this call reuses it.
    with llm._call_scope("embed", action=action, ticker=ticker, route="", max_tokens=0) as att:
        return _embed_in_scope(texts, att)


def _embed_in_scope(texts: Sequence[str], att: dict[str, Any]) -> list[list[float]]:

    if _hash_allowed():
        return [_hash_embed(t) for t in texts]
    att.update(requested_provider="openai", requested_model=EMBEDDING_MODEL,
               model_resolution="fixed")

    def _row(error: str, *, started: float | None = None) -> None:
        # `update_last_usage=False`: `last_usage()` describes the last CHAT
        # call on this thread, and callers read it right after one to cache
        # the spend. An embedding lookup in between (a retrieval tool during
        # an analyst turn) would otherwise be billed as the analyst's call.
        ms = int((time.monotonic() - started) * 1000) if started is not None else 0
        _record_row("openai", EMBEDDING_MODEL, 0, 0, duration_ms=ms, success=False,
                          error=error, update_last_usage=False)

    if not _is_openai_available():
        _row("skipped:client_unavailable")
        raise EmbeddingUnavailable("no embedding provider configured in production")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
    except Exception as exc:
        _row("skipped:client_unavailable")
        raise EmbeddingUnavailable(f"openai client unavailable: {type(exc).__name__}") from None
    # Unchanged in position, and outside every `try`: a stale memo or
    # industry lease must cancel the dispatch. `LeaseLost` subclasses
    # BaseException, so no `except Exception` could swallow it anyway.
    from .regen_lease import assert_current
    assert_current()
    from .industry_lease import assert_current as assert_industry_current
    assert_industry_current()
    started = time.monotonic()
    try:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=list(texts))
    except Exception as exc:
        # Type only, and `from None`: a provider exception can carry the
        # request body or a key, and a chained cause would be logged too.
        _row(f"provider_error:{type(exc).__name__}", started=started)
        raise EmbeddingUnavailable(f"openai embeddings failed: {type(exc).__name__}") from None
    ms = int((time.monotonic() - started) * 1000)
    used = getattr(getattr(resp, "usage", None), "total_tokens", None)
    # None when the response omits usage: the meter then estimates, while
    # the row records 0 rather than an estimate it would present as billed.
    billed: int | None = (used if isinstance(used, int) and not isinstance(used, bool)
                          and used >= 0 else None)
    served = getattr(resp, "model", None)
    served_model = served if isinstance(served, str) and served else None
    try:
        vecs = [list(d.embedding) for d in resp.data]
    except Exception:
        vecs = None
    if vecs is None or len(vecs) != len(texts) or any(len(v) != EMBEDDING_DIM for v in vecs):
        # Billed all the same, so the row carries the tokens it reported.
        _record_row("openai", EMBEDDING_MODEL, billed or 0, 0, duration_ms=ms,
                          success=False, error="unexpected_shape", served_model=served_model,
                          update_last_usage=False)
        raise EmbeddingUnavailable("openai embeddings returned an unexpected shape")
    _record_row("openai", EMBEDDING_MODEL, billed or 0, 0, duration_ms=ms,
                      served_model=served_model, update_last_usage=False)
    meter = _usage_meter.get()
    if meter is not None:
        meter.calls += 1
        if billed is not None:
            meter.tokens += billed
        else:
            meter.estimated_tokens += sum(count_tokens(t) for t in texts)
    return vecs


def embed_one(text: str, *, action: str | None = None, ticker: str | None = None) -> list[float]:
    from ..agents import llm

    with llm._call_scope("embed_one", action=action, ticker=ticker, route="", max_tokens=0):
        return embed([text], action=action, ticker=ticker)[0]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
#
# Chunk budgets are stated in *model* tokens, so they have to be measured in
# model tokens. The previous chunker counted whitespace-separated words and
# called them tokens, which is not a rounding error on this corpus: measured
# against `cl100k_base`, ordinary 10-K prose runs ~1.1 tokens per word, an
# MD&A sentence dense with figures ("increased 12.4% to $394,328 million")
# runs ~1.8, and a segment table row runs ~4.1. A 500-"word" chunk was
# therefore anywhere from 550 to 2,000 real tokens — silently over budget
# exactly where the retrieval corpus is densest.
#
# `tiktoken` is already a pinned dependency, so exactness costs nothing but a
# lookup. The encoding is resolved *from the embedding model in use* rather
# than hard-coded, so swapping `EMBEDDING_MODEL` cannot leave the chunker
# measuring against the wrong vocabulary.

# Characters per token when the encoder is unavailable. Measured across the
# four text shapes above, `cl100k_base` yields 5.9 chars/token for prose,
# 5.1 for risk-factor boilerplate, 3.3 for figure-dense MD&A and 2.6 for
# table rows. 3.6 sits near the dense end on purpose: over-estimating tokens
# yields chunks a little smaller than the budget, while under-estimating
# yields chunks over it, and only one of those degrades retrieval.
_FALLBACK_CHARS_PER_TOKEN = 3.6

# An ASCII character costs at most one token; Unicode code points can cost
# several. Only short ASCII strings skip encoding. Above
# `_MAX_CHARS_PER_TOKEN` x budget nothing plausibly fits, so we say so
# without encoding. Between the two we encode — a bounded slice, never the
# document. This is what keeps `iter_chunks` from tokenizing a 5 MB 10-K to
# find out it is larger than 500 tokens.
_MAX_CHARS_PER_TOKEN = 8.0

# The `cl100k_base` BPE ranks, committed to the repo (FIX-012). tiktoken
# otherwise downloads them from openaipublic.blob.core.windows.net on first
# use and caches them under the system temp dir, which made exact token
# counting depend on the network:
#
#   * CI's netguard refuses that download, so a fresh runner silently fell
#     back to the character heuristic and skipped every encoder-specific
#     test — a green run said nothing about real token budgets.
#   * On Render, a fresh container re-fetched on every boot, and a boot
#     during a blob-storage or egress outage chunked a whole session on the
#     heuristic.
#
# The file name is tiktoken's cache key — sha1 of the ranks URL — because
# tiktoken's `read_file_cached` looks for exactly `<TIKTOKEN_CACHE_DIR>/<key>`.
# tiktoken verifies it against the SHA-256 pinned in `tiktoken_ext` and, on a
# mismatch, DELETES it and re-fetches; `.gitattributes` marks it `-text` so
# no EOL normalisation can trigger that, and `test_tiktoken_vendored` fails
# on a hash drift or a tiktoken upgrade that moves the pin.
TIKTOKEN_RANKS_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
VENDORED_TIKTOKEN_DIR = Path(__file__).resolve().parents[1] / "data" / "tiktoken"
VENDORED_TIKTOKEN_FILE = VENDORED_TIKTOKEN_DIR / hashlib.sha1(TIKTOKEN_RANKS_URL.encode()).hexdigest()


def _use_vendored_tiktoken_cache() -> None:
    """Point tiktoken's disk cache at the committed ranks, if they are there.

    `setdefault`, not assignment: an operator-set `TIKTOKEN_CACHE_DIR` wins.
    A missing file changes nothing, so a checkout without it behaves exactly
    as before (download, or the heuristic). Only `embeddings` uses tiktoken
    in this app, so redirecting the process-wide cache touches nothing else.

    Never raises: `_encoding()` promises that, and `Path.is_file` still
    raises on errors other than "not there" (EACCES on the directory, EIO).
    An unreadable vendored dir is treated like a missing one.
    """
    try:
        present = VENDORED_TIKTOKEN_FILE.is_file()
    except OSError as exc:
        log.warning("vendored tiktoken ranks unreadable (%s); using tiktoken defaults", exc)
        return
    if present:
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(VENDORED_TIKTOKEN_DIR))


@lru_cache(maxsize=1)
def _encoding():
    """The tiktoken encoding for `EMBEDDING_MODEL`, or None.

    Never raises. The ranks are read from the vendored copy above; were it
    missing or corrupt, `tiktoken` would fetch them over the network, and a
    worker without egress would otherwise take the whole process down inside
    a filing index. Returning None puts the chunker on the calibrated
    character heuristic, which is worse and still correct.
    """
    # Before the import, and inside the lru_cache, so it runs once per
    # process and before tiktoken can resolve its cache directory.
    _use_vendored_tiktoken_cache()
    try:
        import tiktoken
    except Exception as exc:  # pragma: no cover — tiktoken is pinned
        log.warning("tiktoken unavailable (%s); chunking on the char heuristic", exc)
        return None
    try:
        return tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except Exception:
        # Unknown model name — every current `text-embedding-3-*` model uses
        # cl100k_base, so try it by name before giving up.
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            log.warning(
                "tiktoken encoding unavailable (%s); chunking on the char heuristic",
                exc,
            )
            return None


def count_tokens(text: str) -> int:
    """Model-token count for `text`; a calibrated estimate if unencodable."""
    if not text:
        return 0
    enc = _encoding()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:  # pragma: no cover — defensive
            pass
    return max(1, int(len(text) / _FALLBACK_CHARS_PER_TOKEN + 0.5))


def _fits(text: str, budget: int) -> bool:
    """`count_tokens(text) <= budget`, short-circuited on length.

    The two length tests are not optimisations of an exact answer, they are
    what makes the exact answer affordable: the encode only ever runs on a
    string already known to be within a small multiple of the budget.
    """
    n = len(text)
    if n <= budget and text.isascii():
        return True
    if n > budget * _MAX_CHARS_PER_TOKEN:
        return False
    return count_tokens(text) <= budget


# ---------------------------------------------------------------------------
# Structure-aware splitting
# ---------------------------------------------------------------------------
#
# A recursive structure splitter: take the largest natural boundary the text
# offers, and only descend to a smaller one for the pieces that are still
# over budget. Section, then paragraph, then line, then sentence, then
# clause, then word, and a hard character cut as the genuine last resort.
#
# The ordering is the whole point on SEC filings. A fixed window cuts
# mid-sentence, mid-table-row and mid-number, which severs a figure from the
# line item it belongs to — the one failure mode that makes a retrieved
# passage actively misleading rather than merely incomplete.

# A section break: a form heading ("Item 1A.", "PART II"), a markdown
# heading, or a run of blank lines. Kept as a lookahead so the heading stays
# attached to the section it introduces.
_SECTION_BREAK = re.compile(
    r"\n(?=\s*(?:ITEM\s+\d|Item\s+\d|PART\s+[IVX]+\b|Part\s+[IVX]+\b|#{1,6}\s))"
    r"|\n\s*\n\s*\n+"
)

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

# Abbreviations whose trailing period is not a sentence end. Financial prose
# is thick with them, and a split after "Inc." or "U.S." strands the subject
# of the sentence in the previous chunk.
_ABBREVIATIONS = frozenset({
    "inc", "corp", "co", "ltd", "llc", "lp", "plc", "no", "nos", "vs", "approx",
    "est", "fig", "figs", "cf", "al", "etc", "mr", "mrs", "ms", "dr", "jr",
    "sr", "st", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sept", "sep",
    "oct", "nov", "dec", "u.s", "e.g", "i.e", "q1", "q2", "q3", "q4",
})

# Candidate sentence end: terminal punctuation, optional closing quote or
# bracket, then whitespace, then something that can start a sentence.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s+(?=[A-Z“‘(\"'•\-])")

_CLAUSE_BREAK = re.compile(r"(?<=[;:])\s+|(?<=,)\s+(?=(?:and|or|but|while|which|including)\b)")

# A table row: two or more runs of collapsed whitespace, a pipe, or a tab.
# Financial tables in EDGAR text come through as column-aligned runs of
# spaces, and splitting inside one puts a number in a different chunk from
# its row label.
_TABLE_ROW = re.compile(r"\|| {2,}\S.* {2,}|\t")

_LINE_BREAK = re.compile(r"\n")

# A bare figure — a number, optionally signed / currencied / parenthesised,
# with or without a trailing unit character. The word-level last resort must
# not put one of these at the end of a chunk, because its unit or its label
# is in the next word.
_BARE_FIGURE = re.compile(r"^[\(\[]?[-+$€£¥]?[\d,.]+[\)\]%]?$")


def _iter_split(text: str, pattern: re.Pattern[str]) -> Iterator[str]:
    """Yield `text` cut at every `pattern` boundary, lazily.

    Two properties the rest of the module depends on:

    * **Lossless.** Each piece runs to the *end* of the boundary match, so
      concatenating the pieces reproduces `text` exactly. Terminal
      punctuation and the whitespace after it therefore stay attached to
      the sentence they belong to, and a chunk assembled by `"".join` reads
      as the document read.
    * **Lazy.** `re.split` materialises every piece of a multi-megabyte
      document at once, which is the allocation this module exists to
      avoid. `finditer` yields one slice at a time, and the "no boundary
      here" case yields the original object rather than a copy of it.
    """
    start = 0
    matched = False
    for m in pattern.finditer(text):
        if m.end() <= start:  # zero-width, or a boundary inside the last one
            continue
        matched = True
        yield text[start:m.end()]
        start = m.end()
    if not matched:
        yield text
        return
    if start < len(text):
        yield text[start:]


def _iter_words(text: str) -> Iterator[str]:
    """Whitespace-delimited words, keeping their trailing whitespace."""
    for m in re.finditer(r"\S+\s*", text):
        yield m.group(0)


def _hard_cut(text: str, budget: int) -> Iterator[str]:
    """Cut an unsplittable run using bounded, measured character slices.

    Unicode and encoded blobs can have several model tokens per character.
    The character estimate only selects a candidate; shrink it until its
    measured token count fits, preserving every original code point.
    """
    width = max(1, int(budget * _FALLBACK_CHARS_PER_TOKEN))
    start = 0
    while start < len(text):
        end = min(len(text), start + width)
        while end - start > 1 and count_tokens(text[start:end]) > budget:
            end = start + max(1, (end - start) // 2)
        yield text[start:end]
        start = end


def _iter_atoms(text: str, budget: int, level: int = 0) -> Iterator[str]:
    """Yield the largest structural pieces of `text` that fit `budget`.

    Descends one rung of the boundary ladder per recursion, so a paragraph
    that fits is never split into sentences and a sentence that fits is
    never split into words. Peak memory is the recursion path plus the
    current piece — never the whole document, and never its token list.
    """
    if _fits(text, budget):
        if text:
            yield text
        return

    if level == 0:
        pieces: Iterable[str] = _iter_split(text, _SECTION_BREAK)
    elif level == 1:
        pieces = _iter_split(text, _PARAGRAPH_BREAK)
    elif level == 2:
        pieces = _iter_split(text, _LINE_BREAK)
    elif level == 3:
        pieces = _iter_sentences(text)
    elif level == 4:
        pieces = _iter_split(text, _CLAUSE_BREAK)
    elif level == 5:
        pieces = _iter_words(text)
    else:
        yield from _hard_cut(text, budget)
        return

    for piece in pieces:
        if piece is text:
            # No boundary of this kind: drop a rung rather than recurse on
            # an identical string.
            yield from _iter_atoms(text, budget, level + 1)
            return
        # A table row is atomic below the line level. Its columns are one
        # record; a figure split away from its row label retrieves as a
        # number with no referent. Honoured up to twice the budget, past
        # which there is no readable chunk to protect.
        if level >= 2 and _TABLE_ROW.search(piece) and _fits(piece, budget * 2):
            if piece.strip():
                yield piece
            continue
        yield from _iter_atoms(piece, budget, level + 1)


def _iter_sentences(text: str) -> Iterator[str]:
    """Lossless, lazy sentence split that survives financial abbreviations.

    Same contract as `_iter_split`: pieces concatenate back to `text`, and
    a text with no sentence boundary yields the original object so the
    caller can tell that this rung of the ladder had nothing to offer.
    """
    start = 0
    matched = False
    for m in _SENTENCE_END.finditer(text):
        if m.end() <= start:
            continue
        if _ends_in_abbreviation(text[start:m.start() + 1]):
            continue
        matched = True
        yield text[start:m.end()]
        start = m.end()
    if not matched:
        yield text
        return
    if start < len(text):
        yield text[start:]


def _ends_in_abbreviation(fragment: str) -> bool:
    """True when `fragment`'s trailing period closes an abbreviation.

    Decimals need no rule here: `_SENTENCE_END` requires whitespace after
    the period, and "$1.5 billion" / "12.4%" have none, so a figure is
    never a candidate boundary in the first place. Abbreviations are the
    case the regex cannot see — "Berkshire Hathaway Inc. reported" would
    otherwise be cut after "Inc.".
    """
    stripped = fragment.rstrip()
    if not stripped.endswith("."):
        return False
    body = stripped[:-1]
    word = body.rsplit(None, 1)[-1] if body.split() else ""
    return word.lower().strip("([\"'") in _ABBREVIATIONS


def _overlap_tail(chunk: str, overlap_tokens: int) -> str:
    """Whole trailing sentences of `chunk`, never more than `overlap_tokens`.

    Fixed-width overlap was the other half of the old chunker's problem: 50
    words back from an arbitrary cut reproduces a sentence *fragment*, so
    the sentence carrying the figure is still severed — now in both chunks.
    Carrying whole sentences means the boundary never falls inside the
    statement a retrieval hit depends on.

    `overlap_tokens` is a cap, including on the first sentence considered.
    Guaranteeing one whole sentence whatever its size sounds harmless and
    is not: a single run-on sentence — or an EDGAR table block, which has
    no `[.!?]` boundary at all and so reads as one "sentence" — is carried
    in full, and the chunk that follows it is then mostly a copy of its
    predecessor. Measured on filing-shaped text, that cost ~1.9x the
    embedding spend and ~1.9x the `doc_chunks` rows, and the near-duplicate
    chunks compete with each other for `vector_store.search`'s top-k.

    When the last sentence alone is over the cap, the overlap degrades one
    rung down the same boundary ladder `_iter_atoms` uses — to whole
    trailing clauses — rather than to an arbitrary word offset. Nothing is
    severed by that: the sentence is intact at the end of `chunk`, and what
    is carried forward is a boundary-aligned lead-in, not the sentence's
    only copy. Below a clause there is nothing meaningful left to carry, so
    the overlap is dropped.
    """
    if overlap_tokens <= 0 or not chunk.strip():
        return ""
    tail: list[str] = []
    total = 0
    for sentence in reversed(list(_iter_sentences(chunk))):
        n = count_tokens(sentence)
        if total + n > overlap_tokens:
            if not tail:
                tail = _clause_tail(sentence, overlap_tokens)
            break
        tail.insert(0, sentence)
        total += n
        if total >= overlap_tokens:
            break
    out = "".join(tail)
    # Never let the overlap be the entire chunk: the next chunk would then
    # start where this one did and the walk would not advance.
    if len(out) >= len(chunk):
        return ""
    return out


def _clause_tail(sentence: str, budget: int) -> list[str]:
    """Whole trailing clauses of one oversized sentence, within `budget`.

    Returns a list of pieces so `_overlap_tail` can join them the way it
    joins sentences. Empty when even the last clause is over budget —
    there is no boundary below this one worth carrying, and a bare word
    suffix is the fragment this module exists to stop producing.
    """
    pieces = list(_iter_split(sentence, _CLAUSE_BREAK))
    if len(pieces) < 2:  # no clause boundary: `_iter_split` yields the whole
        return []
    tail: list[str] = []
    total = 0
    for piece in reversed(pieces):
        n = count_tokens(piece)
        if total + n > budget:
            break
        tail.insert(0, piece)
        total += n
    return tail


def iter_chunks(
    text: str, *, target_tokens: int = 500, overlap_tokens: int = 50,
) -> Iterator[str]:
    """Stream structure-aware, token-budgeted chunks of `text`.

    A generator on purpose. `filing_memory.index_filing` used to accumulate
    every chunk of a filing in one list before writing a single row, on a
    512 MiB worker that Render has OOM-killed twice; this hands the caller
    one chunk at a time so the peak working set is a chunk, not a 10-K.

    `target_tokens` is a budget, not a hard cap: a table row is kept whole
    up to twice it (see `_iter_atoms`), because a split row retrieves worse
    than a long one.
    """
    if not text or not text.strip():
        return
    target = max(1, int(target_tokens))
    # Half the budget is the ceiling on overlap: past that a chunk is mostly
    # a copy of its predecessor, and the walk slows to a crawl on long docs.
    overlap = max(0, min(int(overlap_tokens), target // 2))

    buf: list[str] = []
    buf_tokens = 0
    for atom in _iter_atoms(text, target):
        n = count_tokens(atom)
        if buf and buf_tokens + n > target:
            # Never end a chunk on a bare figure. Reached only when the
            # boundary has already fallen through to word level inside one
            # oversized sentence or table row, and there it matters most:
            # "$394,328" in one chunk and "million in fiscal 2026" in the
            # next retrieves as a number with no unit and no referent.
            # Push the trailing figures into the next chunk instead; the
            # loop still advances because at least one atom always stays.
            pushed: list[str] = []
            pushed_tokens = 0
            while len(buf) > 1 and _BARE_FIGURE.match(buf[-1].strip()):
                last_tokens = count_tokens(buf[-1])
                # A numeric run may itself fill a chunk. Keep every figure,
                # but never move more than fits beside the incoming atom.
                if pushed_tokens + last_tokens + n > target:
                    break
                pushed.insert(0, buf.pop())
                pushed_tokens += last_tokens
            chunk = "".join(buf)
            if chunk.strip():
                yield chunk
            # The overlap is context, and context never costs the next chunk
            # its budget: ask for only what is left after the atom that
            # forced this flush (and any figures pushed along with it). A
            # long sentence therefore carries less overlap, where the
            # alternative is a chunk well over target — `buf` is rebuilt as
            # carry + pushed + atom, so an unbudgeted carry lands in the
            # chunk whole.
            room = target - n - sum(count_tokens(piece) for piece in pushed)
            carry = _overlap_tail(chunk, min(overlap, max(0, room)))
            buf = ([carry] if carry else []) + pushed
            buf_tokens = sum(count_tokens(piece) for piece in buf)
        buf.append(atom)
        buf_tokens += n
    if buf:
        chunk = "".join(buf)
        if chunk.strip():
            yield chunk


def chunk_text(
    text: str, *, target_tokens: int = 500, overlap_tokens: int = 50,
) -> list[str]:
    """List form of `iter_chunks`, for callers that want one.

    Kept because the retrieval interface must not change — every existing
    caller passes a section or a transcript block and wants a list back.
    Prefer `iter_chunks` for anything document-sized.
    """
    return list(iter_chunks(
        text, target_tokens=target_tokens, overlap_tokens=overlap_tokens,
    ))
