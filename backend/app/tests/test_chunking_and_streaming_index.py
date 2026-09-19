"""Chunking for retrieval over SEC filings, and the memory it is allowed to use.

Two things are being asserted here, and they are the same change seen from
two ends.

**The chunk boundaries.** The previous chunker was a fixed window over
`text.split()`: 500 whitespace words, 50 words of overlap. Words are not
tokens — measured against `cl100k_base`, the encoding the embedding model
actually uses, ordinary filing prose runs ~1.1 tokens per word, a
figure-dense MD&A sentence ~1.8 and a segment table row ~4.1 — so a "500
token" chunk was anywhere from 550 to 2,000 real tokens, and most over
budget exactly where the corpus is densest. Worse, the window had no idea
what it was cutting: it split mid-sentence, mid-table-row and between a
number and its label, which is the one failure mode that makes a retrieved
passage actively misleading rather than merely incomplete. A chunk that
reads "Greater China 72,559" with the year header in a different chunk is a
figure with no referent.

**The memory.** `text.split()` materialises a word list for the whole
document, and `filing_memory.index_filing` then accumulated *every* chunk of
a filing in one list and asked for one embedding call over all of them —
on a 512 MiB worker Render has OOM-killed twice. The fix is a generator end
to end: `iter_chunks` yields, `_iter_filing_chunks` yields, and
`vector_store.upsert_source` embeds and inserts in batches. Batching is the
part that needed care, because re-indexing a filing must still *replace*
its prior chunks exactly — the tests below check both halves of that: a
re-index leaves no stale chunk behind, and a stream that fails half way
leaves the previous index intact rather than half-deleted.
"""
from __future__ import annotations

import inspect
import time

import pytest

from app.database import SessionLocal
from app.models import DocChunk, FilingDoc
from app.services import embeddings as emb
from app.services import filing_memory, vector_store

TARGET = 120
OVERLAP = 24


def _suffix() -> str:
    return str(time.perf_counter_ns())[-9:]


# A figure-dense paragraph of the kind that broke the word-count budget.
_MDNA_SENTENCES = [
    f"Total net sales in segment {i} increased 12.4% to ${394 + i},{i:03d} million "
    f"in fiscal 2026 from ${350 + i},{i:03d} million in fiscal 2025, while gross "
    f"margin expanded {200 + i} bps to 4{i % 10}.2%."
    for i in range(60)
]
MDNA = " ".join(_MDNA_SENTENCES)

SEGMENT_TABLE = (
    "Segment            2026       2025     Change\n"
    "Americas        169,658    155,105       8.7%\n"
    "Europe           95,118     94,294       0.9%\n"
    "Greater China    72,559     74,200      (2.2)%\n"
    "Japan            26,694     25,977       2.8%\n"
)


# ---------------------------------------------------------------------------
# Token budgeting
# ---------------------------------------------------------------------------

def test_the_budget_is_counted_in_model_tokens_not_words():
    """The defect, stated as the property that now holds.

    `MDNA` is the shape the old chunker was worst on. Chunking it to a
    120-token budget and measuring the result with the embedding model's own
    encoding is the whole test: the word-proxy chunker produced chunks ~1.8x
    over, and no assertion about words could have caught it.
    """
    chunks = emb.chunk_text(MDNA, target_tokens=TARGET, overlap_tokens=OVERLAP)

    assert chunks
    over = [c for c in chunks if emb.count_tokens(c) > TARGET]
    assert not over, (
        f"{len(over)} of {len(chunks)} chunks exceed the {TARGET}-token budget; "
        f"largest was {max(emb.count_tokens(c) for c in over)}"
    )


def test_a_word_budget_would_have_blown_the_token_budget_on_this_text():
    """Guards the premise, so the test above cannot quietly become vacuous.

    If this corpus ever stopped being token-dense the budget test would pass
    for the wrong reason. Assert the gap directly instead.
    """
    words = MDNA.split()[:TARGET]
    assert emb.count_tokens(" ".join(words)) > TARGET * 1.4, (
        "this fixture is no longer token-dense, so the budget test above no "
        "longer distinguishes a token budget from a word budget"
    )


def test_the_encoding_follows_the_embedding_model_rather_than_a_hardcoded_name():
    enc = emb._encoding()
    if enc is None:
        pytest.skip("tiktoken encoding unavailable in this environment")
    import tiktoken
    assert enc.name == tiktoken.encoding_for_model(emb.EMBEDDING_MODEL).name


def test_chunking_survives_an_unloadable_tokenizer(monkeypatch):
    """The worker must not die because a tokenizer download failed.

    `tiktoken` fetches BPE ranks over the network on first use. A worker
    that boots without that cache and without egress would otherwise take
    the whole process down inside a filing index — so the degraded path is
    a calibrated character heuristic, never an exception.
    """
    monkeypatch.setattr(emb, "_encoding", lambda: None)

    assert emb.count_tokens("Total net sales increased 12.4%.") > 0
    chunks = emb.chunk_text(MDNA, target_tokens=TARGET, overlap_tokens=OVERLAP)
    assert chunks
    assert all(c.strip() for c in chunks)
    # The heuristic is deliberately pessimistic (it over-counts prose rather
    # than under-counting figures), so chunks land at or under budget.
    assert all(emb.count_tokens(c) <= TARGET for c in chunks)


# ---------------------------------------------------------------------------
# Structure awareness
# ---------------------------------------------------------------------------

def test_chunks_break_on_sentence_boundaries_not_mid_sentence():
    chunks = emb.chunk_text(MDNA, target_tokens=TARGET, overlap_tokens=OVERLAP)

    for c in chunks[:-1]:
        assert c.rstrip().endswith("."), (
            f"chunk ends mid-sentence: ...{c[-70:]!r}"
        )


def test_a_figure_is_never_severed_from_its_label():
    """The worst failure mode for financial retrieval, asserted directly."""
    chunks = emb.chunk_text(MDNA, target_tokens=TARGET, overlap_tokens=OVERLAP)
    joined = "\n<<CHUNK BOUNDARY>>\n".join(chunks)

    for sentence in _MDNA_SENTENCES:
        assert any(sentence.strip() in c for c in chunks), (
            f"a sentence was split across chunks, so its figures lost their "
            f"subject: {sentence!r}\nboundaries:\n{joined[:400]}"
        )


def test_a_table_row_is_kept_whole():
    """A row is one record. Split it and a number retrieves with no referent."""
    chunks = emb.chunk_text(SEGMENT_TABLE, target_tokens=16, overlap_tokens=4)

    for row in SEGMENT_TABLE.strip().split("\n"):
        assert any(row.strip() in c for c in chunks), (
            f"table row was split across chunks: {row!r} -> {chunks}"
        )


def test_a_chunk_never_ends_on_a_bare_number():
    """The last-resort boundary rule, at the only level that can reach it.

    Word-level splitting only happens inside one oversized sentence or
    table row, and that is exactly where a number can be severed from its
    unit: "$394,328" in one chunk and "million in fiscal 2026" in the next
    retrieves as a figure with no unit and no referent.
    """
    run = " ".join(
        f"${394_000 + i} million and ${350_000 + i} million in segment {i} and"
        for i in range(40)
    )
    chunks = emb.chunk_text("Net sales were " + run, target_tokens=60, overlap_tokens=10)

    assert len(chunks) > 5
    for chunk in chunks:
        last = chunk.strip().split()[-1]
        assert not emb._BARE_FIGURE.match(last), (
            f"chunk ends on a bare figure ({last!r}): ...{chunk[-80:]!r}"
        )


def test_paragraph_and_section_structure_is_preferred_over_a_fixed_window():
    doc = (
        "Item 1A. Risk Factors.\n\n"
        "The Company faces intense competition from Apple Inc. and others.\n\n"
        "Item 7. Management's Discussion and Analysis.\n\n"
        + MDNA
    )
    chunks = emb.chunk_text(doc, target_tokens=TARGET, overlap_tokens=OVERLAP)

    # The short risk-factor section fits well inside one chunk, so it must
    # not be glued to the middle of the MD&A the way a fixed window would.
    assert any(
        "Item 1A. Risk Factors." in c and "Apple Inc. and others." in c
        for c in chunks
    )
    # "Inc." is an abbreviation, not a sentence end.
    assert not any(c.rstrip().endswith("Apple Inc.") for c in chunks)


def test_overlap_carries_whole_trailing_sentences():
    """Fixed word overlap reproduced a fragment; boundary-aligned overlap
    reproduces the sentence that carries the fact.

    Run at an overlap budget that a whole `MDNA` sentence fits inside — the
    regime this property actually holds in. The budget is a cap, so what
    happens when a sentence does *not* fit is a separate question, asserted
    directly below.
    """
    wide = max(emb.count_tokens(s) for s in _MDNA_SENTENCES) + 2
    chunks = emb.chunk_text(MDNA, target_tokens=TARGET, overlap_tokens=wide)
    assert len(chunks) > 2

    for earlier, later in zip(chunks, chunks[1:]):
        head = later.split(". ")[0] + "."
        assert head.strip() in earlier, (
            "the chunk boundary is not carrying a whole sentence forward"
        )
        assert any(head.strip() == s.strip() for s in _MDNA_SENTENCES), (
            f"the carried lead-in is not a whole sentence: {head!r}"
        )


# ---------------------------------------------------------------------------
# The overlap budget is a cap
# ---------------------------------------------------------------------------

def test_the_overlap_never_exceeds_its_budget():
    """The defect: one whole sentence was carried whatever its size.

    `_overlap_tail` walked the chunk's sentences backwards and exempted the
    first one it considered from the budget, to guarantee at least one whole
    sentence of overlap. On ordinary filing prose that is a ~10% overshoot.
    On the shapes this module exists for it is not: a run-on risk factor, or
    an EDGAR table block — which carries no `[.!?]` boundary at all and so
    reads as one enormous "sentence" — was carried into the next chunk
    whole, and the overlap ran an order of magnitude over its budget.
    """
    huge = "The Company faces " + " ".join(
        f"risk factor {i} with exposure of ${i},{i:03d} million" for i in range(40)
    ) + ". "
    chunk = "Net sales rose 12.4%. " + huge
    assert emb.count_tokens(huge) > 10 * OVERLAP, "fixture is not oversized"

    tail = emb._overlap_tail(chunk, OVERLAP)

    assert emb.count_tokens(tail) <= OVERLAP, (
        f"the overlap is {emb.count_tokens(tail)} tokens against a budget of "
        f"{OVERLAP}: {tail[:120]!r}"
    )
    assert chunk.endswith(tail), "the overlap must stay a suffix of the chunk"


def test_an_oversized_sentence_degrades_to_a_clause_not_to_nothing():
    """Where the cap sends the overlap when a whole sentence will not fit.

    Dropping the overlap entirely would be the easy cap, and it would quietly
    disable overlap across most of the corpus: filing sentences are routinely
    longer than a 50-token budget. So it steps one rung down the same
    boundary ladder `_iter_atoms` uses — whole trailing clauses — which is
    still boundary-aligned, unlike the arbitrary word offset this chunker
    replaced.
    """
    sentence = (
        "Total net sales increased 12.4% to $394,328 million in fiscal 2026 "
        "from $350,000 million in fiscal 2025, while gross margin expanded "
        "200 bps to 40.2%. "
    )
    chunk = "The prior sentence. " + sentence
    budget = emb.count_tokens(sentence) - 5  # too small for the whole sentence

    tail = emb._overlap_tail(chunk, budget)

    assert tail, "a clause-aligned overlap was available and was not carried"
    assert emb.count_tokens(tail) <= budget
    assert tail.startswith("while gross margin"), (
        f"the overlap did not start at a clause boundary: {tail!r}"
    )


def _sentence_filling(budget: int) -> str:
    """One unsplittable sentence just under `budget` tokens.

    Built to the *measured* size rather than to a fixed clause count. A
    clause count only lands near the budget for one tokenizer: where
    `count_tokens` falls back to its character heuristic (no tiktoken BPE
    cache and no egress to fetch one — the state every CI runner starts
    in), the same 38 clauses measure 506 tokens instead of 461 and the
    fixture stops fitting a chunk. Measuring keeps the case this test
    exists for — a sentence that nearly fills a chunk — true under either
    counter, instead of reporting an encoder-availability problem as a
    failure of the carry cap.
    """
    clauses: list[str] = []
    while True:
        i = len(clauses)
        candidate = clauses + [f"risk factor {i} with exposure of ${i},{i:03d} million"]
        text = "The Company faces " + " ".join(candidate) + ". "
        if emb.count_tokens(text) > budget:
            return "The Company faces " + " ".join(clauses) + ". "
        clauses = candidate


def test_a_long_sentence_does_not_double_the_corpus():
    """The cost the cap exists to stop, measured end to end at the defaults.

    An uncapped carry does not just overshoot the overlap — it lands in the
    next chunk whole, so the chunk itself blows the token budget too. Before
    the cap this text produced 845-token chunks against a 500-token budget
    and 1.8x the content, which is 1.8x the embedding spend, 1.8x the
    `doc_chunks` rows, and near-duplicate chunks competing with each other
    for `vector_store.search`'s top-k.
    """
    sentence = _sentence_filling(460)
    assert 50 < emb.count_tokens(sentence) < 500, "fixture must fit one chunk"
    doc = ("Net sales rose 12.4%. " + sentence + sentence) * 3

    chunks = emb.chunk_text(doc, target_tokens=500, overlap_tokens=50)
    sizes = [emb.count_tokens(c) for c in chunks]

    assert max(sizes) <= 500, (
        f"a chunk ran to {max(sizes)} tokens against a 500-token budget: the "
        f"overlap is being carried into it unbudgeted"
    )
    assert sum(sizes) <= emb.count_tokens(doc) * 1.25, (
        f"the corpus was duplicated {sum(sizes) / emb.count_tokens(doc):.2f}x"
    )


def test_every_chunk_is_a_contiguous_slice_of_the_source():
    """No reflowing: the chunker must not invent or drop whitespace, or the
    text a hit shows the user stops matching the filing."""
    for chunk in emb.chunk_text(MDNA, target_tokens=TARGET, overlap_tokens=OVERLAP):
        assert chunk in MDNA


def test_empty_and_whitespace_input_yield_nothing():
    assert emb.chunk_text("") == []
    assert emb.chunk_text("   \n\n  ") == []


def test_a_single_unsplittable_run_is_cut_rather_than_dropped():
    blob = "A" * 20_000
    chunks = emb.chunk_text(blob, target_tokens=TARGET, overlap_tokens=OVERLAP)
    assert chunks
    assert "".join(chunks) == blob


# ---------------------------------------------------------------------------
# Bounded memory
# ---------------------------------------------------------------------------

def test_the_splitter_is_a_generator_that_does_not_read_the_whole_document():
    """The memory property, asserted by how much work one chunk costs.

    A 10-K is multiple megabytes. Pulling the first chunk must cost work
    proportional to a chunk, not to the document — which is exactly what
    `text.split()` and an eager `re.split` could not do.
    """
    doc = MDNA * 200
    stream = emb.iter_chunks(doc, target_tokens=TARGET, overlap_tokens=OVERLAP)
    assert inspect.isgenerator(stream)

    calls: list[int] = []
    real = emb.count_tokens

    def counting(text: str) -> int:
        calls.append(len(text))
        return real(text)

    emb.count_tokens = counting
    try:
        first = next(stream)
        after_one = len(calls)
        rest = sum(1 for _ in stream)
    finally:
        emb.count_tokens = real

    assert first.strip()
    assert rest > 100, "fixture is too small to distinguish lazy from eager"
    assert after_one < len(calls) / 10, (
        f"producing one chunk cost {after_one} of {len(calls)} token counts — "
        "the splitter is reading the whole document before yielding"
    )


def test_no_single_token_count_ever_encodes_the_whole_document():
    """`count_tokens` on a 5 MB string would allocate its entire token list.

    The guard is a length short-circuit, and it is load-bearing rather than
    an optimisation: without it the first thing `iter_chunks` does to a 10-K
    is tokenize all of it.
    """
    doc = "word " * 2_000_000
    assert not emb._fits(doc, 500)

    seen: list[int] = []
    real_encoding = emb._encoding()
    if real_encoding is None:
        pytest.skip("tiktoken encoding unavailable in this environment")

    class Spy:
        name = real_encoding.name

        def encode(self, text, **kwargs):
            seen.append(len(text))
            return real_encoding.encode(text, **kwargs)

    emb._encoding.cache_clear()
    original = emb._encoding
    emb._encoding = lambda: Spy()
    try:
        next(emb.iter_chunks(doc, target_tokens=500, overlap_tokens=50))
    finally:
        emb._encoding = original
        emb._encoding.cache_clear()

    assert seen, "the encoder was never consulted"
    assert max(seen) <= 500 * 8, (
        f"a single encode saw {max(seen)} characters; the budget short-circuit "
        "is not bounding what reaches the tokenizer"
    )


# ---------------------------------------------------------------------------
# Streaming into the vector store
# ---------------------------------------------------------------------------

@pytest.fixture()
def filing():
    ticker = f"ZZCHK{_suffix()}"[:12]
    with SessionLocal() as db:
        row = FilingDoc(
            ticker=ticker,
            accession_number=f"0000001234-26-{_suffix()}",
            filing_type="10-K",
            sections={"mda": MDNA, "risk_factors": MDNA},
            raw_text=MDNA,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        row_id = row.id
    with SessionLocal() as db:
        yield db.get(FilingDoc, row_id)
    with SessionLocal() as db:
        db.query(DocChunk).filter(DocChunk.source_id == row_id,
                                  DocChunk.source_type == "filing").delete()
        db.query(FilingDoc).filter(FilingDoc.id == row_id).delete()
        db.commit()


def _chunk_rows(filing_id: int) -> list[DocChunk]:
    with SessionLocal() as db:
        return db.query(DocChunk).filter(
            DocChunk.source_type == "filing",
            DocChunk.source_id == filing_id,
        ).all()


def test_index_filing_hands_the_store_a_stream_not_a_list(filing, monkeypatch):
    captured: dict = {}
    real = vector_store.upsert_source

    def spy(**kwargs):
        captured["chunks"] = kwargs["chunks"]
        return real(**kwargs)

    monkeypatch.setattr(filing_memory.vector_store, "upsert_source", spy)
    written = filing_memory.index_filing(filing)

    assert written > 0
    assert inspect.isgenerator(captured["chunks"]), (
        "index_filing built the whole chunk list again; the accumulation is "
        "what the 512 MiB worker could not afford"
    )


def test_embedding_happens_in_bounded_batches(filing, monkeypatch):
    """One embed call over every chunk of a filing is the allocation spike."""
    sizes: list[int] = []
    real = vector_store.emb_svc.embed

    def spy(texts):
        sizes.append(len(texts))
        return real(texts)

    monkeypatch.setattr(vector_store.emb_svc, "embed", spy)
    written = vector_store.upsert_source(
        ticker=filing.ticker, source_type="filing", source_id=filing.id,
        chunks=filing_memory._iter_filing_chunks(filing), batch_size=8,
    )

    assert written > 8
    assert len(sizes) > 1, "the whole filing was embedded in a single call"
    assert max(sizes) <= 8


def test_reindexing_replaces_the_prior_chunks_exactly(filing):
    """Idempotency is the property batching most easily breaks."""
    first = filing_memory.index_filing(filing)
    assert first > 0
    assert len(_chunk_rows(filing.id)) == first

    second = filing_memory.index_filing(filing)
    assert second == first
    assert len(_chunk_rows(filing.id)) == first, (
        "re-indexing duplicated chunks instead of replacing them"
    )

    # And a filing whose content changed must leave nothing of the old one.
    with SessionLocal() as db:
        row = db.get(FilingDoc, filing.id)
        row.sections = {"mda": "The Company sold its data centre segment."}
        db.commit()
        changed = db.get(FilingDoc, filing.id)
        third = filing_memory.index_filing(changed)

    rows = _chunk_rows(filing.id)
    assert third == len(rows) == 1
    assert "data centre segment" in rows[0].text
    assert not any("Total net sales" in r.text for r in rows), (
        "chunks from the previous version of the filing survived the re-index"
    )


def test_a_stream_that_fails_half_way_leaves_the_previous_index_intact(filing):
    """The delete and every batch's insert share one transaction.

    Without that, a provider hiccup part way through a long filing would
    leave the ticker with no chunks at all — worse than the stale ones.
    """
    before = filing_memory.index_filing(filing)
    assert before > 4

    def exploding():
        for i, chunk in enumerate(filing_memory._iter_filing_chunks(filing)):
            if i >= 3:
                raise RuntimeError("provider died mid-stream")
            yield chunk

    assert vector_store.upsert_source(
        ticker=filing.ticker, source_type="filing", source_id=filing.id,
        chunks=exploding(), batch_size=2,
    ) == 0
    assert len(_chunk_rows(filing.id)) == before, (
        "a failed re-index destroyed the chunks it could not replace"
    )


def test_an_empty_stream_does_not_delete_the_existing_chunks(filing):
    before = filing_memory.index_filing(filing)
    assert before > 0

    assert vector_store.upsert_source(
        ticker=filing.ticker, source_type="filing", source_id=filing.id,
        chunks=iter(()),
    ) == 0
    assert len(_chunk_rows(filing.id)) == before


def test_token_count_on_the_stored_row_is_a_token_count(filing):
    filing_memory.index_filing(filing)
    rows = _chunk_rows(filing.id)

    assert rows
    for row in rows:
        assert row.token_count == emb.count_tokens(row.text)
        assert row.token_count > len(row.text.split()), (
            "token_count is still counting whitespace words on figure-dense text"
        )


def test_chunk_text_still_returns_a_list_for_the_transcript_path():
    """`filing_memory.index_transcript` and any other caller keep working."""
    out = emb.chunk_text(MDNA, target_tokens=TARGET, overlap_tokens=OVERLAP)
    assert isinstance(out, list)
    assert all(isinstance(c, str) for c in out)


@pytest.mark.parametrize("unit", ["營業收入增長風險市場", "🧑🏽‍💻🚀", "aZ3_9/"])
def test_dense_unsplittable_runs_respect_measured_budget_without_losing_text(unit):
    text = unit * 1200
    chunks = emb.chunk_text(text, target_tokens=500, overlap_tokens=0)
    assert "".join(chunks) == text
    assert all(emb.count_tokens(chunk) <= 500 for chunk in chunks)


@pytest.mark.skipif(
    emb._encoding() is None,
    reason=(
        "asserts a property of the real encoder: without tiktoken's cl100k_base "
        "ranks (not cached, and the netguard blocks the fetch) count_tokens IS a "
        "code-point heuristic, so there is nothing here to measure"
    ),
)
def test_short_unicode_is_measured_in_tokens_not_code_points():
    text = "🚀" * 100
    assert len(text) <= 120 < emb.count_tokens(text)
    assert not emb._fits(text, 120)


def test_numeric_run_carry_cannot_overfill_the_next_paragraph():
    text = "business " * 100 + "123 " * 200 + "\n\n" + "growth " * 473
    chunks = emb.chunk_text(text, target_tokens=500, overlap_tokens=0)
    assert "".join(chunks) == text
    assert max(map(emb.count_tokens, chunks)) <= 500
