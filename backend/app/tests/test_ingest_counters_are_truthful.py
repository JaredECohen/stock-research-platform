"""`backfill_ticker` counts writes that happened, not rows it looked at.

`backfill_ticker`'s docstring promised "net write counts" and "idempotent:
re-running on unchanged data is a no-op". The second half was true of the
database and false of the number: `_ingest_filings` and `_ingest_transcripts`
both did `written += 1` on the UPDATE branch, so re-ingesting a row that had
not changed by one character still counted as a write.

That is not a cosmetic inaccuracy. It is the reason a dead pipeline looked
healthy for the system's entire life. The nightly `history_backfill` note read

    tickers=166 fp=0 filings=1660 transcripts=166 errors=0

every single night — 1660 being 166 tickers x the SEC provider's hard 10-filing
cap, a number that could only appear if nothing was being filtered and nothing
was new. `fp=0` sat right next to it and was *truthful*, because
`_upsert_financial_period` compares before it writes. The one counter that
would have shown the filings pipeline had been frozen since the first day was
the one that could not report zero.

So these tests assert the counters can say "nothing happened" — and that they
still say something when something does.
"""
from __future__ import annotations

import pytest

from app.database import SessionLocal
from app.models import EarningsTranscript, FilingDoc, FinancialPeriod
from app.monitoring import history_backfill
from app.services import history_service

TICKER = "NVDA"


@pytest.fixture(autouse=True)
def _quiet_indexing(monkeypatch):
    """Skip the post-insert vector indexing.

    Inserting a filing fires `filing_memory.post_pass` (chunk embedding plus
    an LLM diff) and a transcript fires `index_transcript`. Both are covered
    by their own tests, neither has anything to do with what a counter
    returns, and leaving them in adds ~100 seconds to this file alone.
    """
    from app.services import filing_memory
    monkeypatch.setattr(filing_memory, "post_pass", lambda row: None)
    monkeypatch.setattr(filing_memory, "index_transcript", lambda row: None)


@pytest.fixture()
def clean_tables():
    def _reset() -> None:
        with SessionLocal() as db:
            history_service._ensure_tables(db)
            db.query(FinancialPeriod).filter(FinancialPeriod.ticker == TICKER).delete()
            db.query(FilingDoc).filter(FilingDoc.ticker == TICKER).delete()
            db.query(EarningsTranscript).filter(EarningsTranscript.ticker == TICKER).delete()
            db.commit()
    _reset()
    yield
    _reset()


def test_a_second_backfill_of_unchanged_data_reports_zero_writes(clean_tables):
    """The contract the docstring always claimed."""
    first = history_service.backfill_ticker(TICKER)
    assert first["filings"] > 0, "fixture should ingest something on the first pass"
    assert first["transcripts"] > 0

    second = history_service.backfill_ticker(TICKER)

    assert second == {"financial_periods": 0, "filings": 0, "transcripts": 0}, (
        "re-running on unchanged provider data must write nothing and say so — "
        f"got {second}"
    )


def test_a_third_backfill_is_still_zero(clean_tables):
    """Not an alternating artefact of `fetched_at` — genuinely stable."""
    history_service.backfill_ticker(TICKER)
    history_service.backfill_ticker(TICKER)
    assert history_service.backfill_ticker(TICKER)["filings"] == 0


def test_a_changed_filing_is_still_counted(clean_tables):
    """The counter must not be zero because it stopped counting.

    A stored row is edited out from under the ingest, so the next pass sees
    provider content that genuinely differs and has to write it.
    """
    history_service.backfill_ticker(TICKER)
    with SessionLocal() as db:
        row = db.query(FilingDoc).filter(FilingDoc.ticker == TICKER).first()
        assert row is not None
        row.raw_text = "a materially different body"
        row.word_count = 5
        db.commit()

    res = history_service.backfill_ticker(TICKER)

    assert res["filings"] == 1, f"a real difference must count as a write: {res}"
    assert res["transcripts"] == 0, "untouched transcripts must stay at zero"
    with SessionLocal() as db:
        restored = db.query(FilingDoc).filter(FilingDoc.ticker == TICKER).first()
        assert restored.raw_text != "a materially different body", (
            "the row should have been rewritten from the provider payload"
        )


def test_a_changed_transcript_is_still_counted(clean_tables):
    history_service.backfill_ticker(TICKER)
    with SessionLocal() as db:
        row = db.query(EarningsTranscript).filter(
            EarningsTranscript.ticker == TICKER,
        ).first()
        assert row is not None
        row.full_text = "different call"
        row.word_count = 2
        db.commit()

    res = history_service.backfill_ticker(TICKER)

    assert res["transcripts"] == 1, f"a real difference must count as a write: {res}"
    assert res["filings"] == 0


def test_a_new_filing_still_counts_as_a_write(clean_tables):
    """Deleting a row simulates a filing the store has never seen."""
    history_service.backfill_ticker(TICKER)
    with SessionLocal() as db:
        row = db.query(FilingDoc).filter(FilingDoc.ticker == TICKER).first()
        db.delete(row)
        db.commit()

    assert history_service.backfill_ticker(TICKER)["filings"] == 1


def test_the_nightly_note_reads_zero_on_an_unchanged_run(clean_tables, monkeypatch):
    """The production symptom, at the loop that produced it.

    `filings=1660` every night was the single piece of telemetry pointing at
    this pipeline, and it was a constant. A number that cannot change cannot
    be a signal.
    """
    notes: list[str] = []
    monkeypatch.setattr(
        history_backfill, "record_run",
        lambda *a, **k: notes.append(k.get("note", "")),
    )

    history_backfill.run_once(TICKER)
    history_backfill.run_once(TICKER)

    assert "fp=0 filings=0 transcripts=0" in notes[-1], (
        f"a quiet night must read as a quiet night: {notes[-1]!r}"
    )
    assert "errors=0" in notes[-1]
