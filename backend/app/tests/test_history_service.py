"""Wave 2 tests — financial history depth.

Covers:
- `backfill_ticker` ingests demo provider data into all three new tables.
- Backfill is idempotent: re-running on unchanged data yields zero writes.
- `get_financial_history` returns long-format rows newest-first.
- Filing accession_number is the unique key (re-ingest doesn't dup).
- Transcript `(ticker, period)` is the unique key.
- `get_filing_text(..., section=...)` extracts a single section.
- Period label parsing handles common shapes (`2024Q4`, `FY2024`, `2024`).
- Read APIs scope by ticker (no cross-talk).
- Backfill loop wires up — `monitoring.history_backfill.run_once(ticker)`
  populates rows when called directly.
"""
from __future__ import annotations

from app.database import SessionLocal
from app.models import EarningsTranscript, FilingDoc, FinancialPeriod
from app.services import history_service


def _reset_tables() -> None:
    with SessionLocal() as db:
        history_service._ensure_tables(db)
        db.query(FinancialPeriod).delete()
        db.query(FilingDoc).delete()
        db.query(EarningsTranscript).delete()
        db.commit()


# ---------------------------------------------------------------------------
# Period label parser
# ---------------------------------------------------------------------------

def test_parse_period_quarterly():
    assert history_service._parse_period("2024Q4") == (2024, 4)
    assert history_service._parse_period("2024-Q1") == (2024, 1)


def test_parse_period_annual():
    assert history_service._parse_period("FY2024") == (2024, None)
    assert history_service._parse_period("2024") == (2024, None)
    assert history_service._parse_period(2023) == (2023, None)


def test_parse_period_garbage():
    assert history_service._parse_period("not-a-period") == (None, None)
    assert history_service._parse_period(None) == (None, None)


# ---------------------------------------------------------------------------
# Backfill mechanics
# ---------------------------------------------------------------------------

def test_backfill_ticker_writes_rows_for_all_three_tables():
    _reset_tables()
    res = history_service.backfill_ticker("NVDA")
    assert res["financial_periods"] > 0, "expected statement rows for NVDA"
    assert res["filings"] > 0, "expected at least one filing for NVDA"
    assert res["transcripts"] > 0, "expected at least one transcript for NVDA"

    with SessionLocal() as db:
        fp_count = db.query(FinancialPeriod).filter(
            FinancialPeriod.ticker == "NVDA",
        ).count()
        fd_count = db.query(FilingDoc).filter(FilingDoc.ticker == "NVDA").count()
        tx_count = db.query(EarningsTranscript).filter(
            EarningsTranscript.ticker == "NVDA",
        ).count()
        assert fp_count >= 4, f"expected ≥4 (annual periods × revenue at minimum), got {fp_count}"
        assert fd_count >= 1
        assert tx_count >= 1


def test_backfill_is_idempotent():
    _reset_tables()
    history_service.backfill_ticker("NVDA")
    second = history_service.backfill_ticker("NVDA")
    # Second call should produce zero net writes — no value/period changes.
    assert second["financial_periods"] == 0
    # Filings + transcripts re-write metadata on every pass (re-fetched_at);
    # what matters is the row count doesn't grow.
    with SessionLocal() as db:
        fd_count = db.query(FilingDoc).filter(FilingDoc.ticker == "NVDA").count()
        tx_count = db.query(EarningsTranscript).filter(
            EarningsTranscript.ticker == "NVDA",
        ).count()
    history_service.backfill_ticker("NVDA")
    with SessionLocal() as db:
        fd_count2 = db.query(FilingDoc).filter(FilingDoc.ticker == "NVDA").count()
        tx_count2 = db.query(EarningsTranscript).filter(
            EarningsTranscript.ticker == "NVDA",
        ).count()
    assert fd_count2 == fd_count
    assert tx_count2 == tx_count


def test_backfill_isolates_tickers():
    """A backfill of MSFT must not change NVDA rows."""
    _reset_tables()
    history_service.backfill_ticker("NVDA")
    with SessionLocal() as db:
        nvda_before = db.query(FinancialPeriod).filter(
            FinancialPeriod.ticker == "NVDA",
        ).count()
    history_service.backfill_ticker("MSFT")
    with SessionLocal() as db:
        nvda_after = db.query(FinancialPeriod).filter(
            FinancialPeriod.ticker == "NVDA",
        ).count()
        msft_count = db.query(FinancialPeriod).filter(
            FinancialPeriod.ticker == "MSFT",
        ).count()
    assert nvda_after == nvda_before
    assert msft_count > 0


# ---------------------------------------------------------------------------
# Read APIs
# ---------------------------------------------------------------------------

def test_get_financial_history_long_format_newest_first():
    _reset_tables()
    history_service.backfill_ticker("NVDA")
    out = history_service.get_financial_history(
        "NVDA", ["revenue", "operating_income"], limit=40,
    )
    assert "revenue" in out and "operating_income" in out
    rev = out["revenue"]
    assert rev, "no revenue rows returned"
    # Newest first: the leading row's period_end (or period as fallback) is
    # the maximum across the series.
    periods = [r["period"] for r in rev]
    assert periods == sorted(periods, reverse=True), (
        f"expected newest-first ordering, got {periods}"
    )
    # Each row carries the long-format keys.
    assert {"period", "value", "fiscal_year", "statement"} <= set(rev[0].keys())
    assert rev[0]["statement"] == "income"


def test_get_recent_filings_newest_first_and_filterable():
    _reset_tables()
    history_service.backfill_ticker("NVDA")
    rows = history_service.get_recent_filings("NVDA", limit=10)
    assert rows, "expected at least one filing"
    # Filing dates should be in non-increasing order (None last).
    seen_none = False
    last_date = None
    for r in rows:
        d = r["filing_date"]
        if d is None:
            seen_none = True
            continue
        assert not seen_none, "None-dated filings must come AFTER dated ones"
        if last_date is not None:
            assert d <= last_date
        last_date = d
    # Filtering by type narrows the set.
    types = {r["filing_type"] for r in rows}
    if "10-K" in types:
        only_10k = history_service.get_recent_filings(
            "NVDA", filing_type="10-K", limit=10,
        )
        assert all(r["filing_type"] == "10-K" for r in only_10k)


def test_get_filing_text_returns_full_record_and_section():
    _reset_tables()
    history_service.backfill_ticker("NVDA")
    listing = history_service.get_recent_filings("NVDA", limit=1)
    assert listing
    accession = listing[0]["accession_number"]
    full = history_service.get_filing_text("NVDA", accession)
    assert full is not None
    assert full["accession_number"] == accession
    assert isinstance(full["sections"], dict)

    # Pick a section the demo data ships and verify the targeted lookup.
    for name in ("risk_factors", "mda", "business_description"):
        if name in (full["sections"] or {}):
            slim = history_service.get_filing_text(
                "NVDA", accession, section=name,
            )
            assert slim is not None
            assert slim["section"] == name
            assert "text" in slim
            return  # one section is enough


def test_get_transcript_latest_and_specific_period():
    _reset_tables()
    history_service.backfill_ticker("NVDA")
    latest = history_service.get_transcript("NVDA")
    assert latest is not None
    assert latest["ticker"] == "NVDA"
    assert latest["period"]
    # Specific lookup round-trips.
    targeted = history_service.get_transcript("NVDA", period=latest["period"])
    assert targeted is not None
    assert targeted["period"] == latest["period"]
    # Transcripts have at least some structured blocks rendered.
    assert isinstance(targeted["blocks"], list)


def test_get_transcript_unknown_ticker_returns_none():
    assert history_service.get_transcript("NEVER_EXISTS_99") is None


def test_get_filing_text_unknown_returns_none():
    assert history_service.get_filing_text("NVDA", "no-such-accession") is None


# ---------------------------------------------------------------------------
# Monitoring wiring
# ---------------------------------------------------------------------------

def test_monitoring_history_backfill_run_once_processes_one_ticker():
    from app.monitoring import history_backfill
    _reset_tables()
    res = history_backfill.run_once("MSFT")
    assert res["tickers_processed"] == 1
    assert res["errors"] == 0
    with SessionLocal() as db:
        msft_count = db.query(FinancialPeriod).filter(
            FinancialPeriod.ticker == "MSFT",
        ).count()
    assert msft_count > 0
