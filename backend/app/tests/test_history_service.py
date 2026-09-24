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
- Phase 6: inserted rows carry a point-in-time `available_at` (the demo
  dataset has no period_end, so the `assumed_fye` rule applies), and a
  restatement never moves it.
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
    # Zero net writes across all three, which is what "idempotent" has to
    # mean for a number that is also telemetry. Filings and transcripts used
    # to count every re-ingest as a write — see
    # `test_ingest_counters_are_truthful.py` for what that cost.
    assert second == {"financial_periods": 0, "filings": 0, "transcripts": 0}
    # And the row counts don't grow either.
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


# ---------------------------------------------------------------------------
# Phase 6 — point-in-time availability on ingest
# ---------------------------------------------------------------------------

def test_backfill_sets_available_at_on_every_inserted_row():
    """Demo statement rows carry `period="2024"` and no period_end, so the
    only derivable date is the assumed fiscal-year end plus the annual
    lag — and it must be there on every row, never NULL."""
    from datetime import date, timedelta

    from app.config import settings

    _reset_tables()
    history_service.backfill_ticker("NVDA")
    with SessionLocal() as db:
        rows = db.query(FinancialPeriod).filter(FinancialPeriod.ticker == "NVDA").all()
    assert rows
    assert all(r.available_at is not None for r in rows)
    assert {r.available_at_source for r in rows} == {"assumed_fye"}
    lag = timedelta(days=settings.scorecard_pit_lag_annual_days)
    for r in rows:
        assert r.fiscal_year is not None
        assert r.available_at == date(r.fiscal_year, 12, 31) + lag
        # never later than when we fetched it
        assert r.available_at <= r.fetched_at.date()


def test_restatement_on_reupsert_keeps_the_original_available_at():
    from datetime import date

    _reset_tables()
    history_service.backfill_ticker("NVDA")
    with SessionLocal() as db:
        row = db.query(FinancialPeriod).filter(
            FinancialPeriod.ticker == "NVDA", FinancialPeriod.line_item == "revenue",
        ).order_by(FinancialPeriod.period.desc()).first()
        assert row is not None
        key = dict(ticker=row.ticker, period=row.period, statement=row.statement, line_item="revenue")
        original_available, original_value = row.available_at, row.value
        assert original_available is not None
        # A restated figure arrives with a different (later) availability date.
        changed = history_service._upsert_financial_period(
            db, **key, value=(original_value or 0.0) + 1.0, period_end=row.period_end,
            fiscal_year=row.fiscal_year, fiscal_quarter=row.fiscal_quarter, source="test",
            available_at=date(2030, 1, 1), available_at_source="provider",
        )
        db.commit()
        assert changed is True
        after = db.query(FinancialPeriod).filter_by(**key).one()
    assert after.value == (original_value or 0.0) + 1.0
    assert after.available_at == original_available
    assert after.available_at_source == "assumed_fye"


def test_upsert_refuses_period_end_change_under_available_at():
    """FIX-006 root cause: an INSERT-only availability must never end up
    before its period end because a later write moved the end (LULU)."""
    from datetime import date

    _reset_tables()
    key = dict(ticker="PEGRD", period="FY2025", statement="balance", line_item="total_debt",
               fiscal_year=2025, fiscal_quarter=None, source="fmp")
    refusals: list = []
    with SessionLocal() as db:
        assert history_service._upsert_financial_period(
            db, **key, value=5.0, period_end=date(2025, 2, 2), available_at=date(2025, 4, 18),
            available_at_source="lag_rule")
        db.flush()
        for new_end in (date(2026, 2, 1), None):
            assert not history_service._upsert_financial_period(
                db, **key, value=6.0, period_end=new_end, refusals=refusals)
        db.commit()
        row = db.query(FinancialPeriod).filter_by(ticker="PEGRD").one()
        assert (row.period_end, row.available_at, row.value) == (date(2025, 2, 2), date(2025, 4, 18), 5.0)
        # Same end: the ordinary same-provider restatement still applies.
        assert history_service._upsert_financial_period(db, **key, value=6.0, period_end=date(2025, 2, 2))
        db.commit()
        assert db.query(FinancialPeriod).filter_by(ticker="PEGRD").one().value == 6.0
    assert [(r["kind"], r["stored_period_end"], r["incoming_period_end"]) for r in refusals] == [
        ("period_end_change_refused", "2025-02-02", "2026-02-01"),
        ("period_end_change_refused", "2025-02-02", None)]


def test_provider_owned_ticker_makes_no_statements_read_and_writes_no_periods(monkeypatch):
    """FIX-005/FIX-006 (§3.12): the anonymous, cache-backed annual read never
    touches a ticker with provider-owned durable history, not even for newer
    periods: the filing-driven refresh (`fundamental_refresh`) fetches those
    from FMP when EDGAR shows them filed. First-contact tickers keep the
    whole legacy ingest."""
    from datetime import date

    from app.services.data_service import get_data_service

    _reset_tables()
    with SessionLocal() as db:
        db.add(FinancialPeriod(ticker="NVDA", period="FY2025", statement="income", line_item="revenue", value=1.0,
                               period_end=date(2025, 1, 26), fiscal_year=2025, source="fmp", currency="USD"))
        db.commit()
    ds = get_data_service()
    original = ds.get_financial_statements
    asked: list[str] = []

    def spy(ticker):
        asked.append(ticker)
        if ticker != "NVDA":
            return original(ticker)
        rows = [("FY2024", "2024-01-28", 7.0), ("FY2025", "2025-01-28", 9.0), ("FY2026", "2026-01-25", 11.0)]
        return {"income": [{"period": p, "period_end": e, "revenue": v, "currency": "USD"} for p, e, v in rows],
                "balance": [], "cash": []}

    monkeypatch.setattr(ds, "get_financial_statements", spy)
    owned = history_service.backfill_ticker("NVDA")
    first_contact = history_service.backfill_ticker("MSFT")
    assert owned["fundamentals"] == "durable" and owned["financial_periods"] == 0
    assert first_contact["financial_periods"] > 0 and "fundamentals" not in first_contact
    assert asked == ["MSFT"]
    with SessionLocal() as db:
        rows = {(r.period, r.source, r.value) for r in db.query(FinancialPeriod).filter_by(ticker="NVDA")}
    assert rows == {("FY2025", "fmp", 1.0)}
