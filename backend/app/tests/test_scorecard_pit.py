"""Phase 6 (Slice A) — point-in-time data foundation for the scorecard.

Covers `services/scorecard_pit`, the `available_at` wiring in
`history_service`, the FMP filing-date pass-through, the scorecard tables
and the backfill CLI. Everything is network-free: statement rows are
synthetic, prices come from the session `DemoProvider`, and the clock is
only ever compared to values the test itself constructed.

Synthetic tickers (`PITA`, `PITB`, …) keep these tests from touching rows
other suites seed for the demo names.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import inspect as sa_inspect

from app.config import settings
from app.database import SessionLocal, engine, init_db
from app.models import (
    FilingDoc,
    FinancialPeriod,
    PriceMonthEnd,
    ScorecardDisagreement,
    ScorecardEvaluation,
    ScorecardRun,
    ScorecardScore,
    ScorecardVersion,
)
from app.providers.fmp_provider import FMPProvider
from app.scripts import scorecard_backfill
from app.services import history_service, scorecard_pit
from app.services.scorecard_pit import FilingDate, derive_available_at

T = "PITA"
T2 = "PITB"


def _clear(*tickers: str) -> None:
    with SessionLocal() as db:
        scorecard_pit._ensure_tables(db)
        for t in tickers:
            db.query(FinancialPeriod).filter(FinancialPeriod.ticker == t).delete()
            db.query(FilingDoc).filter(FilingDoc.ticker == t).delete()
            db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker == t).delete()
        db.commit()


def _seed_period(
    db, *, ticker: str, period: str, line: str = "revenue", value: float = 1.0,
    statement: str = "income", period_end: date | None = None,
    fiscal_year: int | None = None, fiscal_quarter: int | None = None,
    available_at: date | None = None, available_at_source: str | None = None,
    fetched_at: datetime | None = None,
) -> None:
    db.add(FinancialPeriod(
        ticker=ticker, period=period, statement=statement, line_item=line,
        value=value, period_end=period_end, fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter, source="test",
        fetched_at=fetched_at or datetime(2026, 9, 1, 12, 0, 0),
        available_at=available_at, available_at_source=available_at_source,
    ))


# ---------------------------------------------------------------------------
# derive_available_at — the rule chain
# ---------------------------------------------------------------------------

def test_provider_date_wins_and_is_tagged_provider():
    got = derive_available_at(
        ticker=T, period_end=date(2025, 6, 30), fiscal_year=2025, fiscal_quarter=None,
        provider_date="2025-08-15",
        filings=[FilingDate("10-K", date(2025, 8, 20), date(2025, 6, 30))],
    )
    assert got == (date(2025, 8, 15), "provider")


def test_provider_datetime_string_is_coerced_to_a_date():
    got = derive_available_at(
        ticker=T, period_end=date(2025, 6, 30), fiscal_year=2025, fiscal_quarter=None,
        provider_date="2025-08-15 16:05:00",
    )
    assert got == (date(2025, 8, 15), "provider")


def test_provider_date_before_period_end_is_garbage_and_falls_through():
    """A filing cannot precede the period it reports on."""
    got = derive_available_at(
        ticker=T, period_end=date(2025, 6, 30), fiscal_year=2025, fiscal_quarter=None,
        provider_date="2025-01-02",
    )
    assert got == (date(2025, 6, 30) + timedelta(days=settings.scorecard_pit_lag_annual_days), "lag_rule")


def test_filing_doc_match_within_seven_days_uses_earliest_filing_date():
    filings = [
        FilingDate("10-K/A", date(2025, 7, 1), date(2025, 6, 30)),   # an amendment never dates the financials
        FilingDate("10-K", date(2025, 8, 29), date(2025, 6, 28)),     # within ±7 days
        FilingDate("10-K", date(2025, 8, 20), date(2025, 7, 5)),      # within ±7 days, earlier filing → wins
        FilingDate("10-Q", date(2025, 5, 1), date(2025, 3, 31)),      # different period
    ]
    got = derive_available_at(
        ticker=T, period_end=date(2025, 6, 30), fiscal_year=2025, fiscal_quarter=None,
        filings=filings,
    )
    assert got == (date(2025, 8, 20), "filing_doc")


def test_filing_doc_outside_window_or_before_period_end_falls_back_to_lag():
    filings = [
        FilingDate("10-K", date(2025, 8, 20), date(2025, 6, 20)),  # 10 days off
        FilingDate("10-K", date(2025, 6, 1), date(2025, 6, 30)),   # filed before the period closed
    ]
    got = derive_available_at(
        ticker=T, period_end=date(2025, 6, 30), fiscal_year=2025, fiscal_quarter=None,
        filings=filings,
    )
    assert got == (date(2025, 6, 30) + timedelta(days=settings.scorecard_pit_lag_annual_days), "lag_rule")


def test_lag_rule_uses_annual_lag_for_fy_rows_and_quarter_lag_for_quarters():
    annual, src_a = derive_available_at(
        ticker=T, period_end=date(2025, 12, 31), fiscal_year=2025, fiscal_quarter=None,
    )
    quarter, src_q = derive_available_at(
        ticker=T, period_end=date(2025, 9, 30), fiscal_year=2025, fiscal_quarter=3,
    )
    assert (src_a, src_q) == ("lag_rule", "lag_rule")
    assert annual == date(2025, 12, 31) + timedelta(days=settings.scorecard_pit_lag_annual_days)
    assert quarter == date(2025, 9, 30) + timedelta(days=settings.scorecard_pit_lag_quarter_days)


def test_lag_defaults_are_the_documented_conservative_values():
    assert settings.scorecard_pit_lag_annual_days == 75
    assert settings.scorecard_pit_lag_quarter_days == 45


def test_assumed_fye_path_for_demo_rows_without_period_end():
    """The demo dataset writes `period="2024"` and no period_end; the
    year end is assumed and the annual lag stacked on top."""
    got = derive_available_at(ticker=T, period_end=None, fiscal_year=2024, fiscal_quarter=None)
    assert got == (date(2024, 12, 31) + timedelta(days=settings.scorecard_pit_lag_annual_days), "assumed_fye")


def test_nothing_derivable_returns_none_pair_not_a_guess():
    assert derive_available_at(ticker=T, period_end=None, fiscal_year=None, fiscal_quarter=None) == (None, None)


def test_fetched_at_is_a_hard_upper_bound_for_every_rule():
    fetched = datetime(2026, 1, 15, 3, 0, 0)
    # lag rule would say 2026-03-16; we fetched it on 2026-01-15, so it was public by then.
    lag = derive_available_at(
        ticker=T, period_end=date(2025, 12, 31), fiscal_year=2025, fiscal_quarter=None, fetched_at=fetched,
    )
    assert lag == (date(2026, 1, 15), "lag_rule")
    # a provider date after fetched_at is likewise capped (source kept for the audit trail)
    prov = derive_available_at(
        ticker=T, period_end=date(2025, 12, 31), fiscal_year=2025, fiscal_quarter=None,
        provider_date="2026-02-01", fetched_at=fetched,
    )
    assert prov == (date(2026, 1, 15), "provider")
    # an earlier date is left alone
    early = derive_available_at(
        ticker=T, period_end=date(2024, 12, 31), fiscal_year=2024, fiscal_quarter=None, fetched_at=fetched,
    )
    assert early == (date(2024, 12, 31) + timedelta(days=settings.scorecard_pit_lag_annual_days), "lag_rule")


def test_derive_loads_filings_from_db_when_not_supplied():
    _clear(T)
    with SessionLocal() as db:
        db.add(FilingDoc(
            ticker=T, accession_number=f"{T}-10K-2025", filing_type="10-K",
            filing_date=date(2025, 8, 22), period_end=date(2025, 6, 30),
        ))
        db.add(FilingDoc(  # an 8-K on the same period end must not date the financials
            ticker=T, accession_number=f"{T}-8K-2025", filing_type="8-K",
            filing_date=date(2025, 7, 2), period_end=date(2025, 6, 30),
        ))
        db.commit()
        got = derive_available_at(
            ticker=T, period_end=date(2025, 6, 30), fiscal_year=2025, fiscal_quarter=None, db=db,
        )
    assert got == (date(2025, 8, 22), "filing_doc")
    _clear(T)


# ---------------------------------------------------------------------------
# history_service wiring — insert sets it, restatement never moves it
# ---------------------------------------------------------------------------

def _ingest(db, rows, *, statement="income"):
    return history_service._ingest_statement_rows(
        db, T, statement, rows, history_service._INCOME_LINES, "test",
    )


def test_history_insert_records_available_at_from_the_provider_date():
    _clear(T)
    with SessionLocal() as db:
        n = _ingest(db, [{
            "period": "FY2025", "period_end": "2025-06-30", "revenue": 100.0,
            "net_income": 10.0, "filing_date": "2025-08-15",
        }])
        db.commit()
        rows = db.query(FinancialPeriod).filter(FinancialPeriod.ticker == T).all()
    assert n == 2
    assert {(r.available_at, r.available_at_source) for r in rows} == {(date(2025, 8, 15), "provider")}
    _clear(T)


def test_history_insert_without_provider_date_uses_lag_rule():
    _clear(T)
    with SessionLocal() as db:
        _ingest(db, [{"period": "FY2025", "period_end": "2025-06-30", "revenue": 100.0}])
        db.commit()
        row = db.query(FinancialPeriod).filter(FinancialPeriod.ticker == T).one()
    assert row.available_at == date(2025, 6, 30) + timedelta(days=settings.scorecard_pit_lag_annual_days)
    assert row.available_at_source == "lag_rule"
    _clear(T)


def test_restatement_overwrites_value_but_never_moves_available_at():
    """The invariant the whole point-in-time story rests on."""
    _clear(T)
    with SessionLocal() as db:
        _ingest(db, [{"period": "FY2025", "period_end": "2025-06-30", "revenue": 100.0, "filing_date": "2025-08-15"}])
        db.commit()
        first = db.query(FinancialPeriod).filter(FinancialPeriod.ticker == T).one()
        original_available, original_fetched = first.available_at, first.fetched_at
    with SessionLocal() as db:
        # A year later the provider restates the figure and carries a new filing date.
        n = _ingest(db, [{"period": "FY2025", "period_end": "2025-06-30", "revenue": 120.0, "filing_date": "2026-08-20"}])
        db.commit()
        restated = db.query(FinancialPeriod).filter(FinancialPeriod.ticker == T).one()
    assert n == 1
    assert restated.value == 120.0
    assert restated.available_at == original_available == date(2025, 8, 15)
    assert restated.available_at_source == "provider"
    assert restated.fetched_at >= original_fetched
    _clear(T)


def test_upsert_of_an_unchanged_value_writes_nothing():
    _clear(T)
    rows = [{"period": "FY2025", "period_end": "2025-06-30", "revenue": 100.0}]
    with SessionLocal() as db:
        assert _ingest(db, rows) == 1
        db.commit()
        assert _ingest(db, rows) == 0
    _clear(T)


# ---------------------------------------------------------------------------
# backfill_available_at — idempotent, counted, never guesses
# ---------------------------------------------------------------------------

def test_backfill_fills_nulls_reports_counts_and_is_idempotent():
    _clear(T, T2)
    with SessionLocal() as db:
        # lag-rule candidate
        _seed_period(db, ticker=T, period="FY2024", period_end=date(2024, 12, 31), fiscal_year=2024)
        # assumed-fye candidate (demo shape)
        _seed_period(db, ticker=T, period="2023", fiscal_year=2023, line="net_income")
        # unresolvable: no period end, no fiscal year
        _seed_period(db, ticker=T, period="garbage", line="ebit")
        # already dated: must not be touched even though the lag rule would say otherwise
        _seed_period(
            db, ticker=T2, period="FY2024", period_end=date(2024, 12, 31), fiscal_year=2024,
            available_at=date(2025, 2, 1), available_at_source="provider",
        )
        # filing-doc candidate on the second ticker
        _seed_period(db, ticker=T2, period="FY2023", period_end=date(2023, 12, 31), fiscal_year=2023)
        db.add(FilingDoc(
            ticker=T2, accession_number=f"{T2}-10K-2023", filing_type="10-K",
            filing_date=date(2024, 2, 20), period_end=date(2023, 12, 31),
        ))
        db.commit()

    first = scorecard_pit.backfill_available_at(tickers=[T, T2])
    assert first["scanned"] == 4
    assert first["filled"] == 3
    assert first["unresolved"] == 1
    assert first["tickers"] == 2
    assert first["by_source"] == {"lag_rule": 1, "assumed_fye": 1, "filing_doc": 1}

    with SessionLocal() as db:
        by_key = {
            (r.ticker, r.period): (r.available_at, r.available_at_source)
            for r in db.query(FinancialPeriod).filter(FinancialPeriod.ticker.in_([T, T2]))
        }
    lag = settings.scorecard_pit_lag_annual_days
    assert by_key[(T, "FY2024")] == (date(2024, 12, 31) + timedelta(days=lag), "lag_rule")
    assert by_key[(T, "2023")] == (date(2023, 12, 31) + timedelta(days=lag), "assumed_fye")
    assert by_key[(T, "garbage")] == (None, None)
    assert by_key[(T2, "FY2024")] == (date(2025, 2, 1), "provider")
    assert by_key[(T2, "FY2023")] == (date(2024, 2, 20), "filing_doc")

    second = scorecard_pit.backfill_available_at(tickers=[T, T2])
    assert second["filled"] == 0
    assert second["scanned"] == 1 and second["unresolved"] == 1
    assert second["by_source"] == {}
    _clear(T, T2)


def test_backfill_bounds_by_fetched_at_for_rows_fetched_before_the_lag_date():
    _clear(T)
    with SessionLocal() as db:
        _seed_period(
            db, ticker=T, period="FY2026", period_end=date(2026, 6, 30), fiscal_year=2026,
            fetched_at=datetime(2026, 7, 20, 4, 0, 0),
        )
        db.commit()
    scorecard_pit.backfill_available_at(tickers=[T])
    with SessionLocal() as db:
        row = db.query(FinancialPeriod).filter(FinancialPeriod.ticker == T).one()
    assert row.available_at == date(2026, 7, 20)
    assert row.available_at_source == "lag_rule"
    _clear(T)


def test_backfill_with_an_empty_ticker_list_is_a_noop():
    assert scorecard_pit.backfill_available_at(tickers=[]) == {
        "scanned": 0, "filled": 0, "unresolved": 0, "tickers": 0, "by_source": {},
    }


# ---------------------------------------------------------------------------
# Month-end prices
# ---------------------------------------------------------------------------

def _series(*dates: str, close: float = 10.0):
    return [{"date": d, "close": close + i, "adjusted_close": close + i} for i, d in enumerate(dates)]


def test_select_month_end_rows_takes_last_row_per_complete_month_only():
    rows = _series("2026-01-05", "2026-01-30", "2026-02-02", "2026-02-27", "2026-03-03", "2026-03-10")
    out = scorecard_pit.select_month_end_rows(rows)
    assert [(r["month_end"], r["price_date"]) for r in out] == [
        (date(2026, 1, 31), date(2026, 1, 30)),
        (date(2026, 2, 28), date(2026, 2, 27)),
    ]
    assert out[0]["close"] == 11.0 and out[0]["adjusted_close"] == 11.0
    # March is partial (last row 03-10, nothing later): skipped, not guessed.


def test_select_month_end_rows_accepts_a_trailing_month_that_ends_on_the_calendar_end():
    rows = _series("2026-03-03", "2026-03-31")
    out = scorecard_pit.select_month_end_rows(rows)
    assert [(r["month_end"], r["price_date"]) for r in out] == [(date(2026, 3, 31), date(2026, 3, 31))]


def test_select_month_end_rows_is_order_independent_and_skips_bad_rows():
    rows = _series("2026-02-28", "2026-01-30", "2026-01-05") + [
        {"date": "not-a-date", "close": 1.0}, {"date": "2026-02-10", "close": None},
        {"date": "2026-03-31", "close": "abc"},  # March has no usable close: no row, not a zero
    ]
    out = scorecard_pit.select_month_end_rows(rows)
    assert [r["month_end"] for r in out] == [date(2026, 1, 31), date(2026, 2, 28)]
    assert out[0]["close"] == 11.0 and out[1]["close"] == 10.0
    assert scorecard_pit.select_month_end_rows([]) == []


def test_sync_price_month_ends_from_the_demo_series_is_idempotent():
    """One row per complete month of the cached 252-day series; a second
    sync over the same series writes nothing and adds no rows."""
    from app.services.data_service import get_data_service
    _clear("NVDA")
    series = get_data_service().get_price_history("NVDA", days=252)
    expected = scorecard_pit.select_month_end_rows(series)
    assert expected, "demo price series should span at least one complete month"

    first = scorecard_pit.sync_price_month_ends("NVDA")
    assert first["months"] == len(expected)
    assert first["written"] == len(expected)
    with SessionLocal() as db:
        rows = db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker == "NVDA").order_by(PriceMonthEnd.month_end).all()
        stored = [(r.month_end, r.price_date, r.close) for r in rows]
    assert stored == [(m["month_end"], m["price_date"], m["close"]) for m in expected]
    assert all(m["month_end"] == date(m["month_end"].year, m["month_end"].month, m["month_end"].day) for m in expected)

    second = scorecard_pit.sync_price_month_ends("NVDA")
    assert second["months"] == len(expected)
    assert second["written"] == 0
    with SessionLocal() as db:
        assert db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker == "NVDA").count() == len(expected)
    _clear("NVDA")


def test_sync_price_month_ends_rewrites_a_changed_close_without_duplicating(monkeypatch):
    _clear(T)
    from app.services import data_service as ds_mod

    class _DS:
        def __init__(self, rows):
            self.rows = rows

        def get_price_history(self, ticker, days=252):
            return self.rows

        def mode(self):
            return "test"

    fake = _DS(_series("2026-01-05", "2026-01-30", "2026-02-02"))
    monkeypatch.setattr(ds_mod, "get_data_service", lambda: fake)
    assert scorecard_pit.sync_price_month_ends(T) == {"months": 1, "written": 1, "skipped": 1}
    fake.rows = _series("2026-01-05", "2026-01-30", "2026-02-02", close=20.0)
    assert scorecard_pit.sync_price_month_ends(T) == {"months": 1, "written": 1, "skipped": 1}
    with SessionLocal() as db:
        rows = db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker == T).all()
    assert len(rows) == 1 and rows[0].close == 21.0 and rows[0].source == "test"
    _clear(T)


def test_sync_price_month_ends_raises_when_the_provider_chain_returns_none(monkeypatch):
    """None from `get_price_history` means every provider failed. That must
    surface as an error — folding it into `{months: 0}` would let the CLI
    and the worker record a clean sync that wrote nothing."""
    _clear(T)
    from app.services import data_service as ds_mod

    class _DownDS:
        def get_price_history(self, ticker, days=252):
            return None

        def mode(self):
            return "test"

    monkeypatch.setattr(ds_mod, "get_data_service", lambda: _DownDS())
    with pytest.raises(scorecard_pit.PriceSeriesUnavailable):
        scorecard_pit.sync_price_month_ends(T)
    with SessionLocal() as db:
        assert db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker == T).count() == 0


def test_sync_price_month_ends_treats_an_empty_series_as_a_legitimate_zero(monkeypatch):
    """An EMPTY series (a listing younger than one complete month) is not
    an outage: it is a successful sync of zero months, no exception."""
    _clear(T)
    from app.services import data_service as ds_mod

    class _EmptyDS:
        def get_price_history(self, ticker, days=252):
            return []

        def mode(self):
            return "test"

    monkeypatch.setattr(ds_mod, "get_data_service", lambda: _EmptyDS())
    assert scorecard_pit.sync_price_month_ends(T) == {"months": 0, "written": 0, "skipped": 0}


# ---------------------------------------------------------------------------
# snapshot_as_of — the point-in-time read
# ---------------------------------------------------------------------------

def test_snapshot_as_of_excludes_future_and_undated_rows_and_counts_them():
    _clear(T)
    with SessionLocal() as db:
        for fy, avail in ((2023, date(2024, 2, 20)), (2024, date(2025, 2, 25)), (2025, date(2026, 2, 27))):
            for stmt, line, val in (("income", "revenue", fy * 1.0), ("balance", "total_assets", fy * 2.0)):
                _seed_period(
                    db, ticker=T, period=f"FY{fy}", statement=stmt, line=line, value=val,
                    period_end=date(fy, 12, 31), fiscal_year=fy,
                    available_at=avail, available_at_source="provider",
                )
        # a row nobody has dated yet
        _seed_period(db, ticker=T, period="FY2022", line="revenue", value=2022.0, period_end=date(2022, 12, 31), fiscal_year=2022)
        db.add(PriceMonthEnd(ticker=T, month_end=date(2025, 1, 31), price_date=date(2025, 1, 31), close=50.0))
        db.add(PriceMonthEnd(ticker=T, month_end=date(2025, 3, 31), price_date=date(2025, 3, 31), close=55.0))
        db.commit()

    snap = scorecard_pit.snapshot_as_of(T, date(2025, 3, 1))
    assert snap["ticker"] == T and snap["as_of"] == date(2025, 3, 1)
    assert [p["period"] for p in snap["periods"]] == ["FY2024", "FY2023"]  # newest first; FY2025 not yet public
    assert snap["latest_period"] == "FY2024"
    assert snap["data_available_at"] == date(2025, 2, 25)
    assert snap["periods"][0]["income"] == {"revenue": 2024.0}
    assert snap["periods"][0]["balance"] == {"total_assets": 4048.0}
    assert snap["periods"][0]["cash"] == {}
    assert snap["excluded"] == {"null_available_at": 1, "after_as_of": 2}
    assert len(snap["rows"]) == 4
    assert snap["price"]["month_end"] == date(2025, 1, 31) and snap["price"]["close"] == 50.0

    # On the availability date itself the row is in (<=, not <).
    on_day = scorecard_pit.snapshot_as_of(T, date(2026, 2, 27))
    assert on_day["latest_period"] == "FY2025"
    assert on_day["excluded"] == {"null_available_at": 1, "after_as_of": 0}
    assert on_day["price"]["month_end"] == date(2025, 3, 31)

    # Before anything was public: empty, explained, no price.
    early = scorecard_pit.snapshot_as_of(T, date(2023, 6, 30))
    assert early["periods"] == [] and early["latest_period"] is None
    assert early["data_available_at"] is None and early["price"] is None
    assert early["excluded"] == {"null_available_at": 1, "after_as_of": 6}
    _clear(T)


def test_snapshot_as_of_dates_a_period_by_its_latest_line():
    """A period is knowable only when its last line is; a later-dated line
    (re-derived after a restatement) pushes the period's availability out."""
    _clear(T)
    with SessionLocal() as db:
        _seed_period(db, ticker=T, period="FY2024", line="revenue", period_end=date(2024, 12, 31), fiscal_year=2024,
                     available_at=date(2025, 2, 1), available_at_source="provider")
        _seed_period(db, ticker=T, period="FY2024", line="net_income", period_end=date(2024, 12, 31), fiscal_year=2024,
                     available_at=date(2025, 3, 1), available_at_source="lag_rule")
        db.commit()
    snap = scorecard_pit.snapshot_as_of(T, date(2025, 2, 15))
    assert snap["periods"][0]["available_at"] == date(2025, 2, 1)
    assert snap["periods"][0]["income"] == {"revenue": 1.0}  # net_income not yet public
    later = scorecard_pit.snapshot_as_of(T, date(2025, 3, 15))
    assert later["periods"][0]["available_at"] == date(2025, 3, 1)
    assert later["periods"][0]["available_at_source"] == "lag_rule"
    assert later["data_available_at"] == date(2025, 3, 1)
    _clear(T)


# ---------------------------------------------------------------------------
# FMP pass-through (additive keys, only when present)
# ---------------------------------------------------------------------------

def test_fmp_row_mappers_pass_filing_dates_through_only_when_present():
    base = {"date": "2025-06-30", "period": "FY", "reportedCurrency": "USD", "revenue": 1}
    for mapper in (FMPProvider._income_row, FMPProvider._balance_row, FMPProvider._cash_row):
        plain = mapper(dict(base))
        assert "filing_date" not in plain and "accepted_date" not in plain
        dated = mapper({**base, "fillingDate": "2025-08-15", "acceptedDate": "2025-08-15 16:05:00"})
        assert dated["filing_date"] == "2025-08-15"
        assert dated["accepted_date"] == "2025-08-15 16:05:00"
        assert dated["period"] == "FY2025" and dated["period_end"] == "2025-06-30"
        only_accepted = mapper({**base, "acceptedDate": "2025-08-15 16:05:00"})
        assert "filing_date" not in only_accepted and only_accepted["accepted_date"] == "2025-08-15 16:05:00"


# ---------------------------------------------------------------------------
# Schema — the six tables and the two columns
# ---------------------------------------------------------------------------

SCORECARD_TABLES = (
    "scorecard_versions", "scorecard_runs", "scorecard_scores",
    "price_month_ends", "scorecard_evaluations", "scorecard_disagreements",
)

# Identity columns: set on every insert and created with the table. Every
# other column must be addable to a live table later by
# `reconcile_missing_columns` (nullable or defaulted) — the only migration
# path this repo has.
_IDENTITY = {
    ScorecardVersion: {"id", "version_key"},
    ScorecardRun: {"id", "run_id", "version_key", "as_of"},
    ScorecardScore: {"id", "run_id", "version_key", "as_of", "ticker"},
    PriceMonthEnd: {"id", "ticker", "month_end", "close"},
    ScorecardEvaluation: {"id", "version_key", "eval_kind"},
    ScorecardDisagreement: {"id", "ticker", "scorecard_score_id", "version_key", "as_of"},
}


def test_scorecard_tables_are_created_by_init_db():
    init_db()
    existing = set(sa_inspect(engine).get_table_names())
    assert set(SCORECARD_TABLES) <= existing
    assert {c["name"] for c in sa_inspect(engine).get_columns("financial_periods")} >= {"available_at", "available_at_source"}


@pytest.mark.parametrize("model", list(_IDENTITY))
def test_scorecard_columns_are_nullable_or_defaulted_beyond_identity(model):
    names = {c.name for c in model.__table__.columns}
    assert _IDENTITY[model] <= names
    for col in model.__table__.columns:
        if col.name in _IDENTITY[model]:
            continue
        assert col.nullable or col.default is not None or col.server_default is not None, f"{model.__tablename__}.{col.name}"


def test_financial_period_pit_columns_are_nullable():
    for name in ("available_at", "available_at_source"):
        assert FinancialPeriod.__table__.columns[name].nullable, name


def test_scorecard_config_defaults():
    assert settings.enable_scorecard is True
    assert settings.enable_scorecard_loop is True
    assert settings.enable_scorecard_disagreement_regen is False
    assert settings.scorecard_disagreement_regen_daily_cap == 3
    assert settings.scorecard_min_sector_n == 5
    assert settings.scorecard_min_leg_n == 15
    assert settings.scorecard_daily_retention_days == 45
    assert settings.scorecard_export_token == ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_is_a_noop_when_there_is_nothing_to_do(capsys):
    assert scorecard_backfill.main(["--available-at", "--tickers", "ZZZNOPE"]) == 0
    out = capsys.readouterr().out
    assert '"filled": 0' in out and '"prices"' not in out


def test_cli_prices_pass_syncs_the_requested_tickers(capsys):
    _clear("NVDA")
    assert scorecard_backfill.main(["--prices", "--tickers", "nvda"]) == 0
    out = capsys.readouterr().out
    assert '"available_at"' not in out
    with SessionLocal() as db:
        assert db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker == "NVDA").count() >= 1
    _clear("NVDA")


def test_cli_counts_a_provider_outage_as_a_failed_ticker_and_exits_1(monkeypatch, capsys):
    """Regression: a None series used to report `months: 0, errors: 0` and
    exit 0. It is now an error, tagged `unavailable` so an operator can
    tell an outage from a genuinely young listing."""
    from app.services import data_service as ds_mod

    class _DownDS:
        def get_price_history(self, ticker, days=252):
            return None

        def mode(self):
            return "test"

    monkeypatch.setattr(ds_mod, "get_data_service", lambda: _DownDS())
    assert scorecard_backfill.main(["--prices", "--tickers", "ZZZNOPE"]) == 1
    report = json.loads(capsys.readouterr().out)["prices"]
    assert report["errors"] == 1 and report["unavailable"] == 1
    assert report["failed_tickers"] == ["ZZZNOPE"]
    assert report["months"] == 0 and report["written"] == 0


def test_cli_runs_both_passes_by_default_and_reports_a_failed_ticker(monkeypatch, capsys):
    def _boom(ticker, **kw):
        raise RuntimeError("provider down")
    monkeypatch.setattr(scorecard_pit, "sync_price_month_ends", _boom)
    assert scorecard_backfill.main(["--tickers", "ZZZNOPE"]) == 1
    out = capsys.readouterr().out
    assert '"available_at"' in out and '"errors": 1' in out
