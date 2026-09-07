"""Structural tests for `services/screener_metrics_service.py`.

Shape and persistence only. ROIC / tax semantics are deliberately not
asserted here — that computation is being reworked separately, and
this file must keep passing on either side of it. What is pinned: the
16-key metric vocabulary, numeric-or-None values, the None contract for
an unknown ticker, no-raise for a company with no financials, and that
`snapshot_universe` upserts rather than duplicates.

Financials come from the DemoProvider through `history_service.
backfill_ticker`, the same path production uses to fill
`financial_periods`.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from sqlalchemy import select

from app.database import SessionLocal
from app.models import Company, ScreenerMetric
from app.services import history_service
from app.services import screener_metrics_service as sms
from app.tests.fixtures.seed_demo_data import run_full_seed

DEMO = "MSFT"
METRIC_KEYS = {
    "ticker", "pe_ttm", "forward_pe", "peg", "ev_ebitda", "ev_revenue",
    "gross_margin", "op_margin", "fcf_margin", "roic", "roe",
    "debt_to_ebitda", "revenue_growth_yoy", "dividend_yield",
    "market_cap", "beta",
}
DERIVED_KEYS = METRIC_KEYS - {"ticker", "market_cap", "beta"}


@pytest.fixture(scope="module", autouse=True)
def _seeded():
    run_full_seed()
    history_service.backfill_ticker(DEMO)


def _is_number_or_none(v: Any) -> bool:
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_returns_the_documented_vocabulary():
    m = sms.compute_metrics(DEMO.lower())
    assert m is not None and set(m) == METRIC_KEYS
    assert m["ticker"] == DEMO
    assert all(_is_number_or_none(v) for k, v in m.items() if k != "ticker")
    assert m["market_cap"] > 0 and m["beta"] is not None
    # Derived from the demo statements — present, not asserted numerically.
    for key in ("pe_ttm", "ev_ebitda", "ev_revenue", "gross_margin", "op_margin",
                "fcf_margin", "roe", "debt_to_ebitda", "revenue_growth_yoy"):
        assert isinstance(m[key], float), key
    assert "roic" in m and _is_number_or_none(m["roic"])      # shape only
    # Documented v1 gaps stay None until estimates / DPS parsing land.
    assert m["forward_pe"] is None and m["peg"] is None and m["dividend_yield"] is None


def test_compute_metrics_unknown_ticker_returns_none():
    assert sms.compute_metrics("ZZZNOPE") is None


def test_company_without_financials_does_not_raise_and_is_skipped():
    ticker = "TSTNOFIN"
    with SessionLocal() as db:
        db.merge(Company(
            ticker=ticker, company_name="No Financials Inc", sector="Industrials",
            industry="Test", market_cap=1.5e9, beta=1.1, universe_tier="auto_analysis",
        ))
        db.commit()
    try:
        m = sms.compute_metrics(ticker)
        assert m is not None and set(m) == METRIC_KEYS
        assert all(m[k] is None for k in DERIVED_KEYS)
        assert m["market_cap"] == 1.5e9 and m["beta"] == 1.1
        report = sms.snapshot_universe()
        assert report["skipped"] >= 1
        with SessionLocal() as db:
            assert db.get(ScreenerMetric, ticker) is None   # never written
    finally:
        with SessionLocal() as db:
            db.query(ScreenerMetric).filter_by(ticker=ticker).delete()
            db.query(Company).filter_by(ticker=ticker).delete()
            db.commit()


# ---------------------------------------------------------------------------
# snapshot_universe
# ---------------------------------------------------------------------------

def test_snapshot_universe_writes_rows_and_upserts_on_rerun():
    with SessionLocal() as db:
        universe = db.execute(
            select(Company.ticker).where(Company.universe_tier == "auto_analysis")
        ).scalars().all()
    first = sms.snapshot_universe()
    assert set(first) == {"written", "skipped", "missing_data"}
    assert first["written"] >= 1
    assert first["written"] + first["skipped"] + first["missing_data"] == len(universe)

    with SessionLocal() as db:
        row = db.get(ScreenerMetric, DEMO)
        assert row is not None and isinstance(row.pe_ttm, float)
        assert row.last_updated is not None
        n_rows = db.query(ScreenerMetric).count()
        stamp = row.last_updated

    second = sms.snapshot_universe()
    assert second["written"] == first["written"]
    with SessionLocal() as db:
        assert db.query(ScreenerMetric).count() == n_rows            # upsert, not insert
        assert db.get(ScreenerMetric, DEMO).last_updated >= stamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Row:
    def __init__(self, line_item: str, value, period_end: date) -> None:
        self.line_item, self.value, self.period_end = line_item, value, period_end


def test_latest_period_value_skips_nulls_in_date_desc_order():
    rows = [
        _Row("revenue", None, date(2025, 6, 30)),
        _Row("revenue", 200.0, date(2024, 6, 30)),
        _Row("revenue", 100.0, date(2023, 6, 30)),
        _Row("net_income", 5.0, date(2025, 6, 30)),
    ]
    assert sms._latest_period_value(rows, "revenue") == 200.0
    assert sms._latest_period_value(rows, "net_income") == 5.0
    assert sms._latest_period_value(rows, "capex") is None


def test_yoy_growth_needs_two_values_and_a_nonzero_base():
    d = date(2025, 6, 30)
    assert sms._yoy_growth([_Row("revenue", 120.0, d), _Row("revenue", 100.0, d)], "revenue") == 0.2
    assert sms._yoy_growth([_Row("revenue", 120.0, d)], "revenue") is None
    assert sms._yoy_growth([_Row("revenue", 120.0, d), _Row("revenue", 0.0, d)], "revenue") is None
    assert sms._yoy_growth([_Row("revenue", 90.0, d), _Row("revenue", -100.0, d)], "revenue") == 1.9
