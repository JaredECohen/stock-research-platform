"""FEAT-001 — `build_series` on directly seeded `FinancialPeriod` rows.

Seeds rows for throwaway tickers (FXA/FXB/…) so no other test's demo
backfill can change a value under us; the one demo-ticker test (the P/E
tie to `comps_history`) backfills NVDA itself first. Prices come from a
monkeypatched `market_data_service.get_price_series`; the clock is the
module's `_utcnow()` seam. Nothing here touches a provider or an LLM.
"""
from __future__ import annotations

import logging
import math
from calendar import monthrange
from datetime import date, datetime

import pytest

from app.database import SessionLocal
from app.finance import comps_history as ch
from app.finance import ratios as R
from app.models import Company, FinancialPeriod
from app.services import fundamentals_catalog as C
from app.services import fundamentals_series_service as fss
from app.services import history_service, market_data_service
from app.tests.factories import make_financial_period, seed_annual_periods

NOW = datetime(2025, 6, 1, 12, 0, 0)
FRESH = datetime(2025, 5, 30)   # fetched_at two days before NOW

FULL = {
    "revenue": 1000.0, "gross_profit": 600.0, "operating_income": 250.0, "net_income": 180.0,
    "ebitda": 300.0, "depreciation_and_amortization": 50.0, "eps_diluted": 1.8,
    "cash_from_operations": 320.0, "capex": -70.0, "free_cash_flow": 250.0,
    "stock_based_compensation": 40.0, "pretax_income": 240.0, "tax_expense": 60.0,
    "total_debt": 400.0, "short_term_debt": 80.0, "long_term_debt": 320.0,
    "shareholders_equity": 600.0, "cash_and_equivalents": 150.0, "short_term_investments": 50.0,
    "weighted_avg_shares_diluted": 100.0,
}


def _scaled(factor: float) -> dict[str, float]:
    return {k: v * factor for k, v in FULL.items()}


def _close(a, b):
    return a is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=0)


@pytest.fixture
def db():
    with SessionLocal() as session:
        history_service._ensure_tables(session)
        yield session


@pytest.fixture
def clean(db):
    """Remove every row of the throwaway tickers before and after a test."""
    tickers = ("FXA", "FXB", "FXC", "FXQ", "FXS", "FXV")

    def _wipe():
        db.query(FinancialPeriod).filter(FinancialPeriod.ticker.in_(tickers)).delete(
            synchronize_session=False)
        db.query(Company).filter(Company.ticker.in_(tickers)).delete(synchronize_session=False)
        db.commit()

    _wipe()
    yield
    _wipe()


@pytest.fixture
def clock(monkeypatch):
    monkeypatch.setattr(fss, "_utcnow", lambda: NOW)
    return NOW


@pytest.fixture
def no_prices(monkeypatch):
    calls: list[tuple[str, int]] = []

    def fake(ticker, days=252):
        calls.append((ticker, days))
        return []

    monkeypatch.setattr(market_data_service, "get_price_series", fake)
    return calls


def _series(resp, ticker, metric):
    return next(s for s in resp.series if s.ticker == ticker and s.metric == metric)


def _values(resp, ticker, metric):
    return [(p.period, p.value, p.reason) for p in _series(resp, ticker, metric).points]


# ---------------------------------------------------------------------------
# Axis, ordering, gaps
# ---------------------------------------------------------------------------

def test_ordering_and_common_axis_with_gap_rows(db, clean, clock):
    seed_annual_periods(db, "FXA", {fy: _scaled(1 + 0.1 * i) for i, fy in enumerate((2021, 2022, 2023, 2024))},
                        fetched_at=FRESH)
    seed_annual_periods(db, "FXB", {2022: _scaled(2.0), 2024: _scaled(2.5)}, fetched_at=FRESH)
    db.commit()

    resp = fss.build_series(["fxb", "FXA", "fxa"], ["revenue", "gross_margin"])
    assert resp.periods == ["FY2021", "FY2022", "FY2023", "FY2024"]
    assert resp.limits.applied.model_dump() == {"companies": 2, "metrics": 2, "years": None}
    assert not resp.limits.capped_by_plan and not resp.unavailable
    # Dedupe keeps first appearance; series are ticker-major in request order.
    assert [(s.ticker, s.metric) for s in resp.series] == [
        ("FXB", "revenue"), ("FXB", "gross_margin"), ("FXA", "revenue"), ("FXA", "gross_margin"),
    ]
    assert [p.period for p in _series(resp, "FXA", "revenue").points] == resp.periods
    assert [v for _, v, _ in _values(resp, "FXA", "revenue")] == pytest.approx([1000.0, 1100.0, 1200.0, 1300.0])
    # FXB has no FY2021 / FY2023 rows: the slots exist on the shared axis,
    # empty with a reason, never zero.
    assert _values(resp, "FXB", "revenue") == [
        ("FY2021", None, "missing_line"), ("FY2022", 2000.0, None),
        ("FY2023", None, "missing_line"), ("FY2024", 2500.0, None),
    ]
    cov = _series(resp, "FXB", "revenue").coverage
    assert cov.model_dump() == {"first": "FY2022", "last": "FY2024", "n": 2, "expected": 4}
    assert _series(resp, "FXA", "revenue").currency == "USD"
    assert _series(resp, "FXA", "gross_margin").currency is None


def test_years_trims_the_axis_but_growth_still_sees_the_prior_year(db, clean, clock):
    seed_annual_periods(db, "FXA", {2021: _scaled(1.0), 2022: _scaled(1.5), 2023: _scaled(1.8)},
                        fetched_at=FRESH)
    db.commit()
    resp = fss.build_series(["FXA"], ["revenue_growth_yoy"], years=2)
    assert resp.periods == ["FY2022", "FY2023"]
    assert resp.limits.applied.years == 2
    vals = _values(resp, "FXA", "revenue_growth_yoy")
    assert vals[0][0] == "FY2022" and _close(vals[0][1], 0.5)
    assert _close(vals[1][1], 0.2)
    # Without the year before, the first point is honestly missing.
    full = fss.build_series(["FXA"], ["revenue_growth_yoy"])
    assert _values(full, "FXA", "revenue_growth_yoy")[0] == ("FY2021", None, "missing_line")


def test_growth_needs_the_immediately_preceding_year(db, clean, clock):
    seed_annual_periods(db, "FXA", {2021: _scaled(1.0), 2023: _scaled(1.8)}, fetched_at=FRESH)
    db.commit()
    resp = fss.build_series(["FXA"], ["revenue_growth_yoy"])
    assert _values(resp, "FXA", "revenue_growth_yoy") == [
        ("FY2021", None, "missing_line"), ("FY2022", None, "missing_line"),
        ("FY2023", None, "missing_line"),
    ]


# ---------------------------------------------------------------------------
# Every reason code is reachable
# ---------------------------------------------------------------------------

def test_reason_codes_reachable(db, clean, clock, no_prices):
    seed_annual_periods(db, "FXA", {
        2022: {**FULL, "revenue": -50.0},                   # prior base <= 0
        2023: {**FULL, "revenue": 0.0, "gross_profit": 10.0},  # denominator <= 0
        2024: {k: v for k, v in FULL.items() if k != "gross_profit"},  # line missing
    }, fetched_at=FRESH)
    # FXS: a dated period, a price, but no share count anywhere.
    seed_annual_periods(db, "FXS", {2024: {k: v for k, v in FULL.items()
                                          if k != "weighted_avg_shares_diluted"}}, fetched_at=FRESH)
    db.commit()

    resp = fss.build_series(["FXA", "FXS", "FXC"], ["revenue_growth_yoy", "gross_margin", "pe_ttm"])
    assert _values(resp, "FXA", "revenue_growth_yoy")[1] == ("FY2023", None, "base_nonpositive")
    assert _values(resp, "FXA", "gross_margin")[1] == ("FY2023", None, "denominator_nonpositive")
    assert _values(resp, "FXA", "gross_margin")[2] == ("FY2024", None, "missing_line")
    # No price rows at all → no_price on every market point, and the price
    # fetch was attempted once per ticker with rows (never for FXC).
    assert all(r == "no_price" for _, _, r in _values(resp, "FXA", "pe_ttm"))
    assert sorted(t for t, _ in no_prices) == ["FXA", "FXS"]
    # FXC has no rows: listed as unavailable with the research remedy,
    # and its points say so rather than being dropped.
    assert [u.model_dump() for u in resp.unavailable] == [
        {"ticker": "FXC", "reason": "not_backfilled", "remedy": fss.NOT_BACKFILLED_REMEDY},
    ]
    assert all(r == "not_backfilled" for _, _, r in _values(resp, "FXC", "gross_margin"))
    assert _series(resp, "FXC", "gross_margin").provenance.model_dump() == {
        "source": "", "fetched_at": None, "stale": False, "stale_reason": None,
    }


def test_no_shares_when_neither_period_nor_company_has_a_count(db, clean, clock, monkeypatch):
    seed_annual_periods(db, "FXS", {2024: {k: v for k, v in FULL.items()
                                          if k != "weighted_avg_shares_diluted"}}, fetched_at=FRESH)
    db.commit()
    monkeypatch.setattr(market_data_service, "get_price_series",
                        lambda t, d=252: [{"date": "2024-12-31", "close": 50.0}])
    resp = fss.build_series(["FXS"], ["pe_ttm"])
    assert _values(resp, "FXS", "pe_ttm") == [("FY2024", None, "no_shares")]


def test_price_fetch_failure_degrades_to_no_price_with_a_warning(db, clean, clock, monkeypatch):
    seed_annual_periods(db, "FXA", {2024: FULL}, fetched_at=FRESH)
    db.commit()

    def boom(t, d=252):
        raise RuntimeError("provider down apikey=sk-secret123456")

    monkeypatch.setattr(market_data_service, "get_price_series", boom)
    resp = fss.build_series(["FXA"], ["pe_ttm", "revenue"])
    assert _values(resp, "FXA", "pe_ttm") == [("FY2024", None, "no_price")]
    assert _values(resp, "FXA", "revenue") == [("FY2024", 1000.0, None)]
    assert resp.warnings == ["FXA: price history unavailable; market-derived metrics show no_price"]


# ---------------------------------------------------------------------------
# Estimated flags on the documented fallbacks
# ---------------------------------------------------------------------------

def test_estimated_flags(db, clean, clock, monkeypatch):
    rows = {k: v for k, v in FULL.items()
            if k not in ("ebitda", "free_cash_flow", "tax_expense", "weighted_avg_shares_diluted")}
    # Undated period → the fiscal year end is estimated from the company's FYE.
    seed_annual_periods(db, "FXA", {2024: rows}, period_end_month=None, fetched_at=FRESH)
    db.add(Company(ticker="FXA", company_name="FXA Corp", sector="Technology", industry="Software",
                   fiscal_year_end="June", shares_outstanding=200.0))
    db.commit()

    def prices(t, d=252):
        return [{"date": "2024-06-28", "close": 40.0}, {"date": "2024-07-01", "close": 99.0}]

    monkeypatch.setattr(market_data_service, "get_price_series", prices)
    resp = fss.build_series(["FXA"], ["ebitda", "free_cash_flow", "roic", "pe_ttm"])
    by = {s.metric: s.points[0] for s in resp.series}
    assert by["ebitda"].estimated and _close(by["ebitda"].value, 300.0)
    assert by["free_cash_flow"].estimated and _close(by["free_cash_flow"].value, 250.0)
    assert by["roic"].estimated and _close(by["roic"].value, 250.0 * 0.79 / 1000.0)
    # Estimated shares (company count) at an estimated June-30 period end,
    # priced off the 06-28 close — not the July one.
    assert by["pe_ttm"].estimated and _close(by["pe_ttm"].value, 40.0 * 200.0 / 180.0)
    assert by["pe_ttm"].period_end is None   # the stored date stays honest


def test_reported_values_are_not_estimated_on_undated_periods(db, clean, clock):
    seed_annual_periods(db, "FXA", {2024: FULL}, period_end_month=None, fetched_at=FRESH)
    db.commit()
    resp = fss.build_series(["FXA"], ["revenue", "fcf_after_sbc"])
    for s in resp.series:
        assert s.points[0].estimated is False and s.points[0].period_end is None


# ---------------------------------------------------------------------------
# fcf_after_sbc and ROIC on stored rows
# ---------------------------------------------------------------------------

def test_fcf_after_sbc_arithmetic_on_stored_rows(db, clean, clock):
    seed_annual_periods(db, "FXA", {
        2023: {"free_cash_flow": 500.0, "stock_based_compensation": 120.0},
        2024: {"free_cash_flow": 500.0},   # SBC missing → not zero
    }, fetched_at=FRESH)
    db.commit()
    resp = fss.build_series(["FXA"], ["fcf_after_sbc"])
    assert _values(resp, "FXA", "fcf_after_sbc") == [
        ("FY2023", 380.0, None), ("FY2024", None, "missing_line"),
    ]
    assert _series(resp, "FXA", "fcf_after_sbc").unit_type == "currency"


def test_roic_delegates_to_ratios_and_none_becomes_a_reason(db, clean, clock, monkeypatch):
    seed_annual_periods(db, "FXA", {2024: FULL}, fetched_at=FRESH)
    db.commit()
    resp = fss.build_series(["FXA"], ["roic"])
    assert _close(_values(resp, "FXA", "roic")[0][1], R.roic(FULL, FULL))

    monkeypatch.setattr(R, "roic_with_provenance", lambda i, b, tax_rate=None: (None, R.ROIC_UNKNOWN_LOSS_MAKER))
    resp = fss.build_series(["FXA"], ["roic"])
    assert _values(resp, "FXA", "roic") == [("FY2024", None, "missing_line")]


# ---------------------------------------------------------------------------
# Staleness and provenance
# ---------------------------------------------------------------------------

def test_stale_when_last_period_is_older_than_15_months(db, clean, clock):
    # FY2023 ending 2023-12-31 is 17 months before NOW (2025-06-01).
    seed_annual_periods(db, "FXA", {2023: FULL}, fetched_at=FRESH, source="fmp")
    db.commit()
    prov = _series(fss.build_series(["FXA"], ["revenue"]), "FXA", "revenue").provenance
    assert prov.stale and "FY2023" in prov.stale_reason and "15 months" in prov.stale_reason
    assert prov.source == "fmp" and prov.fetched_at == FRESH


def test_stale_when_fetched_at_is_older_than_400_days(db, clean, clock):
    # FY2024 ending 2024-12-31 is 5 months old — fresh — but the rows were
    # fetched 500 days ago.
    seed_annual_periods(db, "FXA", {2024: FULL}, fetched_at=datetime(2024, 1, 15))
    db.commit()
    prov = _series(fss.build_series(["FXA"], ["revenue"]), "FXA", "revenue").provenance
    assert prov.stale and "400 days" in prov.stale_reason and "15 months" not in prov.stale_reason


def test_fresh_rows_are_not_stale_and_sources_aggregate(db, clean, clock):
    seed_annual_periods(db, "FXA", {2023: FULL}, fetched_at=datetime(2024, 3, 1), source="alpha_vantage")
    seed_annual_periods(db, "FXA", {2024: FULL}, fetched_at=FRESH, source="fmp")
    db.commit()
    prov = _series(fss.build_series(["FXA"], ["revenue"]), "FXA", "revenue").provenance
    assert prov.model_dump() == {"source": "alpha_vantage+fmp", "fetched_at": FRESH,
                                 "stale": False, "stale_reason": None}


def test_stale_threshold_constants_live_in_the_catalog():
    assert C.STALE_LAST_PERIOD_MONTHS == 15 and C.STALE_FETCHED_AT_DAYS == 400


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_stable_across_calls_and_changes_with_a_value(db, clean, clock):
    seed_annual_periods(db, "FXA", {2023: FULL, 2024: _scaled(1.2)}, fetched_at=FRESH)
    db.commit()
    a = fss.build_series(["FXA"], ["revenue", "net_margin"])
    b = fss.build_series(["FXA"], ["revenue", "net_margin"])
    assert a.fingerprint == b.fingerprint and len(a.fingerprint) == 64

    # A re-fetch that changes nothing but fetched_at must not move it.
    db.query(FinancialPeriod).filter(FinancialPeriod.ticker == "FXA").update(
        {"fetched_at": datetime(2025, 5, 31)}, synchronize_session=False)
    db.commit()
    assert fss.build_series(["FXA"], ["revenue", "net_margin"]).fingerprint == a.fingerprint

    row = db.query(FinancialPeriod).filter(FinancialPeriod.ticker == "FXA",
                                           FinancialPeriod.fiscal_year == 2024,
                                           FinancialPeriod.line_item == "revenue").one()
    row.value = row.value + 1.0
    db.commit()
    assert fss.build_series(["FXA"], ["revenue", "net_margin"]).fingerprint != a.fingerprint
    # Selection shape is part of it too.
    assert fss.build_series(["FXA"], ["revenue"]).fingerprint != a.fingerprint
    assert fss.build_series(["FXA"], ["revenue", "net_margin"], normalize="indexed").fingerprint != a.fingerprint


# ---------------------------------------------------------------------------
# Indexed view
# ---------------------------------------------------------------------------

def test_indexed_rebases_currency_only_and_refuses_nonpositive_bases(db, clean, clock):
    seed_annual_periods(db, "FXA", {2022: _scaled(1.0), 2023: _scaled(1.5), 2024: _scaled(2.0)},
                        fetched_at=FRESH)
    seed_annual_periods(db, "FXB", {2022: {**FULL, "net_income": -10.0}, 2023: {**FULL, "net_income": 20.0}},
                        fetched_at=FRESH)
    db.commit()
    resp = fss.build_series(["FXA", "FXB"], ["revenue", "gross_margin", "net_income"], normalize="indexed")
    assert resp.normalize == "indexed"
    rev = _series(resp, "FXA", "revenue")
    assert rev.indexed and [p.value for p in rev.points] == pytest.approx([100.0, 150.0, 200.0])
    gm = _series(resp, "FXA", "gross_margin")
    assert not gm.indexed and _close(gm.points[0].value, 0.6)
    # FXB's first valued net income is a loss: no rebase, and the gap row
    # keeps its own reason.
    ni = _series(resp, "FXB", "net_income")
    assert not ni.indexed
    assert [(p.value, p.reason) for p in ni.points] == [
        (None, "base_nonpositive"), (None, "base_nonpositive"), (None, "missing_line"),
    ]


# ---------------------------------------------------------------------------
# Provider label shapes: what the live fallback chain actually writes
# ---------------------------------------------------------------------------

def _av_report(date_str: str, **fields):
    """One AlphaVantage `annualReports` entry (strings, like the API)."""
    base = {"fiscalDateEnding": date_str, "reportedCurrency": "USD"}
    base.update({k: str(v) for k, v in fields.items()})
    return base


def _seed_legacy_av_year(db, ticker: str, date_str: str, scale: float) -> None:
    """Seed the historical malformed shape for read compatibility.

    New history ingestion rejects date strings as fiscal labels. This fixture
    deliberately models rows written before that guard instead of requiring the
    current producer to keep creating malformed fiscal years.
    """
    from app.providers.alpha_vantage_provider import AlphaVantageProvider as AV
    income = AV._income_row(_av_report(
        date_str, totalRevenue=1000 * scale, grossProfit=600 * scale,
        operatingIncome=250 * scale, netIncome=180 * scale, ebitda=300 * scale,
        incomeBeforeTax=240 * scale, incomeTaxExpense=60 * scale,
    ))
    balance = AV._balance_row(_av_report(
        date_str, totalShareholderEquity=600 * scale, shortLongTermDebtTotal=400 * scale,
        cashAndCashEquivalentsAtCarryingValue=150 * scale, shortTermInvestments=50 * scale,
    ))
    cash = AV._cash_row(_av_report(
        date_str, operatingCashflow=320 * scale, capitalExpenditures=70 * scale,
        stockBasedCompensation=40 * scale,
    ))
    # Sanity: these are the shapes the finding describes.
    assert income["period"].endswith("Q4") and balance["period"] == date_str
    for statement, rows, lines in (
        ("income", [income], history_service._INCOME_LINES),
        ("balance", [balance], history_service._BALANCE_LINES),
        ("cash", [cash], history_service._CASH_LINES),
    ):
        for row in rows:
            for line in lines:
                if row.get(line) is not None:
                    db.add(FinancialPeriod(ticker=ticker, period=row["period"], period_end=date.fromisoformat(date_str),
                        fiscal_year=int(date_str[:4]) if statement == "income" else int(date_str.replace("-", "")),
                        fiscal_quarter=4 if statement == "income" else None, statement=statement, line_item=line,
                        value=row[line], currency="USD", source="alpha_vantage", fetched_at=FRESH))


def test_alpha_vantage_shaped_annual_rows_are_annual_not_not_backfilled(db, clean, clock, caplog):
    """The high finding: AV labels annual income rows `2024Q4` (fiscal_quarter=4)
    and balance/cash rows `2024-12-31` (fiscal_year=20241231). They are the
    annual data and must render, not collapse into a false `not_backfilled`."""
    _seed_legacy_av_year(db, "FXV", "2023-12-31", 1.0)
    _seed_legacy_av_year(db, "FXV", "2024-12-31", 1.1)
    db.commit()
    stored = db.query(FinancialPeriod).filter(FinancialPeriod.ticker == "FXV").all()
    assert {r.period for r in stored} == {"2023Q4", "2024Q4", "2023-12-31", "2024-12-31"}
    assert {r.fiscal_quarter for r in stored} == {4, None}

    with caplog.at_level(logging.WARNING, logger="app.services.fundamentals_series_service"):
        resp = fss.build_series(["FXV"], ["revenue", "free_cash_flow", "fcf_after_sbc", "net_debt"])
    assert resp.periods == ["FY2023", "FY2024"]
    assert resp.unavailable == [] and resp.warnings == []
    assert not [r for r in caplog.records if "FXV" in r.getMessage()]
    rev = _values(resp, "FXV", "revenue")
    assert rev[0] == ("FY2023", 1000.0, None) and _close(rev[1][1], 1100.0)
    fcf = _series(resp, "FXV", "free_cash_flow")
    assert _close(fcf.points[0].value, 250.0) and fcf.points[0].period_end == date(2023, 12, 31)
    assert not fcf.points[0].estimated  # a stored date, not an estimated FYE
    assert _close(_series(resp, "FXV", "fcf_after_sbc").points[0].value, 210.0)
    assert _close(_series(resp, "FXV", "net_debt").points[0].value, 400.0 - 150.0 - 50.0)
    assert fcf.provenance.source == "alpha_vantage" and not fcf.provenance.stale


def test_fmp_month_derived_fallback_label_is_annual(db, clean, clock):
    """FMP's `_period_label` falls back to `2024Q4` when the API omits
    `period` — still an annual statement (the endpoint defaults to annual)."""
    from app.providers.fmp_provider import FMPProvider
    label = FMPProvider._period_label("2024-12-31", None)
    assert label == "2024Q4"
    seed_annual_periods(db, "FXQ", {2023: FULL}, fetched_at=FRESH)
    db.add(make_financial_period("FXQ", 2024, "revenue", 1200.0, period=label, fiscal_quarter=4,
                                 period_end=date(2024, 12, 31), fetched_at=FRESH))
    db.commit()
    resp = fss.build_series(["FXQ"], ["revenue", "revenue_growth_yoy"])
    assert resp.periods == ["FY2023", "FY2024"] and resp.warnings == []
    assert _values(resp, "FXQ", "revenue")[1] == ("FY2024", 1200.0, None)
    assert _close(_series(resp, "FXQ", "revenue_growth_yoy").points[1].value, 0.2)


def test_real_quarterly_signature_is_excluded_with_a_warning(db, clean, clock, caplog):
    """Several distinct periods for one line in one fiscal year is quarterly
    data whatever the labels say: excluded, counted, and echoed in `warnings`."""
    seed_annual_periods(db, "FXQ", {2023: FULL}, fetched_at=FRESH)
    for q, month in ((1, 3), (2, 6), (3, 9), (4, 12)):
        db.add(make_financial_period("FXQ", 2024, "revenue", 300.0, period=f"2024Q{q}", fiscal_quarter=q,
                                     period_end=date(2024, month, monthrange(2024, month)[1]),
                                     fetched_at=FRESH))
    db.commit()

    with caplog.at_level(logging.WARNING, logger="app.services.fundamentals_series_service"):
        resp = fss.build_series(["FXQ"], ["revenue"])
    assert resp.periods == ["FY2023"]
    assert _values(resp, "FXQ", "revenue") == [("FY2023", 1000.0, None)]
    assert resp.warnings == [
        "FXQ: 4 stored row(s) excluded from annual output (quarterly-labelled=4, ambiguous=0); "
        "some fiscal years may show missing_line"
    ]
    records = [r for r in caplog.records if "excluded" in r.getMessage() and "FXQ" in r.getMessage()]
    assert len(records) == 1 and records[0].levelno == logging.WARNING
    assert "excluded 4 non-annual row(s)" in records[0].getMessage()
    assert "quarterly-labelled=4" in records[0].getMessage()


def test_stray_quarter_months_from_the_fiscal_year_end_is_excluded(db, clean, clock):
    """A lone `2025Q2` row (June end) for a December-FYE company is a quarter
    that the annual filing has not caught up with, not FY2025."""
    db.add(Company(ticker="FXQ", company_name="FXQ Corp", sector="Technology", industry="Software",
                   fiscal_year_end="December"))
    seed_annual_periods(db, "FXQ", {2023: FULL, 2024: FULL}, fetched_at=FRESH)
    db.add(make_financial_period("FXQ", 2025, "revenue", 260.0, period="2025Q2", fiscal_quarter=2,
                                 period_end=date(2025, 6, 30), fetched_at=FRESH))
    db.commit()
    resp = fss.build_series(["FXQ"], ["revenue"])
    assert resp.periods == ["FY2023", "FY2024"]
    assert len(resp.warnings) == 1 and "quarterly-labelled=1" in resp.warnings[0]


def test_stray_quarter_uses_the_modal_month_when_the_profile_is_missing(db, clean, clock):
    # No Company row: the reference month comes from the other rows.
    seed_annual_periods(db, "FXQ", {2022: FULL, 2023: FULL, 2024: FULL}, fetched_at=FRESH)
    db.add(make_financial_period("FXQ", 2025, "revenue", 260.0, period="2025Q1", fiscal_quarter=1,
                                 period_end=date(2025, 3, 31), fetched_at=FRESH))
    db.commit()
    resp = fss.build_series(["FXQ"], ["revenue"])
    assert resp.periods == ["FY2022", "FY2023", "FY2024"]
    assert len(resp.warnings) == 1 and "quarterly-labelled=1" in resp.warnings[0]


def test_fifty_two_week_year_drift_across_a_month_boundary_is_still_annual(db, clean, clock):
    """Retail fiscal years end on the Saturday nearest 31 Jan, so consecutive
    years land in January and February: within tolerance, never a quarter."""
    db.add(Company(ticker="FXQ", company_name="FXQ Retail", sector="Consumer", industry="Retail",
                   fiscal_year_end="January"))
    db.add(make_financial_period("FXQ", 2024, "revenue", 500.0, period="2024Q1", fiscal_quarter=1,
                                 period_end=date(2024, 2, 3), fetched_at=FRESH))
    db.add(make_financial_period("FXQ", 2025, "revenue", 550.0, period="2025Q1", fiscal_quarter=1,
                                 period_end=date(2025, 1, 31), fetched_at=FRESH))
    db.commit()
    resp = fss.build_series(["FXQ"], ["revenue"])
    assert resp.warnings == []
    assert _values(resp, "FXQ", "revenue") == [("FY2024", 500.0, None), ("FY2025", 550.0, None)]


def test_fy_label_wins_over_a_duplicate_provider_label_without_a_warning(db, clean, clock, caplog):
    """FMP (`FY2024`) and AlphaVantage (`2024Q4`) both wrote FY2024 revenue:
    the explicit label is used, the other is a duplicate — not a mislabelled feed."""
    seed_annual_periods(db, "FXQ", {2024: FULL}, fetched_at=FRESH)
    db.add(make_financial_period("FXQ", 2024, "revenue", 999.0, period="2024Q4", fiscal_quarter=4,
                                 period_end=date(2024, 12, 31), source="alpha_vantage", fetched_at=FRESH))
    db.commit()
    with caplog.at_level(logging.WARNING, logger="app.services.fundamentals_series_service"):
        resp = fss.build_series(["FXQ"], ["revenue"])
    assert _values(resp, "FXQ", "revenue") == [("FY2024", 1000.0, None)]
    assert resp.warnings == []
    assert not [r for r in caplog.records if "FXQ" in r.getMessage()]
    assert _series(resp, "FXQ", "revenue").provenance.source == "test"


def test_undatable_row_is_ambiguous_and_counted(db, clean, clock, caplog):
    seed_annual_periods(db, "FXQ", {2023: FULL}, fetched_at=FRESH)
    # Q-shaped label, no period_end, no usable fiscal_year column: nothing
    # says which fiscal year this belongs to.
    row = make_financial_period("FXQ", 2024, "net_income", 1.0, period="garbage", fiscal_quarter=None,
                                fetched_at=FRESH)
    row.fiscal_year = None
    db.add(row)
    db.commit()
    with caplog.at_level(logging.WARNING, logger="app.services.fundamentals_series_service"):
        resp = fss.build_series(["FXQ"], ["revenue", "net_income"])
    assert resp.periods == ["FY2023"]
    assert resp.warnings == [
        "FXQ: 1 stored row(s) excluded from annual output (quarterly-labelled=0, ambiguous=1); "
        "some fiscal years may show missing_line"
    ]
    assert "ambiguous=1" in caplog.records[-1].getMessage()


def test_all_rows_excluded_is_unavailable_with_an_honest_remedy(db, clean, clock):
    for q, month in ((1, 3), (2, 6)):
        db.add(make_financial_period("FXQ", 2024, "revenue", 300.0, period=f"2024Q{q}", fiscal_quarter=q,
                                     period_end=date(2024, month, 30), fetched_at=FRESH))
    db.commit()
    resp = fss.build_series(["FXQ"], ["revenue"])
    assert resp.periods == []
    assert [u.reason for u in resp.unavailable] == ["not_backfilled"]
    assert resp.unavailable[0].remedy == fss.NOT_ANNUAL_REMEDY.format(excluded=2)
    assert "none of it is annual-shaped (2 row(s) excluded)" in resp.unavailable[0].remedy
    assert len(resp.warnings) == 1


def test_annual_label_year_accepts_fy_and_bare_year_only():
    assert fss._annual_label_year("FY2024", 2024, None) == 2024
    assert fss._annual_label_year("2024", 2024, None) == 2024
    assert fss._annual_label_year("2024", None, None) == 2024
    assert fss._annual_label_year("2024Q3", 2024, 3) is None
    assert fss._annual_label_year("2024Q3", 2024, None) is None
    assert fss._annual_label_year("FY2024", 2024, 4) is None
    assert fss._annual_label_year("FY2023", 2024, None) is None
    assert fss._annual_label_year("garbage", None, None) is None


def test_fiscal_year_of_derives_from_label_then_date_then_column():
    assert fss._fiscal_year_of("FY2024", None, 2024, None) == (2024, True)
    assert fss._fiscal_year_of("2024", None, None, None) == (2024, True)
    # AV income: Q-label + quarter column + date → the date's year, not explicit.
    assert fss._fiscal_year_of("2024Q4", date(2024, 12, 31), 2024, 4) == (2024, False)
    # AV balance/cash: date label parsed into a nonsense fiscal_year column.
    assert fss._fiscal_year_of("2024-12-31", date(2024, 12, 31), 20241231, None) == (2024, False)
    # January FYE: the fiscal year is the calendar year the period ended in.
    assert fss._fiscal_year_of("2025Q1", date(2025, 1, 31), 2025, 1) == (2025, False)
    # No date: a plausible fiscal_year column is the last resort.
    assert fss._fiscal_year_of("2024Q4", None, 2024, 4) == (2024, False)
    assert fss._fiscal_year_of("2024-12-31", None, 20241231, None) == (None, False)
    assert fss._fiscal_year_of("garbage", None, None, None) == (None, False)


def test_month_distance_is_circular():
    assert fss._month_distance(12, 1) == 1
    assert fss._month_distance(1, 12) == 1
    assert fss._month_distance(3, 12) == 3
    assert fss._month_distance(6, 6) == 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_unknown_metric_and_empty_selection_raise(db, clean):
    with pytest.raises(ValueError):
        fss.build_series(["FXA"], ["revenue", "bogus"])
    with pytest.raises(ValueError):
        fss.build_series([], ["revenue"])
    with pytest.raises(ValueError):
        fss.build_series(["FXA"], ["revenue"], normalize="log")
    with pytest.raises(ValueError):
        fss.build_series(["FXA"], ["revenue"], years=0)


def test_empty_store_is_a_valid_empty_response(db, clean, clock):
    resp = fss.build_series(["FXA"], ["revenue"], years=5, capped_by_plan=True)
    assert resp.periods == [] and resp.limits.capped_by_plan
    assert [u.ticker for u in resp.unavailable] == ["FXA"]
    assert _series(resp, "FXA", "revenue").points == []
    assert resp.catalog_version == C.CATALOG_VERSION and resp.as_of == NOW


# ---------------------------------------------------------------------------
# Regression tie to shipped code: P/E for a demo ticker == comps_history
# ---------------------------------------------------------------------------

def test_market_pe_matches_comps_history_recomputation_for_demo_ticker(monkeypatch):
    from app.tests.fixtures.seed_demo_data import seed_companies
    seed_companies()
    history_service.backfill_ticker("NVDA")

    with SessionLocal() as db:
        company = db.get(Company, "NVDA")
        assert company is not None
    # Demo rows carry no period_end, so the series service estimates the
    # fiscal year end from the company's FYE month (or December).
    month = fss._MONTHS.get((company.fiscal_year_end or "").strip().lower(), 12)
    fy = 2024
    period_end = date(fy, month, monthrange(fy, month)[1])

    prices = [
        {"date": date(fy - 1, 12, 20).isoformat(), "close": 90.0},
        {"date": (period_end).isoformat(), "close": 123.45},
        {"date": date(fy, 12, 31).isoformat(), "close": 200.0},
    ]
    if period_end == date(fy, 12, 31):
        prices.pop()
    monkeypatch.setattr(market_data_service, "get_price_series", lambda t, d=252: prices)
    monkeypatch.setattr(fss, "_utcnow", lambda: datetime(2025, 3, 1))

    resp = fss.build_series(["NVDA"], ["pe_ttm", "ev_ebitda", "fcf_yield"], years=1)
    assert resp.periods == [f"FY{fy}"]
    ours = {s.metric: s.points[0] for s in resp.series}
    assert ours["pe_ttm"].value is not None and ours["pe_ttm"].estimated  # estimated period end

    # comps_history's recomputation on the same per-period dict and the
    # same close-on-or-before pairing.
    history = history_service.get_financial_history("NVDA", list(ch._NEEDED_LINES) + [C.SHARES_LINE])
    row: dict[str, float | None] = {}
    for line, entries in history.items():
        for e in entries:
            if e["fiscal_year"] == fy:
                row[line] = e["value"]
    price = ch._closing_price_for(prices, period_end.isoformat())
    assert price == 123.45
    market_cap = price * row[C.SHARES_LINE]
    theirs = ch._recompute_per_period_row(row, market_cap)
    for metric, key in (("pe_ttm", "pe"), ("ev_ebitda", "ev_ebitda"), ("fcf_yield", "fcf_yield")):
        assert theirs[key] is not None
        assert math.isclose(ours[metric].value, theirs[key], rel_tol=1e-12), metric
