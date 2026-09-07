"""ROIC tax-rate resolution (Notion P1: ROIC when tax data is missing).

Before this, `ratios.roic` always deflated operating income by 21% and
the screener re-derived ROIC inline with `tax/pretax if pretax > 0 else
0.21` — so a loss-maker with no usable tax line got a fabricated
negative ROIC that screen rules then ranked on. The rules under test:

- effective rate = tax/pretax, credible only for pretax > 0 and a
  result inside [0, 0.5];
- profitable + no credible rate → 21% statutory fallback, exactly;
- loss-maker + no credible rate → None (never invent a rate);
- loss-maker + credible rate → a real negative ROIC;
- no operating income / no invested capital → None;
- the screener and comps consume the shared definition.
"""
from __future__ import annotations

import math

import pytest

from app.finance import ratios as R


BALANCE = {"total_debt": 400.0, "shareholders_equity": 600.0}  # invested = 1000


def _close(a, b):
    return a is not None and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# effective_tax_rate
# ---------------------------------------------------------------------------

def test_effective_rate_is_tax_over_pretax():
    assert _close(R.effective_tax_rate({"pretax_income": 200.0, "tax_expense": 46.0}), 0.23)


@pytest.mark.parametrize("income", [
    {"pretax_income": 200.0},                      # tax missing
    {"tax_expense": 46.0},                         # pretax missing
    {"pretax_income": None, "tax_expense": 46.0},  # explicit null
    {"pretax_income": 0.0, "tax_expense": 0.0},    # zero denominator
    {"pretax_income": -100.0, "tax_expense": 10.0},   # loss year: not a rate
    {"pretax_income": -100.0, "tax_expense": -21.0},  # loss year with refund
    {"pretax_income": 200.0, "tax_expense": -10.0},   # refund on a profit
    {"pretax_income": 100.0, "tax_expense": 62.0},    # 0.62: above the ceiling
    {"pretax_income": 1.0, "tax_expense": 30.0},      # tiny denominator
])
def test_effective_rate_not_credible(income):
    assert R.effective_tax_rate(income) is None


def test_effective_rate_ceiling_is_inclusive():
    assert _close(R.effective_tax_rate({"pretax_income": 100.0, "tax_expense": 50.0}), 0.50)
    assert R.effective_tax_rate({"pretax_income": 100.0, "tax_expense": 50.01}) is None
    assert _close(R.effective_tax_rate({"pretax_income": 100.0, "tax_expense": 0.0}), 0.0)


# ---------------------------------------------------------------------------
# roic / roic_with_provenance — every branch with explicit numbers
# ---------------------------------------------------------------------------

def test_profitable_with_credible_effective_rate():
    income = {"operating_income": 100.0, "pretax_income": 90.0, "tax_expense": 27.0}  # 30%
    value, source = R.roic_with_provenance(income, BALANCE)
    assert source == R.ROIC_EFFECTIVE_RATE
    assert _close(value, 100.0 * 0.70 / 1000.0)
    assert _close(R.roic(income, BALANCE), 0.07)


def test_profitable_with_missing_tax_fields_uses_statutory_21_exactly():
    income = {"operating_income": 100.0}
    value, source = R.roic_with_provenance(income, BALANCE)
    assert source == R.ROIC_STATUTORY_FALLBACK
    assert R.STATUTORY_TAX_RATE_FALLBACK == 0.21
    assert _close(value, 100.0 * (1 - 0.21) / 1000.0)  # 0.079


def test_profitable_with_incredible_rate_falls_back_to_statutory():
    income = {"operating_income": 100.0, "pretax_income": 100.0, "tax_expense": 62.0}
    value, source = R.roic_with_provenance(income, BALANCE)
    assert source == R.ROIC_STATUTORY_FALLBACK
    assert _close(value, 0.079)


def test_loss_maker_with_credible_rate_gets_negative_roic():
    # Positive pretax despite an operating loss (e.g. large non-operating
    # gain); the effective rate is real, so ROIC is a real negative.
    income = {"operating_income": -50.0, "pretax_income": 20.0, "tax_expense": 5.0}  # 25%
    value, source = R.roic_with_provenance(income, BALANCE)
    assert source == R.ROIC_EFFECTIVE_RATE
    assert _close(value, -50.0 * 0.75 / 1000.0)  # -0.0375


def test_loss_maker_with_refund_is_unknown_not_fabricated():
    income = {"operating_income": -50.0, "pretax_income": -60.0, "tax_expense": -12.0}
    assert R.effective_tax_rate(income) is None
    value, source = R.roic_with_provenance(income, BALANCE)
    assert value is None
    assert source == R.ROIC_UNKNOWN_LOSS_MAKER
    assert R.roic(income, BALANCE) is None


def test_loss_maker_with_incredible_rate_is_unknown():
    income = {"operating_income": -50.0, "pretax_income": 100.0, "tax_expense": 62.0}
    assert R.roic_with_provenance(income, BALANCE) == (None, R.ROIC_UNKNOWN_LOSS_MAKER)


def test_loss_maker_with_no_tax_fields_is_unknown():
    assert R.roic_with_provenance({"operating_income": -50.0}, BALANCE) == (
        None, R.ROIC_UNKNOWN_LOSS_MAKER,
    )


def test_zero_operating_income_with_no_rate_is_unknown():
    # Breakeven is not "profitable": nothing justifies the statutory guess.
    assert R.roic_with_provenance({"operating_income": 0.0}, BALANCE) == (
        None, R.ROIC_UNKNOWN_LOSS_MAKER,
    )


def test_explicit_tax_rate_overrides_resolution():
    income = {"operating_income": -50.0}  # would otherwise be unknown
    value, source = R.roic_with_provenance(income, BALANCE, tax_rate=0.10)
    assert source == R.ROIC_EFFECTIVE_RATE
    assert _close(value, -50.0 * 0.90 / 1000.0)


def test_no_operating_income_is_none():
    assert R.roic_with_provenance({"pretax_income": 90.0, "tax_expense": 27.0}, BALANCE) == (
        None, R.ROIC_NO_OPERATING_INCOME,
    )


@pytest.mark.parametrize("balance", [
    {},
    {"total_debt": 0.0, "shareholders_equity": 0.0},
    {"shareholders_equity": -300.0, "total_debt": 300.0},   # exactly zero
    {"shareholders_equity": -500.0, "short_term_debt": 100.0, "long_term_debt": 100.0},
])
def test_invested_capital_at_or_below_zero_is_none(balance):
    income = {"operating_income": 100.0, "pretax_income": 90.0, "tax_expense": 27.0}
    assert R.roic_with_provenance(income, balance) == (None, R.ROIC_NO_INVESTED_CAPITAL)


def test_invested_capital_falls_back_to_short_plus_long_debt():
    balance = {"short_term_debt": 150.0, "long_term_debt": 250.0, "shareholders_equity": 600.0}
    assert _close(R.invested_capital(balance), 1000.0)
    assert _close(R.roic({"operating_income": 100.0}, balance), 0.079)


# ---------------------------------------------------------------------------
# Screener + comps consume the shared definition
# ---------------------------------------------------------------------------

def _seed_nvda_periods():
    """Company row + long-format financial_periods for NVDA from the demo
    provider, which is what `compute_metrics` reads."""
    from app.services.history_service import backfill_ticker
    from app.tests.fixtures.seed_demo_data import seed_companies

    seed_companies()
    backfill_ticker("NVDA")


def _latest_stored(ticker: str, line_item: str) -> float:
    from app.services.history_service import get_financial_history

    rows = get_financial_history(ticker, line_items=[line_item])[line_item]
    return float(rows[0]["value"])  # newest period first


def _strip_tax_lines(monkeypatch, *, negate_operating_income: bool):
    """Make the screener see NVDA without pretax/tax lines (and optionally
    as a loss-maker) without mutating the shared sqlite rows."""
    from app.services import screener_metrics_service as sms

    real = sms._latest_period_value

    def _patched(rows, line_item):
        if line_item in ("pretax_income", "tax_expense"):
            return None
        v = real(rows, line_item)
        if negate_operating_income and line_item == "operating_income" and v is not None:
            return -v
        return v

    monkeypatch.setattr(sms, "_latest_period_value", _patched)


def test_screener_roic_matches_shared_definition():
    from app.services import screener_metrics_service as sms

    _seed_nvda_periods()
    m = sms.compute_metrics("NVDA")
    assert m is not None and m["roic"] is not None
    income = {k: _latest_stored("NVDA", k)
              for k in ("operating_income", "pretax_income", "tax_expense")}
    balance = {k: _latest_stored("NVDA", k)
               for k in ("total_debt", "shareholders_equity")}
    assert math.isclose(m["roic"], R.roic(income, balance), rel_tol=1e-9)


def test_screener_profitable_ticker_without_tax_lines_gets_statutory_roic(monkeypatch):
    from app.services import screener_metrics_service as sms

    _seed_nvda_periods()
    _strip_tax_lines(monkeypatch, negate_operating_income=False)
    m = sms.compute_metrics("NVDA")
    assert m is not None and m["roic"] is not None
    op = _latest_stored("NVDA", "operating_income")
    invested = _latest_stored("NVDA", "total_debt") + _latest_stored("NVDA", "shareholders_equity")
    expected = op * (1 - R.STATUTORY_TAX_RATE_FALLBACK) / invested
    assert math.isclose(m["roic"], expected, rel_tol=1e-9)


def test_screener_loss_maker_without_tax_lines_keeps_roic_null(monkeypatch):
    from app.models import ScreenerMetric
    from app.services import screener_metrics_service as sms

    _seed_nvda_periods()
    _strip_tax_lines(monkeypatch, negate_operating_income=True)
    m = sms.compute_metrics("NVDA")
    assert m is not None
    assert m["op_margin"] < 0, "fixture should present NVDA as a loss-maker"
    assert m["roic"] is None, "an invented tax rate must not manufacture a negative ROIC"
    # The column tolerates the NULL — the row must not be forced to 0.
    assert ScreenerMetric.__table__.c.roic.nullable is True


def test_comps_still_builds_for_demo_universe():
    from app.services.valuation_service import build_comps

    res = build_comps("NVDA", force_refresh=True)
    assert res is not None
    assert res.target.roic is not None and res.target.roic > 0
    assert any(p.roic is not None for p in res.peers)
    assert res.median.roic is not None
