"""DCF engine tests."""
from __future__ import annotations

from app.finance.dcf import build_full_dcf, run_dcf
from app.schemas import DCFAssumptions
from app.services.valuation_service import default_dcf_assumptions


def _msft_assumptions() -> DCFAssumptions:
    a = default_dcf_assumptions("MSFT")
    assert a is not None
    return a


def test_default_assumptions_are_within_sane_bounds():
    a = _msft_assumptions()
    assert 0.06 <= a.wacc <= 0.14
    assert 0.0 <= a.tax_rate <= 0.30
    assert 0.005 <= a.capex_pct_revenue <= 0.20
    assert a.base_revenue > 0
    assert a.diluted_shares > 0


def test_dcf_produces_three_scenarios_with_ordering():
    a = _msft_assumptions()
    res = build_full_dcf("MSFT", a)
    assert res.bull.implied_share_price >= res.base.implied_share_price
    assert res.base.implied_share_price >= res.bear.implied_share_price
    # Bear should still produce a positive price for a profitable mega-cap
    assert res.bear.implied_share_price > 0


def test_dcf_projections_have_explicit_horizon():
    a = _msft_assumptions()
    res = run_dcf(a)
    assert len(res.projections) == 5
    for p in res.projections:
        assert p.revenue > 0
        assert p.discount_factor > 0


def test_sensitivity_tables_built():
    a = _msft_assumptions()
    res = build_full_dcf("MSFT", a)
    assert len(res.sensitivities) == 3
    assert all(len(s.cells) == 25 for s in res.sensitivities)
