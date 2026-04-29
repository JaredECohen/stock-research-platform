"""Comps engine tests."""
from app.services.valuation_service import build_comps


def test_nvda_comps_has_peers_and_interpretation():
    res = build_comps("NVDA")
    assert res is not None
    assert len(res.peers) >= 3
    assert res.target.ticker == "NVDA"
    assert res.median.ticker == "MEDIAN"
    assert res.interpretation


def test_premium_discount_signs_make_sense():
    res = build_comps("NVDA")
    assert res is not None
    # NVDA should be at a premium on EV/EBITDA in the demo dataset
    delta = res.premium_discount.get("ev_ebitda")
    assert delta is None or delta > -0.5  # not deeply discounted
