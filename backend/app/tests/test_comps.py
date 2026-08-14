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


def test_partial_financials_peer_is_skipped_not_crashed(monkeypatch):
    """A peer with an income statement but no balance sheet must cost that
    peer, not the whole comps result.

    Regression test for the bug behind four "flaky" comps tests. The peer
    loop guarded only `income`, then indexed `p["balance"][-1]` and
    `p["cash"][-1]` unguarded, so a partial-financials peer raised
    IndexError out of `build_comps`. It looked order-dependent because
    whether a given peer had partial data depended on what earlier tests
    had populated — but it was a live bug, and partial data is routine in
    production (a provider can serve one statement endpoint and miss
    another).
    """
    from app.services import valuation_service as vs

    real = vs.get_full_financials

    def _partial_for_one_peer(ticker, *a, **k):
        fin = real(ticker, *a, **k)
        peers = vs.get_peers("NVDA")
        if peers and ticker == peers[0]:
            fin = dict(fin)
            fin["balance"] = []  # income present, balance missing
        return fin

    monkeypatch.setattr(vs, "get_full_financials", _partial_for_one_peer)

    res = vs.build_comps("NVDA", force_refresh=True)
    assert res is not None, "one bad peer must not sink the whole comps result"
    broken = vs.get_peers("NVDA")[0]
    assert broken not in {p.ticker for p in res.peers}
    assert res.target.ticker == "NVDA"


def test_target_with_partial_financials_returns_none(monkeypatch):
    """The target is not skippable — but it must return None rather than
    raising IndexError, so callers get "no comps" instead of an exception."""
    from app.services import valuation_service as vs

    real = vs.get_full_financials

    def _target_missing_cash(ticker, *a, **k):
        fin = dict(real(ticker, *a, **k))
        if ticker == "NVDA":
            fin["cash"] = []
        return fin

    monkeypatch.setattr(vs, "get_full_financials", _target_missing_cash)
    assert vs.build_comps("NVDA", force_refresh=True) is None
