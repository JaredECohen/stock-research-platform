"""Tests for the Fama-French + momentum factor regression module."""
from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pytest

from app.providers.base import ProviderStatus
from app.services import factor_analytics
from app.services.data_service import get_data_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trading_days(n: int, start: date = date(2023, 1, 2)) -> List[date]:
    out: List[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:  # mon-fri
            out.append(d)
        d += timedelta(days=1)
    return out


def _build_synthetic_factor_returns(n: int) -> Dict[str, List[Dict[str, Any]]]:
    """Generate a deterministic FF+momentum factor return panel for n days."""
    rng = random.Random(0)
    dates = _trading_days(n)
    payload: Dict[str, List[Dict[str, Any]]] = {}
    for sid, mu, sigma in [
        ("KFR.MKT_RF.D", 0.0003, 0.010),
        ("KFR.SMB.D",    0.0001, 0.005),
        ("KFR.HML.D",    0.0001, 0.005),
        ("KFR.RMW.D",    0.0001, 0.004),
        ("KFR.CMA.D",    0.0000, 0.004),
        ("KFR.MOM.D",    0.0001, 0.007),
        ("KFR.RF.D",     0.0002, 0.0),
    ]:
        rows = [
            {"date": d.isoformat(), "value": rng.gauss(mu, sigma)}
            for d in dates
        ]
        payload[sid] = rows
    return payload


def _prices_from_returns(
    factor_payload: Dict[str, List[Dict[str, Any]]],
    *,
    market_beta: float,
    value_beta: float,
    seed: int = 42,
    alpha_daily: float = 0.0002,
) -> List[Dict[str, Any]]:
    """Synthesize a price series whose true betas match the inputs.

    asset_excess[t] = alpha + market_beta * MKT_RF[t] + value_beta * HML[t] + noise[t]
    asset[t]        = asset_excess[t] + RF[t]
    Then convert to cumulative price levels.
    """
    rng = random.Random(seed)
    mkt = {p["date"]: p["value"] for p in factor_payload["KFR.MKT_RF.D"]}
    hml = {p["date"]: p["value"] for p in factor_payload["KFR.HML.D"]}
    rf = {p["date"]: p["value"] for p in factor_payload["KFR.RF.D"]}
    dates = sorted(mkt.keys())
    price = 100.0
    prices: List[Dict[str, Any]] = [{"date": dates[0], "close": price}]
    for d in dates[1:]:
        excess = (alpha_daily + market_beta * mkt[d] + value_beta * hml[d]
                  + rng.gauss(0.0, 0.002))
        ret = excess + rf[d]
        price = price * (1.0 + ret)
        prices.append({"date": d, "close": price})
    return prices


class _StubFactorProvider:
    """In-memory provider returning synthetic FF + RF series for tests.

    Also satisfies the price-history call so the convenience function
    `factor_analytics.compute_for_ticker(ticker)` can be exercised
    end-to-end without hitting the network.
    """
    name = "stub-factor"

    def __init__(self, factor_payload: Dict[str, List[Dict[str, Any]]],
                 prices: Optional[List[Dict[str, Any]]] = None) -> None:
        self._factors = factor_payload
        self._prices = prices or []

    def status(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, configured=True, healthy=True)

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        rows = self._factors.get(series_id)
        if not rows:
            return None
        return {
            "series_id": series_id,
            "name": series_id,
            "units": "decimal_return",
            "points": rows,
        }

    def get_price_history(self, ticker: str, days: int = 252):
        return list(self._prices)

    # BaseProvider stubs
    def get_company_profile(self, *_a, **_k): return None
    def get_financial_statements(self, *_a, **_k): return None
    def get_ratios(self, *_a, **_k): return None
    def get_key_metrics(self, *_a, **_k): return None
    def get_earnings(self, *_a, **_k): return None
    def get_earnings_transcripts(self, *_a, **_k): return None
    def get_filings(self, *_a, **_k): return None
    def get_news(self, *_a, **_k): return None
    def get_estimates(self, *_a, **_k): return None
    def list_tickers(self) -> List[str]: return []
    def list_macro_series(self) -> List[Dict[str, Any]]: return []


@pytest.fixture
def synthetic_factor_world():
    """Install a stub provider serving synthetic FF + a known-beta price."""
    factors = _build_synthetic_factor_returns(252)
    prices = _prices_from_returns(factors, market_beta=1.2, value_beta=-0.3)
    ds = get_data_service()
    stub = _StubFactorProvider(factors, prices)
    ds.register_test_provider(stub)
    # Clear the in-process bundle cache + provider_cache for these series
    try:
        from app.providers.ken_french_provider import _BUNDLE_CACHE
        _BUNDLE_CACHE.clear()
    except Exception:
        pass
    try:
        from app.services import provider_cache
        for sid in ["KFR.MKT_RF.D", "KFR.SMB.D", "KFR.HML.D",
                    "KFR.RMW.D", "KFR.CMA.D", "KFR.MOM.D", "KFR.RF.D"]:
            provider_cache.invalidate("macro", f"catalog:{sid}")
    except Exception:
        pass
    yield stub
    from app.tests.fixtures.demo_provider import DemoProvider
    ds.register_test_provider(DemoProvider())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_estimate_factor_profile_recovers_planted_betas(synthetic_factor_world):
    profile = factor_analytics.estimate_factor_profile(
        "FAKE", synthetic_factor_world._prices, min_observations=60,
    )
    assert profile is not None
    # Should recover the planted market beta (1.2) and value beta (-0.3)
    # within a tolerance reflecting the added noise.
    mkt_beta = profile.betas.get("KFR.MKT_RF.D")
    hml_beta = profile.betas.get("KFR.HML.D")
    assert mkt_beta is not None
    assert hml_beta is not None
    assert abs(mkt_beta - 1.2) < 0.15, f"recovered market beta {mkt_beta}"
    assert abs(hml_beta - (-0.3)) < 0.20, f"recovered HML beta {hml_beta}"
    # Market should be the primary factor by abs(beta)
    assert profile.primary_factor == "KFR.MKT_RF.D"
    assert profile.r_squared > 0.5


def test_factor_profile_emits_narrative_hints(synthetic_factor_world):
    profile = factor_analytics.estimate_factor_profile(
        "FAKE", synthetic_factor_world._prices,
    )
    assert profile is not None
    hints = profile.narrative_hints
    assert hints, "expected at least one narrative hint"
    # First hint should mention R² + observation count
    assert "FF5+momentum regression explains" in hints[0]
    assert "Market beta" in " ".join(hints[:3])


def test_empty_inputs_return_none():
    assert factor_analytics.estimate_factor_profile("FAKE", None) is None
    assert factor_analytics.estimate_factor_profile("FAKE", []) is None


def test_compute_for_ticker_uses_data_service(synthetic_factor_world):
    out = factor_analytics.compute_for_ticker("FAKE")
    assert out is not None
    assert out["ticker"] == "FAKE"
    assert "betas" in out
    assert "narrative_hints" in out


def test_overlay_wrapper_returns_consistent_shape(synthetic_factor_world):
    from app.services import sector_overlays
    bundle = sector_overlays.compute_factor_overlay(
        "FAKE", profile={"sector": "Information Technology"},
    )
    assert bundle["available"] is True
    assert "factor_profile" in bundle
    assert bundle["narrative_hints"]


def test_overlay_unavailable_when_factor_data_missing():
    """Without the stub, the regression should fail-soft to available=False."""
    from app.services import sector_overlays
    # Use a ticker the demo provider doesn't carry, AND no stub installed
    # for factor data (the autouse demo provider doesn't serve KFR.* IDs).
    bundle = sector_overlays.compute_factor_overlay(
        "ZZZZ_UNKNOWN_FAKE_TICKER",
    )
    assert bundle["available"] is False
    assert "reason" in bundle
