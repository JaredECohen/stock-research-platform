"""Wave 3E tests — self-historical valuation comparison.

Covers:
- Pivot helper: long-format `get_financial_history` rows reshape into
  per-period dicts.
- `_closing_price_for`: picks the price on or just before the target
  date; returns None when the price series is empty or wholly future.
- `build_history_stats` returns None when the target has fewer than
  `min_periods` usable periods.
- With a synthetic 20-period history where the current EV/EBITDA sits
  at a high percentile, `current_percentile["ev_ebitda"]` is close to
  1.0 and `current_vs_own_median["ev_ebitda"]` is positive.
- `compute_comps` + `valuation_service.build_comps` integration: NVDA
  demo run produces a `result.history` (or cleanly None when sparse).
- Comps agent integration: when both peer and own-history premium
  agree, the agent's headline reflects the agreement; when they
  diverge, the headline calls out the divergence.
"""
from __future__ import annotations

from app.agents.comps_agent import run_comps_agent
from app.finance import comps_history as ch
from app.schemas import (
    AgentFinding,
    CompsHistoryStats,
    CompsResult,
    CompsRow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_pivot_long_to_per_period_groups_by_period_end():
    history = {
        "revenue": [
            {"period": "2024Q1", "period_end": "2024-03-31", "value": 100.0,
             "fiscal_year": 2024, "fiscal_quarter": 1, "currency": "USD",
             "statement": "income"},
            {"period": "2024Q2", "period_end": "2024-06-30", "value": 110.0,
             "fiscal_year": 2024, "fiscal_quarter": 2, "currency": "USD",
             "statement": "income"},
        ],
        "operating_income": [
            {"period": "2024Q1", "period_end": "2024-03-31", "value": 25.0,
             "fiscal_year": 2024, "fiscal_quarter": 1, "currency": "USD",
             "statement": "income"},
        ],
    }
    rows = ch._pivot_long_to_per_period(history)
    assert len(rows) == 2
    # Sorted oldest-first.
    assert rows[0]["period_end"] == "2024-03-31"
    assert rows[0]["revenue"] == 100.0
    assert rows[0]["operating_income"] == 25.0
    assert rows[1]["period_end"] == "2024-06-30"
    assert rows[1]["revenue"] == 110.0
    # The Q2 row didn't have an operating_income line; that's fine.
    assert rows[1].get("operating_income") is None


def test_pivot_drops_periods_without_period_end():
    history = {
        "revenue": [
            {"period": "2024Q1", "period_end": None, "value": 100.0},
            {"period": "2024Q2", "period_end": "2024-06-30", "value": 110.0},
        ],
    }
    rows = ch._pivot_long_to_per_period(history)
    assert len(rows) == 1
    assert rows[0]["period_end"] == "2024-06-30"


def test_closing_price_for_picks_on_or_before():
    rows = [
        {"date": "2024-06-25", "close": 100.0},
        {"date": "2024-06-28", "close": 102.5},
        {"date": "2024-07-05", "close": 110.0},  # after target
    ]
    out = ch._closing_price_for(rows, "2024-06-30")
    assert out == 102.5


def test_closing_price_for_returns_none_when_all_future():
    rows = [{"date": "2024-12-01", "close": 100.0}]
    out = ch._closing_price_for(rows, "2024-01-01")
    assert out is None


def test_percentile_in_assigns_top_when_above_max():
    out = ch._percentile_in([10.0, 12.0, 14.0, 16.0], 20.0)
    assert out == 1.0


def test_percentile_in_handles_empty():
    assert ch._percentile_in([], 5.0) == 0.5


# ---------------------------------------------------------------------------
# build_history_stats end-to-end
# ---------------------------------------------------------------------------

def test_build_history_stats_returns_none_when_history_is_too_short(monkeypatch):
    """Only 4 periods → below min_periods of 8 → None."""
    from app.services import history_service
    monkeypatch.setattr(
        history_service, "get_financial_history",
        lambda *args, **kwargs: {
            "revenue": [
                {"period": f"2021Q{i}", "period_end": f"2021-{3*i:02d}-15",
                 "value": 100.0, "fiscal_year": 2021, "fiscal_quarter": i,
                 "currency": "USD", "statement": "income"}
                for i in range(1, 5)
            ],
        },
    )
    target_row = CompsRow(ticker="X", company_name="X", market_cap=1_000.0)
    out = ch.build_history_stats("X", target_row, lookback_quarters=20, min_periods=8)
    assert out is None


def test_build_history_stats_with_synthetic_history(monkeypatch):
    """A target whose current EV/EBITDA is well above its own historical
    distribution should land near the top of the percentile rank."""
    # Build 20 periods where revenue grows steadily, ebitda margin holds,
    # and net debt is roughly zero. Vary multiples by varying market cap.
    long_format = {
        "revenue": [], "gross_profit": [], "operating_income": [],
        "ebitda": [], "net_income": [], "shareholders_equity": [],
        "total_assets": [], "cash_and_equivalents": [],
        "short_term_debt": [], "long_term_debt": [],
        "depreciation_and_amortization": [], "free_cash_flow": [],
        "weighted_avg_shares_diluted": [],
    }
    for i in range(20):
        period = f"2020Q{(i % 4) + 1}"
        period_end = f"2020-{((i % 4) + 1) * 3:02d}-30"
        # Drift period ends across multiple years so they're unique.
        year = 2020 + (i // 4)
        period_end = f"{year}-{((i % 4) + 1) * 3:02d}-30"
        period = f"{year}Q{(i % 4) + 1}"
        rev = 100.0 + i * 5
        ebitda = rev * 0.30
        ni = rev * 0.20
        shares = 1_000.0  # constant
        common = {
            "period": period, "period_end": period_end,
            "fiscal_year": year, "fiscal_quarter": (i % 4) + 1,
            "currency": "USD", "statement": "income",
        }
        long_format["revenue"].append({**common, "value": rev})
        long_format["gross_profit"].append({**common, "value": rev * 0.6})
        long_format["operating_income"].append({**common, "value": rev * 0.25})
        long_format["ebitda"].append({**common, "value": ebitda})
        long_format["net_income"].append({**common, "value": ni})
        long_format["shareholders_equity"].append({**common, "value": 500.0})
        long_format["total_assets"].append({**common, "value": 800.0})
        long_format["cash_and_equivalents"].append({**common, "value": 50.0})
        long_format["short_term_debt"].append({**common, "value": 0.0})
        long_format["long_term_debt"].append({**common, "value": 0.0})
        long_format["depreciation_and_amortization"].append({**common, "value": rev * 0.05})
        long_format["free_cash_flow"].append({**common, "value": rev * 0.18})
        long_format["weighted_avg_shares_diluted"].append({**common, "value": shares})

    # Synthetic price series: stays low for 19 of 20 periods, then spikes.
    # → the *current* multiple at the latest period_end is much higher than
    # any other point in the series → top percentile.
    price_rows = []
    base_price = 50.0
    for i in range(20):
        year = 2020 + (i // 4)
        period_end = f"{year}-{((i % 4) + 1) * 3:02d}-30"
        # Pre-spike low closes.
        price_rows.append({"date": period_end, "close": base_price + i * 0.5})

    from app.services import history_service, market_data_service
    monkeypatch.setattr(history_service, "get_financial_history",
                        lambda *args, **kwargs: long_format)
    monkeypatch.setattr(market_data_service, "get_price_series",
                        lambda *args, **kwargs: price_rows)

    # Live target row: EV/EBITDA at a much higher level than the history.
    # ev/ebitda historic ~ (price * shares) / ebitda — at the last period
    # ebitda = 195 * 0.30 = 58.5 and price ~ 60 → market cap 60_000 →
    # EV/EBITDA ~ 60_000 / 58.5 ~ 1025.
    # Force the live multiple way above that.
    target_row = CompsRow(
        ticker="X", company_name="X",
        market_cap=200_000.0,  # 2x the highest historical
        ev_ebitda=2_000.0,
        operating_margin=0.25,
        revenue_growth=0.05,
    )
    out = ch.build_history_stats("X", target_row, lookback_quarters=20, min_periods=8)
    assert out is not None
    assert out.lookback_periods == 20
    # Current EV/EBITDA (2000) sits above every historical point → percentile ~ 1.0.
    assert out.current_percentile.get("ev_ebitda", 0.0) >= 0.9
    # Premium to own history is positive.
    assert out.current_vs_own_median.get("ev_ebitda", 0.0) > 0
    assert "premium" in out.interpretation.lower() or "percentile" in out.interpretation.lower()


# ---------------------------------------------------------------------------
# Service + agent integration
# ---------------------------------------------------------------------------

def test_build_comps_attaches_history_when_available():
    """A live build_comps call against the demo data — `history` may be None
    (demo annuals don't reach the 8-period floor) but the field is at
    least present and never raises."""
    from app.services.valuation_service import build_comps
    result = build_comps("NVDA", force_refresh=True)
    if result is None:
        return  # peer set unavailable in demo set; skip
    # The new field exists on the response — present but possibly None.
    assert hasattr(result, "history")
    if result.history is not None:
        assert result.history.lookback_periods >= 8


def test_comps_agent_calls_out_divergence_when_lenses_disagree():
    """Synthetic CompsResult: peer says premium (+10%), own-history says
    discount (-10%) on EV/EBITDA → headline must flag divergence."""
    target = CompsRow(
        ticker="X", company_name="X", market_cap=100.0,
        ev_ebitda=20.0, operating_margin=0.20, revenue_growth=0.10,
    )
    median = CompsRow(
        ticker="MEDIAN", company_name="Peer Median",
        ev_ebitda=18.0, operating_margin=0.18, revenue_growth=0.08,
    )
    history = CompsHistoryStats(
        lookback_periods=20, lookback_label="20 quarters",
        own_median={"ev_ebitda": 22.0, "operating_margin": 0.20, "revenue_growth": 0.10},
        own_p25={}, own_p75={},
        current_percentile={"ev_ebitda": 0.20},
        current_vs_own_median={"ev_ebitda": -0.10},
        interpretation="Discount to history.",
    )
    comps = CompsResult(
        target=target, peers=[target], median=median,
        premium_discount={"ev_ebitda": 0.10},
        target_percentiles={}, interpretation="peer interpretation",
        history=history,
    )
    finding = run_comps_agent({"ticker": "X"}, comps)
    assert isinstance(finding, AgentFinding)
    assert "divergence" in finding.headline.lower()
    # Confidence should be slightly lower (0.65) when lenses disagree.
    assert finding.confidence == 0.65


def test_comps_agent_bumps_confidence_when_lenses_agree():
    target = CompsRow(
        ticker="X", company_name="X", market_cap=100.0,
        ev_ebitda=22.0, operating_margin=0.22, revenue_growth=0.12,
    )
    median = CompsRow(
        ticker="MEDIAN", company_name="Peer Median",
        ev_ebitda=18.0, operating_margin=0.18, revenue_growth=0.08,
    )
    history = CompsHistoryStats(
        lookback_periods=20, lookback_label="20 quarters",
        own_median={"ev_ebitda": 18.0, "operating_margin": 0.18, "revenue_growth": 0.08},
        own_p25={}, own_p75={},
        current_percentile={"ev_ebitda": 0.85},
        current_vs_own_median={"ev_ebitda": 0.22},
        interpretation="Premium to history.",
    )
    comps = CompsResult(
        target=target, peers=[target], median=median,
        premium_discount={"ev_ebitda": 0.22},
        target_percentiles={}, interpretation="peer interpretation",
        history=history,
    )
    finding = run_comps_agent({"ticker": "X"}, comps)
    assert finding.confidence == 0.75
    assert "premium" in finding.headline.lower()
    assert "BOTH" in finding.headline


def test_comps_agent_runs_without_history_unchanged():
    """When `history` is None the agent must still produce a finding."""
    target = CompsRow(
        ticker="X", company_name="X", market_cap=100.0,
        ev_ebitda=20.0, operating_margin=0.20,
    )
    median = CompsRow(
        ticker="MEDIAN", company_name="Peer Median",
        ev_ebitda=18.0, operating_margin=0.18,
    )
    comps = CompsResult(
        target=target, peers=[target], median=median,
        premium_discount={"ev_ebitda": 0.10},
        target_percentiles={}, interpretation="peer interpretation",
        history=None,
    )
    finding = run_comps_agent({"ticker": "X"}, comps)
    assert finding.confidence == 0.7  # default when no history
    assert "Peer-relative" in finding.headline
