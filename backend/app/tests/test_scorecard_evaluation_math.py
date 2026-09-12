"""Tests for finance/scorecard_evaluation_math — quintile long/short spread,
spread statistics and the FF5+momentum regression helper.

All panels are synthetic (numpy RandomState(0)); nothing touches the
database or a provider.

- planted spread: top quintile built to outperform → positive spread,
  t > 2, monotone quintiles, every month qualifies
- legs skipped below 15 names (and the skip is reported, not silent)
- coverage / missing-score / missing-return rows are excluded per month
- deterministic quantile assignment with tie-breaking
- spread statistics: None with a reason for <1 / <2 months, zero dispersion
- factor regression: planted alpha/betas recovered; `insufficient` flag
  below 24 months while the fit is still reported; all-None when there are
  fewer months than parameters
- KFR monthly alignment by return month, missing months reported
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from app.finance import scorecard_evaluation_math as sem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _month_ends(n: int, start: date = date(2021, 1, 31)) -> list[str]:
    out: list[str] = []
    y, m = start.year, start.month
    for _ in range(n):
        nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        out.append((nxt - timedelta(days=1)).isoformat())
        y, m = nxt.year, nxt.month
    return out


def _planted_panel(
    *,
    n_months: int = 36,
    n_names: int = 120,
    slope: float = 0.01,
    noise: float = 0.02,
    seed: int = 0,
) -> list[sem.PanelObservation]:
    """Forward return = slope·score + noise, so the top quintile outperforms."""
    rng = np.random.RandomState(seed)
    obs: list[sem.PanelObservation] = []
    for as_of in _month_ends(n_months):
        scores = rng.normal(size=n_names)
        rets = slope * scores + rng.normal(scale=noise, size=n_names)
        for i in range(n_names):
            obs.append(sem.PanelObservation(
                as_of=as_of, ticker=f"T{i:03d}", score=float(scores[i]),
                forward_return=float(rets[i]), coverage=1.0,
            ))
    return obs


# ---------------------------------------------------------------------------
# Quantile assignment + return month
# ---------------------------------------------------------------------------

def test_assign_quantiles_deterministic_with_ties():
    scores = [0.0, 0.0, 0.0, 1.0, 2.0]
    tickers = ["C", "A", "B", "D", "E"]
    buckets = sem.assign_quantiles(scores, tickers, n_quantiles=5)
    # Ties broken by ticker: A, B, C get buckets 1, 2, 3 in that order.
    assert dict(zip(tickers, buckets)) == {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    # Uneven split: the extra names land in the bottom buckets, so the
    # top (long) bucket is the short one.
    buckets = sem.assign_quantiles(list(range(74)), [f"t{i:02d}" for i in range(74)], n_quantiles=5)
    counts = [buckets.count(q) for q in range(1, 6)]
    assert counts == [15, 15, 15, 15, 14]


def test_return_month_rolls_forward():
    assert sem.return_month("2024-01-31") == "2024-02"
    assert sem.return_month("2024-12-31") == "2025-01"
    assert sem.return_month("2024-06-30T00:00:00") == "2024-07"


def test_panel_observation_from_mapping_coerces_types():
    row = {"as_of": date(2024, 1, 31), "ticker": "AAPL", "score": "1.5", "forward_return": None}
    obs = sem.PanelObservation.from_mapping(row)
    assert obs.as_of == "2024-01-31"
    assert obs.score == 1.5
    assert obs.forward_return is None
    assert obs.coverage == 1.0
    assert sem.PanelObservation.from_mapping({"as_of": "2024-01-31", "ticker": "X", "score": float("nan")}).score is None


# ---------------------------------------------------------------------------
# Quintile long/short
# ---------------------------------------------------------------------------

def test_planted_spread_is_recovered():
    result = sem.quintile_long_short(_planted_panel())
    assert result["n_months"] == 36
    assert result["n_skipped"] == 0
    assert result["skipped_months"] == []
    assert result["mean_spread"] > 0.02  # slope 0.01 × ~2.8 z-gap between legs
    assert result["t_stat"] > 2.0
    assert result["sharpe_annualized"] > 1.0
    assert result["hit_rate"] > 0.8
    assert result["monotonic"] is True
    means = [row["mean_ret"] for row in result["quantile_table"]]
    assert means == sorted(means)
    assert [row["q"] for row in result["quantile_table"]] == [1, 2, 3, 4, 5]
    assert all(row["n_months"] == 36 for row in result["quantile_table"])
    first = result["months"][0]
    assert first["as_of"] == "2021-01-31"
    assert first["return_month"] == "2021-02"
    assert first["n_long"] == 24 and first["n_short"] == 24
    assert first["spread"] == pytest.approx(first["long_ret"] - first["short_ret"])
    assert set(first["quantile_returns"]) == {"1", "2", "3", "4", "5"}
    assert first["quantile_returns"]["5"] == pytest.approx(first["long_ret"])
    assert result["caveats"] == list(sem.EVALUATION_CAVEATS)
    assert any("unadjusted" in c for c in result["caveats"])
    assert any("survivorship" in c for c in result["caveats"])
    assert result["max_drawdown"] <= 0.0
    assert result["cumulative_spread"] > 0.0


def test_no_signal_panel_has_no_spread():
    result = sem.quintile_long_short(_planted_panel(slope=0.0, n_months=48))
    assert abs(result["t_stat"]) < 2.0
    assert result["monotonic"] is False or abs(result["mean_spread"]) < 0.01


def test_legs_skipped_below_minimum_names():
    # 74 eligible names → bottom bucket has 14 (< 15) → month skipped.
    thin = _planted_panel(n_months=3, n_names=74)
    full = _planted_panel(n_months=3, n_names=75)
    thin_result = sem.quintile_long_short(thin)
    full_result = sem.quintile_long_short(full)
    assert thin_result["n_months"] == 0
    assert thin_result["n_skipped"] == 3
    assert thin_result["mean_spread"] is None
    assert "no months qualified" in thin_result["stats_note"]
    assert thin_result["monotonic"] is None
    assert all(row["mean_ret"] is None for row in thin_result["quantile_table"])
    skip = thin_result["skipped_months"][0]
    assert skip["as_of"] == "2021-01-31"
    assert skip["n_eligible"] == 74
    assert "75 needed" in skip["reason"]
    assert full_result["n_months"] == 3
    assert full_result["months"][0]["n_long"] == 15
    assert full_result["months"][0]["n_short"] == 15
    # The rule is parametrised, not hardcoded to the default universe.
    assert sem.quintile_long_short(thin, min_leg=10)["n_months"] == 3


def test_leg_rule_reports_when_a_leg_is_thin_after_bucketing():
    # 76 names: buckets 16/15/15/15/15 → both legs ≥ 15, included.
    ok = sem.quintile_long_short(_planted_panel(n_months=1, n_names=76))
    assert ok["n_months"] == 1
    assert ok["months"][0]["n_short"] == 16 and ok["months"][0]["n_long"] == 15
    # Quartiles, custom minimum: 65 names / 4 → 17,16,16,16 → both legs ≥ 16.
    res = sem.quintile_long_short(_planted_panel(n_months=1, n_names=65), n_quantiles=4, min_leg=16)
    assert res["n_months"] == 1
    assert res["months"][0]["n_short"] == 17 and res["months"][0]["n_long"] == 16
    # 67 names / 4 with min_leg 17 → 68 needed → skipped by the pre-check.
    # Because floor division never makes a bucket smaller than n // n_q,
    # the pre-check is exact: once it passes, no leg can be thin.
    res = sem.quintile_long_short(_planted_panel(n_months=1, n_names=67), n_quantiles=4, min_leg=17)
    assert res["n_months"] == 0
    assert "68 needed for 17 per leg" in res["skipped_months"][0]["reason"]
    for n_names in range(60, 100):
        buckets = sem.assign_quantiles(list(range(n_names)), [f"t{i}" for i in range(n_names)], n_quantiles=4)
        assert min(buckets.count(q) for q in range(1, 5)) == n_names // 4


def test_coverage_and_missing_values_are_excluded_per_month():
    panel = _planted_panel(n_months=2, n_names=80)
    as_of = panel[0].as_of
    # Knock out 6 rows in the first month: 2 low coverage, 2 no score, 2 no return.
    edited: list[sem.PanelObservation] = []
    n_hit = 0
    for obs in panel:
        if obs.as_of == as_of and n_hit < 6:
            kind = n_hit % 3
            obs = sem.PanelObservation(
                as_of=obs.as_of, ticker=obs.ticker,
                score=None if kind == 1 else obs.score,
                forward_return=None if kind == 2 else obs.forward_return,
                coverage=0.2 if kind == 0 else obs.coverage,
            )
            n_hit += 1
        edited.append(obs)
    result = sem.quintile_long_short(edited)
    assert result["n_months"] == 1  # 74 eligible in month 1 → skipped
    assert result["skipped_months"][0]["n_eligible"] == 74
    assert result["skipped_months"][0]["n_excluded"] == 6
    assert result["months"][0]["as_of"] != as_of
    assert result["months"][0]["n_excluded"] == 0


def test_universe_ew_is_mean_of_eligible_returns():
    panel = _planted_panel(n_months=1, n_names=100)
    result = sem.quintile_long_short(panel)
    expected = float(np.mean([o.forward_return for o in panel]))
    assert result["months"][0]["universe_ew"] == pytest.approx(expected)
    assert result["months"][0]["n_eligible"] == 100


def test_quintile_long_short_validates_parameters():
    with pytest.raises(ValueError):
        sem.quintile_long_short([], n_quantiles=1)
    with pytest.raises(ValueError):
        sem.quintile_long_short([], min_leg=0)
    empty = sem.quintile_long_short([])
    assert empty["n_months"] == 0 and empty["months"] == []
    assert empty["mean_spread"] is None


# ---------------------------------------------------------------------------
# Spread statistics
# ---------------------------------------------------------------------------

def test_spread_statistics_small_samples_are_none_with_reason():
    empty = sem.spread_statistics([])
    assert empty["mean_spread"] is None and "no months" in empty["stats_note"]
    one = sem.spread_statistics([0.02])
    assert one["mean_spread"] == pytest.approx(0.02)
    assert one["hit_rate"] == 1.0
    assert one["stdev"] is None and one["t_stat"] is None and one["sharpe_annualized"] is None
    assert "one month only" in one["stats_note"]
    flat = sem.spread_statistics([0.01, 0.01, 0.01])
    assert flat["stdev"] == 0.0
    assert flat["sharpe_annualized"] is None and flat["t_stat"] is None
    assert "zero dispersion" in flat["stats_note"]


def test_spread_statistics_closed_forms():
    spreads = [0.02, -0.01, 0.03, 0.00, 0.01, -0.02]
    stats = sem.spread_statistics(spreads)
    arr = np.array(spreads)
    assert stats["mean_spread"] == pytest.approx(arr.mean())
    assert stats["stdev"] == pytest.approx(arr.std(ddof=1))
    assert stats["sharpe_annualized"] == pytest.approx(arr.mean() / arr.std(ddof=1) * math.sqrt(12))
    assert stats["t_stat"] == pytest.approx(arr.mean() / (arr.std(ddof=1) / math.sqrt(6)))
    assert stats["hit_rate"] == pytest.approx(3 / 6)
    wealth = np.cumprod(1 + arr)
    assert stats["cumulative_spread"] == pytest.approx(wealth[-1] - 1)
    assert stats["max_drawdown"] == pytest.approx((wealth / np.maximum.accumulate(wealth) - 1).min())
    assert stats["stats_note"] is None


# ---------------------------------------------------------------------------
# Factor regression
# ---------------------------------------------------------------------------

def _factor_panel(n: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    sigmas = np.array([0.045, 0.030, 0.030, 0.025, 0.020, 0.040])
    return rng.normal(size=(n, 6)) * sigmas


def test_factor_regression_recovers_planted_alpha_and_betas():
    n = 240
    F = _factor_panel(n)
    rng = np.random.RandomState(1)
    true_betas = np.array([0.3, -0.5, 0.8, 0.0, 0.2, -0.4])
    alpha = 0.004
    y = alpha + F @ true_betas + rng.normal(scale=0.005, size=n)
    dates = _month_ends(n)
    result = sem.factor_regression(y, F, dates=dates, series="spread")
    assert result["series"] == "spread"
    assert result["insufficient"] is False
    assert result["n_months"] == n
    assert result["start"] == dates[0] and result["end"] == dates[-1]
    assert result["alpha_monthly"] == pytest.approx(alpha, abs=0.0015)
    assert result["alpha_annualized"] == pytest.approx(result["alpha_monthly"] * 12)
    assert result["alpha_t"] > 2.0
    for name, beta in zip(sem.FF6_FACTOR_NAMES, true_betas):
        assert result["betas"][name] == pytest.approx(beta, abs=0.1), name
    assert abs(result["beta_t"]["RMW"]) < 2.5  # the zero beta is not "found"
    assert result["beta_t"]["HML"] > 5.0
    assert result["r_squared"] > 0.8
    assert result["reasons"] == []


def test_factor_regression_insufficient_flag_still_reports_fit():
    n = 18
    F = _factor_panel(n)
    y = 0.002 + F @ np.array([0.5, 0, 0, 0, 0, 0]) + np.random.RandomState(2).normal(scale=0.003, size=n)
    result = sem.factor_regression(y, F)
    assert result["insufficient"] is True
    assert result["alpha_monthly"] is not None
    assert result["betas"]["MKT_RF"] == pytest.approx(0.5, abs=0.2)
    assert any("24 required" in r for r in result["reasons"])
    assert result["start"] is None and result["end"] is None


def test_factor_regression_too_few_months_is_all_none():
    F = _factor_panel(6)
    result = sem.factor_regression(np.zeros(6) + 0.01, F)
    assert result["insufficient"] is True
    assert result["alpha_monthly"] is None and result["alpha_t"] is None
    assert all(v is None for v in result["betas"].values())
    assert result["r_squared"] is None
    assert any("cannot identify" in r for r in result["reasons"])


def test_factor_regression_drops_non_finite_and_validates_shapes():
    n = 40
    F = _factor_panel(n)
    y = F[:, 0] * 0.5 + 0.001
    y[3] = np.nan
    F[7, 2] = np.inf
    result = sem.factor_regression(y, F, dates=_month_ends(n))
    assert result["n_months"] == 38
    assert any("2 months dropped" in r for r in result["reasons"])
    assert result["start"] == _month_ends(n)[0]
    with pytest.raises(ValueError):
        sem.factor_regression(y[:10], F)
    with pytest.raises(ValueError):
        sem.factor_regression(y, F, factor_names=("a", "b"))
    with pytest.raises(ValueError):
        sem.factor_regression(y, F, dates=["x"])


def test_factor_regression_custom_factor_set():
    rng = np.random.RandomState(3)
    n = 60
    f = rng.normal(scale=0.03, size=(n, 1))
    y = 0.001 + 1.2 * f[:, 0] + rng.normal(scale=0.002, size=n)
    result = sem.factor_regression(y, f, factor_names=("MKT_RF",))
    assert set(result["betas"]) == {"MKT_RF"}
    assert result["betas"]["MKT_RF"] == pytest.approx(1.2, abs=0.05)


# ---------------------------------------------------------------------------
# KFR monthly alignment
# ---------------------------------------------------------------------------

def test_align_monthly_factors_joins_on_return_month():
    months = [
        {"as_of": "2024-01-31", "return_month": "2024-02", "spread": 0.01, "long_ret": 0.03},
        {"as_of": "2024-02-29", "return_month": "2024-03", "spread": -0.02, "long_ret": 0.00},
        {"as_of": "2024-03-31", "spread": 0.015, "long_ret": 0.02},  # return_month derived → 2024-04
        {"as_of": "2024-04-30", "return_month": "2024-05", "spread": None, "long_ret": 0.01},
    ]
    points: dict[str, list[dict[str, object]]] = {}
    for sid in sem.KFR_MONTHLY_FACTOR_IDS.values():
        points[sid] = [
            {"date": "2024-02-29", "value": 0.01},
            {"date": "2024-03-31", "value": 0.02},
            {"date": "2024-04-30", "value": 0.03},
        ]
    # RF is missing for April.
    points[sem.KFR_MONTHLY_RF_ID] = [{"date": "2024-02-29", "value": 0.004}, {"date": "2024-03-31", "value": 0.004}]
    aligned = sem.align_monthly_factors(months, points)
    assert aligned["return_months"] == ["2024-02", "2024-03"]
    assert aligned["returns"] == [0.01, -0.02]
    assert aligned["factor_names"] == list(sem.FF6_FACTOR_NAMES)
    assert aligned["factors"][0] == [0.01] * 6
    assert aligned["rf"] == [0.004, 0.004]
    assert aligned["n_aligned"] == 2
    reasons = {m["return_month"]: m["reason"] for m in aligned["missing_months"]}
    assert "KFR.RF.M" in reasons["2024-04"]
    assert "no finite spread" in reasons["2024-05"]
    # Without RF the April month aligns on the factors alone.
    no_rf = sem.align_monthly_factors(months, points, rf_id=None, value_key="long_ret")
    assert no_rf["return_months"] == ["2024-02", "2024-03", "2024-04"]
    assert no_rf["rf"] == [None, None, None]
    assert no_rf["returns"] == [0.03, 0.00, 0.02]


def test_align_then_regress_end_to_end():
    n = 48
    F = _factor_panel(n)
    rng = np.random.RandomState(4)
    y = 0.003 + F @ np.array([0.5, 0, 0.4, 0, 0, -0.3]) + rng.normal(scale=0.004, size=n)
    as_ofs = _month_ends(n, start=date(2019, 12, 31))
    months = [{"as_of": as_of, "spread": float(y[i])} for i, as_of in enumerate(as_ofs)]
    # Factor points live in the *return* month: as_of + 1.
    ret_months = _month_ends(n, start=date(2020, 1, 31))
    points = {
        sid: [{"date": ret_months[i], "value": float(F[i, j])} for i in range(n)]
        for j, sid in enumerate(sem.KFR_MONTHLY_FACTOR_IDS.values())
    }
    points[sem.KFR_MONTHLY_RF_ID] = [{"date": rm, "value": 0.001} for rm in ret_months]
    aligned = sem.align_monthly_factors(months, points)
    assert aligned["n_aligned"] == n
    result = sem.factor_regression(aligned["returns"], aligned["factors"], dates=aligned["return_months"])
    assert result["insufficient"] is False
    assert result["betas"]["MKT_RF"] == pytest.approx(0.5, abs=0.1)
    assert result["betas"]["MOM"] == pytest.approx(-0.3, abs=0.1)
    assert result["start"] == "2020-01" and result["end"] == "2023-12"


def test_constants_are_the_documented_defaults():
    assert sem.DEFAULT_N_QUANTILES == 5
    assert sem.DEFAULT_MIN_LEG == 15
    assert sem.DEFAULT_MIN_COVERAGE == 0.6
    assert sem.DEFAULT_MIN_REGRESSION_OBS == 24
    assert sem.KFR_MONTHLY_FACTOR_IDS["MKT_RF"] == "KFR.MKT_RF.M"
    assert sem.KFR_MONTHLY_RF_ID == "KFR.RF.M"
