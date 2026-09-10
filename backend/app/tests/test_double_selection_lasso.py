"""Tests for the numpy-only double-selection LASSO (finance/double_selection_lasso).

Everything is deterministic (numpy RandomState(0)) and network/DB-free:

- soft-threshold + single-feature closed form  β = S(xᵀy/n, λ)
- λ = 0 reproduces OLS to 1e-6
- standardisation is invertible and lasso(λ=0) on standardised X maps back
  to OLS on raw X
- BCH λ decreases in n, increases in p and σ
- ols_hc: HC1 and cluster SEs are finite, match the classical SE under
  homoskedasticity, and a rank-deficient design does not raise
- synthetic DGP: coef_d recovered, x₁ selected, naive coefficient biased
  upward, full-OLS comparison reported
- subsumed verdict when d is (almost) a control
- insufficient_data when the panel is too short / too small
- degenerate (constant) controls produce no NaN/inf and are never selected
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.finance import double_selection_lasso as dsl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _panel(
    *,
    n_months: int = 30,
    n_names: int = 100,
    p: int = 30,
    beta_d: float = 0.5,
    d_noise: float = 1.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """The plan's DGP:  y = beta_d·d + 1.0·x₁ − 0.8·x₂ + ε,  d = 0.7·x₁ + 0.3·x₃ + η."""
    rng = np.random.RandomState(seed)
    n = n_months * n_names
    X = rng.normal(size=(n, p))
    eta = rng.normal(scale=d_noise, size=n)
    d = 0.7 * X[:, 0] + 0.3 * X[:, 2] + eta
    eps = rng.normal(size=n)
    y = beta_d * d + 1.0 * X[:, 0] - 0.8 * X[:, 1] + eps
    months = np.repeat(np.arange(n_months), n_names)
    return {"y": y, "d": d, "X": X, "months": months}


# ---------------------------------------------------------------------------
# Soft threshold + closed forms
# ---------------------------------------------------------------------------

def test_soft_threshold_closed_form():
    assert dsl.soft_threshold(3.0, 1.0) == 2.0
    assert dsl.soft_threshold(-3.0, 1.0) == -2.0
    assert dsl.soft_threshold(0.5, 1.0) == 0.0
    assert dsl.soft_threshold(-0.5, 1.0) == 0.0
    assert dsl.soft_threshold(1.0, 1.0) == 0.0


def test_single_feature_matches_closed_form():
    rng = np.random.RandomState(0)
    n = 400
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(size=n)
    Xs, _ = dsl.standardize(x)
    y_c = y - y.mean()
    lam = 0.3
    fit = dsl.lasso_cd(Xs, y_c, lam)
    expected = dsl.soft_threshold(float(Xs[:, 0] @ y_c) / n, lam)
    assert fit.converged
    assert fit.beta[0] == pytest.approx(expected, abs=1e-9)


def test_lambda_zero_reproduces_ols():
    rng = np.random.RandomState(0)
    n, p = 500, 5
    X = rng.normal(size=(n, p))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + rng.normal(size=n)
    Xs, _ = dsl.standardize(X)
    y_c = y - y.mean()
    fit = dsl.lasso_cd(Xs, y_c, 0.0, tol=1e-10)
    ols = _ols(Xs, y_c)
    assert fit.converged
    assert np.max(np.abs(fit.beta - ols)) < 1e-6


def test_warm_start_and_iteration_accounting():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(200, 4))
    y = X[:, 0] + rng.normal(size=200)
    Xs, _ = dsl.standardize(X)
    cold = dsl.lasso_cd(Xs, y - y.mean(), 0.05)
    warm = dsl.lasso_cd(Xs, y - y.mean(), 0.05, beta0=cold.beta)
    assert np.allclose(cold.beta, warm.beta, atol=1e-6)
    assert warm.n_iter <= cold.n_iter
    with pytest.raises(ValueError):
        dsl.lasso_cd(Xs, y - y.mean(), -0.1)
    with pytest.raises(ValueError):
        dsl.lasso_cd(Xs, y[:10], 0.1)


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

def test_standardize_scaling_and_invertibility():
    rng = np.random.RandomState(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(300, 4)) * np.array([1.0, 10.0, 0.1, 100.0])
    Xs, stats = dsl.standardize(X)
    n = X.shape[0]
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose((Xs ** 2).sum(axis=0), n)
    assert not stats.degenerate.any()
    # Invert the column transform.
    assert np.allclose(Xs * stats.scale + stats.mean, X)
    # Coefficients fitted on (Xs, centred y) map back to OLS on (1, X_raw).
    y = X @ np.array([0.5, -0.2, 4.0, 0.01]) + 2.0 + rng.normal(size=n)
    fit = dsl.lasso_cd(Xs, y - y.mean(), 0.0, tol=1e-12)
    beta_raw, intercept = dsl.unstandardize_coefficients(fit.beta, stats, y_mean=float(y.mean()))
    ols = _ols(np.column_stack([np.ones(n), X]), y)
    assert np.allclose(beta_raw, ols[1:], atol=1e-6)
    assert intercept == pytest.approx(ols[0], abs=1e-6)


def test_standardize_constant_column_is_zeroed_and_flagged():
    X = np.column_stack([np.ones(50), np.arange(50, dtype=float)])
    Xs, stats = dsl.standardize(X)
    assert bool(stats.degenerate[0]) is True
    assert bool(stats.degenerate[1]) is False
    assert np.all(Xs[:, 0] == 0.0)
    assert np.isfinite(Xs).all()
    beta_raw, _ = dsl.unstandardize_coefficients(np.array([7.0, 1.0]), stats)
    assert beta_raw[0] == 0.0  # never a coefficient for a constant column
    with pytest.raises(ValueError):
        dsl.standardize(np.zeros((0, 2)))


# ---------------------------------------------------------------------------
# BCH penalty
# ---------------------------------------------------------------------------

def test_bch_lambda_monotone_in_n_p_sigma():
    assert dsl.bch_lambda(1000, 30, 1.0) < dsl.bch_lambda(100, 30, 1.0)
    assert dsl.bch_lambda(100, 60, 1.0) > dsl.bch_lambda(100, 30, 1.0)
    assert dsl.bch_lambda(100, 30, 2.0) == pytest.approx(2.0 * dsl.bch_lambda(100, 30, 1.0))
    # Closed form check against the formula in the docstring.
    from statistics import NormalDist
    expected = 1.1 * 1.0 * NormalDist().inv_cdf(1 - 0.05 / 60) / math.sqrt(100)
    assert dsl.bch_lambda(100, 30, 1.0) == pytest.approx(expected)
    assert dsl.bch_lambda(100, 30, 0.0) == 0.0
    with pytest.raises(ValueError):
        dsl.bch_lambda(0, 30, 1.0)
    with pytest.raises(ValueError):
        dsl.bch_lambda(100, 30, 1.0, alpha=1.5)
    with pytest.raises(ValueError):
        dsl.bch_lambda(100, 30, float("nan"))


def test_bch_lasso_selects_true_support_and_iterates_sigma():
    rng = np.random.RandomState(0)
    n, p = 2000, 30
    X = rng.normal(size=(n, p))
    y = 1.0 * X[:, 0] - 0.8 * X[:, 1] + rng.normal(size=n)
    Xs, _ = dsl.standardize(X)
    fit, sigma = dsl.bch_lasso(Xs, y - y.mean())
    support = fit.support()
    assert 0 in support and 1 in support
    assert len(support) <= 6  # BCH is conservative: few false positives
    # Iterated sigma ends near the true noise level (1.0), well below stdev(y).
    assert 0.8 < sigma < 1.25
    assert sigma < float(np.std(y))
    assert fit.lam == pytest.approx(dsl.bch_lambda(n, p, sigma, c=1.1, alpha=0.05), rel=0.2)


def test_bch_lasso_constant_outcome_is_all_zero():
    Xs, _ = dsl.standardize(np.random.RandomState(0).normal(size=(50, 3)))
    fit, sigma = dsl.bch_lasso(Xs, np.zeros(50))
    assert sigma == 0.0
    assert fit.support() == []


# ---------------------------------------------------------------------------
# OLS with robust SEs
# ---------------------------------------------------------------------------

def test_ols_hc_matches_classical_under_homoskedasticity():
    rng = np.random.RandomState(0)
    n = 5000
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = X @ np.array([1.0, 2.0, -1.0]) + rng.normal(size=n)
    clusters = np.arange(n) % 40
    res = dsl.ols_hc(X, y, clusters=clusters)
    classical = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (res.residuals @ res.residuals) / (n - 3))
    assert np.allclose(res.coef, [1.0, 2.0, -1.0], atol=0.1)
    assert np.allclose(res.se_hc1, classical, rtol=0.1)
    assert res.se_cluster is not None
    assert np.allclose(res.se_cluster, classical, rtol=0.2)
    assert res.n_clusters == 40
    assert res.rank == 3
    assert res.r_squared is not None and 0.5 < res.r_squared < 1.0


def test_ols_hc_rank_deficient_design_does_not_raise():
    rng = np.random.RandomState(0)
    n = 100
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x, 2.0 * x])  # duplicated control
    y = x + rng.normal(size=n)
    res = dsl.ols_hc(X, y, clusters=np.arange(n) % 5)
    assert res.rank == 2
    assert np.isfinite(res.coef).all()
    assert np.isfinite(res.se_hc1).all()
    assert res.se_cluster is not None and np.isfinite(res.se_cluster).all()


def test_ols_hc_constant_outcome_reports_no_r_squared():
    X = np.column_stack([np.ones(20), np.arange(20, dtype=float)])
    res = dsl.ols_hc(X, np.full(20, 3.0))
    assert res.r_squared is None
    with pytest.raises(ValueError):
        dsl.ols_hc(X, np.zeros(19))
    with pytest.raises(ValueError):
        dsl.ols_hc(X, np.zeros(20), clusters=np.zeros(3))


def test_two_sided_normal_p():
    assert dsl.two_sided_normal_p(0.0) == pytest.approx(1.0)
    assert dsl.two_sided_normal_p(1.96) == pytest.approx(0.05, abs=1e-3)
    assert dsl.two_sided_normal_p(-1.96) == pytest.approx(0.05, abs=1e-3)


def test_demean_within_groups_vector_and_matrix():
    groups = np.array([0, 0, 1, 1, 1])
    v = np.array([1.0, 3.0, 2.0, 4.0, 6.0])
    out = dsl.demean_within_groups(v, groups)
    assert np.allclose(out, [-1.0, 1.0, -2.0, 0.0, 2.0])
    M = np.column_stack([v, 2 * v])
    outM = dsl.demean_within_groups(M, groups)
    assert np.allclose(outM[:, 1], 2 * out)
    with pytest.raises(ValueError):
        dsl.demean_within_groups(v, groups[:3])


# ---------------------------------------------------------------------------
# Double selection on the synthetic DGP
# ---------------------------------------------------------------------------

def test_double_selection_recovers_planted_effect():
    panel = _panel()
    names = [f"c{j}" for j in range(30)]
    res = dsl.double_selection(panel["y"], panel["d"], panel["X"], panel["months"], control_names=names)

    assert res.verdict == dsl.VERDICT_INDEPENDENT
    assert res.direction == "positive"
    assert res.coef_d is not None and 0.35 <= res.coef_d <= 0.65
    assert 0 in res.selected_union  # x₁ is in S₁ ∪ S₂
    assert 0 in res.selected_d  # it drives the treatment
    assert 1 in res.selected_y  # x₂ drives the outcome
    assert "c0" in res.selected_names
    # Naive y ~ d is biased upward because x₁ pushes both d and y.
    assert res.naive_coef is not None and res.naive_coef > 0.75
    assert res.full_ols_coef is not None and 0.35 <= res.full_ols_coef <= 0.65
    assert res.t_stat is not None and res.t_stat > 2.0
    assert res.p_value is not None and res.p_value < 0.05
    assert res.se_cluster_month is not None and res.se_cluster_month > 0
    assert res.se_hc1 is not None and res.se_hc1 > 0
    assert res.lambda_y is not None and res.lambda_d is not None
    assert res.n_obs == 3000 and res.n_months == 30 and res.p_controls == 30
    assert res.n_dropped == 0
    assert "carries information" in res.interpretation
    payload = res.to_dict()
    assert payload["verdict"] == dsl.VERDICT_INDEPENDENT
    assert payload["selected_names"] == res.selected_names


def test_double_selection_subsumed_when_treatment_is_a_control():
    rng = np.random.RandomState(0)
    n_months, n_names, p = 30, 100, 10
    n = n_months * n_names
    X = rng.normal(size=(n, p))
    d = 0.9 * X[:, 0] + rng.normal(scale=0.05, size=n)  # d ≈ x₁
    y = 1.0 * X[:, 0] + rng.normal(size=n)  # d has no effect once x₁ is in
    months = np.repeat(np.arange(n_months), n_names)
    res = dsl.double_selection(y, d, X, months)
    assert res.verdict == dsl.VERDICT_SUBSUMED
    assert res.direction == "none"
    assert 0 in res.selected_union
    assert res.t_stat is not None and abs(res.t_stat) < 2.0
    # The naive regression would have called this a strong signal.
    assert res.naive_coef is not None and res.naive_coef > 0.9
    assert "subsumed" in res.interpretation


def test_double_selection_negative_direction_is_named():
    panel = _panel(beta_d=-0.5)
    res = dsl.double_selection(panel["y"], panel["d"], panel["X"], panel["months"])
    assert res.verdict == dsl.VERDICT_INDEPENDENT
    assert res.direction == "negative"
    assert "contrary" in res.interpretation


def test_double_selection_insufficient_months():
    panel = _panel(n_months=12, n_names=200)  # 2,400 obs but only 12 months
    res = dsl.double_selection(panel["y"], panel["d"], panel["X"], panel["months"])
    assert res.verdict == dsl.VERDICT_INSUFFICIENT
    assert res.coef_d is None and res.t_stat is None
    assert res.n_months == 12 and res.n_obs == 2400
    assert any("12 months" in r for r in res.reasons)
    assert res.interpretation.startswith("No verdict")


def test_double_selection_insufficient_observations():
    panel = _panel(n_months=30, n_names=20)  # 30 months but only 600 obs
    res = dsl.double_selection(panel["y"], panel["d"], panel["X"], panel["months"])
    assert res.verdict == dsl.VERDICT_INSUFFICIENT
    assert any("600 observations" in r for r in res.reasons)
    # Lowering the floors lets the arithmetic run on the same panel.
    res2 = dsl.double_selection(panel["y"], panel["d"], panel["X"], panel["months"], min_obs=500)
    assert res2.verdict != dsl.VERDICT_INSUFFICIENT
    assert res2.coef_d is not None


def test_double_selection_degenerate_controls_no_nan():
    panel = _panel()
    X = panel["X"].copy()
    X[:, 5] = 3.0  # constant control
    X[:, 6] = 0.0  # another constant control
    res = dsl.double_selection(panel["y"], panel["d"], X, panel["months"])
    payload = res.to_dict()
    for key, value in payload.items():
        if isinstance(value, float):
            assert math.isfinite(value), key
    assert 5 not in res.selected_union and 6 not in res.selected_union
    assert any("2 constant control column(s)" in r for r in res.reasons)
    assert res.verdict == dsl.VERDICT_INDEPENDENT
    assert res.coef_d is not None and 0.35 <= res.coef_d <= 0.65


def test_double_selection_drops_non_finite_rows_and_counts_them():
    panel = _panel()
    y = panel["y"].copy()
    X = panel["X"].copy()
    y[0] = np.nan
    X[1, 3] = np.inf
    res = dsl.double_selection(y, panel["d"], X, panel["months"])
    assert res.n_dropped == 2
    assert res.n_obs == 2998
    assert any("2 rows dropped" in r for r in res.reasons)
    assert res.verdict == dsl.VERDICT_INDEPENDENT


def test_double_selection_no_controls_is_naive_ols():
    panel = _panel()
    X = np.zeros((len(panel["y"]), 0))
    res = dsl.double_selection(panel["y"], panel["d"], X, panel["months"])
    assert res.p_controls == 0
    assert res.selected_union == []
    assert res.coef_d == pytest.approx(res.naive_coef)
    assert res.full_ols_coef == pytest.approx(res.naive_coef)


def test_double_selection_constant_treatment_is_insufficient():
    panel = _panel()
    d = np.full(len(panel["y"]), 1.0)
    res = dsl.double_selection(panel["y"], d, panel["X"], panel["months"])
    assert res.verdict == dsl.VERDICT_INSUFFICIENT
    assert any("constant" in r for r in res.reasons)


def test_double_selection_shape_validation():
    panel = _panel(n_months=2, n_names=5, p=3)
    with pytest.raises(ValueError):
        dsl.double_selection(panel["y"], panel["d"][:5], panel["X"], panel["months"])
    with pytest.raises(ValueError):
        dsl.double_selection(panel["y"], panel["d"], panel["X"][:5], panel["months"])
    with pytest.raises(ValueError):
        dsl.double_selection(panel["y"], panel["d"], panel["X"], panel["months"][:5])
    with pytest.raises(ValueError):
        dsl.double_selection(panel["y"], panel["d"], panel["X"], panel["months"], control_names=["a"])


def test_verdict_vocabulary_is_closed():
    assert set(dsl.VERDICTS) == {"independent_information", "subsumed", "insufficient_data"}
