"""Double-selection LASSO (Belloni, Chernozhukov & Hansen 2014) in plain numpy.

The question the scorecard evaluation asks is narrow: does the fundamental
scorecard `d` (overall z at month end t) carry information about next-month
relative returns `y` *beyond* a set of well-known controls `X` (size, value,
momentum, reversal, beta, profitability, investment, leverage, sector)?

The naive `y ~ d` regression cannot answer that, because `d` is built from
the same fundamentals the controls summarise. Putting every control into one
OLS is honest but noisy when the panel is small (the curated universe is a
few hundred names). BCH double selection sits between the two:

    1. lasso(y ~ X)  -> S1  (controls that predict the outcome)
    2. lasso(d ~ X)  -> S2  (controls that predict the treatment)
    3. OLS  y ~ d + X[S1 ∪ S2]  and read the coefficient on d

Selecting on *both* equations is what keeps the post-selection estimate from
being biased by a control that is weakly related to y but strongly related
to d (the classic omitted-variable failure of single-selection).

Everything here is deterministic arithmetic on arrays the caller hands in.
There is no database access, no provider call, and numpy is imported inside
the functions so importing this module costs nothing on the idle web
process (the repo keeps numpy off module scope for that reason).

Conventions
-----------
* Lasso objective:  (1/2n)·‖y − Xβ‖² + λ·‖β‖₁, solved by cyclic coordinate
  descent. With ‖x_j‖² = n (see `standardize`) the update is the textbook
  β_j ← S(x_jᵀ r₋ⱼ / n, λ).
* BCH penalty:  λ = c·σ·Φ⁻¹(1 − α/(2p)) / √n  in that same scaling, with σ
  estimated iteratively from lasso residuals (start at stdev(y), 2 refits).
* Standard errors:  HC1 (White with the n/(n−k) small-sample factor) and,
  when month identifiers are supplied, month-clustered with the usual
  G/(G−1)·(n−1)/(n−k) correction. The panel's observations within a month
  share the cross-sectional shock, so the clustered SE is the one the
  verdict uses.
* Verdicts:  `independent_information` when |t| ≥ 2 on the clustered SE,
  `subsumed` when |t| < 2, `insufficient_data` when the panel is too short
  or too small for a t-statistic to mean anything. The verdict is a
  scenario read of one sample, not a recommendation.

Missing numbers are reported as None with a reason, never as 0.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # numpy stays a lazy import at runtime (see module docstring)
    import numpy as np

# ---------------------------------------------------------------------------
# Verdict vocabulary — consumers (evaluation service, UI) import these rather
# than re-typing the strings.
# ---------------------------------------------------------------------------
VERDICT_INDEPENDENT = "independent_information"
VERDICT_SUBSUMED = "subsumed"
VERDICT_INSUFFICIENT = "insufficient_data"
VERDICTS: tuple[str, ...] = (VERDICT_INDEPENDENT, VERDICT_SUBSUMED, VERDICT_INSUFFICIENT)

# Sample-size floors below which a verdict is not drawn. 24 months is the
# same floor the FF6 regression uses; 2,000 observations is roughly one year
# of a ~170-name universe — fewer than that and the clustered t-statistic
# rests on too few effective clusters to trust.
DEFAULT_MIN_MONTHS = 24
DEFAULT_MIN_OBS = 2000
DEFAULT_T_THRESHOLD = 2.0

_NORMAL = NormalDist()


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Standardization:
    """Column statistics needed to map standardised coefficients back.

    `scale` is the population standard deviation of each column, so the
    standardised column satisfies ‖x_j‖² = n. Constant (degenerate) columns
    get scale 1 and are zeroed in the standardised matrix; `degenerate`
    records which ones so a caller never reports a coefficient for them.
    """

    mean: np.ndarray
    scale: np.ndarray
    degenerate: np.ndarray


def standardize(X: Any) -> tuple[np.ndarray, Standardization]:
    """Centre each column and scale it so ‖x_j‖² = n.

    Returns the standardised matrix and the statistics needed to invert the
    transform (`unstandardize_coefficients`). A column with zero variance
    (or a non-finite scale) is zeroed rather than divided by zero — the
    lasso then simply never selects it.
    """
    import numpy as np

    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    n = arr.shape[0]
    if n == 0:
        raise ValueError("standardize: X has no rows")
    mean = arr.mean(axis=0)
    centered = arr - mean
    scale = np.sqrt((centered ** 2).sum(axis=0) / n)
    degenerate = ~np.isfinite(scale) | (scale <= 0.0)
    safe_scale = np.where(degenerate, 1.0, scale)
    standardized = centered / safe_scale
    standardized[:, degenerate] = 0.0
    return standardized, Standardization(mean=mean, scale=safe_scale, degenerate=degenerate)


def unstandardize_coefficients(
    beta_std: Any,
    stats: Standardization,
    *,
    y_mean: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Map coefficients fitted on standardised X (and centred y) back to raw
    units. Returns `(beta_raw, intercept)` such that
    `y ≈ intercept + X_raw @ beta_raw`.
    """
    import numpy as np

    beta = np.asarray(beta_std, dtype=float)
    beta_raw = beta / stats.scale
    beta_raw = np.where(stats.degenerate, 0.0, beta_raw)
    intercept = float(y_mean - float(stats.mean @ beta_raw))
    return beta_raw, intercept


# ---------------------------------------------------------------------------
# Lasso by cyclic coordinate descent
# ---------------------------------------------------------------------------

def soft_threshold(a: float, lam: float) -> float:
    """S(a, λ) = sign(a)·max(|a| − λ, 0)."""
    if a > lam:
        return a - lam
    if a < -lam:
        return a + lam
    return 0.0


@dataclass(frozen=True)
class LassoFit:
    beta: np.ndarray
    lam: float
    n_iter: int
    converged: bool

    def support(self, *, tol: float = 1e-12) -> list[int]:
        """Indices of non-zero coefficients (the selected set)."""
        return [int(j) for j in range(len(self.beta)) if abs(float(self.beta[j])) > tol]


def lasso_cd(
    X: Any,
    y: Any,
    lam: float,
    *,
    tol: float = 1e-7,
    max_iter: int = 5000,
    beta0: Any | None = None,
) -> LassoFit:
    """Minimise (1/2n)·‖y − Xβ‖² + λ·‖β‖₁ by cyclic coordinate descent.

    No intercept is fitted: centre `y` (and `X`) first. `X` is expected to
    be standardised (‖x_j‖² = n) but the update divides by ‖x_j‖²/n so an
    unscaled matrix still converges to the right solution; zero columns get
    β_j = 0. With λ = 0 the iteration converges to the OLS solution for a
    full-rank design. Warm starts via `beta0` make the iterated BCH refits
    cheap. Convergence is declared when the largest coefficient change in a
    full sweep drops below `tol`; `converged=False` is reported honestly
    rather than raised, because a near-converged lasso is still a usable
    selector.
    """
    import numpy as np

    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    n, p = Xa.shape
    if ya.shape != (n,):
        raise ValueError(f"lasso_cd: y has shape {ya.shape}, expected ({n},)")
    if lam < 0:
        raise ValueError("lasso_cd: lam must be non-negative")

    col_sq = (Xa ** 2).sum(axis=0) / n  # 1.0 for standardised columns
    beta = np.zeros(p) if beta0 is None else np.array(beta0, dtype=float, copy=True)
    if beta.shape != (p,):
        raise ValueError(f"lasso_cd: beta0 has shape {beta.shape}, expected ({p},)")
    beta[col_sq <= 0.0] = 0.0
    resid = ya - Xa @ beta

    n_iter = 0
    converged = False
    while n_iter < max_iter:
        n_iter += 1
        max_delta = 0.0
        for j in range(p):
            if col_sq[j] <= 0.0:
                continue
            old = float(beta[j])
            # x_jᵀ r₋ⱼ / n where r₋ⱼ is the residual with x_j's contribution
            # added back — computed incrementally so each step is O(n).
            rho = float(Xa[:, j] @ resid) / n + col_sq[j] * old
            new = soft_threshold(rho, lam) / float(col_sq[j])
            if new != old:
                resid -= Xa[:, j] * (new - old)
                beta[j] = new
                delta = abs(new - old)
                if delta > max_delta:
                    max_delta = delta
        if max_delta < tol:
            converged = True
            break
    return LassoFit(beta=beta, lam=float(lam), n_iter=n_iter, converged=converged)


# ---------------------------------------------------------------------------
# BCH penalty
# ---------------------------------------------------------------------------

def bch_lambda(n: int, p: int, sigma: float, *, c: float = 1.1, alpha: float = 0.05) -> float:
    """Belloni–Chernozhukov–Hansen penalty for the (1/2n)‖·‖² + λ‖β‖₁ scaling.

        λ = c · σ · Φ⁻¹(1 − α / (2p)) / √n

    It shrinks with √n and grows (slowly) with p, which is the whole point:
    the penalty is a bound on the noise level of x_jᵀε/n rather than a
    tuning knob. σ = 0 (a constant outcome) returns 0 — the lasso then has
    nothing to fit and every coefficient is zero regardless.
    """
    if n < 1 or p < 1:
        raise ValueError("bch_lambda: n and p must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("bch_lambda: alpha must lie in (0, 1)")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("bch_lambda: sigma must be a finite non-negative number")
    if sigma == 0.0:
        return 0.0
    quantile = _NORMAL.inv_cdf(1.0 - alpha / (2.0 * p))
    return float(c * sigma * quantile / math.sqrt(n))


def bch_lasso(
    X: Any,
    y: Any,
    *,
    c: float = 1.1,
    alpha: float = 0.05,
    n_refits: int = 2,
    tol: float = 1e-7,
    max_iter: int = 5000,
) -> tuple[LassoFit, float]:
    """Lasso with the BCH penalty and an iterated σ estimate.

    σ starts at stdev(y), the lasso is fitted, σ is re-estimated from the
    residuals (RSS / (n − |S|)), and the fit is repeated `n_refits` times
    with warm starts. Returns the final fit and the σ it was fitted with.
    Expects centred `y` and standardised `X`.
    """
    import numpy as np

    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    n, p = Xa.shape
    sigma = float(np.std(ya)) if n else 0.0
    if p == 0 or not math.isfinite(sigma) or sigma == 0.0:
        return LassoFit(beta=np.zeros(p), lam=0.0, n_iter=0, converged=True), sigma

    fit: LassoFit | None = None
    for _ in range(n_refits + 1):
        lam = bch_lambda(n, p, sigma, c=c, alpha=alpha)
        fit = lasso_cd(Xa, ya, lam, tol=tol, max_iter=max_iter, beta0=None if fit is None else fit.beta)
        resid = ya - Xa @ fit.beta
        df = max(n - len(fit.support()), 1)
        sigma_new = float(math.sqrt(float(resid @ resid) / df))
        if not math.isfinite(sigma_new) or sigma_new <= 0.0:
            break
        sigma = sigma_new
    assert fit is not None  # n_refits + 1 >= 1 iterations always run
    return fit, sigma


# ---------------------------------------------------------------------------
# OLS with heteroskedasticity- and cluster-robust standard errors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OlsResult:
    coef: np.ndarray
    se_hc1: np.ndarray
    se_cluster: np.ndarray | None
    residuals: np.ndarray
    r_squared: float | None
    n_obs: int
    n_params: int
    n_clusters: int | None
    rank: int


def ols_hc(X: Any, y: Any, *, clusters: Any | None = None) -> OlsResult:
    """OLS via least squares with HC1 and (optionally) cluster-robust SEs.

    `X` must already contain an intercept column if one is wanted. The
    coefficient vector is the minimum-norm least-squares solution, so a
    rank-deficient design (e.g. a duplicated control) does not raise; the
    sandwich uses the pseudo-inverse of XᵀX for the same reason, and the
    affected standard errors come out large rather than NaN.

    `r_squared` is None when y has no variance — there is nothing to
    explain, and reporting 0.0 would read as "explained nothing".
    """
    import numpy as np

    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    n, k = Xa.shape
    if ya.shape != (n,):
        raise ValueError(f"ols_hc: y has shape {ya.shape}, expected ({n},)")
    if n == 0:
        raise ValueError("ols_hc: no observations")

    coef, _residuals, rank, _sv = np.linalg.lstsq(Xa, ya, rcond=None)
    resid = ya - Xa @ coef
    xtx_inv = np.linalg.pinv(Xa.T @ Xa)
    scores = Xa * resid[:, None]  # row i is x_i · e_i

    dof_factor = n / (n - k) if n > k else 1.0
    meat = scores.T @ scores
    cov_hc1 = xtx_inv @ meat @ xtx_inv * dof_factor
    se_hc1 = np.sqrt(np.clip(np.diag(cov_hc1), 0.0, None))

    se_cluster: np.ndarray | None = None
    n_clusters: int | None = None
    if clusters is not None:
        cl = np.asarray(clusters)
        if cl.shape != (n,):
            raise ValueError(f"ols_hc: clusters has shape {cl.shape}, expected ({n},)")
        _uniq, inverse = np.unique(cl, return_inverse=True)
        n_clusters = int(len(_uniq))
        cluster_scores = np.zeros((n_clusters, k))
        np.add.at(cluster_scores, inverse, scores)
        meat_c = cluster_scores.T @ cluster_scores
        if n_clusters > 1 and n > k:
            adj = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        else:
            adj = 1.0
        cov_c = xtx_inv @ meat_c @ xtx_inv * adj
        se_cluster = np.sqrt(np.clip(np.diag(cov_c), 0.0, None))

    tss = float(((ya - ya.mean()) ** 2).sum())
    rss = float(resid @ resid)
    r_squared = (1.0 - rss / tss) if tss > 0.0 else None

    return OlsResult(
        coef=coef, se_hc1=se_hc1, se_cluster=se_cluster, residuals=resid,
        r_squared=r_squared, n_obs=int(n), n_params=int(k),
        n_clusters=n_clusters, rank=int(rank),
    )


def two_sided_normal_p(t: float) -> float:
    """Two-sided p-value under the normal approximation, 2·(1 − Φ(|t|))."""
    return float(2.0 * (1.0 - _NORMAL.cdf(abs(t))))


def demean_within_groups(A: Any, group_ids: Any) -> np.ndarray:
    """Subtract each group's mean from its rows (a time fixed effect when
    groups are months). Works on a vector or a matrix; returns a new array.
    """
    import numpy as np

    arr = np.asarray(A, dtype=float)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[:, None]
    ids = np.asarray(group_ids)
    if ids.shape != (arr.shape[0],):
        raise ValueError(f"demean_within_groups: group_ids has shape {ids.shape}, expected ({arr.shape[0]},)")
    _uniq, inverse = np.unique(ids, return_inverse=True)
    sums = np.zeros((len(_uniq), arr.shape[1]))
    np.add.at(sums, inverse, arr)
    counts = np.bincount(inverse, minlength=len(_uniq)).astype(float)
    means = sums / counts[:, None]
    out = arr - means[inverse]
    return out[:, 0] if squeeze else out


# ---------------------------------------------------------------------------
# Double selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoubleSelectionResult:
    """Outcome of one double-selection run.

    Statistics are None (with an entry in `reasons`) when they cannot be
    computed; `verdict` is always one of `VERDICTS`. `selected_*` hold
    control indices into the caller's X (names in `selected_names` when the
    caller supplied `control_names`).
    """

    coef_d: float | None
    se_hc1: float | None
    se_cluster_month: float | None
    t_stat: float | None
    p_value: float | None
    selected_y: list[int]
    selected_d: list[int]
    selected_union: list[int]
    selected_names: list[str]
    lambda_y: float | None
    lambda_d: float | None
    sigma_y: float | None
    sigma_d: float | None
    n_obs: int
    n_months: int
    p_controls: int
    n_dropped: int
    naive_coef: float | None
    full_ols_coef: float | None
    verdict: str
    direction: str
    interpretation: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coef_d": self.coef_d,
            "se_hc1": self.se_hc1,
            "se_cluster_month": self.se_cluster_month,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "selected_y": list(self.selected_y),
            "selected_d": list(self.selected_d),
            "selected_union": list(self.selected_union),
            "selected_names": list(self.selected_names),
            "lambda_y": self.lambda_y,
            "lambda_d": self.lambda_d,
            "sigma_y": self.sigma_y,
            "sigma_d": self.sigma_d,
            "n_obs": self.n_obs,
            "n_months": self.n_months,
            "p_controls": self.p_controls,
            "n_dropped": self.n_dropped,
            "naive_coef": self.naive_coef,
            "full_ols_coef": self.full_ols_coef,
            "verdict": self.verdict,
            "direction": self.direction,
            "interpretation": self.interpretation,
            "reasons": list(self.reasons),
        }


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _insufficient(
    *, n_obs: int, n_months: int, p_controls: int, n_dropped: int, reasons: list[str],
) -> DoubleSelectionResult:
    return DoubleSelectionResult(
        coef_d=None, se_hc1=None, se_cluster_month=None, t_stat=None, p_value=None,
        selected_y=[], selected_d=[], selected_union=[], selected_names=[],
        lambda_y=None, lambda_d=None, sigma_y=None, sigma_d=None,
        n_obs=n_obs, n_months=n_months, p_controls=p_controls, n_dropped=n_dropped,
        naive_coef=None, full_ols_coef=None,
        verdict=VERDICT_INSUFFICIENT, direction="none",
        interpretation="No verdict: " + "; ".join(reasons) + ".",
        reasons=list(reasons),
    )


def double_selection(
    y: Any,
    d: Any,
    X: Any,
    month_ids: Any,
    *,
    control_names: Sequence[str] | None = None,
    c: float = 1.1,
    alpha: float = 0.05,
    min_months: int = DEFAULT_MIN_MONTHS,
    min_obs: int = DEFAULT_MIN_OBS,
    t_threshold: float = DEFAULT_T_THRESHOLD,
    tol: float = 1e-7,
    max_iter: int = 5000,
) -> DoubleSelectionResult:
    """Estimate the effect of treatment `d` on outcome `y` with BCH double
    selection over controls `X`, clustered by `month_ids`.

    `y`, `d`: length-n vectors (next-month relative return; scorecard z).
    `X`: n×p control matrix (may be empty, p = 0). Rows with a non-finite
    value in any of the three are dropped and counted in `n_dropped`.
    `month_ids`: length-n labels; the clustered SE and the month count for
    the sufficiency rule both come from here, which is why it is required.

    The returned coefficient is per unit of `d` in the caller's units; the
    controls are standardised internally, which leaves the coefficient on
    `d` unchanged.
    """
    import numpy as np

    ya = np.asarray(y, dtype=float).reshape(-1)
    da = np.asarray(d, dtype=float).reshape(-1)
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa[:, None] if Xa.size else Xa.reshape(len(ya), 0)
    n_in = len(ya)
    if da.shape != (n_in,):
        raise ValueError(f"double_selection: d has shape {da.shape}, expected ({n_in},)")
    if Xa.shape[0] != n_in:
        raise ValueError(f"double_selection: X has {Xa.shape[0]} rows, expected {n_in}")
    months = np.asarray(month_ids)
    if months.shape != (n_in,):
        raise ValueError(f"double_selection: month_ids has shape {months.shape}, expected ({n_in},)")
    p = int(Xa.shape[1])
    names = list(control_names) if control_names is not None else [f"x{j}" for j in range(p)]
    if len(names) != p:
        raise ValueError(f"double_selection: {len(names)} control names for {p} controls")

    keep = np.isfinite(ya) & np.isfinite(da)
    if p:
        keep &= np.isfinite(Xa).all(axis=1)
    n_dropped = int((~keep).sum())
    ya, da, Xa, months = ya[keep], da[keep], Xa[keep], months[keep]
    n = int(len(ya))
    n_months = int(len(np.unique(months))) if n else 0

    reasons: list[str] = []
    if n_dropped:
        reasons.append(f"{n_dropped} rows dropped for non-finite values")
    sufficiency: list[str] = []
    if n_months < min_months:
        sufficiency.append(f"{n_months} months in panel, {min_months} required")
    if n < min_obs:
        sufficiency.append(f"{n} observations in panel, {min_obs} required")
    # Even when a caller lowers the floors, the arithmetic needs a handful
    # of rows and some variation in d to produce anything at all.
    if n < 3:
        sufficiency.append("fewer than 3 usable observations")
    elif float(np.std(da)) == 0.0:
        sufficiency.append("treatment d is constant")
    if sufficiency:
        return _insufficient(n_obs=n, n_months=n_months, p_controls=p, n_dropped=n_dropped,
                             reasons=reasons + sufficiency)

    # --- selection -------------------------------------------------------
    y_c = ya - ya.mean()
    d_c = da - da.mean()
    if p:
        Xs, stats = standardize(Xa)
        sigma_y: float | None
        sigma_d: float | None
        fit_y, sigma_y = bch_lasso(Xs, y_c, c=c, alpha=alpha, tol=tol, max_iter=max_iter)
        fit_d, sigma_d = bch_lasso(Xs, d_c, c=c, alpha=alpha, tol=tol, max_iter=max_iter)
        selected_y = fit_y.support()
        selected_d = fit_d.support()
        lambda_y: float | None = fit_y.lam
        lambda_d: float | None = fit_d.lam
        usable = [j for j in range(p) if not bool(stats.degenerate[j])]
        n_degenerate = p - len(usable)
        if n_degenerate:
            reasons.append(f"{n_degenerate} constant control column(s) ignored")
    else:
        Xs = Xa
        selected_y, selected_d = [], []
        lambda_y = lambda_d = None
        sigma_y = sigma_d = None
        usable = []
    union = sorted(set(selected_y) | set(selected_d))

    # --- post-selection OLS ---------------------------------------------
    ones = np.ones(n)
    design = np.column_stack([ones, da] + ([Xs[:, union]] if union else []))
    post = ols_hc(design, ya, clusters=months)
    coef_d = _finite_or_none(float(post.coef[1]))
    se_hc1 = _finite_or_none(float(post.se_hc1[1]))
    se_cluster = _finite_or_none(float(post.se_cluster[1])) if post.se_cluster is not None else None

    naive = ols_hc(np.column_stack([ones, da]), ya)
    naive_coef = _finite_or_none(float(naive.coef[1]))

    full_ols_coef: float | None = None
    if len(usable) + 2 < n:
        full = ols_hc(np.column_stack([ones, da, Xs[:, usable]]) if usable else np.column_stack([ones, da]), ya)
        full_ols_coef = _finite_or_none(float(full.coef[1]))
    else:
        reasons.append("full-control OLS skipped: more parameters than observations")

    # --- inference ---------------------------------------------------------
    se_for_t = se_cluster
    if se_for_t is None or se_for_t <= 0.0:
        if se_hc1 is not None and se_hc1 > 0.0:
            se_for_t = se_hc1
            reasons.append("cluster SE unavailable; t-statistic uses HC1 SE")
        else:
            se_for_t = None
    t_stat = (coef_d / se_for_t) if (coef_d is not None and se_for_t) else None
    p_value = two_sided_normal_p(t_stat) if t_stat is not None else None

    k_sel = len(union)
    if t_stat is None:
        verdict = VERDICT_INSUFFICIENT
        direction = "none"
        reasons.append("standard error of the treatment coefficient is not finite")
        interpretation = "No verdict: the post-selection standard error could not be computed."
    elif abs(t_stat) >= t_threshold:
        verdict = VERDICT_INDEPENDENT
        direction = "positive" if t_stat > 0 else "negative"
        sign_note = (
            "higher scores line up with higher next-month relative returns"
            if t_stat > 0 else
            "higher scores line up with LOWER next-month relative returns — the sign is contrary to the score's intent"
        )
        interpretation = (
            f"With {k_sel} of {p} controls selected, a one-unit-higher scorecard z is associated with "
            f"{coef_d:+.4f} next-month relative return (month-clustered t = {t_stat:.2f}, n = {n}, "
            f"{n_months} months); {sign_note}. The scorecard carries information the selected controls "
            f"do not, in this sample."
        )
    else:
        verdict = VERDICT_SUBSUMED
        direction = "none"
        interpretation = (
            f"With {k_sel} of {p} controls selected, the scorecard coefficient {coef_d:+.4f} "
            f"(month-clustered t = {t_stat:.2f}, n = {n}, {n_months} months) is not distinguishable "
            f"from zero; in this sample the scorecard's information is subsumed by the selected controls."
        )

    return DoubleSelectionResult(
        coef_d=coef_d, se_hc1=se_hc1, se_cluster_month=se_cluster,
        t_stat=_finite_or_none(t_stat) if t_stat is not None else None,
        p_value=p_value,
        selected_y=selected_y, selected_d=selected_d, selected_union=union,
        selected_names=[names[j] for j in union],
        lambda_y=lambda_y, lambda_d=lambda_d, sigma_y=sigma_y, sigma_d=sigma_d,
        n_obs=n, n_months=n_months, p_controls=p, n_dropped=n_dropped,
        naive_coef=naive_coef, full_ols_coef=full_ols_coef,
        verdict=verdict, direction=direction, interpretation=interpretation,
        reasons=reasons,
    )
