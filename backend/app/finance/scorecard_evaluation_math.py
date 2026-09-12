"""Scorecard evaluation arithmetic: quintile long/short spread and the
FF5+momentum regression of that spread.

Pure functions over a month-end panel the caller assembles from
`scorecard_scores` + `price_month_ends` (the wiring lives in the evaluation
service; nothing here touches the database or a provider). numpy is
imported inside functions so the module is free to import on the web
process.

Sizing follows the universe as it is (a few hundred names, ~170 today):
legs are QUINTILES, and a month whose long or short leg would hold fewer
than `min_leg` names (15) is skipped and reported, not silently thinned.
Nothing here hardcodes the universe size.

Observed vs. interpreted: the month table and the quintile table are
observations of the sample; `EVALUATION_CAVEATS` says what those
observations cannot tell you. Every statistic that cannot be computed is
None with a reason, never 0.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from .double_selection_lasso import ols_hc

# The evaluation is honest only with these attached. The evaluation service
# renders them verbatim; the UI must not paraphrase them away.
EVALUATION_CAVEATS: tuple[str, ...] = (
    "Prices are unadjusted for splits and dividends where the provider supplies no adjusted close "
    "(FMP-sourced month ends), so returns across corporate actions are distorted.",
    "The universe is today's constituents applied to every historical month (survivorship bias): "
    "names that left the universe are absent from the months they were in it.",
    "Sector labels are today's labels applied to every historical month.",
    "Fundamentals are restated values re-timed to their original availability, not the figures "
    "originally reported at the time.",
    "Legs are equal-weight, rebalanced monthly, with no transaction costs, borrow fees or slippage.",
    "Model output for research and education: a scenario read of one sample, not a recommendation.",
)

# FF5 + momentum, in the column order the regression reports them. The
# monthly Ken French series ids are the catalog's (`KFR.<name>.M`); the
# evaluation service fetches them through `factor_analytics._full_series_points`.
FF6_FACTOR_NAMES: tuple[str, ...] = ("MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM")
KFR_MONTHLY_FACTOR_IDS: dict[str, str] = {name: f"KFR.{name}.M" for name in FF6_FACTOR_NAMES}
KFR_MONTHLY_RF_ID = "KFR.RF.M"

DEFAULT_N_QUANTILES = 5
DEFAULT_MIN_LEG = 15
DEFAULT_MIN_COVERAGE = 0.6
DEFAULT_MIN_REGRESSION_OBS = 24
MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Panel input
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PanelObservation:
    """One (month-end, ticker) row of the evaluation panel.

    `as_of` is the formation month end (ISO date); `forward_return` is the
    simple return from that month end to the next one, or None when the
    price store has no pair. `score` is the scorecard's overall z at
    `as_of` (None when the ticker had no score that month).
    """

    as_of: str
    ticker: str
    score: float | None
    forward_return: float | None
    coverage: float | None = 1.0

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> PanelObservation:
        as_of = row["as_of"]
        return cls(
            as_of=as_of.isoformat() if isinstance(as_of, date) else str(as_of),
            ticker=str(row["ticker"]),
            score=_as_float(row.get("score")),
            forward_return=_as_float(row.get("forward_return")),
            coverage=_as_float(row.get("coverage", 1.0)),
        )


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def return_month(as_of: str) -> str:
    """'YYYY-MM' of the month whose return follows formation at `as_of`.

    A score formed at the 2024-01-31 close earns its forward return over
    February 2024, so the factor month to align to is 2024-02, not 2024-01.
    """
    d = date.fromisoformat(str(as_of)[:10])
    year, month = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return f"{year:04d}-{month:02d}"


# ---------------------------------------------------------------------------
# Quintile long/short
# ---------------------------------------------------------------------------

def assign_quantiles(scores: Sequence[float], tickers: Sequence[str], *, n_quantiles: int) -> list[int]:
    """Bucket a cross-section into 1..n_quantiles by ascending score.

    Ties are broken by ticker so the bucketing is deterministic; bucket
    sizes differ by at most one, and floor division puts the extra names in
    the bottom buckets, so the top (long) leg is the smaller one and the
    `min_leg` rule bites there first.
    """
    order = sorted(range(len(scores)), key=lambda i: (scores[i], tickers[i]))
    n = len(order)
    buckets = [0] * n
    for rank, i in enumerate(order):
        buckets[i] = (rank * n_quantiles) // n + 1
    return buckets


def quintile_long_short(
    observations: Sequence[PanelObservation],
    *,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    min_leg: int = DEFAULT_MIN_LEG,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, Any]:
    """Equal-weight top-minus-bottom quantile spread, month by month.

    Per month: keep rows with a score, a forward return and coverage ≥
    `min_coverage`; bucket by score; long the top bucket, short the bottom.
    A month whose long or short leg holds fewer than `min_leg` names is
    recorded under `skipped_months` with the reason and contributes nothing
    to the statistics. Returns a JSON-ready dict (see keys below).
    """
    import numpy as np

    if n_quantiles < 2:
        raise ValueError("quintile_long_short: n_quantiles must be at least 2")
    if min_leg < 1:
        raise ValueError("quintile_long_short: min_leg must be at least 1")

    by_month: dict[str, list[PanelObservation]] = {}
    for obs in observations:
        by_month.setdefault(str(obs.as_of), []).append(obs)

    months: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    pooled: dict[int, list[float]] = {q: [] for q in range(1, n_quantiles + 1)}

    for as_of in sorted(by_month):
        rows = by_month[as_of]
        # (ticker, score, forward_return) for rows with every input present.
        eligible: list[tuple[str, float, float]] = [
            (r.ticker, r.score, r.forward_return) for r in rows
            if r.score is not None and r.forward_return is not None
            and r.coverage is not None and r.coverage >= min_coverage
        ]
        n_eligible = len(eligible)
        n_excluded = len(rows) - n_eligible
        if n_eligible < n_quantiles * min_leg:
            skipped.append({
                "as_of": as_of,
                "reason": f"{n_eligible} eligible names; {n_quantiles * min_leg} needed for {min_leg} per leg",
                "n_eligible": n_eligible,
                "n_excluded": n_excluded,
            })
            continue
        scores = [score for _t, score, _r in eligible]
        rets = np.array([ret for _t, _s, ret in eligible], dtype=float)
        buckets = assign_quantiles(scores, [ticker for ticker, _s, _r in eligible], n_quantiles=n_quantiles)
        bucket_arr = np.array(buckets)
        long_mask = bucket_arr == n_quantiles
        short_mask = bucket_arr == 1
        n_long, n_short = int(long_mask.sum()), int(short_mask.sum())
        # Unreachable once the pre-check passes (floor division never makes a
        # bucket smaller than n // n_quantiles); kept so a future change to
        # the bucketing cannot silently produce a thin leg.
        if n_long < min_leg or n_short < min_leg:
            skipped.append({
                "as_of": as_of,
                "reason": f"leg below minimum: long {n_long}, short {n_short}, minimum {min_leg}",
                "n_eligible": n_eligible,
                "n_excluded": n_excluded,
            })
            continue
        quantile_returns: dict[str, float] = {}
        for q in range(1, n_quantiles + 1):
            mask = bucket_arr == q
            q_ret = float(rets[mask].mean())
            quantile_returns[str(q)] = q_ret
            pooled[q].append(q_ret)
        long_ret = float(rets[long_mask].mean())
        short_ret = float(rets[short_mask].mean())
        months.append({
            "as_of": as_of,
            "return_month": return_month(as_of),
            "long_ret": long_ret,
            "short_ret": short_ret,
            "spread": long_ret - short_ret,
            "n_long": n_long,
            "n_short": n_short,
            "n_eligible": n_eligible,
            "n_excluded": n_excluded,
            "universe_ew": float(rets.mean()),
            "quantile_returns": quantile_returns,
        })

    quantile_table = [
        {
            "q": q,
            "mean_ret": (float(np.mean(pooled[q])) if pooled[q] else None),
            "n_months": len(pooled[q]),
        }
        for q in range(1, n_quantiles + 1)
    ]
    means = [row["mean_ret"] for row in quantile_table]
    monotonic: bool | None
    if any(m is None for m in means):
        monotonic = None
    else:
        monotonic = all(means[i] < means[i + 1] for i in range(len(means) - 1))  # type: ignore[operator]

    stats = spread_statistics([m["spread"] for m in months])
    return {
        "n_quantiles": n_quantiles,
        "min_leg": min_leg,
        "min_coverage": min_coverage,
        "months": months,
        "skipped_months": skipped,
        "n_months": len(months),
        "n_skipped": len(skipped),
        **stats,
        "quantile_table": quantile_table,
        "monotonic": monotonic,
        "caveats": list(EVALUATION_CAVEATS),
    }


def spread_statistics(spreads: Sequence[float]) -> dict[str, Any]:
    """Mean, sample stdev, annualised Sharpe, t-stat, hit rate and maximum
    drawdown of a monthly spread series. Statistics that need at least two
    months (or a non-zero stdev) are None with `stats_note` explaining why.
    """
    import numpy as np

    arr = np.array([float(s) for s in spreads], dtype=float)
    n = int(len(arr))
    out: dict[str, Any] = {
        "mean_spread": None, "stdev": None, "sharpe_annualized": None, "t_stat": None,
        "hit_rate": None, "max_drawdown": None, "cumulative_spread": None, "stats_note": None,
    }
    if n == 0:
        out["stats_note"] = "no months qualified; statistics not computed"
        return out
    out["mean_spread"] = float(arr.mean())
    out["hit_rate"] = float((arr > 0).mean())
    wealth = np.cumprod(1.0 + arr)
    out["cumulative_spread"] = float(wealth[-1] - 1.0)
    out["max_drawdown"] = float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
    if n < 2:
        out["stats_note"] = "one month only; dispersion statistics need at least two"
        return out
    stdev = float(arr.std(ddof=1))
    out["stdev"] = stdev
    if stdev > 0.0:
        out["sharpe_annualized"] = float(arr.mean() / stdev * math.sqrt(MONTHS_PER_YEAR))
        out["t_stat"] = float(arr.mean() / (stdev / math.sqrt(n)))
    else:
        out["stats_note"] = "spread has zero dispersion; Sharpe and t-stat undefined"
    return out


# ---------------------------------------------------------------------------
# FF5 + momentum regression
# ---------------------------------------------------------------------------

def factor_regression(
    returns: Sequence[float],
    factors: Any,
    *,
    factor_names: Sequence[str] = FF6_FACTOR_NAMES,
    min_obs: int = DEFAULT_MIN_REGRESSION_OBS,
    dates: Sequence[str] | None = None,
    series: str = "spread",
    periods_per_year: int = MONTHS_PER_YEAR,
) -> dict[str, Any]:
    """OLS of a monthly return series on an intercept plus aligned factors.

    `factors` is n×k, columns in `factor_names` order and already aligned to
    `returns` (see `align_monthly_factors`). Standard errors are HC1. The
    fit is still reported when n < `min_obs` — with `insufficient: True` so
    the UI can say so — but not when there are fewer observations than
    parameters plus one, where every statistic is None with a reason.
    Alpha is annualised arithmetically (×12), matching the daily ×252
    convention in `services/factor_analytics`.
    """
    import numpy as np

    y = np.asarray([float(r) for r in returns], dtype=float)
    F = np.asarray(factors, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    names = list(factor_names)
    if F.shape[0] != len(y):
        raise ValueError(f"factor_regression: {F.shape[0]} factor rows for {len(y)} returns")
    if F.shape[1] != len(names):
        raise ValueError(f"factor_regression: {F.shape[1]} factor columns for {len(names)} names")
    if dates is not None and len(dates) != len(y):
        raise ValueError("factor_regression: dates must align with returns")

    keep = np.isfinite(y) & np.isfinite(F).all(axis=1)
    n_dropped = int((~keep).sum())
    y, F = y[keep], F[keep]
    kept_dates = [d for d, k in zip(dates, keep) if k] if dates is not None else []
    n, k = int(len(y)), int(F.shape[1])

    result: dict[str, Any] = {
        "series": series,
        "alpha_monthly": None, "alpha_annualized": None, "alpha_t": None, "alpha_se": None,
        "betas": {name: None for name in names}, "beta_t": {name: None for name in names},
        "r_squared": None,
        "n_months": n,
        "start": kept_dates[0] if kept_dates else None,
        "end": kept_dates[-1] if kept_dates else None,
        "insufficient": n < min_obs,
        "reasons": [],
    }
    if n_dropped:
        result["reasons"].append(f"{n_dropped} months dropped for non-finite values")
    if n < min_obs:
        result["reasons"].append(f"{n} months in sample, {min_obs} required for a reliable read")
    if n < k + 2:
        result["reasons"].append(f"{n} months cannot identify {k + 1} parameters; regression not run")
        return result

    design = np.column_stack([np.ones(n), F])
    fit = ols_hc(design, y)
    alpha = float(fit.coef[0])
    alpha_se = float(fit.se_hc1[0])
    result["alpha_monthly"] = alpha
    result["alpha_annualized"] = alpha * periods_per_year
    result["alpha_se"] = alpha_se
    result["alpha_t"] = (alpha / alpha_se) if alpha_se > 0.0 else None
    for i, name in enumerate(names):
        beta = float(fit.coef[i + 1])
        se = float(fit.se_hc1[i + 1])
        result["betas"][name] = beta
        result["beta_t"][name] = (beta / se) if se > 0.0 else None
    result["r_squared"] = fit.r_squared
    if result["alpha_t"] is None:
        result["reasons"].append("alpha standard error is zero; t-stat undefined")
    return result


def align_monthly_factors(
    months: Sequence[Mapping[str, Any]],
    factor_points: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    value_key: str = "spread",
    factor_ids: Mapping[str, str] = KFR_MONTHLY_FACTOR_IDS,
    rf_id: str | None = KFR_MONTHLY_RF_ID,
) -> dict[str, Any]:
    """Inner-join spread months to Ken French monthly points by return month.

    `months` are the `months` entries from `quintile_long_short` (each has
    `return_month` 'YYYY-MM' and the value under `value_key`); when an entry
    lacks `return_month` it is derived from `as_of`. `factor_points` maps a
    series id to `{date, value}` points whose `date` is any ISO date inside
    the month (the KFR provider emits end-of-month dates). Months missing
    any factor (or the risk-free rate when `rf_id` is given) are listed in
    `missing_months` and left out. Output arrays are plain lists so the
    result is JSON-ready and feeds `factor_regression` directly.
    """
    names = list(factor_ids)
    by_month: dict[str, dict[str, float]] = {}
    needed_ids = [factor_ids[name] for name in names] + ([rf_id] if rf_id else [])
    for sid in needed_ids:
        table: dict[str, float] = {}
        for point in factor_points.get(sid, ()):
            raw_date = point.get("date")
            value = _as_float(point.get("value"))
            if raw_date is None or value is None:
                continue
            key = str(raw_date)[:7]
            if len(key) == 7 and key[4] == "-":
                table[key] = value
        by_month[sid] = table

    aligned_months: list[str] = []
    returns: list[float] = []
    factor_rows: list[list[float]] = []
    rf: list[float | None] = []
    missing: list[dict[str, Any]] = []
    for entry in months:
        value = _as_float(entry.get(value_key))
        rm = entry.get("return_month") or (return_month(entry["as_of"]) if entry.get("as_of") else None)
        if value is None or rm is None:
            missing.append({"return_month": rm, "reason": f"no finite {value_key}"})
            continue
        rm = str(rm)
        absent = [sid for sid in needed_ids if rm not in by_month[sid]]
        if absent:
            missing.append({"return_month": rm, "reason": "factor month missing: " + ", ".join(absent)})
            continue
        aligned_months.append(rm)
        returns.append(value)
        factor_rows.append([by_month[factor_ids[name]][rm] for name in names])
        rf.append(by_month[rf_id][rm] if rf_id else None)

    return {
        "return_months": aligned_months,
        "returns": returns,
        "factors": factor_rows,
        "factor_names": names,
        "rf": rf,
        "n_aligned": len(aligned_months),
        "missing_months": missing,
    }
