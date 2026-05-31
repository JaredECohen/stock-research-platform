"""Fama-French + momentum factor regression of a single asset's returns.

Pure functions on top of `data_catalog_service.fetch_series` for the
Ken French factor returns. The flagship entry point is:

    estimate_factor_profile(ticker, price_history) -> FactorProfile | None

It aligns the asset's daily simple returns to the FF5+momentum daily
factor returns, runs an OLS regression of excess returns on the six
factors (MKT_RF, SMB, HML, RMW, CMA, MOM) using numpy.linalg.lstsq,
and returns betas + R² + per-factor attributed excess return.

The output is a small dict that the sector / risk agent can embed in a
finding's `data` payload directly, plus a `narrative_hints` list that
reads cleanly inline in a memo.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Factor IDs the regression uses (in this order). Each maps to a Ken
# French daily series via the catalog.
_FACTOR_IDS_DAILY: List[str] = [
    "KFR.MKT_RF.D",  # MKT_RF
    "KFR.SMB.D",     # SMB
    "KFR.HML.D",     # HML
    "KFR.RMW.D",     # RMW
    "KFR.CMA.D",     # CMA
    "KFR.MOM.D",     # MOM
]
_RF_ID_DAILY = "KFR.RF.D"
_FACTOR_LABELS: Dict[str, str] = {
    "KFR.MKT_RF.D": "Market",
    "KFR.SMB.D": "Size (SMB)",
    "KFR.HML.D": "Value (HML)",
    "KFR.RMW.D": "Profitability (RMW)",
    "KFR.CMA.D": "Investment (CMA)",
    "KFR.MOM.D": "Momentum (MOM)",
}


@dataclass
class FactorProfile:
    ticker: str
    observations: int
    start_date: Optional[str]
    end_date: Optional[str]
    alpha_daily: float
    alpha_annualized: float
    r_squared: float
    betas: Dict[str, float]
    attributed_excess_return: Dict[str, float]
    total_excess_return: float
    primary_factor: Optional[str]
    narrative_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_factor_profile(
    ticker: str,
    price_history: Optional[List[Dict[str, Any]]],
    *,
    min_observations: int = 60,
) -> Optional[FactorProfile]:
    """Run a 6-factor regression of `ticker`'s daily excess returns.

    `price_history` is a list of `{date, close}` (or `{date, adj_close}`)
    rows — the same shape `data_service.get_price_history` returns. We
    convert to daily returns, align to the FF daily factor series, drop
    NaN rows, and lstsq-fit excess_returns = a + B·F + e.

    Returns None when there isn't enough overlap to fit (e.g. <60 days,
    or the FF data hasn't been fetched yet and the network is down).
    """
    asset_returns = _daily_returns_from_prices(price_history)
    if not asset_returns:
        return None

    try:
        from . import data_catalog_service
    except Exception:  # pragma: no cover
        return None

    # Fetch all factor series + RF.
    factor_points: Dict[str, List[Dict[str, Any]]] = {}
    for sid in _FACTOR_IDS_DAILY + [_RF_ID_DAILY]:
        snap = data_catalog_service.fetch_series(sid)
        if snap is None or snap.error or not snap.sample_points:
            # If any factor is missing the regression can't run.
            log.debug("Factor regression aborted — missing %s", sid)
            return None
        # We want the full history, not just the last 24 points the
        # snapshot trims — so refetch via the underlying provider directly.
        full = _full_series_points(sid)
        if not full:
            return None
        factor_points[sid] = full

    aligned = _align_returns(asset_returns, factor_points)
    if len(aligned) < min_observations:
        return None

    try:
        import numpy as np
    except Exception:  # pragma: no cover — numpy is a hard dep already
        return None

    dates = [row["date"] for row in aligned]
    asset_arr = np.array([row["asset"] for row in aligned], dtype=float)
    rf_arr = np.array([row["rf"] for row in aligned], dtype=float)
    excess = asset_arr - rf_arr
    factor_matrix = np.column_stack([
        np.ones(len(aligned)),
        *[np.array([row[sid] for row in aligned], dtype=float) for sid in _FACTOR_IDS_DAILY],
    ])

    try:
        coefficients, *_ = np.linalg.lstsq(factor_matrix, excess, rcond=None)
    except np.linalg.LinAlgError:
        return None

    fitted = factor_matrix @ coefficients
    residual = excess - fitted
    total_ss = float(np.sum((excess - excess.mean()) ** 2))
    r_squared = 1.0 - (float(np.sum(residual ** 2)) / total_ss) if total_ss else 0.0

    alpha_daily = float(coefficients[0])
    betas: Dict[str, float] = {}
    attributed: Dict[str, float] = {}
    for i, sid in enumerate(_FACTOR_IDS_DAILY):
        beta = float(coefficients[i + 1])
        # Total attributed excess return = beta * sum(factor returns)
        factor_returns_sum = float(np.sum(np.array([row[sid] for row in aligned], dtype=float)))
        betas[sid] = beta
        attributed[sid] = beta * factor_returns_sum

    total_excess = float(np.sum(excess))
    primary_factor = max(betas, key=lambda sid: abs(betas[sid])) if betas else None

    narrative = _build_narrative_hints(
        ticker=ticker, betas=betas, attributed=attributed,
        total_excess=total_excess, r_squared=r_squared,
        alpha_daily=alpha_daily, obs=len(aligned),
    )

    return FactorProfile(
        ticker=ticker.upper(),
        observations=len(aligned),
        start_date=dates[0],
        end_date=dates[-1],
        alpha_daily=alpha_daily,
        alpha_annualized=alpha_daily * 252,
        r_squared=float(r_squared),
        betas=betas,
        attributed_excess_return=attributed,
        total_excess_return=total_excess,
        primary_factor=primary_factor,
        narrative_hints=narrative,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _daily_returns_from_prices(
    price_history: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if not price_history:
        return []
    # Accept either {date, close} or {date, adj_close}. Sort ascending.
    rows = sorted(
        [r for r in price_history if r.get("date")],
        key=lambda r: r["date"],
    )
    out: List[Dict[str, Any]] = []
    prev: Optional[float] = None
    for row in rows:
        price = row.get("adj_close") or row.get("close") or row.get("price")
        if price is None:
            continue
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue
        if prev is not None and prev != 0:
            out.append({"date": str(row["date"])[:10], "ret": (price - prev) / prev})
        prev = price
    return out


def _full_series_points(series_id: str) -> Optional[List[Dict[str, Any]]]:
    """Fetch the full point history for a series (not just the last 24)."""
    try:
        from . import data_catalog_service
        from . import provider_cache
    except Exception:  # pragma: no cover
        return None
    cached = provider_cache.get("macro", f"catalog:{series_id}", ttl_seconds=None)
    if isinstance(cached, dict):
        points = cached.get("points") or []
        if points:
            return points
    # Trigger a fetch to populate the cache, then read the points back.
    data_catalog_service.fetch_series(series_id)
    cached = provider_cache.get("macro", f"catalog:{series_id}", ttl_seconds=None)
    if isinstance(cached, dict):
        return cached.get("points") or []
    return None


def _align_returns(
    asset_returns: List[Dict[str, Any]],
    factor_points: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Inner-join by date across the asset returns + every factor series."""
    by_date: Dict[str, Dict[str, Any]] = {
        row["date"]: {"date": row["date"], "asset": row["ret"]} for row in asset_returns
    }
    for sid, points in factor_points.items():
        per_date = {str(p["date"])[:10]: p.get("value") for p in points if p.get("date") is not None}
        keep: Dict[str, Dict[str, Any]] = {}
        for d, row in by_date.items():
            v = per_date.get(d)
            if v is None or not isinstance(v, (int, float)) or math.isnan(v):
                continue
            row[_align_key(sid)] = float(v)
            keep[d] = row
        by_date = keep
    # All series must be present for the row to survive.
    return sorted(by_date.values(), key=lambda r: r["date"])


def _align_key(series_id: str) -> str:
    if series_id == _RF_ID_DAILY:
        return "rf"
    return series_id


def _build_narrative_hints(
    *, ticker: str, betas: Dict[str, float], attributed: Dict[str, float],
    total_excess: float, r_squared: float, alpha_daily: float, obs: int,
) -> List[str]:
    hints: List[str] = []
    hints.append(
        f"FF5+momentum regression explains {r_squared * 100:.0f}% of {ticker}'s "
        f"excess return variance over the last {obs} trading days."
    )
    mkt_beta = betas.get("KFR.MKT_RF.D")
    if mkt_beta is not None:
        hints.append(f"Market beta of {mkt_beta:.2f}.")
    # Pick top 2 factors by abs(beta), excluding market
    other_factors = sorted(
        [(sid, b) for sid, b in betas.items() if sid != "KFR.MKT_RF.D"],
        key=lambda x: abs(x[1]), reverse=True,
    )[:2]
    for sid, beta in other_factors:
        if abs(beta) < 0.05:
            continue
        label = _FACTOR_LABELS.get(sid, sid)
        direction = _factor_signal_label(sid=sid, beta=beta)
        hints.append(f"{label} beta {beta:+.2f} ({direction}).")
    if abs(total_excess) > 0:
        # Share of excess return explained by each factor's contribution
        contributions = [(sid, contrib) for sid, contrib in attributed.items() if contrib]
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        if contributions:
            top_sid, top_contrib = contributions[0]
            label = _FACTOR_LABELS.get(top_sid, top_sid)
            share = top_contrib / total_excess if total_excess else 0.0
            hints.append(
                f"{label} contributed {top_contrib:+.2%} of excess return "
                f"({share * 100:+.0f}% of the total over the window)."
            )
    if abs(alpha_daily) > 1e-5:
        annualized = alpha_daily * 252
        hints.append(f"Unexplained alpha: {annualized:+.1%} annualized.")
    return hints


def _factor_signal_label(*, sid: str, beta: float) -> str:
    if sid == "KFR.HML.D":
        return "value tilt" if beta >= 0 else "growth tilt"
    if sid == "KFR.MOM.D":
        return "momentum tilt" if beta >= 0 else "contrarian tilt"
    if sid == "KFR.SMB.D":
        return "small-cap tilt" if beta >= 0 else "large-cap tilt"
    if sid == "KFR.RMW.D":
        return "profitability tilt" if beta >= 0 else "low-profitability tilt"
    if sid == "KFR.CMA.D":
        return "conservative-investment tilt" if beta >= 0 else "aggressive-investment tilt"
    if sid == "KFR.MKT_RF.D":
        return "higher market sensitivity" if beta >= 0 else "defensive market tilt"
    return ""


def compute_for_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    """Convenience: fetch price history for a ticker, then run the regression.

    Returns the FactorProfile as a dict, or None if data unavailable.
    """
    try:
        from .data_service import get_data_service
    except Exception:  # pragma: no cover
        return None
    history = get_data_service().get_price_history(ticker, days=400)
    profile = estimate_factor_profile(ticker, history)
    return profile.to_dict() if profile else None
