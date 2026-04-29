"""Risk metrics for portfolios and individual securities."""
from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Dict, List, Optional


def daily_returns(closes: List[float]) -> List[float]:
    if not closes or len(closes) < 2:
        return []
    return [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
        if closes[i - 1]
    ]


def annualized_volatility(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return pstdev(returns) * math.sqrt(252)


def annualized_return(returns: List[float]) -> float:
    if not returns:
        return 0.0
    avg = mean(returns)
    return ((1 + avg) ** 252) - 1


def sharpe_ratio(returns: List[float], rf_annual: float = 0.04) -> float:
    if not returns:
        return 0.0
    vol = annualized_volatility(returns)
    if vol == 0:
        return 0.0
    return (annualized_return(returns) - rf_annual) / vol


def max_drawdown(closes: List[float]) -> float:
    if not closes:
        return 0.0
    peak = closes[0]
    max_dd = 0.0
    for px in closes:
        if px > peak:
            peak = px
        dd = (px - peak) / peak if peak else 0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def correlation(a: List[float], b: List[float]) -> Optional[float]:
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = a[-n:], b[-n:]
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    if den == 0:
        return None
    return num / den


def beta(stock_returns: List[float], market_returns: List[float]) -> Optional[float]:
    n = min(len(stock_returns), len(market_returns))
    if n < 2:
        return None
    s, m = stock_returns[-n:], market_returns[-n:]
    ms, mm = mean(s), mean(m)
    cov = sum((x - ms) * (y - mm) for x, y in zip(s, m)) / n
    var = sum((y - mm) ** 2 for y in m) / n
    if var == 0:
        return None
    return cov / var


def portfolio_volatility(weights: Dict[str, float], vols: Dict[str, float], corr: Dict[str, Dict[str, float]]) -> float:
    """sigma_p^2 = sum_i sum_j w_i w_j sigma_i sigma_j rho_ij"""
    tickers = list(weights.keys())
    var = 0.0
    for i in tickers:
        for j in tickers:
            wi, wj = weights.get(i, 0), weights.get(j, 0)
            si, sj = vols.get(i, 0), vols.get(j, 0)
            r = corr.get(i, {}).get(j, 1.0 if i == j else 0.3)
            var += wi * wj * si * sj * r
    return math.sqrt(max(0.0, var))


def concentration_metrics(weights: Dict[str, float]) -> Dict[str, float]:
    if not weights:
        return {"top_3": 0.0, "top_5": 0.0, "hhi": 0.0, "n_effective": 0.0}
    sorted_w = sorted(weights.values(), reverse=True)
    hhi = sum(w * w for w in sorted_w)
    return {
        "top_3": sum(sorted_w[:3]),
        "top_5": sum(sorted_w[:5]),
        "hhi": hhi,
        "n_effective": (1 / hhi) if hhi else 0.0,
    }
