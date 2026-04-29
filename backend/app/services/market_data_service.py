"""Market data convenience layer (prices + lightweight stats)."""
from __future__ import annotations

from typing import Dict, List, Optional

from .data_service import get_data_service
from ..finance import risk as risk_lib


def get_price_series(ticker: str, days: int = 252) -> List[Dict]:
    return get_data_service().get_price_history(ticker, days) or []


def get_close_series(ticker: str, days: int = 252) -> List[float]:
    rows = get_price_series(ticker, days)
    return [r.get("close") or r.get("adjusted_close") for r in rows if r.get("close") is not None]


def get_basic_stats(ticker: str) -> Dict:
    closes = get_close_series(ticker)
    if not closes:
        return {}
    rets = risk_lib.daily_returns(closes)
    return {
        "annualized_volatility": risk_lib.annualized_volatility(rets),
        "annualized_return": risk_lib.annualized_return(rets),
        "sharpe": risk_lib.sharpe_ratio(rets),
        "max_drawdown": risk_lib.max_drawdown(closes),
        "last_close": closes[-1],
        "n_obs": len(closes),
    }
