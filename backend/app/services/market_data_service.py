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


def get_current_price(ticker: str) -> Optional[float]:
    """Live intraday price with EOD-close fallback.

    Returns the freshest price available: a 60s-cached quote during
    market hours, or yesterday's close if the quote chain misses (or
    when an as-of backtest is active and `get_quote` short-circuits
    to None).
    """
    quote = get_data_service().get_quote(ticker)
    if quote and quote.get("price") is not None:
        return float(quote["price"])
    closes = get_close_series(ticker, days=5)
    return closes[-1] if closes else None


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
