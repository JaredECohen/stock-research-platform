"""Market data convenience layer (prices + lightweight stats)."""
from __future__ import annotations

from ..finance import risk as risk_lib
from .data_service import get_data_service


def get_price_series(ticker: str, days: int = 252) -> list[dict]:
    return get_data_service().get_price_history(ticker, days) or []


def get_close_series(ticker: str, days: int = 252) -> list[float]:
    rows = get_price_series(ticker, days)
    return [r.get("close") or r.get("adjusted_close") for r in rows if r.get("close") is not None]


def get_current_price(ticker: str) -> float | None:
    """Live intraday price with EOD-close fallback.

    Returns the freshest price available: a provider quote under the
    calendar-aware policy (`quote_service`: 15 minutes in session, until
    the next open after the close, 60 s inside a memo run; a stale row is
    taken only while it is within those 15 minutes), or the last close when
    there is no current quote. `get_quote` answers None for an as-of
    backtest, a malformed ticker, or an exchange calendar that cannot load
    (tzdata missing), so each of those lands on the close here rather than
    raising into the DCF defaults.
    """
    quote = get_data_service().get_quote(ticker)
    if quote and quote.get("price") is not None:
        return float(quote["price"])
    closes = get_close_series(ticker, days=5)
    return closes[-1] if closes else None


def get_basic_stats(ticker: str) -> dict:
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
