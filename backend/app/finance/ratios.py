"""Ratio computations used across the comps and screener engines."""
from __future__ import annotations

from typing import Optional, Tuple


def safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None or d == 0:
        return None
    return n / d


def gross_margin(income: dict) -> Optional[float]:
    return safe_div(income.get("gross_profit"), income.get("revenue"))


def operating_margin(income: dict) -> Optional[float]:
    return safe_div(income.get("operating_income"), income.get("revenue"))


def net_margin(income: dict) -> Optional[float]:
    return safe_div(income.get("net_income"), income.get("revenue"))


def ebitda(income: dict, cash_flow: dict) -> Optional[float]:
    op = income.get("operating_income")
    da = cash_flow.get("depreciation_and_amortization") if cash_flow else None
    if op is None:
        return None
    return op + (da or 0.0)


def ebitda_margin(income: dict, cash_flow: dict) -> Optional[float]:
    val = ebitda(income, cash_flow)
    return safe_div(val, income.get("revenue"))


def fcf_margin(cash_flow: dict, income: dict) -> Optional[float]:
    return safe_div(cash_flow.get("free_cash_flow"), income.get("revenue"))


def revenue_growth(prior: dict, current: dict) -> Optional[float]:
    p, c = prior.get("revenue"), current.get("revenue")
    if not p or p == 0:
        return None
    return (c - p) / abs(p)


def roe(income: dict, balance: dict) -> Optional[float]:
    return safe_div(income.get("net_income"), balance.get("shareholders_equity"))


def roa(income: dict, balance: dict) -> Optional[float]:
    return safe_div(income.get("net_income"), balance.get("total_assets"))


# U.S. federal statutory corporate rate (TCJA, 2018-). Used only as a
# last resort for *profitable* companies whose filings don't carry a
# credible tax line: the true effective rate for S&P 500 names clusters
# around it, so it is a defensible estimate of NOPAT. It is never applied
# to a loss-maker — see `roic_with_provenance` for why.
STATUTORY_TAX_RATE_FALLBACK = 0.21

# An effective rate above this is not a rate we can trust to deflate
# operating income. Real ongoing rates top out in the mid-30s even for
# companies in high-tax jurisdictions; ratios above 0.5 come from tiny
# pretax denominators (a near-breakeven year), one-off charges (deferred
# tax revaluations, repatriation tolls) or refunds netted against
# pretax — none of which say anything about next year's NOPAT.
EFFECTIVE_TAX_RATE_CEILING = 0.50

# Provenance labels returned by `roic_with_provenance`. The screener
# stores/logs these so a NULL ROIC can be told apart by cause.
ROIC_EFFECTIVE_RATE = "effective_rate"
ROIC_STATUTORY_FALLBACK = "statutory_fallback"
ROIC_UNKNOWN_LOSS_MAKER = "unknown_loss_maker"
ROIC_NO_OPERATING_INCOME = "no_operating_income"
ROIC_NO_INVESTED_CAPITAL = "no_invested_capital"


def effective_tax_rate(income: dict) -> Optional[float]:
    """`tax_expense / pretax_income`, or None when the ratio isn't credible.

    Credible means: both lines present, positive pretax income (a loss
    year's "rate" is meaningless — a refund over a negative denominator
    is a positive number that describes nothing), and a result inside
    [0, EFFECTIVE_TAX_RATE_CEILING]. Negative tax on positive pretax is
    a refund/credit, not a rate, and is likewise rejected.
    """
    pretax = income.get("pretax_income")
    tax = income.get("tax_expense")
    if pretax is None or tax is None or pretax <= 0:
        return None
    rate = tax / pretax
    if rate < 0.0 or rate > EFFECTIVE_TAX_RATE_CEILING:
        return None
    return rate


def invested_capital(balance: dict) -> float:
    """Debt + equity. `total_debt` wins when reported; otherwise the
    short/long split is summed. Missing lines count as zero."""
    debt = (balance.get("total_debt") or 0) or (
        (balance.get("short_term_debt") or 0) + (balance.get("long_term_debt") or 0)
    )
    equity = balance.get("shareholders_equity") or 0
    return debt + equity


def roic_with_provenance(
    income: dict, balance: dict, tax_rate: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """ROIC = operating_income * (1 - tax_rate) / (debt + equity), plus a
    label saying how the tax rate was resolved (or why ROIC is None).

    Tax-rate resolution, in order:
      1. explicit `tax_rate` argument (caller knows better);
      2. the company's own effective rate when credible;
      3. `STATUTORY_TAX_RATE_FALLBACK` — but only for positive operating
         income. For a loss-maker with no credible rate we return None
         rather than deflating the loss by an invented 21%: the sign of
         the resulting "ROIC" would be right but its magnitude would be
         fabricated, and a screener rule like `roic < -0.05` would then
         rank companies on a number nobody measured. Missing data must
         surface as missing.

    The single source of truth for ROIC: the screener and comps engines
    must call this (or `roic`) rather than re-deriving NOPAT locally.
    """
    op = income.get("operating_income")
    if op is None:
        return None, ROIC_NO_OPERATING_INCOME
    invested = invested_capital(balance)
    if invested <= 0:
        return None, ROIC_NO_INVESTED_CAPITAL

    if tax_rate is not None:
        rate, source = tax_rate, ROIC_EFFECTIVE_RATE
    else:
        eff = effective_tax_rate(income)
        if eff is not None:
            rate, source = eff, ROIC_EFFECTIVE_RATE
        elif op > 0:
            rate, source = STATUTORY_TAX_RATE_FALLBACK, ROIC_STATUTORY_FALLBACK
        else:
            return None, ROIC_UNKNOWN_LOSS_MAKER
    return op * (1 - rate) / invested, source


def roic(income: dict, balance: dict, tax_rate: Optional[float] = None) -> Optional[float]:
    """Value-only view of `roic_with_provenance`; same resolution rules."""
    return roic_with_provenance(income, balance, tax_rate)[0]


def net_debt(balance: dict) -> float:
    cash = (balance.get("cash_and_equivalents") or 0) + (balance.get("short_term_investments") or 0)
    debt = (balance.get("total_debt") or 0) or (
        (balance.get("short_term_debt") or 0) + (balance.get("long_term_debt") or 0)
    )
    return debt - cash


def enterprise_value(market_cap: float, balance: dict) -> Optional[float]:
    if market_cap is None:
        return None
    return market_cap + net_debt(balance)


def ev_revenue(market_cap: float, balance: dict, income: dict) -> Optional[float]:
    ev = enterprise_value(market_cap, balance)
    return safe_div(ev, income.get("revenue"))


def ev_ebitda(market_cap: float, balance: dict, income: dict, cash_flow: dict) -> Optional[float]:
    ev = enterprise_value(market_cap, balance)
    return safe_div(ev, ebitda(income, cash_flow))


def pe_ratio(market_cap: float, income: dict) -> Optional[float]:
    return safe_div(market_cap, income.get("net_income"))


def p_fcf(market_cap: float, cash_flow: dict) -> Optional[float]:
    return safe_div(market_cap, cash_flow.get("free_cash_flow"))


def fcf_yield(market_cap: float, cash_flow: dict) -> Optional[float]:
    val = safe_div(cash_flow.get("free_cash_flow"), market_cap)
    return val
