"""Ratio computations used across the comps and screener engines."""
from __future__ import annotations

from typing import Optional


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


def roic(income: dict, balance: dict, tax_rate: float = 0.21) -> Optional[float]:
    op = income.get("operating_income")
    if op is None:
        return None
    nopat = op * (1 - tax_rate)
    debt = (balance.get("total_debt") or 0) or (
        (balance.get("short_term_debt") or 0) + (balance.get("long_term_debt") or 0)
    )
    equity = balance.get("shareholders_equity") or 0
    invested = debt + equity
    if invested <= 0:
        return None
    return nopat / invested


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
