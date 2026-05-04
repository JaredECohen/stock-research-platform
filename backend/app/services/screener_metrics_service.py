"""Compute + persist per-ticker raw metrics for rule-based screening.

Wave 9b Phase 4a. Snapshots the 15-metric vocabulary chosen in the plan
(`docs/UNIVERSE_REFACTOR_PLAN.md` §6.3) into the `screener_metrics`
table. Run nightly alongside `recompute_screener_scores()`; the custom
screen endpoint reads directly from this table.

All metrics derive from data already in the cache:
    - `companies` (market_cap, beta, shares_outstanding)
    - `financial_periods` (income/balance/cash long-format)

Metrics that require sell-side estimates (forward_pe, peg) or
dividend-per-share parsing (dividend_yield) are left None for v1; the
table accepts NULLs and the screener treats unknown values as "fails
the rule" rather than excluding the row.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..database import session_scope
from ..models import Company, FinancialPeriod, ScreenerMetric

log = logging.getLogger(__name__)


def _latest_period_value(
    rows: List[FinancialPeriod], line_item: str,
) -> Optional[float]:
    """Most-recent non-null value for `line_item` from a sorted (desc) list."""
    for r in rows:
        if r.line_item == line_item and r.value is not None:
            return float(r.value)
    return None


def _yoy_growth(
    rows: List[FinancialPeriod], line_item: str,
) -> Optional[float]:
    values = [float(r.value) for r in rows if r.line_item == line_item and r.value is not None]
    if len(values) < 2 or values[1] == 0:
        return None
    return (values[0] - values[1]) / abs(values[1])


def compute_metrics(ticker: str) -> Optional[Dict[str, Any]]:
    """Compute the 15-metric snapshot for one ticker. Returns None when
    insufficient data — caller skips the upsert."""
    ticker = ticker.upper()
    with session_scope() as db:
        company: Optional[Company] = db.get(Company, ticker)
        if company is None:
            return None
        company_kwargs = dict(
            market_cap=company.market_cap,
            beta=company.beta,
        )
        # Project to plain tuples so we don't carry ORM instances out
        # of the session (avoids DetachedInstanceError on attribute
        # access).
        raw = db.execute(
            select(
                FinancialPeriod.statement,
                FinancialPeriod.line_item,
                FinancialPeriod.value,
                FinancialPeriod.period_end,
                FinancialPeriod.period,
            )
            .where(FinancialPeriod.ticker == ticker)
            .order_by(FinancialPeriod.period_end.desc().nulls_last(),
                      FinancialPeriod.period.desc())
        ).all()

    # Group by statement; preserve the (date-desc) ordering above so
    # the first matching row is the most recent value.
    Row = type("Row", (), {})  # lightweight value holder
    def _wrap(t):
        r = Row()
        r.statement, r.line_item, r.value, r.period_end, r.period = t
        return r
    rows = [_wrap(r) for r in raw]
    income = [r for r in rows if r.statement == "income"]
    balance = [r for r in rows if r.statement == "balance"]
    cash = [r for r in rows if r.statement == "cash"]

    revenue = _latest_period_value(income, "revenue")
    gross_profit = _latest_period_value(income, "gross_profit")
    op_income = _latest_period_value(income, "operating_income")
    ebitda = _latest_period_value(income, "ebitda") or _latest_period_value(income, "ebit")
    net_income = _latest_period_value(income, "net_income")
    pretax = _latest_period_value(income, "pretax_income")
    tax = _latest_period_value(income, "tax_expense")
    fcf = _latest_period_value(cash, "free_cash_flow")
    cash_balance = _latest_period_value(balance, "cash_and_equivalents") or 0
    short_inv = _latest_period_value(balance, "short_term_investments") or 0
    total_debt = _latest_period_value(balance, "total_debt")
    if total_debt is None:
        st = _latest_period_value(balance, "short_term_debt") or 0
        lt = _latest_period_value(balance, "long_term_debt") or 0
        total_debt = (st + lt) or None
    equity = _latest_period_value(balance, "shareholders_equity")

    market_cap = company_kwargs["market_cap"]
    enterprise_value = None
    if market_cap is not None:
        enterprise_value = market_cap + (total_debt or 0) - (cash_balance + short_inv)

    def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None or b == 0:
            return None
        return a / b

    metrics = dict(
        ticker=ticker,
        pe_ttm=_safe_div(market_cap, net_income),
        forward_pe=None,   # requires estimates
        peg=None,          # requires forward_pe + growth
        ev_ebitda=_safe_div(enterprise_value, ebitda),
        ev_revenue=_safe_div(enterprise_value, revenue),
        gross_margin=_safe_div(gross_profit, revenue),
        op_margin=_safe_div(op_income, revenue),
        fcf_margin=_safe_div(fcf, revenue),
        roic=None,         # filled below
        roe=_safe_div(net_income, equity),
        debt_to_ebitda=_safe_div(total_debt, ebitda),
        revenue_growth_yoy=_yoy_growth(income, "revenue"),
        dividend_yield=None,  # requires dividend-per-share parsing
        market_cap=market_cap,
        beta=company_kwargs["beta"],
    )

    # ROIC = NOPAT / (debt + equity); approximate NOPAT from operating income.
    if op_income is not None and pretax and pretax != 0:
        tax_rate = (tax or 0) / pretax if pretax > 0 else 0.21
        nopat = op_income * (1 - max(0.0, min(0.5, tax_rate)))
        invested = (total_debt or 0) + (equity or 0)
        if invested > 0:
            metrics["roic"] = nopat / invested

    return metrics


def snapshot_universe() -> Dict[str, int]:
    """Recompute + upsert metrics for every `auto_analysis` ticker.

    Returns counts: `{written, skipped, missing_data}`.
    """
    written = skipped = missing = 0
    with session_scope() as db:
        tickers = [
            t for (t,) in db.execute(
                select(Company.ticker).where(Company.universe_tier == "auto_analysis")
            ).all()
        ]

    for ticker in tickers:
        m = compute_metrics(ticker)
        if m is None:
            missing += 1
            continue
        # Need at least one non-null derived metric for the row to be useful.
        if all(v is None for k, v in m.items() if k not in ("ticker", "market_cap", "beta")):
            skipped += 1
            continue
        with session_scope() as db:
            existing = db.get(ScreenerMetric, ticker)
            if existing is None:
                db.add(ScreenerMetric(**m, last_updated=datetime.utcnow()))
            else:
                for k, v in m.items():
                    setattr(existing, k, v)
                existing.last_updated = datetime.utcnow()
        written += 1
    return {"written": written, "skipped": skipped, "missing_data": missing}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    summary = snapshot_universe()
    log.info("screener_metrics snapshot: %s", summary)
    print(summary)
