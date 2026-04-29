"""Discounted cash flow engine.

A self-contained, dependency-light implementation that takes assumption
dictionaries and returns Pydantic-typed scenarios + sensitivities. It is
exercised both by the /api/dcf endpoint and the valuation agent.
"""
from __future__ import annotations

import copy
from typing import Iterable, List

from ..schemas import (
    DCFAssumptions,
    DCFResult,
    DCFScenario,
    DCFSensitivity,
    DCFYearProjection,
    SensitivityCell,
)


# ---------------------------------------------------------------------------
# Default assumptions from historical data
# ---------------------------------------------------------------------------

def _safe_div(n: float, d: float, default: float = 0.0) -> float:
    if d in (None, 0) or d == 0.0:
        return default
    try:
        return n / d
    except ZeroDivisionError:
        return default


def _avg(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _trend(values: List[float]) -> float:
    """Naive trend = average year-over-year growth, last 3 periods."""
    if not values or len(values) < 2:
        return 0.0
    growths = []
    for i in range(1, len(values)):
        prev, cur = values[i - 1], values[i]
        if prev and prev != 0:
            growths.append((cur - prev) / abs(prev))
    if not growths:
        return 0.0
    last = growths[-3:] if len(growths) >= 3 else growths
    return sum(last) / len(last)


def derive_default_assumptions(
    income_statements: List[dict],
    cash_flows: List[dict],
    balance_sheets: List[dict],
    *,
    current_price: float,
    diluted_shares: float,
    risk_free_rate: float = 0.042,
    equity_risk_premium: float = 0.05,
    beta: float = 1.0,
    pretax_cost_of_debt: float = 0.055,
    target_debt_weight: float = 0.15,
) -> DCFAssumptions:
    """Build sane base-case assumptions from a few years of statements."""
    # Sort oldest -> newest
    income_statements = sorted(income_statements, key=lambda r: r.get("period", ""))
    cash_flows = sorted(cash_flows, key=lambda r: r.get("period", ""))
    balance_sheets = sorted(balance_sheets, key=lambda r: r.get("period", ""))

    revenues = [r.get("revenue", 0.0) or 0.0 for r in income_statements]
    op_incomes = [r.get("operating_income", 0.0) or 0.0 for r in income_statements]
    net_incomes = [r.get("net_income", 0.0) or 0.0 for r in income_statements]
    pretax = [r.get("pretax_income", 0.0) or 0.0 for r in income_statements]
    tax_exp = [r.get("tax_expense", 0.0) or 0.0 for r in income_statements]
    capex_vals = [abs(r.get("capex", 0.0) or 0.0) for r in cash_flows]
    da_vals = [r.get("depreciation_and_amortization", 0.0) or 0.0 for r in cash_flows]
    nwc_vals = [r.get("change_in_working_capital", 0.0) or 0.0 for r in cash_flows]

    base_revenue = revenues[-1] if revenues else 0.0
    growth_trend = _trend(revenues)
    # Cap growth between -5% and 30% to avoid runaway extrapolation
    growth_trend = max(-0.05, min(0.30, growth_trend))

    # Fade growth toward terminal over 5 years
    base_growth: List[float] = []
    for i in range(5):
        weight = (5 - i) / 5
        g = growth_trend * weight + 0.04 * (1 - weight)
        base_growth.append(round(g, 4))

    op_margin_recent = _avg([_safe_div(o, r) for o, r in zip(op_incomes[-3:], revenues[-3:]) if r])
    op_margin_recent = max(0.02, min(0.55, op_margin_recent or 0.18))
    base_margin = [round(op_margin_recent + (0.005 * i), 4) for i in range(5)]
    base_margin = [min(m, 0.60) for m in base_margin]

    eff_tax = _avg([_safe_div(t, p) for t, p in zip(tax_exp[-3:], pretax[-3:]) if p])
    eff_tax = max(0.10, min(0.30, eff_tax or 0.21))

    da_pct = _avg([_safe_div(d, r) for d, r in zip(da_vals[-3:], revenues[-3:]) if r]) or 0.05
    capex_pct = _avg([_safe_div(c, r) for c, r in zip(capex_vals[-3:], revenues[-3:]) if r]) or 0.05
    nwc_pct = _avg([_safe_div(c, r) for c, r in zip(nwc_vals[-3:], revenues[-3:]) if r]) or 0.02

    last_bs = balance_sheets[-1] if balance_sheets else {}
    cash = (last_bs.get("cash_and_equivalents", 0) or 0) + (last_bs.get("short_term_investments", 0) or 0)
    debt = (last_bs.get("total_debt", 0) or 0) or (
        (last_bs.get("short_term_debt", 0) or 0) + (last_bs.get("long_term_debt", 0) or 0)
    )
    net_debt = debt - cash

    # WACC: equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt
    cost_of_equity = risk_free_rate + beta * equity_risk_premium
    after_tax_cod = pretax_cost_of_debt * (1 - eff_tax)
    equity_weight = 1 - target_debt_weight
    wacc = equity_weight * cost_of_equity + target_debt_weight * after_tax_cod
    wacc = max(0.06, min(0.14, wacc))

    return DCFAssumptions(
        revenue_growth=base_growth,
        operating_margin=base_margin,
        tax_rate=round(eff_tax, 4),
        da_pct_revenue=round(max(0.005, min(0.15, da_pct)), 4),
        capex_pct_revenue=round(max(0.005, min(0.15, capex_pct)), 4),
        nwc_pct_revenue=round(max(-0.05, min(0.10, nwc_pct)), 4),
        terminal_growth=0.025,
        exit_ebitda_multiple=15.0,
        wacc=round(wacc, 4),
        base_revenue=base_revenue,
        net_debt=net_debt,
        diluted_shares=diluted_shares or 0.0,
        current_price=current_price or 0.0,
    )


# ---------------------------------------------------------------------------
# Project + value
# ---------------------------------------------------------------------------

def _project_year(prev_revenue: float, year: int, assumptions: DCFAssumptions) -> DCFYearProjection:
    growth = assumptions.revenue_growth[min(year - 1, len(assumptions.revenue_growth) - 1)]
    margin = assumptions.operating_margin[min(year - 1, len(assumptions.operating_margin) - 1)]
    revenue = prev_revenue * (1 + growth)
    ebit = revenue * margin
    nopat = ebit * (1 - assumptions.tax_rate)
    da = revenue * assumptions.da_pct_revenue
    capex = revenue * assumptions.capex_pct_revenue
    change_nwc = (revenue - prev_revenue) * assumptions.nwc_pct_revenue
    fcff = nopat + da - capex - change_nwc
    discount_factor = 1 / ((1 + assumptions.wacc) ** year)
    return DCFYearProjection(
        year=year,
        revenue=revenue,
        ebit=ebit,
        nopat=nopat,
        da=da,
        capex=capex,
        change_nwc=change_nwc,
        fcff=fcff,
        discount_factor=discount_factor,
        pv_fcff=fcff * discount_factor,
    )


def run_dcf(assumptions: DCFAssumptions, *, scenario_name: str = "base", label: str = "Base case") -> DCFScenario:
    """Run a single DCF scenario, returning a fully populated DCFScenario."""
    years = max(len(assumptions.revenue_growth), len(assumptions.operating_margin))
    if years == 0:
        years = 5
    projections: List[DCFYearProjection] = []
    prev_revenue = assumptions.base_revenue or 1.0
    for year in range(1, years + 1):
        proj = _project_year(prev_revenue, year, assumptions)
        projections.append(proj)
        prev_revenue = proj.revenue

    pv_explicit = sum(p.pv_fcff for p in projections)
    last = projections[-1]

    # Terminal: Gordon Growth on FCFF
    gordon_denom = (assumptions.wacc - assumptions.terminal_growth)
    gordon_denom = gordon_denom if gordon_denom > 0.005 else 0.005
    tv_gordon = (last.fcff * (1 + assumptions.terminal_growth)) / gordon_denom

    # Terminal: exit EBITDA multiple
    final_ebitda = last.ebit + last.da
    tv_exit = final_ebitda * assumptions.exit_ebitda_multiple

    pv_terminal_gordon = tv_gordon / ((1 + assumptions.wacc) ** years)
    pv_terminal_exit = tv_exit / ((1 + assumptions.wacc) ** years)

    ev_gordon = pv_explicit + pv_terminal_gordon
    ev_exit = pv_explicit + pv_terminal_exit
    ev_blended = (ev_gordon + ev_exit) / 2

    equity_value = ev_blended - assumptions.net_debt
    if assumptions.diluted_shares and assumptions.diluted_shares > 0:
        implied_share_price = equity_value / assumptions.diluted_shares
    else:
        implied_share_price = 0.0
    upside_pct = 0.0
    if assumptions.current_price:
        upside_pct = (implied_share_price - assumptions.current_price) / assumptions.current_price

    return DCFScenario(
        name=scenario_name,  # type: ignore[arg-type]
        label=label,
        assumptions=assumptions,
        projections=projections,
        pv_explicit=pv_explicit,
        terminal_value_gordon=tv_gordon,
        terminal_value_exit_multiple=tv_exit,
        pv_terminal_gordon=pv_terminal_gordon,
        pv_terminal_exit=pv_terminal_exit,
        enterprise_value_gordon=ev_gordon,
        enterprise_value_exit=ev_exit,
        enterprise_value_blended=ev_blended,
        equity_value=equity_value,
        implied_share_price=implied_share_price,
        upside_pct=upside_pct,
    )


def _bull_assumptions(base: DCFAssumptions) -> DCFAssumptions:
    a = copy.deepcopy(base)
    a.revenue_growth = [min(0.40, g + 0.04) for g in a.revenue_growth]
    a.operating_margin = [min(0.65, m + 0.02) for m in a.operating_margin]
    a.terminal_growth = min(0.04, a.terminal_growth + 0.005)
    a.exit_ebitda_multiple = a.exit_ebitda_multiple + 3.0
    a.wacc = max(0.05, a.wacc - 0.005)
    return a


def _bear_assumptions(base: DCFAssumptions) -> DCFAssumptions:
    a = copy.deepcopy(base)
    a.revenue_growth = [max(-0.10, g - 0.04) for g in a.revenue_growth]
    a.operating_margin = [max(0.01, m - 0.03) for m in a.operating_margin]
    a.terminal_growth = max(0.01, a.terminal_growth - 0.005)
    a.exit_ebitda_multiple = max(5.0, a.exit_ebitda_multiple - 3.0)
    a.wacc = a.wacc + 0.01
    return a


# ---------------------------------------------------------------------------
# Sensitivities
# ---------------------------------------------------------------------------

def _build_sensitivity(
    base: DCFAssumptions,
    *,
    name: str,
    row_axis: str,
    col_axis: str,
    rows: List[float],
    cols: List[float],
    setter,
) -> DCFSensitivity:
    cells: List[SensitivityCell] = []
    for r in rows:
        for c in cols:
            assumptions = copy.deepcopy(base)
            setter(assumptions, r, c)
            scenario = run_dcf(assumptions, scenario_name="base", label="sens")
            cells.append(SensitivityCell(
                row_label=f"{r:.2%}" if abs(r) < 1 else f"{r:.1f}",
                col_label=f"{c:.2%}" if abs(c) < 1 else f"{c:.1f}",
                value=scenario.implied_share_price,
            ))
    return DCFSensitivity(
        name=name,
        row_axis=row_axis,
        col_axis=col_axis,
        rows=rows,
        cols=cols,
        cells=cells,
    )


def build_default_sensitivities(base: DCFAssumptions) -> List[DCFSensitivity]:
    sens: List[DCFSensitivity] = []

    def set_wacc_terminal(a: DCFAssumptions, w: float, g: float) -> None:
        a.wacc = w
        a.terminal_growth = g

    sens.append(_build_sensitivity(
        base,
        name="WACC vs Terminal Growth",
        row_axis="WACC",
        col_axis="Terminal Growth",
        rows=[round(base.wacc - 0.02 + 0.01 * i, 4) for i in range(5)],
        cols=[0.015, 0.02, 0.025, 0.03, 0.035],
        setter=set_wacc_terminal,
    ))

    def set_wacc_exit(a: DCFAssumptions, w: float, m: float) -> None:
        a.wacc = w
        a.exit_ebitda_multiple = m

    sens.append(_build_sensitivity(
        base,
        name="WACC vs Exit EBITDA Multiple",
        row_axis="WACC",
        col_axis="Exit EBITDA",
        rows=[round(base.wacc - 0.02 + 0.01 * i, 4) for i in range(5)],
        cols=[max(5.0, base.exit_ebitda_multiple - 5 + 2.5 * i) for i in range(5)],
        setter=set_wacc_exit,
    ))

    def set_growth_margin(a: DCFAssumptions, g: float, m: float) -> None:
        a.revenue_growth = [g - 0.005 * i for i in range(len(a.revenue_growth))]
        a.operating_margin = [m for _ in a.operating_margin]

    base_g0 = base.revenue_growth[0] if base.revenue_growth else 0.08
    base_m_last = base.operating_margin[-1] if base.operating_margin else 0.25
    sens.append(_build_sensitivity(
        base,
        name="Year-1 Revenue Growth vs Terminal Op Margin",
        row_axis="Yr1 Growth",
        col_axis="Term Op Margin",
        rows=[round(base_g0 - 0.04 + 0.02 * i, 4) for i in range(5)],
        cols=[round(max(0.01, base_m_last - 0.04 + 0.02 * i), 4) for i in range(5)],
        setter=set_growth_margin,
    ))

    return sens


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def build_full_dcf(ticker: str, base_assumptions: DCFAssumptions) -> DCFResult:
    """Run base/bull/bear scenarios + sensitivities and synthesize a summary."""
    base = run_dcf(base_assumptions, scenario_name="base", label="Base case")
    bull = run_dcf(_bull_assumptions(base_assumptions), scenario_name="bull", label="Bull case")
    bear = run_dcf(_bear_assumptions(base_assumptions), scenario_name="bear", label="Bear case")
    sens = build_default_sensitivities(base_assumptions)

    summary_parts: List[str] = []
    if base_assumptions.current_price:
        summary_parts.append(
            f"Base case implied price ${base.implied_share_price:,.2f} "
            f"vs current ${base_assumptions.current_price:,.2f} "
            f"({base.upside_pct:+.1%})"
        )
    summary_parts.append(
        f"Bull ${bull.implied_share_price:,.2f} ({bull.upside_pct:+.1%}) | "
        f"Bear ${bear.implied_share_price:,.2f} ({bear.upside_pct:+.1%})"
    )
    return DCFResult(
        ticker=ticker,
        current_price=base_assumptions.current_price,
        base=base,
        bull=bull,
        bear=bear,
        sensitivities=sens,
        summary=". ".join(summary_parts),
    )
