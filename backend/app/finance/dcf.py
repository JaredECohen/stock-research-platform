"""Discounted cash flow engine.

A self-contained, dependency-light implementation that takes assumption
dictionaries and returns Pydantic-typed scenarios + sensitivities. It is
exercised both by the /api/dcf endpoint and the valuation agent.
"""
from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Optional

from ..schemas import (
    DCFAssumptions,
    DCFGuardrail,
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


def _consensus_growth_path(estimates: Optional[Dict]) -> Optional[List[float]]:
    """Wave 8I — derive a 5-year revenue-growth path from analyst
    consensus estimates when present. Returns None when the estimates
    payload doesn't carry usable revenue rows.

    Accepts a few common shapes:
    - `estimates["revenue"]` as a list of {period, value} (chronological).
    - `estimates["revenue_growth"]` as a list of floats (already deltas).
    - `estimates["analyst_growth"]` (single float) — used as a flat path.
    Falls through to None on anything else; caller fades historical
    trend instead.
    """
    if not isinstance(estimates, dict):
        return None
    # Direct growth list — easiest case.
    rg = estimates.get("revenue_growth")
    if isinstance(rg, list) and rg:
        out: List[float] = []
        for v in rg[:5]:
            try:
                out.append(round(max(-0.20, min(0.50, float(v))), 4))
            except (TypeError, ValueError):
                continue
        if len(out) >= 1:
            # Pad with the last value if fewer than 5 periods.
            while len(out) < 5:
                out.append(out[-1])
            return out
    # Single analyst growth number — apply as flat path.
    g = estimates.get("analyst_growth")
    if isinstance(g, (int, float)):
        gv = round(max(-0.20, min(0.50, float(g))), 4)
        return [gv] * 5
    # Revenue-level estimates → derive YoY deltas.
    rev_rows = estimates.get("revenue")
    if isinstance(rev_rows, list) and len(rev_rows) >= 2:
        vals: List[float] = []
        for r in rev_rows:
            if isinstance(r, dict):
                v = r.get("value") or r.get("revenue")
            else:
                v = r
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            vals.append(v)
        if len(vals) >= 2:
            deltas = []
            for i in range(1, min(6, len(vals))):
                prev, cur = vals[i - 1], vals[i]
                if prev:
                    deltas.append(round(max(-0.20, min(0.50, (cur - prev) / abs(prev))), 4))
            if deltas:
                while len(deltas) < 5:
                    deltas.append(deltas[-1])
                return deltas[:5]
    return None


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
    analyst_estimates: Optional[Dict] = None,
    margin_mean_reversion: bool = False,
    cohort_op_margin: Optional[float] = None,
) -> DCFAssumptions:
    """Build sane base-case assumptions from a few years of statements.

    Wave 8I: when `analyst_estimates` is supplied, the consensus growth
    path is the *starting point* for `revenue_growth` instead of the
    historical-trend extrapolation. Locked decision: the AI agent
    layer (Wave 5A `dcf_updater`) is responsible for diverging from
    consensus and must justify each change with a rationale (per-cycle
    delta still capped at ±20%). The default builder defers to consensus.

    Wave 10 — `margin_mean_reversion=True` swaps the held-flat margin
    path for a glide that reverts the trailing-3yr operating margin
    toward `cohort_op_margin` (or 18% as a generic fallback) over the
    5-year forecast. Recommended for cyclicals at peak — without
    this the DCF systematically over-values names whose trailing-3yr
    margin happens to be the cycle high.
    """
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

    # Wave 8I: consensus first, historical fallback.
    consensus_path = _consensus_growth_path(analyst_estimates)
    if consensus_path is not None:
        base_growth = consensus_path
    else:
        growth_trend = _trend(revenues)
        # Cap growth between -5% and 30% to avoid runaway extrapolation
        growth_trend = max(-0.05, min(0.30, growth_trend))
        # Fade growth toward terminal over 5 years
        base_growth = []
        for i in range(5):
            weight = (5 - i) / 5
            g = growth_trend * weight + 0.04 * (1 - weight)
            base_growth.append(round(g, 4))

    # Wave 8Q — default-preserve current profitability. Previously baked
    # +50bps/yr expansion into every base margin which produced
    # systematically bearish DCFs (later years' FCFs got higher implied
    # margins than reality, so the discount applied to current price felt
    # too large). Now hold the trailing 3-yr average flat across the
    # explicit forecast; the LLM updater (Wave 5A) is the only thing
    # allowed to diverge, and only with a per-field rationale.
    op_margin_recent = _avg([_safe_div(o, r) for o, r in zip(op_incomes[-3:], revenues[-3:]) if r])
    op_margin_recent = max(0.02, min(0.65, op_margin_recent or 0.18))
    if margin_mean_reversion:
        # Glide from trailing margin → cohort/ secular norm over 5y.
        # Cohort target supplied by caller (sector_research) when
        # available; else 18% as a "typical mature US large-cap"
        # anchor that's mid-pack across the S&P 500.
        target = max(0.05, min(0.45, cohort_op_margin or 0.18))
        base_margin = []
        for i in range(5):
            weight = (5 - i) / 5  # 1.0 → 0.2 over 5 years
            m = op_margin_recent * weight + target * (1 - weight)
            base_margin.append(round(m, 4))
    else:
        base_margin = [round(op_margin_recent, 4)] * 5

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
    # Wave 10j — terminal value uses Gordon Growth as the headline.
    # The exit-multiple value is still computed for transparency + the
    # exit-multiple sensitivity table, but it does NOT influence the
    # core implied price. Per Damodaran: exit multiples smuggle a
    # market-cycle assumption into a fundamentals model and shouldn't
    # be averaged with Gordon. The user's `exit_ebitda_multiple`
    # assumption now only feeds the sensitivity table downstream.
    # `enterprise_value_blended` is kept as a field name for backward
    # compatibility, but its value is now Gordon (not a 50/50 blend).
    ev_blended = ev_gordon

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


def _scenario_with_exit_terminal(
    assumptions: DCFAssumptions, exit_multiple: float,
) -> float:
    """Run a single DCF scenario but use the *exit-multiple terminal*
    instead of Gordon Growth. Returns the implied share price.

    Wave 10j — used by `build_exit_multiple_sensitivity` to show what
    the implied price would be if the user picked exit-multiple
    terminal instead of the default Gordon. This is the *only* place
    `exit_ebitda_multiple` actually moves the implied price.
    """
    a = copy.deepcopy(assumptions)
    a.exit_ebitda_multiple = exit_multiple
    years = max(len(a.revenue_growth), len(a.operating_margin)) or 5
    projections: List[DCFYearProjection] = []
    prev_revenue = a.base_revenue or 1.0
    for year in range(1, years + 1):
        proj = _project_year(prev_revenue, year, a)
        projections.append(proj)
        prev_revenue = proj.revenue
    pv_explicit = sum(p.pv_fcff for p in projections)
    last = projections[-1]
    final_ebitda = last.ebit + last.da
    tv_exit = final_ebitda * exit_multiple
    pv_terminal_exit = tv_exit / ((1 + a.wacc) ** years)
    ev = pv_explicit + pv_terminal_exit
    equity_value = ev - a.net_debt
    if a.diluted_shares and a.diluted_shares > 0:
        return equity_value / a.diluted_shares
    return 0.0


def build_exit_multiple_sensitivity(
    base: DCFAssumptions,
) -> DCFSensitivity:
    """Wave 10j — what would the implied price be under exit-multiple
    terminal across a range of multiples × bear/base/bull?

    Headline DCF uses Gordon Growth (Wave 10j). This sensitivity is
    the cross-check: 5 multiples (centered on the user's input,
    bracketed ±5x) × 3 scenarios (bear / base / bull). Lets the user
    see the dispersion an exit-multiple framing would produce without
    contaminating the core DCF with the multiple assumption.
    """
    bull = _bull_assumptions(base)
    bear = _bear_assumptions(base)

    # Five multiples centered on user input, bracketed.
    base_m = max(5.0, float(base.exit_ebitda_multiple))
    half_span = max(2.5, base_m * 0.4)
    multiples: List[float] = [
        round(max(3.0, base_m + (i - 2) * (half_span / 2)), 1)
        for i in range(5)
    ]

    cells: List[SensitivityCell] = []
    scenarios = [
        ("bear", bear),
        ("base", base),
        ("bull", bull),
    ]
    for m in multiples:
        for col_label, scen in scenarios:
            implied = _scenario_with_exit_terminal(scen, m)
            cells.append(SensitivityCell(
                row_label=f"{m:.1f}x",
                col_label=col_label,
                value=implied,
            ))
    return DCFSensitivity(
        name="Exit Multiple Sensitivity (cross-check vs Gordon headline)",
        row_axis="Exit EBITDA",
        col_axis="Scenario",
        rows=multiples,
        cols=[0.0, 1.0, 2.0],  # placeholder numeric for legacy schema
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

    # Wave 10j — exit-multiple cross-check (5 multiples × bear/base/bull).
    # The core DCF uses Gordon Growth; this surface tells the user
    # "if you preferred exit-multiple terminal, the dispersion would
    # look like this." Replaces the prior 50/50 blend.
    sens.append(build_exit_multiple_sensitivity(base))

    return sens


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def check_dcf_realism(
    base: DCFScenario, *, ticker: str = "",
) -> List[DCFGuardrail]:
    """Wave 10 — sanity-check the DCF against cohort distribution.

    The user's frustration: "valuations seem off." This guardrail
    surfaces the most common failure modes so the PM (and the user)
    can audit the model before trusting the implied price.

    Checks:
    1. **Terminal-value disagreement.** When Gordon and exit-multiple
       blended-50/50 disagree by >25%, the model's two terminal
       methods are talking past each other — the assumptions haven't
       been thought through. Flagged at WARN.
    2. **Implied Y5 EV/EBITDA outside cohort.** When the year-5
       implied EV/EBITDA exceeds the comps cohort 90th percentile,
       the model is implicitly pricing the company as a category
       outlier. Flagged at WARN.
    3. **Negative or absurd implied price.** Implied share price
       <= 0, or > 10x current price (likely a units / share count
       error). Flagged at ERROR.
    4. **Operating margin > 60%.** If the projected steady-state
       margin tops 60%, that's an extraordinary claim — flagged at
       WARN unless it's a software / payments name (caller can
       suppress via business-context if needed).
    """
    guardrails: List[DCFGuardrail] = []

    # 1) Terminal value disagreement
    if base.enterprise_value_gordon and base.enterprise_value_exit:
        a = base.enterprise_value_gordon
        b = base.enterprise_value_exit
        denom = (abs(a) + abs(b)) / 2 or 1.0
        disagreement = abs(a - b) / denom
        if disagreement > 0.25:
            guardrails.append(DCFGuardrail(
                severity="warn",
                metric="terminal_disagreement",
                message=(
                    f"Gordon vs exit-multiple terminal values disagree "
                    f"{disagreement:.0%}. Reconsider terminal growth or "
                    f"exit EBITDA multiple — model is internally split."
                ),
                value=disagreement,
            ))

    # 2) Implied Y5 EV/EBITDA vs cohort
    try:
        if base.projections and ticker:
            last_proj = base.projections[-1]
            final_ebitda = last_proj.ebit + last_proj.da
            if final_ebitda > 0:
                implied_y5_multiple = (
                    base.enterprise_value_blended / final_ebitda
                )
                cohort_p90 = _cohort_p90_ev_ebitda(ticker)
                if cohort_p90 and implied_y5_multiple > cohort_p90:
                    guardrails.append(DCFGuardrail(
                        severity="warn",
                        metric="implied_y5_ev_ebitda",
                        message=(
                            f"Year-5 implied EV/EBITDA {implied_y5_multiple:.1f}x "
                            f"exceeds cohort 90th percentile ({cohort_p90:.1f}x). "
                            f"Model is pricing this as a category outlier."
                        ),
                        value=implied_y5_multiple,
                        cohort_p90=cohort_p90,
                    ))
    except Exception:  # pragma: no cover — guardrails never fail loudly
        pass

    # 3) Absurd implied price
    if base.implied_share_price <= 0:
        guardrails.append(DCFGuardrail(
            severity="error",
            metric="implied_share_price",
            message="Implied share price is non-positive — check net debt and share count inputs.",
            value=base.implied_share_price,
        ))
    elif (
        base.assumptions.current_price
        and base.implied_share_price > base.assumptions.current_price * 10
    ):
        guardrails.append(DCFGuardrail(
            severity="error",
            metric="implied_share_price_runaway",
            message=(
                f"Implied share price ${base.implied_share_price:,.0f} is >10x current "
                f"${base.assumptions.current_price:,.0f}. Likely a units or share-count "
                f"input error."
            ),
            value=base.implied_share_price,
        ))

    # 4) Operating margin > 60% steady-state
    if base.assumptions.operating_margin:
        peak_margin = max(base.assumptions.operating_margin)
        if peak_margin > 0.60:
            guardrails.append(DCFGuardrail(
                severity="warn",
                metric="peak_operating_margin",
                message=(
                    f"Peak projected operating margin {peak_margin:.0%} is unusual — "
                    f"defensible for top-tier software/payments names; flag for "
                    f"others."
                ),
                value=peak_margin,
            ))

    return guardrails


def _cohort_p90_ev_ebitda(ticker: str) -> Optional[float]:
    """Pull the cohort 90th-percentile EV/EBITDA from comps.

    Returns None when comps are unavailable (bare ticker, sparse
    universe). Defensive — never raises.
    """
    try:
        from ..services.valuation_service import build_comps
        comps = build_comps(ticker)
        if comps is None:
            return None
        peer_multiples = sorted(
            p.ev_ebitda for p in comps.peers if p.ev_ebitda and p.ev_ebitda > 0
        )
        if len(peer_multiples) < 3:
            return None
        # 90th percentile via linear interpolation
        idx = 0.9 * (len(peer_multiples) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(peer_multiples) - 1)
        frac = idx - lo
        return peer_multiples[lo] * (1 - frac) + peer_multiples[hi] * frac
    except Exception:
        return None


def build_full_dcf(ticker: str, base_assumptions: DCFAssumptions) -> DCFResult:
    """Run base/bull/bear scenarios + sensitivities and synthesize a summary."""
    base = run_dcf(base_assumptions, scenario_name="base", label="Base case")
    bull = run_dcf(_bull_assumptions(base_assumptions), scenario_name="bull", label="Bull case")
    bear = run_dcf(_bear_assumptions(base_assumptions), scenario_name="bear", label="Bear case")
    sens = build_default_sensitivities(base_assumptions)
    guardrails = check_dcf_realism(base, ticker=ticker)

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
    if guardrails:
        n_warn = sum(1 for g in guardrails if g.severity == "warn")
        n_err = sum(1 for g in guardrails if g.severity == "error")
        summary_parts.append(f"⚠ Model warnings: {n_err} errors, {n_warn} warns")
    return DCFResult(
        ticker=ticker,
        current_price=base_assumptions.current_price,
        base=base,
        bull=bull,
        bear=bear,
        sensitivities=sens,
        summary=". ".join(summary_parts),
        guardrails=guardrails,
    )
