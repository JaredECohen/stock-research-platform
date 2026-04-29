"""Peer-comparable analysis engine."""
from __future__ import annotations

from statistics import median
from typing import Dict, List, Optional

from ..schemas import CompsResult, CompsRow
from . import ratios as R


def _percentile_rank(values: List[float], target: float) -> float:
    """Return percentile of `target` within `values` (0-1)."""
    clean = [v for v in values if v is not None]
    if not clean:
        return 0.5
    below = sum(1 for v in clean if v < target)
    return below / len(clean)


def build_row(
    ticker: str,
    company_name: str,
    market_cap: Optional[float],
    income: Dict,
    balance: Dict,
    cash_flow: Dict,
    prior_income: Optional[Dict] = None,
) -> CompsRow:
    return CompsRow(
        ticker=ticker,
        company_name=company_name,
        market_cap=market_cap,
        revenue_growth=R.revenue_growth(prior_income, income) if prior_income else None,
        gross_margin=R.gross_margin(income),
        operating_margin=R.operating_margin(income),
        ebitda_margin=R.ebitda_margin(income, cash_flow),
        roic=R.roic(income, balance),
        pe=R.pe_ratio(market_cap, income) if market_cap else None,
        ev_revenue=R.ev_revenue(market_cap, balance, income) if market_cap else None,
        ev_ebitda=R.ev_ebitda(market_cap, balance, income, cash_flow) if market_cap else None,
        p_fcf=R.p_fcf(market_cap, cash_flow) if market_cap else None,
        fcf_yield=R.fcf_yield(market_cap, cash_flow) if market_cap else None,
    )


def compute_comps(target: CompsRow, peers: List[CompsRow]) -> CompsResult:
    fields = [
        "revenue_growth", "gross_margin", "operating_margin", "ebitda_margin",
        "roic", "pe", "ev_revenue", "ev_ebitda", "p_fcf", "fcf_yield",
    ]

    median_data: Dict[str, Optional[float]] = {}
    for f in fields:
        vals = [getattr(p, f) for p in peers if getattr(p, f) is not None]
        median_data[f] = median(vals) if vals else None

    median_row = CompsRow(
        ticker="MEDIAN",
        company_name="Peer Median",
        market_cap=None,
        **{f: median_data[f] for f in fields},
    )

    target_percentiles: Dict[str, float] = {}
    premium_discount: Dict[str, float] = {}
    for f in fields:
        peer_vals = [getattr(p, f) for p in peers if getattr(p, f) is not None]
        target_val = getattr(target, f)
        if target_val is not None and peer_vals:
            target_percentiles[f] = round(_percentile_rank(peer_vals, target_val), 3)
            med = median_data[f]
            if med and med != 0:
                premium_discount[f] = round((target_val - med) / abs(med), 3)

    interpretation_lines: List[str] = []
    if "ev_ebitda" in premium_discount:
        delta = premium_discount["ev_ebitda"]
        if delta > 0.05:
            interpretation_lines.append(
                f"Trades at a {delta:+.1%} premium to peers on EV/EBITDA."
            )
        elif delta < -0.05:
            interpretation_lines.append(
                f"Trades at a {delta:+.1%} discount to peers on EV/EBITDA."
            )
        else:
            interpretation_lines.append("In-line with peer median EV/EBITDA.")
    if "operating_margin" in premium_discount:
        d = premium_discount["operating_margin"]
        if d > 0.05:
            interpretation_lines.append(
                "Operating margin runs above peer median, supporting a quality premium."
            )
        elif d < -0.05:
            interpretation_lines.append(
                "Operating margin trails peer median, which weakens the case for a premium."
            )
    if "revenue_growth" in premium_discount:
        d = premium_discount["revenue_growth"]
        if d > 0.05:
            interpretation_lines.append("Growth is above peer median.")
        elif d < -0.05:
            interpretation_lines.append("Growth is below peer median.")

    if not interpretation_lines:
        interpretation_lines.append("Peer metrics broadly in line with the target.")

    return CompsResult(
        target=target,
        peers=peers,
        median=median_row,
        target_percentiles=target_percentiles,
        premium_discount=premium_discount,
        interpretation=" ".join(interpretation_lines),
    )
