"""Portfolio construction.

Given an investment scenario and a universe of scored candidates, produce a
diversified model portfolio with explicit constraints (max position size,
sector caps, exclusions). Deterministic and inspectable — no optimization
solver needed for a self-contained demo.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from ..schemas import ModelPortfolio, PortfolioHolding, PortfolioRequest


SCENARIO_KEYWORDS: Dict[str, Dict[str, float]] = {
    "soft_landing": {
        "tech": 1.15, "consumer": 1.10, "financials": 1.05,
        "healthcare": 1.0, "energy": 0.95, "industrials": 1.05,
    },
    "recession": {
        "consumer_staples": 1.20, "healthcare": 1.15, "utilities": 1.20,
        "tech": 0.85, "financials": 0.85, "energy": 0.95, "consumer": 0.85,
    },
    "sticky_inflation": {
        "energy": 1.20, "materials": 1.10, "financials": 1.10,
        "tech": 0.90, "consumer": 0.90, "real_estate": 0.85,
    },
    "falling_rates": {
        "tech": 1.20, "real_estate": 1.20, "financials": 1.0,
        "consumer": 1.10, "utilities": 1.10,
    },
    "ai_capex_boom": {
        "tech": 1.30, "industrials": 1.10, "energy": 1.05,
        "consumer": 0.95, "financials": 0.95,
    },
}


def _sector_bucket(sector: str) -> str:
    s = (sector or "").lower()
    if "tech" in s or "communication" in s or "semiconductor" in s:
        return "tech"
    if "financ" in s or "bank" in s or "insurance" in s:
        return "financials"
    if "consumer staples" in s:
        return "consumer_staples"
    if "consumer" in s or "retail" in s or "restaurant" in s:
        return "consumer"
    if "health" in s or "pharma" in s:
        return "healthcare"
    if "energy" in s or "oil" in s:
        return "energy"
    if "industrial" in s or "transportation" in s:
        return "industrials"
    if "utilit" in s:
        return "utilities"
    if "material" in s:
        return "materials"
    if "real estate" in s or "reit" in s:
        return "real_estate"
    return "other"


def detect_scenario(market_view: str) -> str:
    v = market_view.lower()
    if "soft landing" in v:
        return "soft_landing"
    if "recession" in v or "downturn" in v:
        return "recession"
    if "sticky" in v or "inflation" in v:
        return "sticky_inflation"
    if "falling rate" in v or "rate cut" in v or "rates fall" in v:
        return "falling_rates"
    if "ai" in v or "artificial intelligence" in v or "capex" in v:
        return "ai_capex_boom"
    return "soft_landing"


def build_portfolio(
    request: PortfolioRequest,
    candidates: List[dict],
    *,
    name: str = "Scenario Portfolio",
) -> ModelPortfolio:
    """Build a diversified model portfolio from a list of candidate dicts.

    Each candidate must have at least: ticker, company_name, sector, pm_score,
    quality, growth, valuation, risk, macro_fit.
    """
    scenario_key = detect_scenario(request.market_view)
    sector_weights = SCENARIO_KEYWORDS.get(scenario_key, {})

    # Filter exclusions
    excluded_sectors = {s.lower() for s in request.excluded_sectors}
    excluded_tickers = {t.upper() for t in request.excluded_tickers}

    eligible = []
    for c in candidates:
        if c["ticker"].upper() in excluded_tickers:
            continue
        if (c.get("sector") or "").lower() in excluded_sectors:
            continue
        eligible.append(c)

    risk_pen = {"conservative": 1.5, "balanced": 1.0, "aggressive": 0.5}[request.risk_level]

    def candidate_score(c: dict) -> float:
        bucket = _sector_bucket(c.get("sector", ""))
        sector_mult = sector_weights.get(bucket, 1.0)
        # higher risk score = lower risk; conservative weights it higher
        risk_component = c.get("risk", 50.0)
        valuation_penalty = max(0.0, 50.0 - c.get("valuation", 50.0)) * 0.3
        s = (
            c.get("pm_score", 50.0) * 0.40
            + c.get("quality", 50.0) * 0.20
            + c.get("growth", 50.0) * 0.10
            + c.get("macro_fit", 50.0) * 0.15
            + risk_component * 0.10 * risk_pen
            + c.get("earnings_momentum", 50.0) * 0.05
            - valuation_penalty
        )
        return s * sector_mult

    desired = {s.lower() for s in request.desired_sectors}
    if desired:
        eligible.sort(key=lambda c: (
            0 if (c.get("sector") or "").lower() in desired else 1,
            -candidate_score(c),
        ))
    else:
        eligible.sort(key=candidate_score, reverse=True)

    n = max(3, min(request.num_holdings, 30))
    max_pos = min(max(0.04, request.max_position_size), 0.40)

    # Sector cap: at most 35% in a single sector for balanced; tighter for conservative
    sector_cap = {"conservative": 0.30, "balanced": 0.40, "aggressive": 0.55}[request.risk_level]

    selected: List[dict] = []
    sector_used: Dict[str, float] = defaultdict(float)
    raw_weights: Dict[str, float] = {}
    for c in eligible:
        if len(selected) >= n:
            break
        bucket = _sector_bucket(c.get("sector", ""))
        # tentative weight proportional to score
        score = max(1.0, candidate_score(c))
        tentative = min(max_pos, score / 1000)
        # Sector cap check
        if sector_used[bucket] + tentative > sector_cap and len(selected) >= n // 2:
            continue
        selected.append(c)
        raw_weights[c["ticker"]] = score
        sector_used[bucket] += tentative

    # If nothing passed (e.g., aggressive exclusions), relax and re-pick
    if not selected and eligible:
        selected = eligible[:n]
        for c in selected:
            raw_weights[c["ticker"]] = max(1.0, candidate_score(c))

    # Normalize weights, enforce max_pos
    total_score = sum(raw_weights.values()) or 1.0
    weights = {t: raw_weights[t] / total_score for t in raw_weights}
    # Clip to max_pos and re-distribute residual
    excess = 0.0
    for t in list(weights):
        if weights[t] > max_pos:
            excess += weights[t] - max_pos
            weights[t] = max_pos
    remainder = [t for t in weights if weights[t] < max_pos]
    if excess > 0 and remainder:
        bump = excess / len(remainder)
        for t in remainder:
            new = min(max_pos, weights[t] + bump)
            excess -= (new - weights[t])
            weights[t] = new
    # final renormalize
    total = sum(weights.values()) or 1.0
    weights = {t: round(w / total, 4) for t, w in weights.items()}

    holdings: List[PortfolioHolding] = []
    sector_allocation: Dict[str, float] = defaultdict(float)
    for c in selected:
        w = weights.get(c["ticker"], 0.0)
        if w <= 0:
            continue
        sector = c.get("sector", "Unknown")
        sector_allocation[sector] += w
        rationale = c.get("one_line_thesis") or f"{sector} exposure with strong PM score ({c.get('pm_score', 0):.0f})."
        holdings.append(PortfolioHolding(
            ticker=c["ticker"],
            company_name=c.get("company_name", c["ticker"]),
            sector=sector,
            weight=round(w, 4),
            rationale=rationale,
            pm_conviction=c.get("pm_score", 0.0),
        ))

    holdings.sort(key=lambda h: h.weight, reverse=True)
    sector_allocation = {k: round(v, 4) for k, v in sector_allocation.items()}

    from .risk import concentration_metrics
    weight_map = {h.ticker: h.weight for h in holdings}
    concentration = {k: round(v, 4) for k, v in concentration_metrics(weight_map).items()}

    risk_notes: List[str] = []
    if concentration.get("hhi", 0) > 0.18:
        risk_notes.append("Portfolio is concentrated; HHI above 0.18.")
    if concentration.get("top_3", 0) > 0.40:
        risk_notes.append("Top 3 holdings exceed 40% — single-name event risk is elevated.")
    if max(sector_allocation.values(), default=0) > 0.40:
        biggest = max(sector_allocation, key=sector_allocation.get)
        risk_notes.append(
            f"{biggest} is the largest sector exposure at {sector_allocation[biggest]:.0%}; "
            "monitor sector-specific drawdowns."
        )
    if request.risk_level == "aggressive":
        risk_notes.append("Aggressive tilt accepts higher volatility for higher expected return dispersion.")
    if not risk_notes:
        risk_notes.append("Diversification looks reasonable; revisit positions on material thesis change.")

    top_drivers: List[str] = []
    for h in holdings[:3]:
        top_drivers.append(f"{h.ticker}: {h.rationale}")

    invalidators: List[str] = [
        f"A reversal of the '{request.market_view}' thesis would unwind the sector tilt.",
        "Significant rerating in one of the top three positions would dominate portfolio P&L.",
        "Sustained credit-spread widening could compress the multiples this construction assumes.",
    ]
    watch_items: List[str] = [
        "Update the screener weekly — refresh PM scores after major macro prints or earnings.",
        "Re-run the risk committee if sector allocation drifts >5% from target.",
        "Add a defensive sleeve if drawdown tolerance is tighter than risk_level suggests.",
    ]

    expected_vol_proxy = 0.18
    if request.risk_level == "conservative":
        expected_vol_proxy = 0.13
    elif request.risk_level == "aggressive":
        expected_vol_proxy = 0.24

    return ModelPortfolio(
        name=name,
        market_view=request.market_view,
        risk_level=request.risk_level,
        holdings=holdings,
        sector_allocation=sector_allocation,
        concentration=concentration,
        expected_volatility=expected_vol_proxy,
        risk_notes=risk_notes,
        top_thesis_drivers=top_drivers,
        what_could_invalidate=invalidators,
        watch_items=watch_items,
    )
