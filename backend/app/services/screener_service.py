"""Screener service: compute factor scores for the universe and persist."""
from __future__ import annotations

from typing import Dict, List, Optional

from ..finance import factor_scores as fs
from ..schemas import ScreenerRow, ScreenerResult
from .data_service import get_data_service
from .fundamentals_service import get_full_financials


# ---------------------------------------------------------------------------
# Theme matchers — used to bias macro_fit per theme
# ---------------------------------------------------------------------------

THEME_BIAS: Dict[str, Dict[str, float]] = {
    "ai_infrastructure": {
        "Technology": 1.4, "Communication Services": 1.1, "Industrials": 1.1, "Utilities": 1.05,
    },
    "falling_rates": {
        "Technology": 1.2, "Real Estate": 1.3, "Consumer Discretionary": 1.1, "Utilities": 1.15,
    },
    "sticky_inflation": {
        "Energy": 1.3, "Financials": 1.15, "Materials": 1.10, "Technology": 0.85,
    },
    "recession_defense": {
        "Healthcare": 1.25, "Consumer Staples": 1.30, "Utilities": 1.20, "Technology": 0.85,
    },
    "high_quality_compounders": {
        "Technology": 1.10, "Consumer Staples": 1.15, "Financials": 1.0, "Healthcare": 1.10,
    },
    "margin_expansion": {
        "Technology": 1.10, "Communication Services": 1.10, "Financials": 1.05,
    },
    "reasonable_valuation_growth": {
        "Technology": 1.0, "Consumer Discretionary": 1.05, "Financials": 1.05, "Healthcare": 1.10,
    },
}


def _theme_label(theme: Optional[str]) -> Optional[str]:
    if not theme:
        return None
    t = theme.lower().replace("-", "_").replace(" ", "_")
    return t if t in THEME_BIAS else theme


def _one_line_thesis(profile: Dict) -> str:
    drivers = profile.get("drivers") or []
    return f"{profile.get('industry', 'Sector')} name leveraged to {drivers[0] if drivers else 'durable demand'}."


def compute_universe_scores(theme: Optional[str] = None) -> ScreenerResult:
    """Compute scores for the entire demo universe with optional theme bias."""
    ds = get_data_service()
    tickers = ds.list_tickers()
    rows: List[ScreenerRow] = []
    theme_key = _theme_label(theme)
    bias = THEME_BIAS.get(theme_key or "", {})

    for ticker in tickers:
        fin = get_full_financials(ticker)
        if not fin.get("income"):
            continue
        profile = fin["profile"]
        ratios = fin["ratios"] or {}
        income = sorted(fin["income"], key=lambda r: r.get("period", ""))
        cash = sorted(fin["cash"], key=lambda r: r.get("period", ""))
        latest_inc = income[-1] if income else {}
        latest_cf = cash[-1] if cash else {}
        market_cap = profile.get("market_cap") or 0
        rev_growth = ratios.get("revenue_growth")
        op_margin = ratios.get("operating_margin")
        gross_margin = ratios.get("gross_margin")
        roic = ratios.get("ROIC")
        ev_ebitda = ratios.get("EV_EBITDA")
        p_fcf = ratios.get("PFCF")
        fcf_y = ratios.get("FCF_yield")
        debt_to_ebitda = ratios.get("debt_to_ebitda")
        beta = profile.get("beta")

        quality = fs.quality_score(roic, op_margin, gross_margin)
        growth = fs.growth_score(rev_growth)
        valuation = fs.valuation_score(ev_ebitda, p_fcf, fcf_y)
        earnings = fin.get("earnings", {})
        surprises = [q.get("surprise_pct", 0) for q in earnings.get("quarters", [])]
        earnings_momentum = fs.earnings_momentum_score(surprises)
        risk = fs.risk_score(beta, debt_to_ebitda, drawdown=-0.20)

        sector = profile.get("sector", "")
        macro_fit_base = 60.0
        catalyst = 65.0 if "AI" in (profile.get("description") or "") else 50.0

        if bias:
            mult = bias.get(sector, 0.95)
            macro_fit = round(min(100, max(0, macro_fit_base * mult)), 1)
            catalyst = round(min(100, max(0, catalyst * mult)), 1)
        else:
            macro_fit = macro_fit_base

        pm_score = round(
            quality * 0.25 + growth * 0.20 + valuation * 0.15 + earnings_momentum * 0.10
            + macro_fit * 0.15 + risk * 0.10 + catalyst * 0.05,
            1,
        )

        rows.append(ScreenerRow(
            rank=0,  # filled later
            ticker=ticker,
            company_name=profile.get("company_name", ticker),
            sector=sector,
            pm_score=pm_score,
            quality=quality,
            growth=growth,
            valuation=valuation,
            earnings_momentum=earnings_momentum,
            risk=risk,
            macro_fit=macro_fit,
            one_line_thesis=_one_line_thesis(profile),
            main_catalyst=(profile.get("drivers") or ["—"])[0],
            main_risk=(profile.get("risks") or ["—"])[0],
            theme=theme_key,
        ))

    rows.sort(key=lambda r: r.pm_score, reverse=True)
    for i, r in enumerate(rows, start=1):
        r.rank = i

    return ScreenerResult(theme=theme_key, rows=rows)


def get_universe_dicts(theme: Optional[str] = None) -> List[Dict]:
    """Return scored universe as a list of dicts (used by portfolio agent)."""
    result = compute_universe_scores(theme)
    return [r.model_dump() for r in result.rows]
