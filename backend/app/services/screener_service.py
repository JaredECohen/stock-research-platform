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

# Minimum ThemeExposure score (0-100) required to keep a row in a
# theme-filtered screen. 30 is intentionally lenient — names with a real
# but secondary link to the theme stay; names with effectively zero
# exposure get dropped.
THEME_EXPOSURE_FLOOR: float = 30.0


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
    """Compute scores for the curated screener universe (auto_analysis only).

    Wave 9b — research-on-demand and demoted-class tickers stay out of
    the screener (per the locked decision in
    `docs/UNIVERSE_DESIGN.md`). They're still individually
    researchable from the Research page.

    Theme filtering: when a theme is active, we filter out names that
    have no real exposure to it — otherwise a high-quality / low-beta
    name (e.g. Altria) would surface under "AI Infrastructure" purely
    on the strength of its quality + risk factors. Two layers:

      1. If the ThemeExposure table has any entries for this theme,
         keep only rows whose `theme_exposure_score` clears
         `THEME_EXPOSURE_FLOOR`. Drops names with zero / low real
         exposure regardless of how strong their fundamentals are.

      2. If exposures are not populated yet, fall back to filtering by
         the sectors the theme actually *boosts* (THEME_BIAS entries
         with `mult > 1.0`). Coarse but honest — no AI-tilted result
         set will include Consumer Defensive names.
    """
    from ..database import SessionLocal
    from ..models import Company
    with SessionLocal() as db:
        tickers = [
            t for (t,) in db.query(Company.ticker).filter(
                Company.universe_tier == "auto_analysis",
            ).all()
        ]
    rows: List[ScreenerRow] = []
    theme_key = _theme_label(theme)
    bias = THEME_BIAS.get(theme_key or "", {})

    # Precompute the theme-exposure map for this theme (one query per
    # screen, not one per ticker) and the set of theme-favored sectors
    # used as the fallback filter when no exposures are on file.
    theme_exposure_map: Dict[str, float] = {}
    if theme_key:
        try:
            from ..models import ThemeExposure
            with SessionLocal() as _db:
                rows_te = _db.query(ThemeExposure.ticker, ThemeExposure.score).filter(
                    ThemeExposure.theme == theme_key,
                ).all()
                theme_exposure_map = {
                    str(t).upper(): float(s or 0.0) for (t, s) in rows_te
                }
        except Exception:  # pragma: no cover — best-effort
            theme_exposure_map = {}
    have_exposure_data = bool(theme_exposure_map)
    favored_sectors = {sec for sec, mult in bias.items() if mult > 1.0}

    for ticker in tickers:
        fin = get_full_financials(ticker)
        if not fin.get("income"):
            continue
        profile = fin["profile"]
        # Theme-relevance filter — see the docstring on this function for
        # why this is here. Skip BEFORE running the (expensive) factor
        # math so theme screens stay fast.
        if theme_key:
            if have_exposure_data:
                exposure = theme_exposure_map.get(ticker.upper(), 0.0)
                if exposure < THEME_EXPOSURE_FLOOR:
                    continue
            elif favored_sectors:
                if profile.get("sector") not in favored_sectors:
                    continue
            # else: theme has no bias/exposure data — leave full universe
            # in (rare; only happens for an unknown theme string).
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
        # Screener doesn't run the LLM earnings extraction per row, so
        # no guidance signal available here — just the beat streak.
        earnings_momentum = fs.earnings_momentum_score(surprises)
        beat_streak_n = fs.beat_streak(surprises)
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

        # Theme exposure score (0-100) attached to the row so the UI can
        # show users WHY a name surfaces under a theme. Pulled from the
        # precomputed map; `None` when no exposure data exists for the
        # theme or this ticker.
        theme_exposure_score: Optional[float] = (
            theme_exposure_map.get(ticker.upper())
            if theme_key and have_exposure_data
            else None
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
            theme_exposure_score=theme_exposure_score,
            beat_streak=beat_streak_n,
        ))

    rows.sort(key=lambda r: r.pm_score, reverse=True)
    for i, r in enumerate(rows, start=1):
        r.rank = i

    return ScreenerResult(theme=theme_key, rows=rows)


def get_universe_dicts(theme: Optional[str] = None) -> List[Dict]:
    """Return scored universe as a list of dicts (used by portfolio agent)."""
    result = compute_universe_scores(theme)
    return [r.model_dump() for r in result.rows]
