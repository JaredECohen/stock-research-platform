"""Portfolio construction service.

Wave 10 — adds a brief-extraction step in front of the candidate
selection so the user's prompt actually shapes the output. The
`PortfolioBrief` (themes, factor tilts, sector targets, beta target,
constraints) drives candidate score weights, sector multipliers, and
ticker filtering before they reach the legacy `build_portfolio`
heuristic. The brief is also returned alongside the portfolio for UI
display + edit / re-run.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from ..finance.portfolio_construction import build_portfolio
from ..schemas import ModelPortfolio, PortfolioBrief, PortfolioRequest
from .portfolio_brief import extract_brief
from .screener_service import get_universe_dicts
from .theme_exposure_service import top_for_theme

log = logging.getLogger(__name__)


def _apply_brief(
    candidates: List[dict], brief: PortfolioBrief,
) -> List[dict]:
    """Reshape candidate scores in-place using the brief.

    Today's `build_portfolio` only knows about a 5-key scenario tag;
    we adjust each candidate's `pm_score`, `macro_fit`, and `quality`
    knobs based on the brief so the same downstream selection logic
    produces materially different portfolios for different prompts.
    """
    # Theme exposure boost — candidates with material exposure to
    # any theme in the brief get a score lift.
    theme_tickers: dict[str, set[str]] = {}
    for theme in brief.themes:
        try:
            tops = top_for_theme(theme, min_score=20.0, limit=50)
            theme_tickers[theme] = {row["ticker"] for row in tops}
        except Exception as exc:  # pragma: no cover
            log.debug("theme exposure lookup failed for %s: %s", theme, exc)
            theme_tickers[theme] = set()
    boosted: List[dict] = []
    for c in candidates:
        c = dict(c)  # shallow copy so we don't mutate caller state
        ticker = c.get("ticker", "").upper()
        # Theme boost (additive, capped).
        theme_hits = sum(
            1 for s in theme_tickers.values() if ticker in s
        )
        if theme_hits and brief.themes:
            c["pm_score"] = min(
                100.0, (c.get("pm_score") or 50.0) + 6.0 * theme_hits,
            )
        # Factor tilts — apply each tilt as a weighted nudge to the
        # corresponding factor.
        tilts = brief.factor_tilts or {}
        if "growth" in tilts:
            c["growth"] = (c.get("growth") or 50.0) + (tilts["growth"] - 0.5) * 30
        if "value" in tilts:
            c["valuation"] = (c.get("valuation") or 50.0) + (tilts["value"] - 0.5) * 30
        if "quality" in tilts:
            c["quality"] = (c.get("quality") or 50.0) + (tilts["quality"] - 0.5) * 30
        if "momentum" in tilts:
            c["earnings_momentum"] = (c.get("earnings_momentum") or 50.0) + (tilts["momentum"] - 0.5) * 30
        # Sector targets — multiply macro_fit by the bias.
        if brief.sector_targets:
            mult = brief.sector_targets.get(c.get("sector", ""), 1.0)
            if mult != 1.0:
                c["macro_fit"] = (c.get("macro_fit") or 50.0) * mult
        # Beta filter — clip out names violating the beta cap (when
        # the candidate carries a beta; otherwise pass through).
        if brief.beta_target is not None and c.get("beta") is not None:
            if c["beta"] > brief.beta_target * 1.1:
                # Penalize rather than drop entirely; let the selector
                # sort it down naturally.
                c["pm_score"] = (c.get("pm_score") or 50.0) * 0.7
        boosted.append(c)
    return boosted


def build_model_portfolio(request: PortfolioRequest) -> ModelPortfolio:
    """Backward-compatible entry. Wave 10 routes through the brief."""
    portfolio, _brief = build_model_portfolio_with_brief(request)
    return portfolio


def build_model_portfolio_with_brief(
    request: PortfolioRequest,
) -> Tuple[ModelPortfolio, PortfolioBrief]:
    """Brief-driven build. Returns (portfolio, brief) so the UI / API
    can render the inferred brief alongside the holdings."""
    brief = extract_brief(request)
    # Layer brief exclusions onto the request so the existing
    # build_portfolio path picks them up.
    excluded_tickers = list(set(
        list(request.excluded_tickers or [])
        + list((brief.exclusions or {}).get("tickers", []))
    ))
    excluded_sectors = list(set(
        list(request.excluded_sectors or [])
        + list((brief.exclusions or {}).get("sectors", []))
    ))
    effective = request.model_copy(update={
        "excluded_tickers": excluded_tickers,
        "excluded_sectors": excluded_sectors,
        "risk_level": brief.risk or request.risk_level,
    })
    candidates: List[dict] = get_universe_dicts(theme=None)
    candidates = _apply_brief(candidates, brief)
    portfolio = build_portfolio(effective, candidates, name="Brief-driven Portfolio")
    return portfolio, brief
