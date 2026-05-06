"""Wave 10 — extract a structured PortfolioBrief from the user's prompt.

Today the portfolio builder produces near-identical portfolios because
the only knob the user's prompt turns is a 5-key scenario tag. This
module reads the prompt as the rich object it actually is — horizon,
risk tolerance, themes, factor tilts, sector targets, exclusions,
beta / yield targets, qualitative constraints — so downstream
scoring + selection can respond.

The brief is the **single source of truth** for portfolio shape. UI
shows it before the portfolio runs so the user can edit / re-run if
the model misread their intent.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..config import settings
from ..schemas import PortfolioBrief, PortfolioRequest

log = logging.getLogger(__name__)


_SUPPORTED_THEMES = [
    "ai_infrastructure", "ai_applications", "energy_transition", "glp1",
    "china_consumer", "data_center_buildout", "cybersecurity",
    "weight_loss", "long_rates_sensitivity", "consumer_credit",
]


def _deterministic_brief(req: PortfolioRequest) -> PortfolioBrief:
    """Light fallback when the LLM is unavailable.

    Reads risk_level + market_view keywords; returns a brief that's
    at least as good as today's behaviour and far from "identical
    output regardless of prompt".
    """
    text = (req.market_view or "").lower()
    horizon_years = {"short": 1, "medium": 3, "long": 5}[req.horizon]
    if "10 year" in text or "decade" in text or "retirement" in text:
        horizon_years = 10
    elif "5 year" in text:
        horizon_years = 5
    elif "next year" in text or "1 year" in text:
        horizon_years = 1

    themes: List[str] = []
    for theme in _SUPPORTED_THEMES:
        primary = theme.replace("_", " ")
        if primary in text or theme in text:
            themes.append(theme)

    factor_tilts: Dict[str, float] = {
        "growth": 0.5, "value": 0.5, "quality": 0.6, "momentum": 0.5,
    }
    if any(w in text for w in ["growth", "high-growth", "compounder"]):
        factor_tilts["growth"] = 0.85
    if any(w in text for w in ["value", "cheap", "undervalued"]):
        factor_tilts["value"] = 0.8
    if any(w in text for w in ["quality", "moat", "wide-moat"]):
        factor_tilts["quality"] = 0.85
    if any(w in text for w in ["momentum", "winning"]):
        factor_tilts["momentum"] = 0.75
    if any(w in text for w in ["dividend", "income", "yield"]):
        factor_tilts["yield"] = 0.7
    if any(w in text for w in ["defensive", "low-beta", "stable"]):
        factor_tilts["quality"] = 0.85
        factor_tilts["growth"] = 0.3

    beta_target: Optional[float] = None
    m = re.search(r"beta\s*(?:under|<|below)?\s*(\d+\.?\d*)", text)
    if m:
        try:
            beta_target = float(m.group(1))
        except ValueError:
            beta_target = None
    if "low-beta" in text or "defensive" in text:
        beta_target = beta_target or 1.0

    yield_target: Optional[float] = None
    m = re.search(r"yield\s*(?:of|over|>)?\s*(\d+\.?\d*)\s*%", text)
    if m:
        try:
            yield_target = float(m.group(1)) / 100.0
        except ValueError:
            yield_target = None

    constraints: List[str] = []
    if "esg" in text:
        constraints.append("ESG-aware")
    if "tax" in text and "efficient" in text:
        constraints.append("tax-efficient")
    if "concentrate" in text or "high conviction" in text:
        constraints.append("concentrated")

    return PortfolioBrief(
        horizon_years=horizon_years,
        risk=req.risk_level,
        themes=themes,
        factor_tilts=factor_tilts,
        sector_targets={s: 1.5 for s in (req.desired_sectors or [])},
        exclusions={
            "tickers": list(req.excluded_tickers or []),
            "sectors": list(req.excluded_sectors or []),
        },
        beta_target=beta_target,
        yield_target=yield_target,
        constraints=constraints,
        rationale=(
            "Brief extracted via keyword fallback (no LLM key)."
        ),
    )


def extract_brief(req: PortfolioRequest) -> PortfolioBrief:
    """Extract a PortfolioBrief from a free-form portfolio request.

    Tries the LLM first; falls back to deterministic keyword reading
    on any failure so the path always succeeds.
    """
    if not getattr(settings, "openai_api_key", None):
        return _deterministic_brief(req)

    schema_hint = {
        "horizon_years": "int (1, 3, 5, 10)",
        "risk": "conservative | balanced | aggressive",
        "themes": "list of theme tags from the supported list",
        "factor_tilts": "dict of factor → 0..1 weight (growth, value, quality, momentum, yield)",
        "sector_targets": "dict of sector → bias multiplier (1.0 = neutral, >1 = overweight, <1 = underweight)",
        "exclusions": "dict with 'tickers' (list of upper-case symbols) and 'sectors' (list of sector names)",
        "beta_target": "float or null",
        "yield_target": "decimal yield target (e.g. 0.04 for 4%) or null",
        "constraints": "list of qualitative constraints (e.g. ESG-aware, tax-efficient, concentrated)",
        "rationale": "1-2 sentence explanation of how you read the prompt",
    }
    prompt = (
        "You are a buy-side PM converting a client's brief into a "
        "structured portfolio specification. Read the request below "
        "and emit the JSON spec — capture the SPECIFIC user intent "
        "(themes, time horizon, risk appetite, factor preferences, "
        "exclusions, beta/yield constraints). The downstream "
        "portfolio engine reads this verbatim, so be specific.\n\n"
        f"Supported themes (use only these tags): {', '.join(_SUPPORTED_THEMES)}\n\n"
        f"Schema:\n{json.dumps(schema_hint, indent=2)}\n\n"
        f"Request:\n- market_view: {req.market_view}\n"
        f"- risk_level (default): {req.risk_level}\n"
        f"- num_holdings: {req.num_holdings}\n"
        f"- horizon: {req.horizon}\n"
        f"- desired_sectors: {req.desired_sectors}\n"
        f"- excluded_sectors: {req.excluded_sectors}\n"
        f"- excluded_tickers: {req.excluded_tickers}\n\n"
        "Return strict JSON matching the schema. Empty values are fine."
    )
    from ..agents import llm
    out = llm.chat_json(
        prompt,
        system=(
            "You are a senior buy-side PM. Be specific and concrete. "
            "If the user asks for 'AI exposure', set themes=['ai_infrastructure', 'ai_applications']. "
            "If they ask for 'retirement', set horizon_years=10 and "
            "tilt to quality + yield + low beta. If they ask for "
            "'high-conviction concentrated bets', tilt to growth + "
            "quality and add 'concentrated' constraint."
        ),
        route="cheap",
    )
    if not isinstance(out, dict):
        return _deterministic_brief(req)
    try:
        return PortfolioBrief(
            horizon_years=int(out.get("horizon_years") or 5),
            risk=out.get("risk") or req.risk_level,
            themes=[t for t in (out.get("themes") or []) if t in _SUPPORTED_THEMES],
            factor_tilts={
                k: float(v) for k, v in (out.get("factor_tilts") or {}).items()
                if isinstance(v, (int, float))
            },
            sector_targets={
                k: float(v) for k, v in (out.get("sector_targets") or {}).items()
                if isinstance(v, (int, float))
            },
            exclusions={
                "tickers": [str(t).upper() for t in (out.get("exclusions") or {}).get("tickers", [])],
                "sectors": [str(s) for s in (out.get("exclusions") or {}).get("sectors", [])],
            },
            beta_target=(
                float(out.get("beta_target")) if out.get("beta_target") is not None else None
            ),
            yield_target=(
                float(out.get("yield_target")) if out.get("yield_target") is not None else None
            ),
            constraints=[str(c) for c in (out.get("constraints") or [])],
            rationale=str(out.get("rationale") or ""),
        )
    except Exception as exc:  # pragma: no cover
        log.warning("brief parse failed; falling back: %s", exc)
        return _deterministic_brief(req)
