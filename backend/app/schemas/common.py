"""Foundational schema types shared by every other schema module.

Rating labels, the score<->label mapping, the chat intent vocabulary and
the company profile shape. Nothing here imports another schema module,
so this is the root of the schema import DAG.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Foundational types
# ---------------------------------------------------------------------------

RatingLabel = Literal[
    "Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish"
]


def rating_from_stock_score(score: float) -> str:
    """Wave 8P — deterministic mapping from quant Stock Score → rating label.

    Locked decision: rating is now a function of the Stock Score (the
    quantitative factor blend) rather than the LLM's PM synthesis. The
    LLM's `confidence_score` separately reflects the agents' conviction
    in the directional call.

      80–100 → Very Bullish
      60–80  → Bullish
      40–60  → Neutral
      20–40  → Bearish
       0–20  → Very Bearish
    """
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "Neutral"
    if s >= 80:
        return "Very Bullish"
    if s >= 60:
        return "Bullish"
    if s >= 40:
        return "Neutral"
    if s >= 20:
        return "Bearish"
    return "Very Bearish"


_RATING_LABEL_TO_SCORE: Dict[str, float] = {
    "Very Bullish": 90.0,
    "Bullish": 70.0,
    "Neutral": 50.0,
    "Bearish": 30.0,
    "Very Bearish": 10.0,
}


def score_from_rating_label(label: Optional[str]) -> float:
    """Inverse of `rating_from_stock_score` — bucket centers on 0-100.

    Used by the PM rating-blend (Option A) so the LLM's directional call
    can be mixed with the quant factor score before label assignment.
    Unknown / missing labels collapse to Neutral (50).
    """
    if not label:
        return 50.0
    return _RATING_LABEL_TO_SCORE.get(str(label).strip(), 50.0)


IntentType = Literal[
    "single_stock_analysis",
    "stock_comparison",
    "thematic_screen",
    "macro_question",
    "portfolio_construction",
    "dcf_analysis",
    "comps_analysis",
    "general_research_chat",
]


class CompanyOut(BaseModel):
    ticker: str
    company_name: str
    exchange: str
    sector: str
    industry: str
    sub_industry: Optional[str] = None
    country: str = "US"
    currency: str = "USD"
    market_cap: Optional[float] = None
    business_description: str = ""
    last_price: Optional[float] = None
    is_etf: bool = False
    beta: Optional[float] = None
    shares_outstanding: Optional[float] = None
    # Universe tier (Phase F + Wave 1B). Frontend uses this to render
    # the appropriate analyze affordance: `auto_analysis` shows a memo
    # immediately, `analyzed_on_demand` shows a cached memo, `data_only`
    # shows an explicit "Analyze this stock" gate.
    universe_tier: str = "data_only"
