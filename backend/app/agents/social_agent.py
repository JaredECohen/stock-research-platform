"""Social media agent — Gemini Flash sentiment scalar.

Returns a single sentiment-extremity score (0..100) plus a contrarian flag,
matching the existing `tools.get_social_sentiment` shape so callers can swap
in this implementation transparently. The Gemini call is gated on the
`GEMINI_API_KEY` being present; when missing, we fall back to the
deterministic stub already shipped in `tools.py` so demo mode keeps working.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

from ..cache import cache_get, cache_put
from ..config import settings
from . import llm
from . import tools as _tools


def _stub(ticker: str) -> Dict[str, Any]:
    return _tools.get_social_sentiment(ticker)


def run(ticker: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    """Compute a sentiment-extremity scalar for `ticker` and cache as social_hot."""
    today_key = f"social_hot:{ticker}:{date.today().isoformat()}"
    if not force_refresh:
        cached = cache_get(today_key, "social_hot", max_age_seconds=24 * 3600)
        if cached and isinstance(cached.payload, dict):
            return cached.payload

    if settings.has_gemini:
        prompt = (
            f"Aggregate the sentiment about {ticker} on social platforms "
            "(Twitter/X, Reddit, StockTwits) over the last 7 days into a single "
            "extremity score (0..100, where 50 is neutral). Return JSON: "
            "{sentiment_extremity: number, contrarian_flag: 'bullish_setup'|'bearish_setup'|'neutral', "
            "rationale: string}. Do NOT quote individual posts."
        )
        out = llm.gemini_chat_json(
            prompt, model=settings.gemini_social_model, max_tokens=400,
        )
        if isinstance(out, dict) and "sentiment_extremity" in out:
            payload = {
                "ticker": ticker,
                "sentiment_extremity": float(out.get("sentiment_extremity", 50)),
                "contrarian_flag": str(out.get("contrarian_flag", "neutral")),
                "rationale": str(out.get("rationale", "Aggregate sentiment scalar."))[:600],
                "source": "gemini",
            }
            cache_put(today_key, "social_hot", payload=payload,
                      sources_used=[f"social:{ticker}"],
                      generated_by="social_agent", cost_tokens=20,
                      ttl_seconds=24 * 3600)
            return payload

    payload = {**_stub(ticker), "source": "stub"}
    cache_put(today_key, "social_hot", payload=payload,
              sources_used=[f"social:{ticker}"],
              generated_by="social_agent", cost_tokens=0,
              ttl_seconds=24 * 3600)
    return payload
