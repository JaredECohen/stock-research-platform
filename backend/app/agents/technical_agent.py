"""Wave 3B — Technical Analyst.

Frames technical indicators as "positioning signals in support of (or
in tension with) the long-term thesis", NOT as standalone trade signals.
By design, this agent's view does NOT override the memo's rating —
fundamentals + valuation + thesis are the rating drivers; technicals
provide entry-timing context only.

Architecture mirrors the other specialist agents:
- Pure-math indicator computation lives in `app/finance/technicals.py`
  so it's testable without an LLM.
- This module wraps the indicators in a structured payload + LLM
  narrative pass + deterministic fallback.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..config import settings
from ..finance.technicals import compute_technical_signals
from ..schemas import AgentFinding, TechnicalSignals
from ..services.market_data_service import get_price_series
from . import llm, prompts

log = logging.getLogger(__name__)


def _format_value(v: Optional[float], fmt: str = "{:,.2f}") -> str:
    return "n/a" if v is None else fmt.format(v)


def _deterministic_summary(profile: Dict[str, Any], sig: TechnicalSignals) -> Dict[str, Any]:
    """LLM-free fallback. Reads the structured signals into a sober narrative
    that explicitly frames technicals as positioning context, not a
    rating-driver."""
    ticker = profile.get("ticker") or ""
    headline_bits: List[str] = []
    headline_bits.append(f"{sig.trend} trend")
    headline_bits.append(f"{sig.momentum} momentum")
    if sig.position_52w is not None:
        headline_bits.append(f"{sig.position_52w * 100:.0f}% of 52w range")
    headline = f"{ticker}: " + " · ".join(headline_bits)

    summary_lines: List[str] = []
    if sig.sma_50 is not None and sig.sma_200 is not None:
        rel = "above" if sig.sma_50_above_200 else "below"
        summary_lines.append(
            f"SMA50 ({_format_value(sig.sma_50)}) is {rel} SMA200 "
            f"({_format_value(sig.sma_200)}) — last close {_format_value(sig.last_price)}."
        )
    if sig.rsi_14 is not None:
        summary_lines.append(f"RSI(14) at {sig.rsi_14:.0f}.")
    if sig.macd_histogram is not None:
        direction = "expanding" if sig.macd_histogram > 0 else "compressing"
        summary_lines.append(
            f"MACD histogram {_format_value(sig.macd_histogram, '{:+.2f}')} ({direction})."
        )
    if sig.bb_position is not None:
        summary_lines.append(
            f"Bollinger position {sig.bb_position * 100:.0f}% within band."
        )
    summary_lines.append(
        "Technical signals are positioning context for the fundamental thesis, "
        "not a standalone trade signal."
    )

    key_points: List[str] = list(sig.notes)
    if not key_points:
        key_points.append(
            f"No actionable technical setup; {sig.trend}/{sig.momentum} is the regime."
        )

    return {
        "headline": headline,
        "summary": " ".join(summary_lines),
        "key_points": key_points,
        "confidence": 0.6,
    }


def run_technical_agent(
    profile: Dict[str, Any], days: int = 300,
    *, prior_round_critique: Optional[str] = None,
) -> AgentFinding:
    """Produce the Technical Analyst finding for `profile`.

    `days` is the number of trailing daily bars to request. 300 gives the
    indicator suite enough headroom (SMA200 + a buffer for MACD signal
    smoothing).
    """
    ticker = (profile.get("ticker") or "").upper()
    if not ticker:
        return AgentFinding(
            agent="Technical Analyst",
            headline="Technical analysis unavailable.",
            summary="No ticker provided.",
            confidence=0.3,
        )

    try:
        rows = get_price_series(ticker, days)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("Technical analyst price fetch failed for %s: %s", ticker, exc)
        rows = []
    raw = compute_technical_signals(rows or [])
    if not raw:
        return AgentFinding(
            agent="Technical Analyst",
            headline=f"{ticker}: insufficient price history for technical read.",
            summary=(
                "Need ≥60 daily bars for the cheaper indicators and ≥234 for the "
                "full suite (SMA200 + MACD signal). Treat this run as fundamentals-only."
            ),
            confidence=0.4,
            sources=[f"prices:{ticker}"],
        )

    signals = TechnicalSignals(**{k: v for k, v in raw.items() if k in TechnicalSignals.model_fields})

    # LLM narrative pass (uses the dedicated tool model). The prompt makes
    # the no-trade-signal framing explicit so the model doesn't drift into
    # buy/sell language. Falls back deterministically if the call fails.
    from .earnings_agent import _critique_block as _q
    payload_for_prompt = signals.model_dump()
    user_prompt = (
        prompts.TECHNICAL_ANALYST_PROMPT.format(
            ticker=ticker,
            sector=profile.get("sector", ""),
        )
        + _q(prior_round_critique)
        + "\n\nTechnical indicators (already computed; do NOT recompute):\n"
        + json.dumps(payload_for_prompt, default=str)[:2000]
    )
    llm_out = llm.chat_json(
        user_prompt, system=prompts.PM_SYSTEM, route="cheap",
        model=settings.openai_tool_model,
    )

    narrative = llm_out if llm_out else _deterministic_summary(profile, signals)

    return AgentFinding(
        agent="Technical Analyst",
        headline=str(narrative.get("headline", ""))[:240],
        summary=str(narrative.get("summary", "")),
        key_points=[str(p) for p in (narrative.get("key_points") or [])][:8],
        confidence=float(narrative.get("confidence", 0.6)),
        sources=[f"prices:{ticker}"],
        data={"signals": signals.model_dump()},
    )
