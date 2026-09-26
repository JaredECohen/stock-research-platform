"""Valuation + DCF agent."""
from __future__ import annotations

import json

from ..config import settings
from ..finance.dcf import fmt_price, fmt_upside
from ..schemas import AgentFinding, DCFResult
from ..services.market_data_service import get_current_price
from . import llm, prompts
from .source_ledger import register_source


def run_valuation_agent(
    profile: dict, ratios: dict, dcf: DCFResult | None,
    *, prior_round_critique: str | None = None,
    scorecard_block: str | None = None,
) -> AgentFinding:
    """Valuation analyst: DCF + multiples, optionally against the scorecard.

    `scorecard_block` is the <= 600-char Fundamental Factor Scorecard read
    from `scorecard_context.prompt_block` (the valuation family is the
    "what is already priced in" leg). It is an EXPLICIT kwarg because this
    agent builds its own payload from named ratio keys — a value stuffed
    into `ratios` would be silently dropped. None / empty leaves the
    payload and the prompt exactly as they were before Phase 6.
    """
    # Pull a fresh intraday quote rather than the 7-day-cached
    # `profile.last_price`. Falls back to last close if the quote
    # chain is unavailable (e.g., backtest as-of context).
    ticker = profile.get("ticker")
    live_price = get_current_price(ticker) if ticker else None
    payload: dict = {
        "ticker": ticker,
        "current_price": live_price if live_price is not None else profile.get("last_price"),
        "PE": ratios.get("PE"),
        "EV_EBITDA": ratios.get("EV_EBITDA"),
        "EV_Revenue": ratios.get("EV_Revenue"),
        "PFCF": ratios.get("PFCF"),
        "FCF_yield": ratios.get("FCF_yield"),
        "ROIC": ratios.get("ROIC"),
        "dcf_summary": dcf.summary if dcf else None,
        "dcf_base_implied": dcf.base.implied_share_price if dcf else None,
        "dcf_bull_implied": dcf.bull.implied_share_price if dcf else None,
        "dcf_bear_implied": dcf.bear.implied_share_price if dcf else None,
        "dcf_base_upside": dcf.base.upside_pct if dcf else None,
    }
    if scorecard_block:
        # Observed rank + model read under a named version; the prompt
        # tells the analyst to reconcile, not defer, to it.
        payload["fundamental_scorecard"] = scorecard_block
    # W2b 7(a): the live quote is fetched here; the scorecard block is the
    # rendered rank the prompt shows. The rest of the payload restates the
    # DCF and ratios `_gather_inputs` already registered under their own
    # refs, so it is not registered a second time here (the ledger dedupes
    # only a repeated (kind, ref, extracted facts) registration, not equal
    # values under different refs).
    register_source("price", f"price:{ticker}", {"current_price": payload["current_price"]})
    if scorecard_block:
        register_source("scorecard", f"scorecard_block:{ticker}", scorecard_block)
    # Wave 7C: discretionary notes tagged for the valuation agent.
    from ..services.research_notes import build_notes_block_for_agent
    notes_block = build_notes_block_for_agent(
        "valuation", profile, extra_query="DCF terminal growth WACC multiple",
    )
    from .earnings_agent import _critique_block as _q
    # Tool-agent role — uses OPENAI_TOOL_MODEL (gpt-5.4 by default).
    llm_out = llm.chat_json(
        prompts.VALUATION_ANALYST_PROMPT
        + _q(prior_round_critique)
        + (("\n\n" + notes_block) if notes_block else "")
        + "\n\nContext:\n" + json.dumps(payload, default=str),
        system=prompts.PM_SYSTEM, route="strong",
        model=settings.openai_tool_model,
        action="analyst.valuation", ticker=profile.get("ticker"),
    )
    if llm_out:
        # Wave 10 — typed citations for the DCF + ratio-based claims.
        from ..schemas import Citation
        evidence: list = []
        if dcf and dcf.summary:
            evidence.append(Citation(
                kind="dcf", ref=str(ticker or ""),
                excerpt=str(dcf.summary)[:300],
            ))
        if ratios:
            for k in ("PE", "EV_EBITDA", "FCF_yield", "ROIC"):
                v = ratios.get(k)
                if isinstance(v, (int, float)):
                    evidence.append(Citation(
                        kind="ratio", ref=k,
                        excerpt=f"{k}={v:.2f}",
                    ))
        return AgentFinding(
            agent="Valuation Analyst",
            headline=llm_out.get("headline", "Valuation view"),
            summary=llm_out.get("summary", ""),
            key_points=llm_out.get("key_points", []),
            confidence=float(llm_out.get("confidence", 0.7)),
            sources=["dcf", "ratios"],
            evidence=evidence[:6],
        )

    # Deterministic fallback
    pe = ratios.get("PE")
    ev_eb = ratios.get("EV_EBITDA")
    fcf_y = ratios.get("FCF_yield")
    base_up = dcf.base.upside_pct if dcf else None
    summary_parts = []
    if pe:
        summary_parts.append(f"P/E {pe:.1f}x")
    if ev_eb:
        summary_parts.append(f"EV/EBITDA {ev_eb:.1f}x")
    if fcf_y is not None:
        summary_parts.append(f"FCF yield {fcf_y:.1%}")
    if base_up is not None:
        summary_parts.append(f"DCF base implies {fmt_upside(base_up, decimals=0)} vs current")
    elif dcf is not None:
        # The DCF ran but has no upside to report (no share count or no
        # quote). Say so in the headline rather than silently dropping it —
        # a reader would otherwise assume the DCF was never run.
        summary_parts.append("DCF upside n/a")

    headline = "; ".join(summary_parts) if summary_parts else "Valuation snapshot"
    key_points = []
    if dcf:
        key_points.append(f"Base case implied price: {fmt_price(dcf.base.implied_share_price)}")
        key_points.append(
            f"Bull case: {fmt_price(dcf.bull.implied_share_price)} | "
            f"Bear case: {fmt_price(dcf.bear.implied_share_price)}"
        )
    if ev_eb and ev_eb > 25:
        key_points.append("Valuation is elevated on EV/EBITDA — rate-sensitive.")
    elif ev_eb and ev_eb < 10:
        key_points.append("EV/EBITDA looks undemanding versus history.")
    if fcf_y and fcf_y > 0.04:
        key_points.append("FCF yield > 4% gives downside support if execution holds.")

    summary = (
        f"Current multiples: {', '.join(summary_parts)}. "
        f"DCF triangulates against multiples; the bull/bear range frames the discount-rate sensitivity. "
        f"Valuation risk increases if terminal growth or margin assumptions slip."
    )
    return AgentFinding(
        agent="Valuation Analyst",
        headline=headline,
        summary=summary,
        key_points=key_points or ["See DCF and comps for detail."],
        # Lower than the LLM path's default — a multiples recap with no
        # interpretation is thinner evidence than a real analyst view.
        confidence=0.5,
        sources=["dcf", "ratios"],
        # The graph promotes this flag into `degraded_agents` so the UI
        # can tell readers the section is a deterministic snapshot, not
        # an analyst read (B3).
        data={
            "deterministic_fallback": (
                "Valuation LLM returned no usable output; deterministic "
                "multiples/DCF snapshot shipped instead."
            ),
        },
    )
