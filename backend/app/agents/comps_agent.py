"""Comps agent.

Two valuation lenses, both surfaced when available (Wave 3E):
- Peer-relative — target metrics vs. peer median (today's set).
- Self-historical — target metrics vs. the target's own multi-year
  distribution. A name can look cheap vs. peers but expensive vs. its
  own history (or vice versa); the agent calls out divergence
  explicitly because that's where the alpha sits.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas import AgentFinding, CompsHistoryStats, CompsResult


def _peer_relative_points(comps: CompsResult) -> List[str]:
    target, median = comps.target, comps.median
    pts: List[str] = []
    if target.ev_ebitda is not None and median.ev_ebitda is not None:
        pts.append(
            f"EV/EBITDA: target {target.ev_ebitda:.1f}x vs peer median "
            f"{median.ev_ebitda:.1f}x."
        )
    if target.operating_margin is not None and median.operating_margin is not None:
        pts.append(
            f"Op margin: {target.operating_margin:.0%} vs peers "
            f"{median.operating_margin:.0%}."
        )
    if target.revenue_growth is not None and median.revenue_growth is not None:
        pts.append(
            f"Growth: {target.revenue_growth:+.0%} vs peers "
            f"{median.revenue_growth:+.0%}."
        )
    if target.fcf_yield is not None and median.fcf_yield is not None:
        pts.append(
            f"FCF yield: {target.fcf_yield:.1%} vs peers "
            f"{median.fcf_yield:.1%}."
        )
    return pts


def _self_history_points(
    history: CompsHistoryStats, target_ev_ebitda: Optional[float],
    target_op_margin: Optional[float], target_revenue_growth: Optional[float],
) -> List[str]:
    pts: List[str] = []
    label = history.lookback_label
    own_med = history.own_median
    pct = history.current_percentile

    if target_ev_ebitda is not None and own_med.get("ev_ebitda") is not None:
        ev_pct = pct.get("ev_ebitda")
        if ev_pct is not None:
            pts.append(
                f"EV/EBITDA vs own {label} median {own_med['ev_ebitda']:.1f}x — "
                f"currently at {ev_pct * 100:.0f}th percentile of own history."
            )
    if target_op_margin is not None and own_med.get("operating_margin") is not None:
        delta = history.current_vs_own_median.get("operating_margin")
        if delta is not None:
            pts.append(
                f"Op margin vs own {label} median "
                f"{own_med['operating_margin']:.0%} — {delta:+.0%} delta."
            )
    if target_revenue_growth is not None and own_med.get("revenue_growth") is not None:
        rg_pct = pct.get("revenue_growth")
        if rg_pct is not None:
            pts.append(
                f"Revenue growth at {rg_pct * 100:.0f}th percentile of "
                f"own {label} range."
            )
    return pts


def _premium_signal(comps: CompsResult, metric: str) -> Optional[str]:
    """For a given metric, return 'premium', 'discount', or None for in-line."""
    val = comps.premium_discount.get(metric)
    if val is None:
        return None
    if val > 0.05:
        return "premium"
    if val < -0.05:
        return "discount"
    return None


def _own_history_signal(history: CompsHistoryStats, metric: str) -> Optional[str]:
    delta = history.current_vs_own_median.get(metric)
    if delta is None:
        return None
    if delta > 0.05:
        return "premium"
    if delta < -0.05:
        return "discount"
    return None


def run_comps_agent(
    profile: Dict, comps: Optional[CompsResult],
    *, prior_round_critique: Optional[str] = None,
) -> AgentFinding:
    if comps is None:
        return AgentFinding(
            agent="Comps Analyst",
            headline="Insufficient peer data for comps.",
            summary="No peer set defined for this ticker in the demo dataset.",
            key_points=[],
            confidence=0.4,
            sources=[],
        )

    ticker = profile.get("ticker", "")
    history = comps.history
    key_points: List[str] = _peer_relative_points(comps)

    # Wave 3E: when self-historical context is present, combine both lenses.
    confidence = 0.7
    summary_parts: List[str] = [
        f"Peer set: {', '.join(p.ticker for p in comps.peers)}. {comps.interpretation}"
    ]
    headline = f"Peer-relative read for {ticker}"

    if history is not None:
        key_points.extend(_self_history_points(
            history,
            comps.target.ev_ebitda,
            comps.target.operating_margin,
            comps.target.revenue_growth,
        ))
        if history.interpretation:
            summary_parts.append(f"Own-history context: {history.interpretation}")

        # Highest-signal divergence call: same axis on EV/EBITDA showing
        # opposite signals across the two lenses is alpha territory.
        peer_sig = _premium_signal(comps, "ev_ebitda")
        own_sig = _own_history_signal(history, "ev_ebitda")
        if peer_sig and own_sig:
            if peer_sig == own_sig:
                # Both lenses agree — bump confidence.
                confidence = 0.75
                headline = (
                    f"{ticker}: EV/EBITDA at a {peer_sig} on BOTH peer "
                    f"and own-history axes."
                )
            else:
                # They disagree — that's the alpha; surface in headline.
                confidence = 0.65
                headline = (
                    f"{ticker}: peer {peer_sig} but own-history {own_sig} "
                    f"on EV/EBITDA — divergence to investigate."
                )

    # Wave 7C: discretionary notes tagged for the comps agent. Stashed
    # on data so the drill-down report can surface them; the existing
    # Wave 3E `data["history"]` payload is preserved.
    from ..services.research_notes import build_notes_block_for_agent
    notes_block = build_notes_block_for_agent(
        "comps", profile, extra_query="EV EBITDA multiple cohort peer premium discount",
    )
    finding_data: Dict[str, Any] = {}
    if history is not None:
        finding_data["history"] = history.model_dump()
    if notes_block:
        finding_data["research_notes"] = notes_block

    # Wave 10 — LLM-driven one-paragraph narrative on top of the
    # structured percentile prose. The structured output reads like a
    # spreadsheet ("EV/EBITDA: target 18.0x vs peer median 16.0x");
    # the narrative says what an investor should *do* with it. Cheap
    # (1 call, ~150 tokens out) but high leverage on memo readability.
    # Surfaced via data["narrative"] so callers can render it as a
    # prose summary above the structured tile.
    from ..config import settings
    if settings.has_llm:
        try:
            from . import llm
            import json as _json
            narr_payload = {
                "ticker": ticker,
                "peer_set": [p.ticker for p in comps.peers],
                "target": comps.target.model_dump(),
                "peer_median": comps.median.model_dump(),
                "premium_discount": comps.premium_discount,
                "history": history.model_dump() if history else None,
                "exposure_peers": [p.ticker for p in (comps.exposure_peers or [])],
            }
            narr = llm.chat_json(
                "Below is a comps payload. Write ONE paragraph (3-4 "
                "sentences) summarizing the comparable valuation read "
                "for an investor: where this name sits vs peers AND "
                "vs its own history; the most consequential premium "
                "or discount; and what the divergence (if any) "
                "between the two lenses implies. Specific numbers, "
                "no filler.\n\n"
                "Return JSON: {narrative: \"<one paragraph>\"}\n\n"
                + _json.dumps(narr_payload, default=str)[:3500],
                system="You are a buy-side analyst. Be concrete.",
                route="cheap",
                model=settings.openai_tool_model,
                max_tokens=300,
            )
            if isinstance(narr, dict) and narr.get("narrative"):
                finding_data["narrative"] = str(narr["narrative"]).strip()
        except Exception:  # pragma: no cover — narrative is best-effort
            pass

    finding = AgentFinding(
        agent="Comps Analyst",
        headline=headline,
        summary=" ".join(summary_parts),
        key_points=key_points,
        confidence=confidence,
        sources=[f"peer:{p.ticker}" for p in comps.peers]
        + ([f"history:{ticker}"] if history is not None else []),
        data=finding_data,
    )

    # Wave 9: re-fire path. PM follow-up gets an LLM enrichment grounded
    # in the same comps payload — round-0 stays deterministic.
    if prior_round_critique:
        from ..config import settings
        if settings.has_llm:
            try:
                from . import llm, prompts
                import json as _json
                payload = {
                    "ticker": ticker,
                    "peer_set": [p.ticker for p in comps.peers],
                    "target_ev_ebitda": comps.target.ev_ebitda,
                    "target_op_margin": comps.target.operating_margin,
                    "target_revenue_growth": comps.target.revenue_growth,
                    "target_fcf_yield": comps.target.fcf_yield,
                    "median": comps.median.model_dump(),
                    "premium_discount": comps.premium_discount,
                    "history": history.model_dump() if history else None,
                    "current_summary": finding.summary,
                }
                llm_out = llm.chat_json(
                    "You are the Comps Analyst answering a senior PM follow-up. "
                    "Address the question directly using cohort math (peer median + "
                    "own-history percentile). Reference specific numbers; do NOT "
                    "contradict the prior summary without articulating what changed "
                    "your read. Return JSON: "
                    "{headline, summary, key_points, confidence}.\n\n"
                    f"PM follow-up: {prior_round_critique}\n\n"
                    "Comps context:\n" + _json.dumps(payload, default=str)[:3000],
                    system=prompts.PM_SYSTEM, route="cheap",
                    model=settings.openai_tool_model,
                )
                if isinstance(llm_out, dict) and llm_out.get("summary"):
                    finding = AgentFinding(
                        agent="Comps Analyst",
                        headline=str(llm_out.get("headline") or headline)[:240],
                        summary=str(llm_out["summary"]),
                        key_points=[str(p) for p in (llm_out.get("key_points") or key_points)][:8],
                        confidence=float(llm_out.get("confidence", confidence)),
                        sources=finding.sources,
                        data=finding_data,
                    )
            except Exception:  # pragma: no cover — defensive
                pass

    return finding
