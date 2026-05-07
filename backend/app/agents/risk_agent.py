"""Risk analyst agent."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import settings
from ..schemas import AgentFinding, RiskItem, RiskRecommendation


def _build_recommendations(
    profile: Dict, ratios: Dict, dcf_summary: Optional[Dict],
) -> List[RiskRecommendation]:
    """Wave 8H — concrete, actionable recs the graph deterministically
    applies. Each rec ties an observable signal to a specific change in
    the final memo."""
    recs: List[RiskRecommendation] = []
    risks = profile.get("risks") or []
    debt_to_eb = ratios.get("debt_to_ebitda")
    ev_ebitda = ratios.get("EV_EBITDA") or 0
    fcf_y = ratios.get("FCF_yield")

    # Leverage — material confidence reduction.
    if debt_to_eb and debt_to_eb > 3.5:
        recs.append(RiskRecommendation(
            target="confidence", direction="lower",
            magnitude="medium" if debt_to_eb < 5 else "large",
            detail=f"Net leverage {debt_to_eb:.1f}x is elevated",
            rationale=(
                f"Debt/EBITDA at {debt_to_eb:.1f}x raises the variance "
                f"on equity returns under any execution slip."
            ),
        ))

    # Valuation stretch — small confidence trim + bear case augmentation.
    if ev_ebitda > 30:
        recs.append(RiskRecommendation(
            target="confidence", direction="lower", magnitude="small",
            detail=f"EV/EBITDA {ev_ebitda:.1f}x is rich",
            rationale=(
                "Stretched multiple amplifies the rerating risk on any "
                "miss; lower confidence by small magnitude."
            ),
        ))
        recs.append(RiskRecommendation(
            target="bear_case", direction="flag", magnitude="medium",
            detail=f"Multiple compression risk: EV/EBITDA {ev_ebitda:.1f}x",
            rationale="Stretched valuation needs to land in the bear case.",
        ))

    # Thin FCF support — flag in bear case.
    if fcf_y is not None and fcf_y < 0.02:
        recs.append(RiskRecommendation(
            target="bear_case", direction="flag", magnitude="small",
            detail=f"Low FCF yield ({fcf_y:.1%}) reduces downside support",
            rationale="Thin FCF cushion means draw-down can be sharper.",
        ))

    # Profile-level high-severity risks → flag as thesis_breakers.
    high_kw = ("competition", "regulator", "antitrust", "concentration",
               "going concern", "fraud")
    for r in risks[:6]:
        rl = r.lower()
        if any(k in rl for k in high_kw):
            recs.append(RiskRecommendation(
                target="thesis_breakers", direction="flag", magnitude="medium",
                detail=r[:120],
                rationale=(
                    "Profile risk matches a thesis-breaker keyword "
                    "(competition / regulator / concentration / fraud); "
                    "must surface in thesis_breakers."
                ),
            ))

    # DCF bear well below current → rating-level signal.
    if dcf_summary and isinstance(dcf_summary, dict):
        bear_text = str(dcf_summary).lower()
        if "downside" in bear_text or "bear" in bear_text:
            recs.append(RiskRecommendation(
                target="bear_case", direction="flag", magnitude="small",
                detail="DCF bear scenario maps explicit downside",
                rationale=(
                    "DCF bear case quantifies what happens if growth + "
                    "margin assumptions slip; should land in bear key_points."
                ),
            ))

    return recs


def run_risk_agent(
    profile: Dict, ratios: Dict, dcf_summary: Optional[Dict] = None,
    *, prior_round_critique: Optional[str] = None,
) -> AgentFinding:
    risks: List[str] = profile.get("risks") or []
    summary_lines: List[str] = []

    debt_to_eb = ratios.get("debt_to_ebitda")
    if debt_to_eb and debt_to_eb > 3.5:
        summary_lines.append(f"Leverage at {debt_to_eb:.1f}x debt/EBITDA is elevated.")
    if (ratios.get("EV_EBITDA") or 0) > 30:
        summary_lines.append("Multiple is high — valuation risk elevated to growth slippage.")
    if (ratios.get("FCF_yield") or 0) < 0.02:
        summary_lines.append("Low FCF yield reduces downside support.")
    if not summary_lines:
        summary_lines.append("Quantitative risk profile looks moderate.")

    key_points = [f"Risk: {r}" for r in risks[:5]]
    if dcf_summary and "bear" in dcf_summary:
        key_points.append("DCF bear case maps the downside if growth and margin slip.")

    # Wave 8H: structured recommendations the graph applies deterministically.
    recommendations = _build_recommendations(profile, ratios, dcf_summary)
    if recommendations:
        # Surface in the prose summary so a reader-of-the-finding sees
        # what the risk lens is actually demanding the memo do.
        summary_lines.append(
            f"{len(recommendations)} structured rec(s) "
            f"applied to the memo (see data.recommendations)."
        )

    # Wave 7C: discretionary notes tagged for the risk agent. Risk has no
    # LLM call, so matched notes ride on `data["research_notes"]` for the
    # drill-down report (Wave 3C) to surface.
    from ..services.research_notes import build_notes_block_for_agent
    notes_block = build_notes_block_for_agent(
        "risk", profile, extra_query="thesis breakers downside survivable",
    )
    finding_data: Dict[str, Any] = {
        "recommendations": [r.model_dump() for r in recommendations],
    }
    if notes_block:
        finding_data["research_notes"] = notes_block

    # Wave 10 — optional LLM thesis-breaker narrative on top of the
    # deterministic numbers. Asks the model to name the SINGLE
    # observation that would invalidate the thesis. Cheap (1 call,
    # <300 tokens) but high-leverage for the user reading the memo:
    # the deterministic risk list is "what's elevated"; this is "what
    # would change my mind." Surfaced via data["narrative"] so the
    # UI can render it as a thesis-breaker callout.
    if settings.has_llm:
        try:
            from . import llm
            payload = {
                "ticker": profile.get("ticker"),
                "sector": profile.get("sector"),
                "industry": profile.get("industry"),
                "key_metrics": {k: ratios.get(k) for k in (
                    "debt_to_ebitda", "EV_EBITDA", "FCF_yield", "ROIC", "PE",
                ) if ratios.get(k) is not None},
                "structural_risks": (profile.get("risks") or [])[:5],
                "dcf_summary": dcf_summary,
                "deterministic_summary": " ".join(summary_lines),
            }
            narr = llm.chat_json(
                "You are the Risk Analyst. The deterministic checks "
                "are already done. Now answer in plain English: what "
                "is the SINGLE observation that would most change "
                "your mind about this thesis? Be specific (cite a "
                "metric, segment, or event) and concise (1-2 "
                "sentences). Return JSON: {thesis_breaker: \"<one "
                "sentence>\", watch_for: [\"<observation 1>\", "
                "\"<observation 2>\", ...]} (up to 3 watch_for items).\n\n"
                + json.dumps(payload, default=str)[:6000],
                system="You are a senior risk analyst. No filler.",
                route="cheap",
                model=settings.openai_tool_model,
                max_tokens=300,
            )
            if isinstance(narr, dict):
                tb = str(narr.get("thesis_breaker") or "").strip()
                wf = [str(w).strip() for w in (narr.get("watch_for") or []) if str(w).strip()][:3]
                if tb or wf:
                    finding_data["narrative"] = {
                        "thesis_breaker": tb,
                        "watch_for": wf,
                    }
        except Exception:  # pragma: no cover — narrative is best-effort
            pass

    # Wave 10 — typed citations for the structural risks + ratio
    # signals the agent is grounding its read on.
    from ..schemas import Citation
    evidence: List[Citation] = []
    for r in (profile.get("risks") or [])[:4]:
        evidence.append(Citation(
            kind="other", ref="profile.risks", excerpt=str(r)[:300],
        ))
    for metric in ("debt_to_ebitda", "EV_EBITDA", "FCF_yield"):
        v = ratios.get(metric)
        if isinstance(v, (int, float)):
            evidence.append(Citation(
                kind="ratio", ref=metric,
                excerpt=f"{metric}={v:.2f}",
            ))
    if dcf_summary:
        evidence.append(Citation(
            kind="dcf", ref=str(profile.get("ticker", "")),
            excerpt=str(dcf_summary)[:300],
        ))

    finding = AgentFinding(
        agent="Risk Analyst",
        headline=f"Risk profile for {profile.get('ticker', '')}",
        summary=" ".join(summary_lines),
        key_points=key_points,
        confidence=0.7,
        sources=[],
        evidence=evidence[:8],
        data=finding_data,
    )

    # Wave 9: re-fire path. Only invoke an LLM enrichment when the PM has
    # asked a follow-up — keeps round-0 cheap & deterministic.
    if prior_round_critique and settings.has_llm:
        try:
            from . import llm, prompts
            payload = {
                "ticker": profile.get("ticker"),
                "ratios": {k: ratios.get(k) for k in (
                    "debt_to_ebitda", "EV_EBITDA", "FCF_yield", "ROIC",
                ) if ratios.get(k) is not None},
                "risks": (profile.get("risks") or [])[:6],
                "current_summary": finding.summary,
                "current_recs": [r.model_dump() for r in recommendations],
            }
            llm_out = llm.chat_json(
                "You are the Risk Analyst answering a senior PM follow-up. "
                "Address the question directly with reference to the leverage / "
                "valuation / FCF data and the structured recommendations already "
                "applied. Return JSON: "
                "{headline, summary, key_points, confidence}.\n\n"
                f"PM follow-up: {prior_round_critique}\n\n"
                "Context:\n" + json.dumps(payload, default=str),
                system=prompts.PM_SYSTEM, route="cheap",
                model=settings.openai_tool_model,
            )
            if isinstance(llm_out, dict) and llm_out.get("summary"):
                finding = AgentFinding(
                    agent="Risk Analyst",
                    headline=str(llm_out.get("headline") or finding.headline)[:240],
                    summary=str(llm_out["summary"]),
                    key_points=[str(p) for p in (llm_out.get("key_points") or finding.key_points)][:8],
                    confidence=float(llm_out.get("confidence", 0.7)),
                    sources=finding.sources,
                    data=finding_data,
                )
        except Exception:  # pragma: no cover — defensive; fall through to deterministic
            pass

    return finding


def derive_risk_items(profile: Dict) -> List[RiskItem]:
    risks = profile.get("risks") or []
    items: List[RiskItem] = []
    for r in risks[:6]:
        sev = "high" if any(k in r.lower() for k in ("competition", "regulator", "antitrust")) else "medium"
        type_ = "company"
        rl = r.lower()
        if "regulat" in rl or "antitrust" in rl or "policy" in rl:
            type_ = "regulatory"
        elif "macro" in rl or "rate" in rl or "recession" in rl:
            type_ = "macro"
        elif "valuation" in rl or "multiple" in rl:
            type_ = "valuation"
        items.append(RiskItem(title=r[:80], detail=r, severity=sev, type=type_))
    return items
