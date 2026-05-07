"""Macro analyst agent — LLM-driven scenario reading + per-company impact.

Architecture:
- `detect_scenario_key` — LLM classifier (regex fallback) that maps free-form
  text to one of the canned scenario archetypes. The archetype is just the
  *seed* template; the LLM is then free to generalize beyond it.
- `run_macro_scenario` — returns a `MacroScenarioResult` whose narrative is
  rewritten by the LLM using the live FRED snapshot, with the canned
  template only used as a deterministic backstop.
- `run_macro_agent` — produces the per-company `AgentFinding` for the memo.
  The prompt carries snapshot + scenario context + company profile so the
  output is sector- and ticker-aware, not a generic regime read.

All three layers consult `OPENAI_MACRO_MODEL` via `settings.openai_macro_model`,
matching the per-agent model wiring that PM/sector/tool agents already use.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import settings
from ..schemas import AgentFinding, MacroScenarioResult
from ..services.macro_service import macro_snapshot
from . import llm, prompts


SCENARIO_TEMPLATES: Dict[str, MacroScenarioResult] = {
    "soft_landing": MacroScenarioResult(
        scenario="Soft landing",
        narrative=(
            "Growth stays positive, inflation glides toward target, and the Fed delivers measured rate cuts. "
            "Real yields ease without breaking employment. Risk assets benefit but earnings dispersion widens."
        ),
        sector_impacts={
            "Technology": "Constructive — AI capex continues, multiple support from real-rate easing.",
            "Communication Services": "Constructive — AI-leveraged ad platforms benefit from rerating.",
            "Consumer Discretionary": "Improving — wage gains real, financing eases.",
            "Financials": "Mixed — NIM compresses but credit holds; capital markets reopen.",
            "Healthcare": "Stable — defensive support fades but pipeline catalysts matter.",
            "Energy": "Range-bound — slower demand offset by capital discipline.",
            "Industrials": "Constructive — infrastructure spend, capex cycle.",
            "Utilities": "Constructive — long-duration relief if long end falls.",
        },
        favored_sectors=["Technology", "Communication Services", "Industrials", "Consumer Discretionary"],
        pressured_sectors=["Energy", "Defensive Staples"],
        suggested_research_views=[
            "Quality compounders with AI exposure",
            "Cyclicals with operating leverage to PMI recovery",
            "Long-duration assets benefiting from real-rate easing",
        ],
        risks=["Re-acceleration in inflation", "Labor market crack", "Credit-spread blowout"],
    ),
    "recession": MacroScenarioResult(
        scenario="Recession",
        narrative=(
            "Activity slows, unemployment rises, earnings cuts broaden. Defensive cash flows outperform. "
            "Risk assets reprice; the Fed pivots faster but credit spreads widen first."
        ),
        sector_impacts={
            "Consumer Staples": "Outperforms — defensive demand.",
            "Healthcare": "Outperforms — payor mix headwinds offset by demand stability.",
            "Utilities": "Outperforms — bond proxy and rate relief.",
            "Technology": "Mixed — capex digestion offsets duration tailwind.",
            "Financials": "Underperforms — credit losses, NIM compression.",
            "Consumer Discretionary": "Underperforms — discretionary spend retracts.",
            "Energy": "Mixed — demand pressure, capital discipline support.",
        },
        favored_sectors=["Consumer Staples", "Healthcare", "Utilities"],
        pressured_sectors=["Financials", "Consumer Discretionary", "Industrials"],
        suggested_research_views=[
            "Defensive compounders with FCF resilience",
            "Healthcare names with low cyclicality",
            "Mega-cap balance sheets that absorb credit shocks",
        ],
        risks=["Policy lag prolonging downturn", "CRE losses contaminating banks"],
    ),
    "sticky_inflation": MacroScenarioResult(
        scenario="Sticky inflation",
        narrative=(
            "Core inflation lingers above target. Long-end yields stay elevated, dampening duration assets. "
            "Pricing power and asset-light models outperform; rate cuts are pushed out."
        ),
        sector_impacts={
            "Energy": "Outperforms — pricing power, capital discipline.",
            "Materials": "Constructive — pricing power.",
            "Financials": "Mixed — higher-for-longer NIM tailwind, credit risk on the other side.",
            "Technology": "Pressured — multiple compression on long-duration cash flows.",
            "Real Estate": "Pressured — cap rate expansion.",
            "Utilities": "Pressured — bond-proxy headwind.",
        },
        favored_sectors=["Energy", "Materials", "Financials"],
        pressured_sectors=["Real Estate", "Utilities", "Long-duration Tech"],
        suggested_research_views=[
            "Pricing-power compounders",
            "Asset-light franchises",
            "Energy and materials with disciplined capital allocation",
        ],
        risks=["Demand destruction from rates", "Currency strength weighing on multinationals"],
    ),
    "falling_rates": MacroScenarioResult(
        scenario="Falling rates",
        narrative=(
            "Real yields ease as inflation glides lower. Long-duration assets benefit; "
            "rate-sensitive consumers re-engage; capital markets reopen."
        ),
        sector_impacts={
            "Technology": "Constructive — duration tailwind, AI capex continues.",
            "Real Estate": "Constructive — cap rate compression.",
            "Consumer Discretionary": "Improving — financing eases.",
            "Utilities": "Constructive — bond-proxy tailwind.",
            "Financials": "Mixed — NIM headwind offset by capital markets.",
        },
        favored_sectors=["Technology", "Real Estate", "Utilities", "Consumer Discretionary"],
        pressured_sectors=["Energy", "Defensive Staples"],
        suggested_research_views=[
            "Long-duration growth with AI exposure",
            "REIT subsectors with re-rating optionality",
            "Capital-markets-leveraged financials",
        ],
        risks=["Rate cuts arrive because growth cracks", "Credit-spread widening"],
    ),
    "ai_capex_boom": MacroScenarioResult(
        scenario="AI capex boom",
        narrative=(
            "Hyperscaler and sovereign AI infrastructure spend remains strong. Power and data-center supply chain "
            "tightens; semiconductors, networking, and electrical infrastructure benefit. Valuation risk concentrates "
            "in the bottleneck names."
        ),
        sector_impacts={
            "Technology": "Outperforms — semiconductors, networking, data-center systems.",
            "Industrials": "Outperforms — power equipment, HVAC, electrical.",
            "Utilities": "Constructive — load growth tailwind.",
            "Energy": "Constructive — data-center power demand.",
        },
        favored_sectors=["Technology", "Industrials", "Utilities"],
        pressured_sectors=["Defensive Staples"],
        suggested_research_views=[
            "AI compute and networking",
            "Power generation and grid suppliers",
            "Data-center REITs with low pre-leasing risk",
        ],
        risks=["Capex digestion phase", "Sovereign AI policy reversals", "Power constraints delaying deployments"],
    ),
}


_SCENARIO_KEYS = list(SCENARIO_TEMPLATES.keys())


def _detect_scenario_key_regex(text: str) -> str:
    """Deterministic fallback used when the LLM is unavailable."""
    t = (text or "").lower()
    if "soft landing" in t:
        return "soft_landing"
    if "recession" in t or "downturn" in t:
        return "recession"
    if "sticky" in t or "inflation stays" in t:
        return "sticky_inflation"
    if "falling rate" in t or "rate cut" in t or "rates fall" in t:
        return "falling_rates"
    if "ai capex" in t or "ai infrastructure" in t or "ai boom" in t:
        return "ai_capex_boom"
    return "soft_landing"


def detect_scenario_key(text: str) -> str:
    """LLM-driven scenario classifier; regex fallback when no LLM is available.

    The LLM is constrained to one of `SCENARIO_TEMPLATES.keys()` so downstream
    code never has to handle a free-form scenario name.
    """
    probs = detect_regime_probabilities(text)
    if probs:
        # Pick the modal regime as the legacy single-tag answer.
        return max(probs.items(), key=lambda kv: kv[1])[0]
    return _detect_scenario_key_regex(text)


def detect_regime_probabilities(text: str) -> Dict[str, float]:
    """Wave 10 — continuous regime probabilities across the 5 archetypes.

    Real macro states are mixtures. Returns a dict
    `{regime_key: probability}` summing to ~1.0. When the LLM isn't
    available, falls back to a one-hot (1.0 on the regex match,
    0.0 elsewhere) so the downstream contract is stable.
    """
    if not settings.has_llm or not text:
        key = _detect_scenario_key_regex(text)
        return {k: (1.0 if k == key else 0.0) for k in _SCENARIO_KEYS}
    out = llm.chat_json(
        "Classify the macro regime as a probability mixture across "
        f"these archetypes: {_SCENARIO_KEYS}. Probabilities should "
        "sum to ~1.0. Use real numbers (not just 0/1) — most regimes "
        "are mixtures.\n\n"
        "Return strict JSON: {\"probabilities\": {\"soft_landing\": "
        "0.5, \"sticky_inflation\": 0.3, \"recession\": 0.2, ...}}.\n\n"
        f"User text:\n{text}",
        route="cheap",
        model=settings.openai_macro_model,
        max_tokens=120,
    )
    raw = (out or {}).get("probabilities") if isinstance(out, dict) else None
    if not isinstance(raw, dict):
        key = _detect_scenario_key_regex(text)
        return {k: (1.0 if k == key else 0.0) for k in _SCENARIO_KEYS}
    cleaned: Dict[str, float] = {}
    for k in _SCENARIO_KEYS:
        v = raw.get(k)
        if isinstance(v, (int, float)) and v > 0:
            cleaned[k] = float(v)
    total = sum(cleaned.values()) or 1.0
    return {k: v / total for k, v in cleaned.items()}


def run_macro_scenario(scenario: str) -> MacroScenarioResult:
    """Live narrative rewrite anchored to the FRED snapshot.

    The canned template still seeds the structure (sector_impacts, favored,
    pressured) so the response shape is stable, but the narrative + suggested
    research views get an LLM pass when one is available.

    Wave 10 — also emits a continuous regime probability mixture
    (`regime_probabilities`). The single `scenario` tag carries the
    modal regime for backward compatibility.
    """
    probs = detect_regime_probabilities(scenario)
    key = (
        max(probs.items(), key=lambda kv: kv[1])[0]
        if probs else _detect_scenario_key_regex(scenario)
    )
    base = SCENARIO_TEMPLATES[key]
    snapshot = macro_snapshot()

    # LLM-driven narrative + research views; deterministic concat fallback.
    if settings.has_llm:
        prompt = (
            "You are the macro analyst. Given the regime archetype and the live "
            "macro snapshot below, rewrite the narrative in 4-6 sentences making "
            "it specific to current numbers. Keep the same regime label. Suggest "
            "3-5 research views that follow from this regime + snapshot.\n\n"
            f"Regime: {base.scenario}\n"
            f"Archetype narrative: {base.narrative}\n"
            f"Live snapshot (FRED): {json.dumps(snapshot, default=str)}\n\n"
            "Return JSON: {\"narrative\": \"<text>\", \"suggested_research_views\": [\"...\", ...]}"
        )
        out = llm.chat_json(
            prompt, system=prompts.MACRO_ANALYST_PROMPT, route="cheap",
            model=settings.openai_macro_model, max_tokens=600,
        )
        if isinstance(out, dict) and (out.get("narrative") or out.get("suggested_research_views")):
            return base.model_copy(update={
                "narrative": out.get("narrative") or base.narrative,
                "suggested_research_views": (
                    out.get("suggested_research_views") or base.suggested_research_views
                ),
                "regime_probabilities": probs,
            })

    # Deterministic fallback: append the live snapshot to the canned narrative.
    if snapshot:
        narrative = base.narrative + (
            f" Current snapshot: Fed Funds {snapshot.get('FEDFUNDS', '—')}%, "
            f"10Y {snapshot.get('DGS10', '—')}%, "
            f"Core sticky CPI {snapshot.get('CORESTICKM159SFRBATL', '—')}%, "
            f"Unemployment {snapshot.get('UNRATE', '—')}%."
        )
        return base.model_copy(update={
            "narrative": narrative,
            "regime_probabilities": probs,
        })
    return base.model_copy(update={"regime_probabilities": probs})


def run_macro_agent(
    profile: Dict, scenario: str = "soft_landing",
    *, prior_round_critique: Optional[str] = None,
) -> AgentFinding:
    """Per-company macro `AgentFinding` for the memo.

    LLM-driven: prompt carries the regime + snapshot + the target company's
    sector, drivers, and risks so the resulting summary is *specific to this
    name*, not a generic regime read. Deterministic template-based fallback
    runs when no LLM is configured.
    """
    s = run_macro_scenario(scenario)
    sector = profile.get("sector", "")
    sector_view = s.sector_impacts.get(sector, "Macro impact mapped via sector framework.")

    if settings.has_llm:
        from .earnings_agent import _critique_block as _q
        snapshot = macro_snapshot()
        # Wave 10 — load the macro primer (anchors and operating
        # principles) + the macro analyst's running notes file as
        # context so the agent reasons like a serious macro PM
        # instead of a textbook.
        from ..prompts import load_prompt
        primer = load_prompt("macro_primer") or ""
        macro_memory_block = ""
        try:
            from ..memory import MacroMemory
            mm = MacroMemory.load_macro()
            macro_memory_block = mm.as_prompt_context(max_chars=2500)
        except Exception:
            macro_memory_block = ""
        primer_block = (("\n\n## PRIMER\n\n" + primer) if primer else "")
        memory_block = (
            ("\n\n## MACRO MEMORY (your running notes)\n\n" + macro_memory_block)
            if macro_memory_block.strip() else ""
        )
        prompt = (
            "Given the regime read and live macro snapshot, write a 4-6 sentence "
            "view focused on THIS COMPANY: how the regime helps or hurts its "
            "thesis, what to watch in the next data print, and which risks "
            "would invalidate the call. Reference at least one snapshot value.\n\n"
            f"Regime: {s.scenario}\n"
            f"Regime narrative: {s.narrative}\n"
            f"Sector default impact ({sector}): {sector_view}\n"
            f"Live snapshot: {json.dumps(snapshot, default=str)}\n"
            f"Company profile: {json.dumps({'ticker': profile.get('ticker'), 'sector': sector, 'industry': profile.get('industry'), 'drivers': profile.get('drivers'), 'risks': profile.get('risks')}, default=str)}\n\n"
            "Return JSON: {headline, summary, key_points (list of strings), confidence (0-1)}."
            + primer_block
            + memory_block
            + _q(prior_round_critique)
        )
        out = llm.chat_json(
            prompt, system=prompts.MACRO_ANALYST_PROMPT, route="cheap",
            model=settings.openai_macro_model, max_tokens=600,
        )
        if isinstance(out, dict) and out.get("summary"):
            return AgentFinding(
                agent="Macro Analyst",
                headline=out.get("headline") or f"Macro scenario: {s.scenario}",
                summary=out["summary"],
                key_points=list(out.get("key_points") or []),
                confidence=float(out.get("confidence", 0.75)),
                sources=["macro_scenario_framework", "fred_snapshot"],
            )

    # Deterministic fallback
    summary = f"Scenario read: {s.narrative} {sector} positioning: {sector_view}"
    key_points = [
        f"Favored sectors: {', '.join(s.favored_sectors)}.",
        f"Pressured sectors: {', '.join(s.pressured_sectors)}.",
        f"Top scenario risks: {'; '.join(s.risks[:3])}.",
    ]
    return AgentFinding(
        agent="Macro Analyst",
        headline=f"Macro scenario: {s.scenario}",
        summary=summary,
        key_points=key_points,
        confidence=0.75,
        sources=["macro_scenario_framework"],
    )
