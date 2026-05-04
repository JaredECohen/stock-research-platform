"""Wave 3C — drill-down long-form agent reports.

Each specialist's `AgentFinding` gets an optional markdown `long_form_report`.
The deterministic build always runs (cheap, no LLM, just composed from the
structured fields the agent already produced). When `ENABLE_LONG_FORM_REPORTS=true`,
the deterministic body is wrapped in an LLM enrichment pass that adds 1-2
paragraphs of context-aware narrative. The deterministic body is the floor;
LLM output is the ceiling — never replaces, only enriches.

Why a dedicated module:
- Keeps each agent's runner focused on its primary structured output.
- One place owns long-form formatting conventions (heading style, bullet
  vs. paragraph balance, evidence-quoting). Easier to A/B-test layouts
  without rewriting seven agents.
- The flag-gated LLM call is batched into a single call site so spend is
  observable at a single grep.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..config import settings
from ..schemas import AgentFinding
from . import llm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic build — always runs
# ---------------------------------------------------------------------------

def _heading_for(agent_name: str, ticker: str) -> str:
    return f"## {agent_name} — {ticker} drill-down"


def _bulleted(points: Optional[list]) -> str:
    if not points:
        return ""
    return "\n".join(f"- {p}" for p in points)


def _format_data_evidence(data: Dict[str, Any]) -> str:
    """Pull a few of the most useful structured fields out of `finding.data`.

    Different agents stash different shapes — this is best-effort: if a
    well-known key exists we render it; otherwise we skip silently.
    """
    if not data:
        return ""
    bits: list[str] = []

    # ─── Sector agent ──────────────────────────────────────────────
    placements = data.get("kpi_placements")
    if isinstance(placements, dict) and placements:
        rows = []
        for kpi, p in list(placements.items())[:6]:
            if not isinstance(p, dict):
                continue
            target = p.get("target")
            median = (p.get("distribution") or {}).get("median")
            quartile = p.get("quartile")
            interp = p.get("interpretation") or ""
            if target is None:
                continue
            target_s = _fmt_metric_value(kpi, target)
            median_s = _fmt_metric_value(kpi, median) if median is not None else "—"
            q_s = f"Q{quartile}" if quartile else ""
            rows.append(
                f"- **{kpi}** — target **{target_s}**, cohort median {median_s}"
                + (f" ({q_s})" if q_s else "")
                + (f": _{interp}_" if interp else "")
            )
        if rows:
            bits.append("### Cohort placement\n" + "\n".join(rows))

    cross = data.get("cross_sector_relevance")
    if isinstance(cross, list) and cross:
        chips = ", ".join(f"`{t}`" for t in cross[:8])
        bits.append(
            "### Cross-sector pull-through\n"
            f"Other sectors with material thesis links: {chips}."
        )

    bb = data.get("bull_bear_analysis")
    if isinstance(bb, dict):
        synth = (bb.get("sector_synthesis") or "").strip()
        kd = (bb.get("key_disagreement") or "").strip()
        lean = bb.get("sector_lean")
        if synth or kd or lean:
            section = ["### Sector synthesis"]
            if synth:
                section.append(synth)
            if kd:
                section.append(f"**Where bulls and bears disagree:** {kd}")
            if lean and lean != "balanced":
                section.append(f"**Sector lean:** {lean} (PM may diverge).")
            ftests = bb.get("falsifiable_tests") or []
            if ftests:
                lines = ["**Falsifiable tests:**"]
                for t in ftests[:4]:
                    if not isinstance(t, dict):
                        continue
                    side = t.get("invalidates_side", "")
                    stmt = (t.get("statement") or "").strip()
                    if stmt:
                        lines.append(f"- _Invalidates {side}:_ {stmt}")
                section.append("\n".join(lines))
            bits.append("\n\n".join(section))

    # ─── Comps agent ──────────────────────────────────────────────
    history = data.get("history")
    if isinstance(history, dict):
        own_med = history.get("own_median") or {}
        cur_pct = history.get("current_percentile") or {}
        cvm = history.get("current_vs_own_median") or {}
        lines = []
        for k in ("ev_ebitda", "operating_margin", "revenue_growth"):
            if own_med.get(k) is None:
                continue
            med_s = _fmt_metric_value(k, own_med[k])
            pct = cur_pct.get(k)
            delta = cvm.get(k)
            extras = []
            if pct is not None:
                extras.append(f"{pct * 100:.0f}th pct of own history")
            if delta is not None:
                extras.append(f"{delta:+.0%} vs own median")
            extras_s = " · ".join(extras)
            lines.append(f"- **{k}** vs own {history.get('lookback_label', 'history')} median {med_s}"
                         + (f" — {extras_s}" if extras_s else ""))
        if lines:
            bits.append("### Self-historical valuation\n" + "\n".join(lines))
        if (history.get("interpretation") or "").strip():
            bits.append(f"_{history['interpretation']}_")

    # ─── Technical agent ──────────────────────────────────────────────
    signals = data.get("signals")
    if isinstance(signals, dict):
        readout: list[str] = []
        if signals.get("trend") and signals.get("momentum"):
            readout.append(
                f"- **Regime:** {signals['trend']} trend, {signals['momentum']} momentum"
            )
        if signals.get("position_52w") is not None:
            readout.append(
                f"- **52-week position:** {signals['position_52w'] * 100:.0f}% of range"
                f" (high ${signals.get('high_52w', 0):,.2f} / low ${signals.get('low_52w', 0):,.2f})"
            )
        if signals.get("rsi_14") is not None:
            label = (
                "overbought" if signals["rsi_14"] > 70
                else "oversold" if signals["rsi_14"] < 30
                else "mid-range"
            )
            readout.append(f"- **RSI(14):** {signals['rsi_14']:.0f} ({label})")
        if signals.get("macd_histogram") is not None:
            readout.append(
                f"- **MACD histogram:** {signals['macd_histogram']:+.2f}"
                f" (signal {signals.get('macd_signal', 0):+.2f})"
            )
        if signals.get("sma_50_above_200") is not None:
            readout.append(
                "- **Cross alignment:** "
                + ("golden-cross (SMA50 > SMA200)" if signals["sma_50_above_200"]
                   else "death-cross (SMA50 < SMA200)")
            )
        if signals.get("bb_position") is not None:
            readout.append(
                f"- **Bollinger position:** {signals['bb_position'] * 100:.0f}% within band"
            )
        if readout:
            bits.append("### Indicator readout\n" + "\n".join(readout))

    # ─── Risk agent ──────────────────────────────────────────────
    recs = data.get("recommendations")
    if isinstance(recs, list) and recs:
        lines = ["### Risk lens recommendations"]
        for r in recs[:6]:
            if not isinstance(r, dict):
                continue
            target = r.get("target", "")
            direction = r.get("direction", "")
            mag = r.get("magnitude", "")
            detail = (r.get("detail") or "").strip()
            rationale = (r.get("rationale") or "").strip()
            if not detail:
                continue
            lines.append(
                f"- **{target}** ({direction}, {mag}): {detail}"
                + (f"  \n  _{rationale}_" if rationale else "")
            )
        if len(lines) > 1:
            bits.append("\n".join(lines))

    applied = data.get("applied_recommendations")
    if isinstance(applied, list) and applied:
        lines = ["### What the memo actually changed"]
        for r in applied[:6]:
            if not isinstance(r, dict):
                continue
            change = r.get("applied_change") or {}
            field = change.get("field") or r.get("target", "")
            if "from" in change and "to" in change:
                lines.append(f"- **{field}**: {change['from']} → {change['to']}")
            elif change.get("appended"):
                lines.append(f"- **{field}** appended: {change['appended']}")
        if len(lines) > 1:
            bits.append("\n".join(lines))

    # ─── Discretionary research notes (Wave 7C) ──────────────────────
    notes_md = data.get("research_notes")
    if isinstance(notes_md, str) and notes_md.strip():
        bits.append(notes_md.strip())

    return "\n\n".join(bits)


def _fmt_metric_value(key: str, value: Any) -> str:
    """Best-effort pretty-printer that picks a reasonable format from the
    metric name. Margin/yield ratios → %, multiples → x, otherwise raw."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    k = (key or "").lower()
    if any(t in k for t in ("margin", "yield", "growth", "_pct", "rate", "tax")):
        return f"{v:.1%}"
    if any(t in k for t in ("ev_", "p_fcf", "pe", "pfcf", "multiple", "ebitda")):
        return f"{v:.1f}x"
    if abs(v) < 1:
        return f"{v:.2f}"
    return f"{v:,.1f}"


def deterministic_long_form(
    finding: AgentFinding, *, ticker: str, agent_name: str,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Compose markdown from the structured fields. Always succeeds; no LLM.

    Wave 8O: per-agent flavor on the opening "## What this analyst is
    looking at" block so the drill-down isn't just a re-run of the
    summary. We pull from `profile` (drivers / risks / sector) and
    agent-specific rules to add real depth.
    """
    parts: list[str] = [_heading_for(agent_name, ticker.upper())]
    if finding.headline.strip():
        parts.append(f"**{finding.headline.strip()}**")

    # Per-agent intro block — what this lens is examining + why.
    intro = _agent_intro(agent_name, profile or {}, finding)
    if intro:
        parts.append("### What this analyst is looking at\n" + intro)

    if finding.summary.strip():
        parts.append("### Summary\n" + finding.summary.strip())

    if finding.key_points:
        parts.append("### Key points\n" + _bulleted(finding.key_points))

    evidence = _format_data_evidence(finding.data or {})
    if evidence:
        parts.append(evidence)

    # What changes the read — agent-specific watch items.
    watch = _what_would_change_view(agent_name, profile or {}, finding)
    if watch:
        parts.append("### What would change this read\n" + watch)

    sources = _format_sources(finding.sources)
    if sources:
        parts.append("### Sources\n" + sources)

    parts.append(f"_Analyst confidence: {int(finding.confidence * 100)}/100._")
    return "\n\n".join(p for p in parts if p)


def _agent_intro(
    agent_name: str, profile: Dict[str, Any], finding: AgentFinding,
) -> str:
    """Per-agent context paragraph. Says what this lens cares about for
    *this specific company* — driver list, risk list, regime, etc."""
    drivers = profile.get("drivers") or []
    risks = profile.get("risks") or []
    sector = profile.get("sector") or ""
    name = profile.get("company_name") or profile.get("ticker") or "this company"
    drivers_phrase = (
        ", ".join(drivers[:3]) if drivers else "core sector drivers"
    )
    risks_phrase = ", ".join(risks[:2]) if risks else "execution"

    if agent_name == "Sector Analyst":
        return (
            f"How {name} sits in the {sector} cohort. The lens reads quartile "
            f"placement on the sector's KPIs (margin, growth, valuation, capex), "
            f"identifies cohort outliers, detects regime, and emits a structured "
            f"bull/bear analysis with falsifiable tests on each side."
        )
    if agent_name == "Earnings Analyst":
        return (
            f"What management actually said. The lens reads the latest call's "
            f"prepared remarks + Q&A, extracts management tone, guidance changes, "
            f"capex commentary, segment color, and bullish/bearish takeaways. "
            f"Especially attentive to: {drivers_phrase}."
        )
    if agent_name == "Filing Analyst":
        return (
            f"What the SEC filing actually discloses. The lens pulls risk "
            f"factors, MD&A highlights, segment + customer-concentration "
            f"disclosures, and any legal/regulatory exposure. Cross-references "
            f"against the thesis-relevant lines that would change a rating."
        )
    if agent_name == "Valuation Analyst":
        return (
            f"What's priced in vs. what isn't. The lens triangulates DCF "
            f"(base/bull/bear scenarios) against trading multiples (P/E, EV/"
            f"EBITDA, P/FCF, FCF yield) and prior-period history. Especially "
            f"sensitive to: terminal-growth + WACC fragility, multiple "
            f"compression risk, FCF cushion."
        )
    if agent_name == "Comps Analyst":
        return (
            f"Two valuation lenses, side by side: peer-relative (vs. cohort "
            f"median) and self-historical (vs. {profile.get('ticker', 'the name')}'s "
            f"own multi-year distribution). Divergence between the two is "
            f"the highest-signal output — when peers say cheap but own-history "
            f"says expensive (or vice versa), that's the alpha to investigate."
        )
    if agent_name == "Macro Analyst":
        return (
            f"How the prevailing macro regime maps to {sector}. The lens "
            f"reads the current FRED snapshot (rates, inflation, unemployment), "
            f"classifies the regime, and translates it into first-order + "
            f"second-order effects on the company's drivers."
        )
    if agent_name == "Risk Analyst":
        return (
            f"Thesis-breaker watch. The lens scans leverage, valuation "
            f"stretch, FCF cushion, customer/regulatory concentration, and "
            f"DCF bear scenario. Profile risks under review: {risks_phrase}. "
            f"Outputs structured recommendations the graph deterministically "
            f"applies (confidence cap, rating downshift, bear augmentation)."
        )
    if agent_name == "Technical Analyst":
        return (
            f"Positioning context — NOT a trade signal. The lens computes "
            f"SMA 50/200, EMA 10/20, RSI(14), MACD(12/26/9), Bollinger Bands, "
            f"VWMA, and 52-week placement. Frames the regime so the user "
            f"knows where the chart sits relative to the fundamental thesis."
        )
    return (
        f"{agent_name}'s framework applied to {name}. Drivers in scope: "
        f"{drivers_phrase}; risks under watch: {risks_phrase}."
    )


def _what_would_change_view(
    agent_name: str, profile: Dict[str, Any], finding: AgentFinding,
) -> str:
    """Per-agent 'what to monitor' list — the falsifiable tests that
    could move the lens's read in either direction."""
    if agent_name == "Sector Analyst":
        return (
            "- A peer's KPI quartile shift (revenue growth, op margin, capex) "
            "would re-anchor cohort placement.\n"
            "- A regime change broadcast by the macro loop (e.g., from "
            "expansion → contraction) would reset the bull/bear lean.\n"
            "- New cross-sector relevance flagged by the news loop would "
            "alter the pull-through map."
        )
    if agent_name == "Earnings Analyst":
        return (
            "- Next earnings call: guidance change, segment growth re-"
            "acceleration or deceleration, capex commentary shift.\n"
            "- Management tone shift (constructive → cautious) on the "
            "next prepared remarks pass."
        )
    if agent_name == "Filing Analyst":
        return (
            "- New 10-K/10-Q/8-K with material risk-factor changes.\n"
            "- Updates to customer-concentration / segment disclosures.\n"
            "- New legal/regulatory item that lands in MD&A."
        )
    if agent_name == "Valuation Analyst":
        return (
            "- DCF assumption updater proposes a >5% shift in WACC or terminal "
            "growth (audit log records the rationale).\n"
            "- EV/EBITDA crosses ±20% vs. self-history median.\n"
            "- A new earnings print moves base-case revenue or op margin."
        )
    if agent_name == "Comps Analyst":
        return (
            "- Peer-relative vs. own-history divergence widens or narrows "
            "(divergence is the alpha signal).\n"
            "- A new peer-side filing materially shifts cohort medians.\n"
            "- Currency translation / one-off charge adjustments to FCF."
        )
    if agent_name == "Risk Analyst":
        return (
            "- Leverage crosses 3.5x debt/EBITDA (triggers confidence cap).\n"
            "- Customer-concentration disclosure crosses 25% top customer.\n"
            "- New regulatory action with named exposure.\n"
            "- DCF bear case widens past the prior version."
        )
    if agent_name == "Technical Analyst":
        return (
            "- Cross-alignment flips (golden ↔ death cross).\n"
            "- RSI sustained above 70 or below 30 across multiple sessions.\n"
            "- Price exits the Bollinger band on either side.\n"
            "- New all-time high or 52-week low."
        )
    if agent_name == "Macro Analyst":
        return (
            "- A FRED snapshot regime change (e.g., real-rate inflection).\n"
            "- A scenario re-classification by the macro loop's prompt."
        )
    return ""


def _format_sources(sources: List[Any]) -> str:
    """Pretty-print the source list, dropping bare `key:` entries with no value."""
    if not sources:
        return ""
    out: list[str] = []
    for s in sources[:10]:
        text = str(s).strip()
        if not text:
            continue
        # Drop empty `kind:` entries.
        if text.endswith(":") or text.endswith(": "):
            continue
        out.append(f"`{text}`")
    if not out:
        return ""
    return ", ".join(out)


# ---------------------------------------------------------------------------
# LLM enrichment — only when ENABLE_LONG_FORM_REPORTS=true
# ---------------------------------------------------------------------------

_ENRICH_PROMPT = (
    "You are writing a drill-down memo paragraph (1-2 paragraphs) that "
    "extends the structured finding below. Stay grounded in the supplied "
    "headline / summary / key_points / data — do NOT invent numbers or "
    "introduce claims not visible in the input. Output PLAIN markdown "
    "(no JSON wrapping, no headings — just paragraphs). Keep it under "
    "180 words."
)


def _enriched_long_form(
    finding: AgentFinding, *, ticker: str, agent_name: str,
    profile: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    payload = {
        "ticker": ticker,
        "agent": agent_name,
        "headline": finding.headline,
        "summary": finding.summary,
        "key_points": finding.key_points,
        "sources": finding.sources[:8],
        "profile_drivers": (profile or {}).get("drivers"),
        "profile_risks": (profile or {}).get("risks"),
    }
    user = (
        _ENRICH_PROMPT + "\n\nFinding payload (JSON):\n"
        + json.dumps(payload, default=str)[:1800]
    )
    try:
        text = llm.chat_text(
            user, system="You are a careful equity research editor.",
            route="cheap", model=settings.openai_tool_model,
        )
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("Long-form enrichment failed for %s/%s: %s", ticker, agent_name, exc)
        return None
    if not text:
        return None
    cleaned = text.strip()
    return cleaned or None


def build_long_form_report(
    finding: AgentFinding, *, ticker: str, agent_name: str,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Public entrypoint. Always returns a markdown string.

    Composes deterministic body + (optionally) an LLM-enriched closing
    paragraph. If the LLM call fails or the flag is off, the deterministic
    body alone is returned.
    """
    body = deterministic_long_form(
        finding, ticker=ticker, agent_name=agent_name, profile=profile,
    )
    if not settings.enable_long_form_reports:
        return body
    extra = _enriched_long_form(
        finding, ticker=ticker, agent_name=agent_name, profile=profile,
    )
    if not extra:
        return body
    return body + "\n\n### Analyst expansion\n" + extra


def attach_long_form(
    finding: Optional[AgentFinding], *, ticker: str, agent_name: str,
    profile: Optional[Dict[str, Any]] = None,
) -> Optional[AgentFinding]:
    """Mutate a finding's `long_form_report` in place and return it.

    No-ops on None so callers don't need to guard the optional
    Technical Analyst / future-optional findings.
    """
    if finding is None:
        return None
    try:
        finding.long_form_report = build_long_form_report(
            finding, ticker=ticker, agent_name=agent_name, profile=profile,
        )
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("Long-form build failed for %s/%s: %s", ticker, agent_name, exc)
    return finding
