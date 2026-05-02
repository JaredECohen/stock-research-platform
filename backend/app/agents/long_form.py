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
from typing import Any, Dict, Optional

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

    # Sector agent: KPI placements + cohort math.
    placements = data.get("kpi_placements")
    if isinstance(placements, dict) and placements:
        sample = list(placements.items())[:4]
        rows = []
        for kpi, p in sample:
            if not isinstance(p, dict):
                continue
            target = p.get("target")
            median = (p.get("distribution") or {}).get("median")
            quartile = p.get("quartile")
            if target is None:
                continue
            rows.append(
                f"- **{kpi}** — target {target}, cohort median {median} (quartile {quartile})"
            )
        if rows:
            bits.append("### Cohort placement\n" + "\n".join(rows))

    # Sector agent: cross-sector relevance.
    cross = data.get("cross_sector_relevance")
    if isinstance(cross, list) and cross:
        bits.append("### Cross-sector pull-through\n" + ", ".join(str(t) for t in cross))

    # Sector agent: bull/bear analysis (synthesis + key disagreement).
    bb = data.get("bull_bear_analysis")
    if isinstance(bb, dict):
        synth = bb.get("sector_synthesis")
        kd = bb.get("key_disagreement")
        bits.append("### Sector synthesis\n" + (synth or "—"))
        if kd:
            bits.append(f"**Key disagreement:** {kd}")

    # Technical agent: structured signals.
    signals = data.get("signals")
    if isinstance(signals, dict):
        rows = []
        for k in (
            "trend", "momentum", "rsi_14", "sma_50_above_200",
            "position_52w", "macd_histogram",
        ):
            if k in signals and signals[k] is not None:
                rows.append(f"- **{k}**: {signals[k]}")
        if rows:
            bits.append("### Indicator readout\n" + "\n".join(rows))

    return "\n\n".join(bits)


def deterministic_long_form(
    finding: AgentFinding, *, ticker: str, agent_name: str,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Compose markdown from the structured fields. Always succeeds; no LLM."""
    parts: list[str] = [_heading_for(agent_name, ticker.upper())]
    parts.append(f"**{finding.headline.strip()}**")
    parts.append(finding.summary.strip() or "—")
    if finding.key_points:
        parts.append("### Key points\n" + _bulleted(finding.key_points))
    evidence = _format_data_evidence(finding.data or {})
    if evidence:
        parts.append(evidence)
    if finding.sources:
        parts.append(
            "### Sources\n" + ", ".join(str(s) for s in finding.sources[:8])
        )
    parts.append(
        f"_Confidence: {finding.confidence:.2f}._"
    )
    return "\n\n".join(p for p in parts if p)


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
