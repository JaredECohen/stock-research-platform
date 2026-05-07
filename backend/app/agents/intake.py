"""Wave 10 — PM intake step.

Before the parallel fan-out, the PM looks at the company profile +
recent news alerts + macro regime and decides which specialists matter
most for *this* memo. Default = run all 8. PM can deprioritize up to
3 specialists per run, with a logged rationale.

Why this matters:
- A regulated bank doesn't need a deep technical read.
- A semis stock with no recent regulatory catalysts doesn't need a
  full filings re-pass on every memo refresh.
- A name with sparse coverage doesn't need a full comps run.

Cost discipline: skipped specialists return a `(skipped=True)` marker
that the memo path turns into a tiny stub `AgentFinding` rather than
running the full LLM call. Saves ~60% of the per-skip cost.

Defensive: when the LLM is unavailable or returns garbage, returns the
"all specialists run" decision so behavior is unchanged.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..config import settings
from . import llm

log = logging.getLogger(__name__)

ALL_SPECIALISTS: List[str] = [
    "sector", "earnings", "filing", "valuation", "comps",
    "macro", "risk", "technical",
]

# Hard-cap: PM may skip at most this many specialists per memo.
# Forces the model to keep the rating defensible against the others.
_MAX_SKIPS = 3


@dataclass
class IntakeDecision:
    skipped: Set[str] = field(default_factory=set)
    rationale: str = ""

    def runs(self, specialist: str) -> bool:
        return specialist not in self.skipped

    def model_dump(self) -> Dict[str, Any]:
        return {
            "skipped": sorted(self.skipped),
            "rationale": self.rationale,
        }


def run_intake(
    profile: Dict[str, Any],
    news_alerts: Optional[List[Dict[str, Any]]] = None,
    *,
    macro_regime: Optional[str] = None,
) -> IntakeDecision:
    """Decide which specialists to run for this memo.

    Default: run all 8. LLM may deprioritize up to `_MAX_SKIPS` with a
    one-line rationale per skip. Returns the decision (caller threads
    it through the fan-out)."""
    if not getattr(settings, "openai_api_key", None):
        return IntakeDecision()
    payload = {
        "ticker": profile.get("ticker"),
        "company_name": profile.get("company_name"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "business_description": (profile.get("business_description") or "")[:1500],
        "drivers": (profile.get("drivers") or [])[:5],
        "risks": (profile.get("risks") or [])[:5],
        "macro_regime": macro_regime,
        "recent_news": [
            {"title": n.get("title"), "severity": n.get("severity")}
            for n in (news_alerts or [])[:5]
        ],
    }
    out = llm.chat_json(
        "You are the PM doing intake on a memo run. Eight specialists "
        "are available — sector, earnings, filing, valuation, comps, "
        "macro, risk, technical. Default: run them all. You may "
        f"DEPRIORITIZE up to {_MAX_SKIPS} specialists for this memo "
        "ONLY when running them adds little to the thesis (e.g., a "
        "regulated bank rarely needs a technical read; a name with "
        "no recent material news doesn't need a fresh filings pass). "
        "Each skip MUST cite a specific reason — generic 'low value' "
        "is not acceptable.\n\n"
        "Return strict JSON: { \"skip\": [\"<specialist>\", ...], "
        "\"rationale\": \"<one paragraph explaining the skips, or "
        "empty string if running all>\" }.\n\n"
        + json.dumps(payload, default=str)[:6000],
        system="You are a cost-aware buy-side PM. Be specific.",
        route="cheap",
    )
    if not isinstance(out, dict):
        return IntakeDecision()
    raw_skip = out.get("skip") or []
    if not isinstance(raw_skip, list):
        return IntakeDecision()
    cleaned = {
        str(s).strip().lower() for s in raw_skip
        if isinstance(s, str) and s.strip().lower() in ALL_SPECIALISTS
    }
    if len(cleaned) > _MAX_SKIPS:
        cleaned = set(list(cleaned)[:_MAX_SKIPS])
    rationale = str(out.get("rationale") or "").strip()[:1000]
    decision = IntakeDecision(skipped=cleaned, rationale=rationale)
    log.info(
        "PM intake for %s: skipped=%s rationale=%s",
        profile.get("ticker"), sorted(decision.skipped), rationale[:100],
    )
    return decision


def stub_finding(specialist: str, rationale: str) -> Dict[str, Any]:
    """Build the placeholder AgentFinding payload for a skipped agent.

    Returns a dict so callers can `AgentFinding(**stub_finding(...))`.
    """
    pretty = {
        "sector": "Sector Analyst", "earnings": "Earnings Analyst",
        "filing": "Filing Analyst", "valuation": "Valuation Analyst",
        "comps": "Comps Analyst", "macro": "Macro Analyst",
        "risk": "Risk Analyst", "technical": "Technical Analyst",
    }.get(specialist, specialist.title())
    return {
        "agent": pretty,
        "headline": f"{pretty} — skipped per PM intake.",
        "summary": (
            rationale or "PM elected to deprioritize this specialist for "
            "this memo. Re-run the memo to force the full fan-out."
        ),
        "key_points": [],
        "confidence": 0.0,
        "sources": [],
        "data": {"intake_skipped": True, "intake_rationale": rationale},
    }
