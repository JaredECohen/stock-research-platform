"""Wave 10 — per-agent influence on the memo's rating.

For every memo, score each specialist's directional pull on the
final rating. Computed *post*-synthesis from each AgentFinding's
confidence + tone — no extra LLM calls, no impact on memo cost.

Powers:
- The track-record dashboard's per-agent attribution chart
  (signed influence, averaged across many memos, surfaces which
  specialists are systematically pulling the right way).
- The PM's self-improvement context, which can later say "your
  technical analyst's pull has been correlated with wrong calls in
  the last quarter — discount it."

Method (deterministic, transparent):
1. Score each finding's tone via simple bullish/bearish keyword
   counts on (headline + summary).
2. Multiply by confidence to bias toward high-conviction calls.
3. Normalize so the largest absolute pull is ±1.0; the rest are
   proportional. An agent at 0 had no directional pull (e.g.,
   intake-skipped, or a balanced finding).
"""
from __future__ import annotations

import logging
import re
from typing import Dict

from ..schemas import AgentFinding

log = logging.getLogger(__name__)


_BULLISH_TERMS = (
    "bullish", "constructive", "tailwind", "outperform", "premium",
    "upside", "growth", "expansion", "compounder", "moat", "wide-moat",
    "raise", "raised", "beat", "accelerat", "win", "winning", "leader",
    "favored", "support", "accretive", "improving",
)
_BEARISH_TERMS = (
    "bearish", "cautious", "headwind", "underperform", "compress",
    "compression", "downside", "decel", "decline", "miss", "missed",
    "guidance cut", "lowered", "elevated risk", "pressured", "pressure",
    "stretched", "expensive", "deteriorat", "loser", "challenge",
)


def _tone_score(text: str) -> float:
    """+1 fully bullish, -1 fully bearish, 0 neutral / mixed."""
    if not text:
        return 0.0
    low = text.lower()
    bull_hits = sum(1 for term in _BULLISH_TERMS if term in low)
    bear_hits = sum(1 for term in _BEARISH_TERMS if term in low)
    total = bull_hits + bear_hits
    if total == 0:
        return 0.0
    return (bull_hits - bear_hits) / total


def _agent_pull(finding: AgentFinding) -> float:
    """Directional pull = tone × confidence. Skipped agents (intake)
    return 0 because their stub has zero confidence + no key_points."""
    if not finding:
        return 0.0
    if (finding.data or {}).get("intake_skipped"):
        return 0.0
    text = " ".join([
        finding.headline or "",
        finding.summary or "",
        " ".join(finding.key_points or []),
    ])
    tone = _tone_score(text)
    return tone * (finding.confidence or 0.0)


def compute_influence(findings: Dict[str, AgentFinding]) -> Dict[str, float]:
    """For each agent in `findings`, compute its signed pull on the
    rating. Results are normalized so the biggest |pull| has
    magnitude 1.0; downstream consumers can present them as a
    bar chart with the strongest mover at full extension.
    """
    raw: Dict[str, float] = {}
    for name, finding in (findings or {}).items():
        raw[name] = _agent_pull(finding)
    max_abs = max((abs(v) for v in raw.values()), default=0.0)
    if max_abs <= 1e-6:
        return {k: 0.0 for k in raw}
    return {k: round(v / max_abs, 4) for k, v in raw.items()}
