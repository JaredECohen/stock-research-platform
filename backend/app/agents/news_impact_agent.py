"""Wave 5B — News-impact agent.

Given a prior memo + a fresh news alert, decide whether the news is
*material to this thesis*. If yes, return a structured patch describing
which fields of the memo should change and why. If not, return
`{material: false}` so the orchestrator drops the alert without a memo
update.

Model choice (locked in MASTER_PLAN §5): Anthropic Haiku 4.5 — cheap +
cross-family with PM's OpenAI synthesis so we get an independent read,
not just an OpenAI echo chamber.

Critic is intentionally NOT run on incremental patches (also locked in
MASTER_PLAN). The patch's `revision_log` is flagged with
`critic_skipped: true` so reviewers know.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..config import settings
from ..schemas import NewsAlert, StockMemoOut
from . import llm

log = logging.getLogger(__name__)


_PROMPT = (
    "You are a news-impact analyst. Given a stock memo and a fresh news "
    "alert, decide whether the news is *material to the thesis* — i.e. "
    "should change rating, confidence, or one of the structured fields.\n\n"
    "Threshold for material:\n"
    "- Earnings preannouncement / guidance change.\n"
    "- M&A / divestiture / leadership change at the target.\n"
    "- Regulatory action with named exposure.\n"
    "- Sector-wide regime shift the target is leveraged to.\n\n"
    "NOT material (return material=false):\n"
    "- Daily price commentary, trader notes, options-flow gossip.\n"
    "- Generic sector news without a clear thesis link.\n"
    "- Re-tellings of facts already in the memo's sources_used.\n\n"
    "If material, propose a PATCH — only the fields that should change. "
    "Do NOT touch fields the news doesn't actually inform. Allowed:\n"
    "- rating_label: one of {Bullish, Mixed Positive, Neutral, Mixed "
    "Negative, Bearish}\n"
    "- confidence_score: 0-100, change by at most 15 points per patch\n"
    "- one_sentence_thesis: rewrite if the thesis itself shifted\n"
    "- bull_case / bear_case: append a single key_point if relevant\n"
    "- key_risks: append a single new RiskItem if a risk is unlocked\n"
    "- final_pm_view: rewrite to acknowledge the news\n\n"
    "Each changed field MUST come with a one-sentence rationale.\n\n"
    "Return strict JSON:\n"
    "{\n"
    '  "material": bool,\n'
    '  "patch": { "field": new_value, ... },\n'
    '  "rationales": { "field": "1-sentence why", ... },\n'
    '  "delta_summary": "1-sentence: what changed and why"\n'
    "}\n"
    "When material=false, patch and rationales are empty {}."
)


# Confidence change cap per patch (locked in MASTER_PLAN). Prevents a
# single news event from flipping a memo from 60 → 25.
MAX_CONFIDENCE_DELTA = 15


def _clamp_patch(memo: StockMemoOut, patch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply hard rules to the LLM-proposed patch:
    - confidence_score change capped to ±MAX_CONFIDENCE_DELTA.
    - rating_label must be one of the allowed labels.
    - Drop unknown fields silently (defense against the LLM going rogue).
    """
    allowed_fields = {
        "rating_label", "confidence_score", "one_sentence_thesis",
        "final_pm_view", "bull_case", "bear_case", "key_risks",
    }
    allowed_ratings = {
        "Bullish", "Mixed Positive", "Neutral", "Mixed Negative", "Bearish",
    }
    cleaned: Dict[str, Any] = {}
    for k, v in (patch or {}).items():
        if k not in allowed_fields:
            continue
        if k == "rating_label":
            if v in allowed_ratings:
                cleaned[k] = v
            continue
        if k == "confidence_score":
            try:
                target = float(v)
            except (TypeError, ValueError):
                continue
            current = float(memo.confidence_score or 0)
            target = max(
                current - MAX_CONFIDENCE_DELTA,
                min(current + MAX_CONFIDENCE_DELTA, target),
            )
            cleaned[k] = max(0.0, min(100.0, target))
            continue
        cleaned[k] = v
    return cleaned


def assess(
    memo: StockMemoOut, alert: NewsAlert,
) -> Dict[str, Any]:
    """Run the news-impact agent. Returns a structured assessment dict
    with `material` (bool), `patch` (dict), `rationales` (dict),
    `delta_summary` (str).

    No LLM available → returns `{material: false}` deterministically:
    on the safe side, we don't push an unverified patch into a live memo.
    """
    if not settings.has_llm:
        return {"material": False, "patch": {}, "rationales": {}, "delta_summary": ""}

    payload = {
        "memo_summary": {
            "ticker": memo.ticker,
            "sector": memo.sector,
            "rating_label": memo.rating_label,
            "confidence_score": memo.confidence_score,
            "one_sentence_thesis": memo.one_sentence_thesis,
            "final_pm_view": memo.final_pm_view,
            "thesis_breakers": [r.title for r in memo.thesis_breakers][:3],
        },
        "alert": {
            "title": alert.title,
            "summary": alert.summary,
            "severity": alert.severity,
            "source": alert.source,
            "published_at": alert.published_at,
        },
    }
    prompt = _PROMPT + "\n\nContext:\n" + json.dumps(payload, default=str)[:3000]
    # Anthropic Haiku via the cross-family cheap route (locked in MASTER_PLAN).
    try:
        out = llm.chat_json(
            prompt, system="You are a careful equity-research news-impact analyst.",
            route="cheap", model=settings.anthropic_cheap_model,
        )
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("news_impact_agent LLM call failed for %s: %s", memo.ticker, exc)
        return {"material": False, "patch": {}, "rationales": {}, "delta_summary": ""}

    if not isinstance(out, dict):
        return {"material": False, "patch": {}, "rationales": {}, "delta_summary": ""}

    material = bool(out.get("material"))
    if not material:
        return {"material": False, "patch": {}, "rationales": {}, "delta_summary": ""}

    patch = _clamp_patch(memo, out.get("patch") or {})
    rationales = {k: str(v) for k, v in (out.get("rationales") or {}).items()
                  if k in patch and v}
    # Discipline: a field without a rationale falls out.
    patch = {k: v for k, v in patch.items() if k in rationales}

    return {
        "material": bool(patch),
        "patch": patch,
        "rationales": rationales,
        "delta_summary": str(out.get("delta_summary") or "")[:240],
    }


def apply_patch(
    memo: StockMemoOut, patch: Dict[str, Any],
) -> StockMemoOut:
    """Return a new memo with `patch` applied.

    Bull/bear case patches APPEND to the existing key_points; they don't
    replace the whole case. Same for key_risks. Other fields replace.
    """
    new = memo.model_copy(deep=True)
    for k, v in patch.items():
        if k == "bull_case" and isinstance(v, dict):
            existing_kp = list(new.bull_case.key_points or [])
            extra = [str(p) for p in (v.get("key_points") or []) if p]
            new.bull_case.key_points = existing_kp + extra
            if v.get("headline"):
                new.bull_case.headline = str(v["headline"])
        elif k == "bear_case" and isinstance(v, dict):
            existing_kp = list(new.bear_case.key_points or [])
            extra = [str(p) for p in (v.get("key_points") or []) if p]
            new.bear_case.key_points = existing_kp + extra
            if v.get("headline"):
                new.bear_case.headline = str(v["headline"])
        elif k == "key_risks" and isinstance(v, list):
            from ..schemas import RiskItem
            for item in v:
                if isinstance(item, dict) and item.get("title"):
                    try:
                        new.key_risks.append(RiskItem(**item))
                    except Exception:
                        continue
        else:
            try:
                setattr(new, k, v)
            except Exception:
                continue
    return new
