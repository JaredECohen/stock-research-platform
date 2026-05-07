"""Wave 10 — earnings quarter-over-quarter delta agent.

Compares the current-quarter `EarningsStructured` extraction (emitted
by the earnings agent) against the prior quarter's extraction for the
same ticker. Surfaces:

- guidance changes that *reversed* a prior quarter's call (a CEO
  defending margins last quarter and walking it back this quarter is
  one of the strongest signals a real PM watches),
- new tone classifications that diverge from prior speakers,
- segments newly defended / pressed,
- forward catalysts that were dropped from the calendar.

Returns an `AgentFinding` with `data["qoq_delta"] = {...}` carrying the
structured comparison so the UI can render a side-by-side timeline.

Defensive: returns None when prior data is missing — the memo path
just doesn't surface a QoQ tile in that case.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..config import settings
from ..database import SessionLocal
from ..models import EarningsTranscript, MemoSnapshot
from ..schemas import AgentFinding

log = logging.getLogger(__name__)


def _prior_structured_extraction(ticker: str, current_period: str) -> Optional[Dict[str, Any]]:
    """Pull the structured extraction from the prior quarter's memo.

    The earnings agent stores it under `data.structured` of the memo's
    `earnings_agent_view`. We walk back through MemoSnapshot rows
    until we find one whose earnings_agent_view.data.structured.period
    is *different* from `current_period` (so a re-patched memo for the
    same quarter doesn't fool us into thinking we have a delta).
    """
    if not ticker:
        return None
    try:
        with SessionLocal() as db:
            rows = db.execute(
                select(MemoSnapshot)
                .where(MemoSnapshot.ticker == ticker.upper())
                .order_by(MemoSnapshot.version.desc())
                .limit(20)
            ).scalars().all()
    except Exception:  # pragma: no cover
        return None
    for row in rows:
        memo = row.memo_json or {}
        if not isinstance(memo, dict):
            continue
        view = memo.get("earnings_agent_view") or {}
        data = (view.get("data") or {}) if isinstance(view, dict) else {}
        struct = data.get("structured") if isinstance(data, dict) else None
        if not isinstance(struct, dict):
            continue
        period = (struct.get("period") or "").strip()
        if period and period != current_period:
            return struct
    return None


def _deterministic_delta(
    current: Dict[str, Any], prior: Dict[str, Any],
) -> Dict[str, Any]:
    """Lightweight comparison for use without an LLM.

    Captures the most mechanical signals — tone shift, count changes
    in guidance / Q&A / catalysts. Real "what reversed?" reasoning
    needs the LLM path.
    """
    def _len(x: Any) -> int:
        return len(x) if isinstance(x, list) else 0

    cur_tone = (current.get("overall_tone") or "").strip()
    prior_tone = (prior.get("overall_tone") or "").strip()
    delta: Dict[str, Any] = {
        "current_period": current.get("period"),
        "prior_period": prior.get("period"),
        "tone_shift": (
            f"{prior_tone} → {cur_tone}"
            if cur_tone and prior_tone and cur_tone != prior_tone else None
        ),
        "guidance_changes_count": {
            "prior": _len(prior.get("guidance_changes")),
            "current": _len(current.get("guidance_changes")),
        },
        "qa_themes_count": {
            "prior": _len(prior.get("qa_themes")),
            "current": _len(current.get("qa_themes")),
        },
        "forward_catalysts_count": {
            "prior": _len(prior.get("forward_catalysts")),
            "current": _len(current.get("forward_catalysts")),
        },
        "most_defended_shift": (
            (prior.get("most_defended_segment") or {}).get("name") !=
            (current.get("most_defended_segment") or {}).get("name")
        ),
        "most_pressed_shift": (
            (prior.get("most_pressed_segment") or {}).get("name") !=
            (current.get("most_pressed_segment") or {}).get("name")
        ),
    }
    return delta


def _llm_delta(
    current: Dict[str, Any], prior: Dict[str, Any], ticker: str,
) -> Optional[Dict[str, Any]]:
    """Ask the LLM to spot the highest-signal differences."""
    if not getattr(settings, "openai_api_key", None):
        return None
    payload = {"current": current, "prior": prior, "ticker": ticker}
    from . import llm
    out = llm.chat_json(
        "Two quarters of structured earnings extractions for the same "
        "company. Identify what materially changed quarter-over-quarter. "
        "Especially flag: claims management *defended* last quarter and "
        "walked back this quarter (or vice versa); tone shifts in "
        "specific speakers; forward catalysts that were silently "
        "dropped; guidance metrics that reversed direction.\n\n"
        "Output JSON: {\n"
        "  reversals: [str, ...],   // 0-3 sharp items, each citing "
        "    the metric / segment / speaker\n"
        "  tone_shifts: [str, ...], // 0-3 items\n"
        "  silent_drops: [str, ...],// catalysts that vanished\n"
        "  net_signal: \"better|worse|mixed|no_change\",\n"
        "  one_line_takeaway: str\n}\n\n"
        + json.dumps(payload, default=str)[:14000],
        system=(
            "You are a senior analyst who has been covering this company "
            "for years. Be specific. No filler."
        ),
        route="cheap",
    )
    return out if isinstance(out, dict) else None


def run_earnings_qoq_delta(
    ticker: str, current_structured: Optional[Dict[str, Any]],
) -> Optional[AgentFinding]:
    """Build a QoQ delta finding. Returns None when prior data is
    unavailable so the memo path can skip the tile."""
    if not isinstance(current_structured, dict):
        return None
    period = (current_structured.get("period") or "").strip()
    prior = _prior_structured_extraction(ticker, period)
    if prior is None:
        return None
    deterministic = _deterministic_delta(current_structured, prior)
    llm_out = _llm_delta(current_structured, prior, ticker) or {}
    reversals = [str(r) for r in (llm_out.get("reversals") or []) if str(r).strip()]
    tone_shifts = [str(r) for r in (llm_out.get("tone_shifts") or []) if str(r).strip()]
    silent_drops = [str(r) for r in (llm_out.get("silent_drops") or []) if str(r).strip()]
    net_signal = (llm_out.get("net_signal") or "no_change").strip()
    takeaway = str(llm_out.get("one_line_takeaway") or "").strip()
    summary_parts: List[str] = []
    if takeaway:
        summary_parts.append(takeaway)
    if deterministic.get("tone_shift"):
        summary_parts.append(f"Overall tone {deterministic['tone_shift']}.")
    if reversals:
        summary_parts.append(f"{len(reversals)} reversal(s) flagged.")
    summary = " ".join(summary_parts) or (
        f"QoQ delta computed vs {prior.get('period', 'prior quarter')}; "
        f"no major reversals detected."
    )
    key_points: List[str] = []
    if reversals:
        key_points.extend(f"Reversed: {r}" for r in reversals[:3])
    if tone_shifts:
        key_points.extend(f"Tone: {r}" for r in tone_shifts[:3])
    if silent_drops:
        key_points.extend(f"Dropped catalyst: {r}" for r in silent_drops[:2])
    return AgentFinding(
        agent="Earnings QoQ Delta",
        headline=(
            f"QoQ delta vs {prior.get('period', '—')}: {net_signal}"
            if net_signal != "no_change" else
            f"QoQ delta vs {prior.get('period', '—')}: no material change"
        ),
        summary=summary,
        key_points=key_points or ["No material differences."],
        confidence=0.7 if llm_out else 0.5,
        sources=[
            f"transcript:{prior.get('period', '')}",
            f"transcript:{current_structured.get('period', '')}",
        ],
        data={
            "qoq_delta": {
                **deterministic,
                "reversals": reversals,
                "tone_shifts": tone_shifts,
                "silent_drops": silent_drops,
                "net_signal": net_signal,
            },
        },
    )
