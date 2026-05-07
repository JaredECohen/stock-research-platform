"""Wave 10 — influence-feedback loop.

For each specialist, compute its *reliability score* — how aligned its
historical influence has been with memo outcomes. A specialist that
consistently pulled bullish on memos that turned out wrong, or pulled
bearish on memos that turned out right, has a low reliability score.

This is the closing-the-loop signal the founder asked for: postmortems
score memos right/wrong, `agent_influence` records who pulled which
direction on each memo, and this service joins them so the PM can
literally read "your technical analyst has been pulling the wrong
way 65% of the time over the last 30 memos — discount it."

Reliability formula:
- For each (memo, specialist) pair where the postmortem verdict is
  `right` or `wrong`, compute alignment:
    - influence > 0 (bullish pull) AND verdict right AND rating bullish
      → +1 (specialist pulled the right way)
    - influence > 0 AND verdict wrong AND rating bullish
      → -1 (specialist pulled into a wrong call)
    - influence < 0 AND verdict right AND rating bearish → +1
    - influence < 0 AND verdict wrong AND rating bearish → -1
    - else → 0 (specialist abstained, or ambiguous)
- Reliability = mean alignment across the last N postmortems.
- Range -1.0 (always pulling wrong) → +1.0 (always pulling right).

Used by `pm_context.build_pm_context` to surface a "specialist
reliability" block in the PM's prompt context. Read-only against
`memo_postmortems` + `memo_snapshots`; never writes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..database import SessionLocal
from ..models import MemoPostmortem, MemoSnapshot

log = logging.getLogger(__name__)


_BULL_RATINGS = {"Very Bullish", "Bullish"}
_BEAR_RATINGS = {"Bearish", "Very Bearish"}


def _alignment(influence: float, rating: str, verdict: str) -> int:
    """+1 if pull aligned with outcome, -1 if pulled wrong way, 0
    when ambiguous (Neutral rating or zero influence)."""
    if abs(influence) < 0.05 or rating not in (_BULL_RATINGS | _BEAR_RATINGS):
        return 0
    if verdict not in {"right", "wrong"}:
        return 0
    bullish_rating = rating in _BULL_RATINGS
    bullish_pull = influence > 0
    pulled_with_rating = (bullish_rating == bullish_pull)
    if verdict == "right":
        return +1 if pulled_with_rating else -1
    return -1 if pulled_with_rating else +1  # verdict == "wrong"


def specialist_reliability(*, lookback: int = 30) -> Dict[str, Any]:
    """Compute per-specialist reliability over the last `lookback`
    postmortemmed memos. Returns:
        {
          n: int,
          per_agent: { agent: { reliability: -1..1, n_evaluated: int } },
        }
    """
    rows: List[Dict[str, Any]] = []
    with SessionLocal() as db:
        pm_rows = db.execute(
            select(MemoPostmortem)
            .where(MemoPostmortem.verdict.in_(["right", "wrong"]))
            .order_by(MemoPostmortem.created_at.desc())
            .limit(lookback)
        ).scalars().all()
        for pm in pm_rows:
            snap = db.execute(
                select(MemoSnapshot)
                .where(MemoSnapshot.id == pm.memo_snapshot_id)
                .limit(1)
            ).scalars().first()
            if snap is None:
                continue
            memo = snap.memo_json or {}
            if not isinstance(memo, dict):
                continue
            rating = str(memo.get("rating_label") or "")
            influence = memo.get("agent_influence") or {}
            if not isinstance(influence, dict):
                continue
            rows.append({
                "rating": rating,
                "verdict": pm.verdict,
                "influence": influence,
            })

    aggregate: Dict[str, List[int]] = {}
    for row in rows:
        for agent, pull in (row["influence"] or {}).items():
            if not isinstance(pull, (int, float)):
                continue
            score = _alignment(float(pull), row["rating"], row["verdict"])
            aggregate.setdefault(agent, []).append(score)

    per_agent: Dict[str, Dict[str, Any]] = {}
    for agent, scores in aggregate.items():
        if not scores:
            continue
        per_agent[agent] = {
            "reliability": sum(scores) / len(scores),
            "n_evaluated": len(scores),
        }
    return {"n": len(rows), "per_agent": per_agent}


def reliability_prompt_block(*, lookback: int = 30, threshold: float = -0.2) -> str:
    """Render a markdown block for `pm_context` flagging specialists
    with reliability below `threshold` (default: -0.2 means a
    specialist whose pulls have been wrong more often than right).

    Returns empty string when no specialists trip the threshold or
    when the lookback window has too few postmortems to be meaningful
    (n < 5 by default).
    """
    stats = specialist_reliability(lookback=lookback)
    if stats.get("n", 0) < 5:
        return ""
    flagged: List[Dict[str, Any]] = []
    for agent, payload in stats.get("per_agent", {}).items():
        if payload.get("reliability", 1.0) <= threshold and payload.get("n_evaluated", 0) >= 3:
            flagged.append({"agent": agent, **payload})
    if not flagged:
        return ""
    flagged.sort(key=lambda r: r["reliability"])
    lines = [
        f"- `{r['agent']}` pulled the **wrong** way "
        f"{int((1 - (r['reliability'] + 1) / 2) * 100)}% of the time "
        f"({r['n_evaluated']} of last {stats['n']} postmortemmed memos)."
        for r in flagged[:5]
    ]
    return (
        "## Specialist reliability (last "
        f"{stats['n']} postmortemmed memos)\n\n"
        + "\n".join(lines)
        + "\n\n_Discount these specialists' contributions when their "
        "pull dominates the rating; demand stronger evidence._"
    )
