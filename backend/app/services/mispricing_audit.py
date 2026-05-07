"""Wave 10 — mispricing-thesis quality audit.

Periodic LLM-judged review of `MispricingThesis` fields across the
memo corpus. Surfaces the kinds of failure modes the design review
warned about:

- Thesis that's just a metric recap, not a real claim.
- Empty `gap` field (no real differentiated view).
- Falsifiers that aren't actually disconfirmable.
- "Consensus view" that the user can't tell from "our view".

Output goes nowhere by default — it's a *signal*, written to
`memo_postmortems`-style records under the new `mispricing_audit`
table or returned by the audit endpoint for ad-hoc review. Feeds
PM prompt iteration and (eventually) postmortem attribution.

This service is read-only against memo data; the only writes are to
`memo_postmortems.lesson` (we attach quality notes there) when the
caller passes `attach_to_postmortems=True`.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..config import settings
from ..database import SessionLocal
from ..models import MemoSnapshot

log = logging.getLogger(__name__)


_AUDIT_PROMPT = """You are auditing the quality of a portfolio
manager's *mispricing thesis* across multiple stock memos. The PM
fills four fields per memo: consensus_view, our_view, gap,
falsifiers (list).

Score each memo on three dimensions, 1-5:

- specificity: 1 = vague metric recap, 5 = cites a precise number /
  segment / catalyst that the market is missing.
- differentiation: 1 = our_view is identical to consensus_view in
  substance, 5 = a clearly distinct claim the market hasn't priced.
- falsifiability: 1 = falsifiers are tautologies, 5 = each falsifier
  is a concrete observation that would unambiguously prove the
  thesis wrong.

Then write a one-line `improvement` field per memo — what would lift
this thesis to a 5/5/5? — and a corpus-wide `pattern_observation`
string at the end naming the most common failure mode you saw.

Return JSON:
{
  per_memo: [
    { ticker, version, specificity: int, differentiation: int,
      falsifiability: int, improvement: str }
  ],
  pattern_observation: str
}
"""


def _gather_memos(*, limit: int = 25) -> List[Dict[str, Any]]:
    """Pull the most recent memos that have a non-empty mispricing
    thesis to audit. Skips memos that pre-date the field."""
    out: List[Dict[str, Any]] = []
    with SessionLocal() as db:
        rows = db.execute(
            select(MemoSnapshot)
            .order_by(MemoSnapshot.generated_at.desc())
            .limit(limit * 4)  # fetch wider; filter for present mispricing
        ).scalars().all()
    for row in rows:
        memo = row.memo_json or {}
        if not isinstance(memo, dict):
            continue
        mp = memo.get("mispricing_thesis") or {}
        if not isinstance(mp, dict):
            continue
        if not (mp.get("our_view") or mp.get("gap")):
            continue
        out.append({
            "ticker": row.ticker,
            "version": row.version,
            "rating": memo.get("rating_label"),
            "thesis": memo.get("one_sentence_thesis", "")[:200],
            "mispricing_thesis": mp,
        })
        if len(out) >= limit:
            break
    return out


def run_audit(*, limit: int = 20) -> Dict[str, Any]:
    """Audit up to `limit` recent memos. Returns the LLM scoring +
    pattern observation. Defensive: empty audit when no memos qualify
    or the LLM is unavailable.
    """
    memos = _gather_memos(limit=limit)
    if not memos:
        return {"audited": 0, "per_memo": [], "pattern_observation": ""}
    if not getattr(settings, "openai_api_key", None):
        return {
            "audited": len(memos),
            "per_memo": [],
            "pattern_observation": "LLM audit skipped — no API key configured.",
        }
    from ..agents import llm
    out = llm.chat_json(
        _AUDIT_PROMPT
        + "\n\nMemos to audit:\n"
        + json.dumps(memos, default=str)[:24000],
        system="You are a senior buy-side editor. Be honest and specific.",
        route="cheap",
        max_tokens=2000,
    )
    if not isinstance(out, dict):
        return {
            "audited": len(memos),
            "per_memo": [],
            "pattern_observation": "Audit failed — LLM returned non-dict.",
        }
    per_memo = []
    for item in (out.get("per_memo") or []):
        if not isinstance(item, dict):
            continue
        per_memo.append({
            "ticker": str(item.get("ticker") or "")[:16],
            "version": int(item.get("version") or 0),
            "specificity": _clamp_int(item.get("specificity"), 1, 5),
            "differentiation": _clamp_int(item.get("differentiation"), 1, 5),
            "falsifiability": _clamp_int(item.get("falsifiability"), 1, 5),
            "improvement": str(item.get("improvement") or "")[:300],
        })
    return {
        "audited": len(memos),
        "per_memo": per_memo,
        "pattern_observation": str(out.get("pattern_observation") or "")[:600],
    }


def _clamp_int(v: Any, lo: int, hi: int) -> Optional[int]:
    try:
        n = int(v)
        return max(lo, min(hi, n))
    except (TypeError, ValueError):
        return None


def persist_audit(audit: Dict[str, Any], aggregate: Dict[str, Any]) -> Optional[int]:
    """Wave 10 — write an audit run to `mispricing_audits` so the PM can
    later read the most-recent `pattern_observation` from its prompt
    context. Returns the new row id, or None on failure (the audit is
    still useful for the API response in that case)."""
    try:
        from datetime import datetime
        from ..database import SessionLocal
        from ..models import MispricingAudit
        with SessionLocal() as db:
            row = MispricingAudit(
                audited_at=datetime.utcnow(),
                n_memos=int(audit.get("audited") or 0),
                pattern_observation=str(audit.get("pattern_observation") or "")[:2000],
                per_memo_scores=audit.get("per_memo") or [],
                aggregate_means=aggregate or {},
                weak_memo_count=int((aggregate or {}).get("weak_memo_count") or 0),
            )
            db.add(row)
            db.commit()
            return row.id
    except Exception as exc:  # pragma: no cover — never block the audit response
        log.warning("mispricing audit persist failed: %s", exc)
        return None


def latest_aggregate(*, max_age_days: int = 14) -> Optional[Dict[str, Any]]:
    """Fetch the most-recent persisted audit's aggregate stats.

    Returns the dict written via `persist_audit` (mean specificity /
    differentiation / falsifiability + weak_memo_count + n) or None
    when no recent audit exists.
    """
    try:
        from datetime import datetime, timedelta
        from sqlalchemy import select
        from ..database import SessionLocal
        from ..models import MispricingAudit
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        with SessionLocal() as db:
            row = db.execute(
                select(MispricingAudit)
                .where(MispricingAudit.audited_at >= cutoff)
                .order_by(MispricingAudit.audited_at.desc())
                .limit(1)
            ).scalars().first()
            if row is None:
                return None
            return row.aggregate_means or {}
    except Exception:  # pragma: no cover
        return None


def prompt_fragment(*, max_age_days: int = 14) -> str:
    """Render a *targeted* PM prompt fragment based on the most-recent
    audit's WEAKEST dimension. Sharper than just dumping the
    pattern_observation — the PM gets concrete guidance on what to
    fix on the current memo.

    Returns empty string when no recent audit exists or all
    dimensions score adequately.
    """
    agg = latest_aggregate(max_age_days=max_age_days)
    if not agg:
        return ""
    candidates = [
        ("specificity", agg.get("mean_specificity"),
         "Be SPECIFIC. Cite a precise number, segment, or catalyst the "
         "market is missing. Generic metric recaps will not pass."),
        ("differentiation", agg.get("mean_differentiation"),
         "Make `our_view` materially DIFFERENT from `consensus_view`. "
         "If you can't, write 'fairly priced on our work' rather than "
         "padding with a near-duplicate of consensus."),
        ("falsifiability", agg.get("mean_falsifiability"),
         "Falsifiers must be CONCRETE observations (metric thresholds, "
         "guidance changes, regulatory rulings). Tautologies do not "
         "count."),
    ]
    # Pick the lowest-scoring dimension where data exists.
    weak = [
        (label, score, guidance) for label, score, guidance in candidates
        if isinstance(score, (int, float)) and score < 4.0
    ]
    if not weak:
        return ""
    weak.sort(key=lambda x: x[1])
    label, score, guidance = weak[0]
    return (
        f"## Sharpen this memo's mispricing thesis "
        f"(audit: weakest dimension = `{label}`, mean {score:.1f}/5)\n\n"
        f"{guidance}"
    )


def latest_pattern_observation(*, max_age_days: int = 14) -> str:
    """Read the most-recent `pattern_observation` for PM self-improvement
    context. Returns empty string when no recent audit exists or the
    observation is empty.
    """
    try:
        from datetime import datetime, timedelta
        from sqlalchemy import select
        from ..database import SessionLocal
        from ..models import MispricingAudit
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        with SessionLocal() as db:
            row = db.execute(
                select(MispricingAudit)
                .where(MispricingAudit.audited_at >= cutoff)
                .where(MispricingAudit.pattern_observation != "")
                .order_by(MispricingAudit.audited_at.desc())
                .limit(1)
            ).scalars().first()
            return (row.pattern_observation or "") if row else ""
    except Exception:  # pragma: no cover
        return ""


def aggregate_scores(audit: Dict[str, Any]) -> Dict[str, Any]:
    """Summary stats across an audit run — mean specificity /
    differentiation / falsifiability, count of "weak" memos
    (any score <=2)."""
    rows = audit.get("per_memo") or []
    if not rows:
        return {
            "mean_specificity": None,
            "mean_differentiation": None,
            "mean_falsifiability": None,
            "weak_memo_count": 0,
            "n": 0,
        }
    n = len(rows)
    means = {}
    for k in ("specificity", "differentiation", "falsifiability"):
        vals = [r[k] for r in rows if isinstance(r.get(k), int)]
        means[f"mean_{k}"] = (sum(vals) / len(vals)) if vals else None
    weak = sum(
        1 for r in rows
        if min(
            r.get("specificity") or 5,
            r.get("differentiation") or 5,
            r.get("falsifiability") or 5,
        ) <= 2
    )
    return {**means, "weak_memo_count": weak, "n": n}
