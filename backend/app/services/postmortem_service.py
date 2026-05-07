"""Wave 10 — postmortem service.

For every memo, two cadences fire:

- **30-day early read.** Drift signal — "the call is going against us;
  here's what's already changed." Light: takes the realized return,
  recent news, and asks the LLM to flag whether the thesis is at risk.
- **90-day full postmortem.** Calibration lesson — "we said X, the
  market did Y, here's why we got it right or wrong, and here's what
  to watch differently next time." Heavy: full memo + outcome + recent
  news, agent-by-agent attribution, lesson written back to the
  company / sector / PM memory files.

Reads `memo_outcomes` (already populated by the nightly outcome
evaluator) and writes `memo_postmortems`. Idempotent on
`(memo_snapshot_id, horizon_days)`.

This service powers the *learning loop* the founder asked for: every
memo eventually feeds back into agent memory, so the next memo is
written with knowledge of what worked and what didn't.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..config import settings
from ..database import SessionLocal
from ..models import (
    MemoOutcome,
    MemoPostmortem,
    MemoSnapshot,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Picking memos to postmortem
# ---------------------------------------------------------------------------

_DEDUPE_WINDOW_DAYS = 14  # rate-limit: 1 postmortem per (ticker, horizon) per window


def _rating_of(snap: MemoSnapshot) -> str:
    """Pull the rating label out of a snapshot's memo_json defensively."""
    memo = snap.memo_json or {}
    if not isinstance(memo, dict):
        return ""
    return str(memo.get("rating_label") or "").strip()


def _prior_snapshot(db, ticker: str, version: int) -> Optional[MemoSnapshot]:
    """The most recent prior version for this ticker (version < current)."""
    return db.execute(
        select(MemoSnapshot)
        .where(MemoSnapshot.ticker == ticker, MemoSnapshot.version < version)
        .order_by(MemoSnapshot.version.desc())
        .limit(1)
    ).scalars().first()


def _recent_postmortem_within(
    db, ticker: str, horizon_days: int, window_days: int,
) -> Optional[MemoPostmortem]:
    """Most recent postmortem for (ticker, horizon) within `window_days`."""
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    return db.execute(
        select(MemoPostmortem)
        .where(
            MemoPostmortem.ticker == ticker,
            MemoPostmortem.horizon_days == horizon_days,
            MemoPostmortem.created_at >= cutoff,
        )
        .order_by(MemoPostmortem.created_at.desc())
        .limit(1)
    ).scalars().first()


def _should_postmortem(
    db, snap: MemoSnapshot, horizon_days: int,
) -> tuple[bool, str]:
    """Dedupe + rate-limit guard.

    Returns (proceed, reason). Skip when:
    1. The prior snapshot has the same rating_label (a memo refresh
       that didn't change the call isn't a new thesis to postmortem).
    2. A postmortem for (ticker, horizon) was written in the last
       `_DEDUPE_WINDOW_DAYS` days (noise cap on high-throughput names).
    """
    # Rating-change skip — only when there IS a prior snapshot to
    # compare against (first memo for the ticker always proceeds).
    prior = _prior_snapshot(db, snap.ticker, snap.version)
    if prior is not None:
        prior_rating = _rating_of(prior)
        new_rating = _rating_of(snap)
        if prior_rating and new_rating and prior_rating == new_rating:
            return False, f"rating unchanged ({new_rating}) vs v{prior.version}"
    # Rate-limit per (ticker, horizon).
    recent = _recent_postmortem_within(
        db, snap.ticker, horizon_days, _DEDUPE_WINDOW_DAYS,
    )
    if recent is not None:
        return False, (
            f"recent postmortem exists "
            f"({recent.created_at.date().isoformat()}, within {_DEDUPE_WINDOW_DAYS}d)"
        )
    return True, "ok"


def _due_memos(horizon_days: int, *, limit: int = 50) -> List[Dict[str, Any]]:
    """Memos with an outcome at this horizon and no postmortem yet, after
    dedupe (rating-change + 14d rate-limit) is applied."""
    out: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    with SessionLocal() as db:
        stmt = (
            select(MemoOutcome, MemoSnapshot)
            .join(MemoSnapshot, MemoOutcome.memo_snapshot_id == MemoSnapshot.id)
            .where(MemoOutcome.horizon_days == horizon_days)
        )
        rows = db.execute(stmt).all()
        for outcome, snap in rows:
            existing = db.execute(
                select(MemoPostmortem).where(
                    MemoPostmortem.memo_snapshot_id == outcome.memo_snapshot_id,
                    MemoPostmortem.horizon_days == horizon_days,
                )
            ).scalars().first()
            if existing is not None:
                continue
            proceed, reason = _should_postmortem(db, snap, horizon_days)
            if not proceed:
                skipped.append({"ticker": snap.ticker, "reason": reason})
                continue
            out.append({"outcome": outcome, "snapshot": snap})
            if len(out) >= limit:
                break
    if skipped:
        log.debug("postmortem dedupe skipped %d memos: %s", len(skipped), skipped[:5])
    return out


# ---------------------------------------------------------------------------
# Verdict + attribution
# ---------------------------------------------------------------------------

_BULL_RATINGS = {"Very Bullish", "Bullish"}
_BEAR_RATINGS = {"Bearish", "Very Bearish"}


def _classify_verdict(rating: str, alpha: Optional[float]) -> str:
    """Right / wrong / mixed / pending. Mirrors `_thesis_held` from
    outcome_service but exposes a richer set of buckets."""
    if alpha is None:
        return "pending"
    if rating in _BULL_RATINGS:
        return "right" if alpha > 0.02 else ("wrong" if alpha < -0.05 else "mixed")
    if rating in _BEAR_RATINGS:
        return "right" if alpha < -0.02 else ("wrong" if alpha > 0.05 else "mixed")
    # Neutral — small alpha is "right" (we said no edge)
    return "right" if abs(alpha) < 0.05 else "mixed"


def _llm_postmortem(
    memo: Dict[str, Any], outcome: MemoOutcome, horizon_days: int,
) -> Optional[Dict[str, Any]]:
    """Ask the LLM for a structured retrospective.

    Returns dict with: `lesson` (markdown body), `agent_attribution`
    (per-specialist credit/blame dict), and `regime_at_memo`.
    """
    if not getattr(settings, "openai_api_key", None):
        return None
    payload = {
        "memo": {
            "ticker": memo.get("ticker"),
            "rating": memo.get("rating_label"),
            "confidence": memo.get("confidence_score"),
            "thesis": memo.get("one_sentence_thesis"),
            "mispricing_thesis": memo.get("mispricing_thesis"),
            "key_risks": memo.get("key_risks"),
            "thesis_breakers": memo.get("thesis_breakers"),
            "specialist_views": {
                k: (memo.get(k) or {}).get("summary", "")
                for k in (
                    "sector_agent_view", "earnings_agent_view",
                    "filing_agent_view", "valuation_agent_view",
                    "comps_agent_view", "macro_sensitivity",
                )
            },
        },
        "outcome": {
            "horizon_days": horizon_days,
            "realized_return": outcome.forward_return,
            "benchmark_return": outcome.benchmark_return,
            "alpha": outcome.alpha,
            "thesis_held": outcome.thesis_held,
        },
    }
    from ..agents import llm
    out = llm.chat_json(
        f"Write a {horizon_days}-day postmortem for this memo. The "
        "user is the PM. Be candid — credit specialists who got it "
        "right, blame specialists who got it wrong. Output JSON:\n\n"
        "{ \"lesson\": \"<3-6 sentence markdown — what we said, what "
        "happened, why, what to remember next time>\",\n"
        "  \"agent_attribution\": { \"sector\": -1..1, \"earnings\": "
        "-1..1, \"filing\": -1..1, \"valuation\": -1..1, \"comps\": "
        "-1..1, \"macro\": -1..1, \"risk\": -1..1 },\n"
        "  \"regime_at_memo\": \"<short tag if knowable, else empty>\","
        "  \"sector_lesson\": \"<one sentence the sector analyst should "
        "internalize, or empty if not generalizable>\" }\n\n"
        "Memo + outcome:\n" + json.dumps(payload, default=str)[:24000],
        system=(
            "You are a senior PM running a postmortem. Be specific, "
            "honest, and concise. No hedging."
        ),
        route="strong",
    )
    if not isinstance(out, dict):
        return None
    return out


def _deterministic_lesson(
    memo: Dict[str, Any], outcome: MemoOutcome, verdict: str, horizon_days: int,
) -> str:
    rating = memo.get("rating_label", "")
    alpha = outcome.alpha if outcome.alpha is not None else 0.0
    return (
        f"{horizon_days}d postmortem ({verdict}). Memo rated {rating}; "
        f"alpha {alpha*100:.1f}% vs benchmark over the window. "
        f"Realized return {outcome.forward_return*100:.1f}%, "
        f"benchmark {outcome.benchmark_return*100:.1f}%."
    )


# ---------------------------------------------------------------------------
# Memory writers
# ---------------------------------------------------------------------------

def _write_lesson_to_memory(
    ticker: str, sector: Optional[str], lesson: str, sector_lesson: str,
) -> None:
    if lesson.strip():
        try:
            from ..memory import CompanyMemory, MemoryEntry
            cm = CompanyMemory.for_ticker(ticker)
            cm.append_entry(MemoryEntry(
                date=date.today().isoformat(),
                trigger="postmortem",
                body=lesson,
            ))
            cm.save()
        except Exception as exc:  # pragma: no cover
            log.warning("postmortem→company memory failed for %s: %s", ticker, exc)
    if sector and sector_lesson.strip():
        try:
            from ..memory import CrossCompanyPattern, SectorMemory
            sm = SectorMemory.for_sector(sector)
            sm.add_pattern(CrossCompanyPattern(
                date=date.today().isoformat(),
                source_company=ticker,
                applies_to=[],
                lesson=sector_lesson.strip(),
            ))
            sm.save()
        except Exception as exc:  # pragma: no cover
            log.warning("postmortem→sector memory failed: %s", exc)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_postmortems(*, horizon_days: int = 90, limit: int = 25) -> Dict[str, Any]:
    """Process up to `limit` memos due for a postmortem at this horizon.

    Returns a small report dict for cron logs / observability.
    """
    due = _due_memos(horizon_days, limit=limit)
    written = 0
    skipped = 0
    for item in due:
        outcome: MemoOutcome = item["outcome"]
        snap: MemoSnapshot = item["snapshot"]
        try:
            memo = snap.memo or {}
            if not isinstance(memo, dict):
                memo = json.loads(memo)
        except Exception:
            skipped += 1
            continue
        verdict = _classify_verdict(memo.get("rating_label", ""), outcome.alpha)
        llm_out = _llm_postmortem(memo, outcome, horizon_days)
        lesson = (llm_out or {}).get("lesson") or _deterministic_lesson(
            memo, outcome, verdict, horizon_days,
        )
        attribution = (llm_out or {}).get("agent_attribution") or {}
        sector_lesson = (llm_out or {}).get("sector_lesson") or ""
        # Wave 10 — prefer the authoritative `macro_regime_at_memo`
        # captured on the memo at creation; fall back to LLM guess.
        regime = (
            str(memo.get("macro_regime_at_memo") or "").strip()
            or (llm_out or {}).get("regime_at_memo") or ""
        )
        try:
            with SessionLocal() as db:
                pm_row = MemoPostmortem(
                    memo_snapshot_id=outcome.memo_snapshot_id,
                    ticker=outcome.ticker,
                    horizon_days=horizon_days,
                    verdict=verdict,
                    lesson=lesson,
                    agent_attribution=attribution if isinstance(attribution, dict) else {},
                    realized_return=outcome.forward_return,
                    benchmark_return=outcome.benchmark_return,
                    regime_at_memo=regime[:32] if regime else None,
                    written_to_memory=False,
                    created_at=datetime.utcnow(),
                )
                db.add(pm_row)
                db.commit()
        except Exception as exc:  # pragma: no cover
            log.warning("postmortem persist failed for memo %s: %s", outcome.memo_snapshot_id, exc)
            skipped += 1
            continue
        # Write the lesson back into memory only on the 90d cadence —
        # the 30d "early read" stays in the DB but doesn't pollute the
        # narrative memory yet.
        if horizon_days >= 90:
            _write_lesson_to_memory(
                outcome.ticker, memo.get("sector"), lesson, sector_lesson,
            )
            try:
                with SessionLocal() as db:
                    row = db.execute(
                        select(MemoPostmortem).where(
                            MemoPostmortem.memo_snapshot_id == outcome.memo_snapshot_id,
                            MemoPostmortem.horizon_days == horizon_days,
                        )
                    ).scalars().first()
                    if row is not None:
                        row.written_to_memory = True
                        db.commit()
            except Exception as exc:  # pragma: no cover
                log.debug("postmortem written_to_memory flag failed: %s", exc)
        written += 1
    return {
        "horizon_days": horizon_days,
        "due": len(due),
        "written": written,
        "skipped": skipped,
    }
