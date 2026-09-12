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
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

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


def _prior_snapshot(db, ticker: str, version: int) -> MemoSnapshot | None:
    """The most recent prior version for this ticker (version < current)."""
    return db.execute(
        select(MemoSnapshot)
        .where(MemoSnapshot.ticker == ticker, MemoSnapshot.version < version)
        .order_by(MemoSnapshot.version.desc())
        .limit(1)
    ).scalars().first()


def _recent_postmortem_within(
    db, ticker: str, horizon_days: int, window_days: int,
) -> MemoPostmortem | None:
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


def _postmortem_exists(db, memo_snapshot_id: int, horizon_days: int) -> bool:
    """Is there already a postmortem for exactly what the constraint keys on?

    One function, used by the due query's anti-join, by the pre-flight check
    and by the write's error classification, so those three can never drift
    apart on which columns they mean.
    """
    return db.execute(
        select(MemoPostmortem.id).where(
            MemoPostmortem.memo_snapshot_id == memo_snapshot_id,
            MemoPostmortem.horizon_days == horizon_days,
        ).limit(1)
    ).first() is not None


def _due_memos(horizon_days: int, *, limit: int = 50) -> list[dict[str, Any]]:
    """The due list alone — see `_scan_due` for what it means."""
    return _scan_due(horizon_days, limit=limit)[0]


def _scan_due(
    horizon_days: int, *, limit: int = 50,
) -> tuple[list[dict[str, Any]], int]:
    """`(due, deduped)` — memos with an outcome at this horizon and no
    postmortem yet, plus how many the policy dedupe held back.

    The count is returned rather than stashed on the module, because the
    loops run in `marketmosaic-worker` while cron-health is served by the
    web service and module-level state does not cross that boundary.

    "Due" means: an outcome exists at this horizon, no postmortem exists for
    this `(snapshot, horizon)`, and the policy dedupe (rating-change skip +
    14-day per-ticker rate limit) lets it through.

    Two properties this function is responsible for, both of them learned
    from `postmortem_loop` reporting `success=False` every night for work
    that was already finished:

    * **The exclusion is the constraint.** `memo_postmortems` is unique on
      `(memo_snapshot_id, horizon_days)`; the exclusion is a `NOT EXISTS`
      on those two columns and nothing else, pushed into the same statement
      rather than run per row afterwards.
    * **The list is keyed like the constraint.** It used to be keyed by
      *outcome row*: one entry per `memo_outcomes` row that survived the
      checks. Those are different keys, and the moment they diverge — a
      duplicated outcome row, anything that puts the same snapshot in the
      list twice — the pass attempts the identical insert twice, the second
      one is rejected by the database, and the rejection is counted as a
      failed postmortem. Deduplicating on the constraint's own key makes
      that impossible to express.

    Ordered by snapshot id so that a `limit`-capped pass takes a defined
    slice of the backlog rather than whatever the engine hands back, and
    the next pass continues from a predictable place.
    """
    out: list[dict[str, Any]] = []
    deduped: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int]] = set()
    with SessionLocal() as db:
        stmt = (
            select(MemoOutcome, MemoSnapshot)
            .join(MemoSnapshot, MemoOutcome.memo_snapshot_id == MemoSnapshot.id)
            .where(MemoOutcome.horizon_days == horizon_days)
            .where(~select(MemoPostmortem.id).where(
                MemoPostmortem.memo_snapshot_id == MemoOutcome.memo_snapshot_id,
                MemoPostmortem.horizon_days == horizon_days,
            ).exists())
            .order_by(MemoOutcome.memo_snapshot_id)
        )
        for outcome, snap in db.execute(stmt).all():
            key = (outcome.memo_snapshot_id, horizon_days)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            proceed, reason = _should_postmortem(db, snap, horizon_days)
            if not proceed:
                deduped.append({"ticker": snap.ticker, "reason": reason})
                continue
            out.append({"outcome": outcome, "snapshot": snap})
            if len(out) >= limit:
                break
    if deduped:
        log.debug(
            "postmortem dedupe skipped %d memos: %s", len(deduped), deduped[:5],
        )
    return out, len(deduped)


# ---------------------------------------------------------------------------
# Verdict + attribution
# ---------------------------------------------------------------------------

_BULL_RATINGS = {"Very Bullish", "Bullish"}
_BEAR_RATINGS = {"Bearish", "Very Bearish"}


def _classify_verdict(rating: str, alpha: float | None) -> str:
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
    memo: dict[str, Any], outcome: MemoOutcome, horizon_days: int,
) -> dict[str, Any] | None:
    """Ask the LLM for a structured retrospective.

    Returns dict with: `lesson` (markdown body), `agent_attribution`
    (per-specialist credit/blame dict), and `regime_at_memo`.
    """
    # Gate on any configured LLM, not OpenAI specifically: `llm.chat_json`
    # picks the active provider itself, and the old OpenAI-only check made
    # every postmortem on an Anthropic-only deployment fall back to the
    # deterministic lesson without ever trying.
    if not settings.has_llm:
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
    memo: dict[str, Any], outcome: MemoOutcome, verdict: str, horizon_days: int,
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
    ticker: str, sector: str | None, lesson: str, sector_lesson: str,
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

def _persist_postmortem(row: MemoPostmortem) -> str:
    """Write one postmortem. Returns "written", "already_done" or "failed".

    An `IntegrityError` on `uq_memo_postmortem_snapshot_horizon` means the
    row this pass was about to write is already there. That is not a
    failure and it is not work — it is the answer "already done", and it
    has to be reported as such. Counting it as a skipped postmortem is what
    made `postmortem_loop` report `success=False` every night for a
    backlog that was fine: 23 of 25 memos a night, each one logged as
    "postmortem persist failed", none of them actually a problem.

    The re-read is what distinguishes the two. An IntegrityError from
    anything else — a foreign key, a NOT NULL — leaves no row behind, and
    that genuinely is a failure.
    """
    try:
        with SessionLocal() as db:
            db.add(row)
            db.commit()
        return "written"
    except IntegrityError:
        try:
            with SessionLocal() as db:
                if _postmortem_exists(db, row.memo_snapshot_id, row.horizon_days):
                    log.info(
                        "postmortem for memo %s at %sd already existed",
                        row.memo_snapshot_id, row.horizon_days,
                    )
                    return "already_done"
        except Exception as exc:  # pragma: no cover — diagnostic only
            log.warning("postmortem existence re-check failed: %s", exc)
        log.warning(
            "postmortem persist rejected for memo %s at %sd",
            row.memo_snapshot_id, row.horizon_days,
        )
        return "failed"
    except Exception as exc:  # pragma: no cover — defensive
        log.warning(
            "postmortem persist failed for memo %s: %s", row.memo_snapshot_id, exc,
        )
        return "failed"


def run_postmortems(*, horizon_days: int = 90, limit: int = 25) -> dict[str, Any]:
    """Process up to `limit` memos due for a postmortem at this horizon.

    Returns a report that distinguishes the four states a due memo can end
    in, because collapsing them is how a healthy backlog came to be
    reported as a nightly failure:

      `written`       a postmortem was created.
      `already_done`  one already existed. Not work, not a failure — and
                      crucially it costs nothing now: the existence check
                      runs *before* the strong-route LLM call, so a memo in
                      this state no longer burns a model round-trip to
                      discover what a SELECT could have said.
      `deduped`       policy held it back (rating unchanged since the prior
                      version, or the 14-day per-ticker rate limit).
      `skipped`       it should have been written and could not be. This is
                      the only one that means the loop is unhealthy.
    """
    due, deduped = _scan_due(horizon_days, limit=limit)
    written = 0
    already_done = 0
    skipped = 0
    for item in due:
        outcome: MemoOutcome = item["outcome"]
        snap: MemoSnapshot = item["snapshot"]
        # Re-read immediately before spending anything. The due list is a
        # snapshot of a query that ran before the first LLM call of a pass
        # that makes one per memo; by the time this memo comes round, an
        # admin re-run or another writer may have covered it.
        with SessionLocal() as db:
            if _postmortem_exists(db, outcome.memo_snapshot_id, horizon_days):
                already_done += 1
                continue
        try:
            # MemoSnapshot stores the report in ``memo_json``.  ``snap.memo``
            # never existed; the AttributeError was swallowed here and made
            # every due postmortem count as skipped forever.
            memo = snap.memo_json or {}
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
        status = _persist_postmortem(MemoPostmortem(
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
        ))
        if status == "already_done":
            already_done += 1
            continue
        if status == "failed":
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
        "already_done": already_done,
        "deduped": deduped,
        "skipped": skipped,
    }
