"""Fundamental Factor Scorecard — what the memo pipeline reads and writes.

Phase 6, slice D. Sits between `services/scorecard_service` (the versioned,
point-in-time score rows the worker writes) and the memo run in
`graph.py`. Four jobs, all cheap DB reads or pure functions — nothing here
computes a score, and nothing here spends LLM money:

1. `load_for_memo` — the latest `ScorecardSummary` for a ticker at the
   memo's as-of date, or None. None is the honest answer when no succeeded
   run has a row: the memo then renders the section as n/a with a reason
   and records a SOFT degradation ("Fundamental Scorecard"), never a
   neutral score. With `ENABLE_SCORECARD=false` it returns None silently
   and the memo is byte-for-byte what it was before this phase.
2. `prompt_block` — the <= 600-char context block the PM synthesis and the
   valuation analyst read. It labels every number as an observed rank or
   a model read (fs-v1), and says in so many words that it is a
   cross-sectional scenario input, not a recommendation.
3. `summarize` — the disagreement detector (plan §5.5): compares the
   memo's FINAL rating bucket and its reconciled valuation verdict with
   the scorecard's percentiles and returns the summary with a
   `ScorecardDisagreementFlag`, or none. A disagreement is a finding, not
   an outage — it never enters `degraded_agents`, and the rating blend is
   untouched (the scorecard informs the memo; it does not move ratings in
   this phase).
4. `persist_disagreement` / `pending_seed_questions` / `mark_reviewed` —
   the `scorecard_disagreements` row keyed on the persisted memo snapshot,
   and the deep-research seed that a flag-gated review regen re-fires
   with (`update_orchestrator.handle_scorecard_disagreements`).

The compounder / inflection profile rule lives here so the memo block and
the /app/scorecard page agree on one number: a sub-composite z at or above
`PROFILE_THRESHOLD_Z` "reads as" that profile. The page carries the same
threshold in `frontend/src/types/scorecard.ts` (`profileThresholdZ`).

Research and education only: every string this module writes into a memo
describes a scenario input, not advice.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from ..config import settings
from ..finance import scorecard_spec
from ..schemas import (
    CritiqueQuestion,
    ScorecardDisagreementFlag,
    ScorecardSummary,
    StockMemoOut,
    score_from_rating_label,
)
from .log_safety import safe_exc

log = logging.getLogger(__name__)

# The memo-facing name. It is what `degraded_agents` shows when no row is on
# file (soft) or the read crashed (hard), so keep it stable.
AGENT_NAME = "Fundamental Scorecard"

# A sub-composite z at or above this reads as that profile. Mirrored by the
# frontend's `SCORECARD_CLIENT_RULES.profileThresholdZ`; change both.
PROFILE_THRESHOLD_Z = 0.5

# Hard cap on the block the PM and the valuation analyst read. The PM
# context already carries the brain file, memories and audit fragments; a
# scorecard read that needs more than this is the page's job, not the
# prompt's.
PROMPT_BLOCK_MAX_CHARS = 600

# A memo that calls the name undervalued while the valuation family sits
# below this universe percentile — or overvalued above the mirror — is a
# valuation contradiction (plan §5.5).
VALUATION_LOW_PCT = 30.0
VALUATION_HIGH_PCT = 70.0

# Row lifecycle for `scorecard_disagreements`.
STATUS_OPEN = "open"
STATUS_QUEUED_REVIEW = "queued_review"
STATUS_REVIEWED = "reviewed"
STATUS_DISMISSED = "dismissed"

# Tag on the regen job's enqueue waypoint, so the daily cap can count the
# jobs this feature — and only this feature — created.
REGEN_SOURCE = "scorecard_disagreement"


def _utcnow() -> datetime:
    """Clock seam so tests can pin `resolved_at` / `created_at`."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# 1. Read
# ---------------------------------------------------------------------------

def load_for_memo(ticker: str, as_of: date | None = None) -> ScorecardSummary | None:
    """The latest score row for `ticker` as a memo summary, or None.

    `as_of` is the memo's backtest date when set: the read then returns the
    latest row dated on or before it, so a reproduced memo sees the score a
    reader could have seen. Raises on a DB failure — the caller wraps it in
    `safe_call` so the crash lands on the banner as a hard degradation,
    distinct from the soft "no row on file" case.
    """
    if not settings.enable_scorecard:
        return None
    from ..services import scorecard_service
    return scorecard_service.latest_summary(ticker, as_of=as_of)


# ---------------------------------------------------------------------------
# 2. Profiles and the prompt block
# ---------------------------------------------------------------------------

def _fmt_z(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.2f}"


def _fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f}th pct"


def profile_reads(summary: ScorecardSummary) -> dict[str, str]:
    """Per-profile verdict under the shared threshold.

    Values: ``"reads"`` (z >= threshold), ``"does_not_read"`` (z below), or
    ``"n/a"`` (the sub-composite could not be computed — too few of its
    inputs were available; that is not the same as "does not read").
    """
    out: dict[str, str] = {}
    for name in ("compounder", "inflection"):
        z = (summary.profiles or {}).get(name)
        if z is None:
            out[name] = "n/a"
        else:
            out[name] = "reads" if float(z) >= PROFILE_THRESHOLD_Z else "does_not_read"
    return out


def profile_headline(summary: ScorecardSummary) -> str:
    """One sentence for the memo: what the sub-composites read as.

    Deliberately names the numbers and the threshold, so a reader can
    check the call against the observed z rather than trust a label.
    """
    reads = profile_reads(summary)
    profiles = summary.profiles or {}
    comp, infl = profiles.get("compounder"), profiles.get("inflection")
    labels = {"compounder": "a compounder", "inflection": "an early inflection"}
    hits = [labels[k] for k in ("compounder", "inflection") if reads[k] == "reads"]
    detail = f"compounder z {_fmt_z(comp)}, inflection z {_fmt_z(infl)}; threshold z >= {PROFILE_THRESHOLD_Z:+.2f}"
    if hits:
        return f"Reads as {' and '.join(hits)} ({detail})."
    if all(v == "n/a" for v in reads.values()):
        return "Profile read n/a — the compounder and inflection sub-composites could not be computed from the available features."
    return f"Reads as neither a compounder nor an early inflection ({detail})."


def _valuation_category(summary: ScorecardSummary):
    return (summary.categories or {}).get(scorecard_spec.FAMILY_VALUATION)


def prompt_block(summary: ScorecardSummary | None) -> str:
    """The <= `PROMPT_BLOCK_MAX_CHARS` block the PM and the valuation
    analyst read. Empty string when there is no score — callers can
    concatenate unconditionally.

    Every line is either an observed rank (percentiles, coverage, the
    fiscal period the inputs came from) or a model read under a named
    version (z, contributions). The last line says what it is not.
    """
    if summary is None:
        return ""
    val = _valuation_category(summary)
    val_pct = val.percentile if val is not None else None
    top_pos = ", ".join(f"{c.feature} {_fmt_z(c.z)}" for c in (summary.top_positive or [])[:3]) or "none"
    top_neg = ", ".join(f"{c.feature} {_fmt_z(c.z)}" for c in (summary.top_negative or [])[:3]) or "none"
    reads = profile_reads(summary)
    profile = ", ".join(
        f"{k}={'yes' if v == 'reads' else ('no' if v == 'does_not_read' else 'n/a')}" for k, v in reads.items()
    )
    stale = " (stale)" if summary.stale else ""
    lines = [
        f"Fundamental scorecard {summary.version_key} as of {summary.as_of.isoformat()}{stale}, "
        f"inputs {summary.latest_period or 'n/a'}, coverage {summary.coverage:.0%}.",
        f"Observed rank: universe {_fmt_pct(summary.universe_percentile)}, sector {_fmt_pct(summary.sector_percentile)}; "
        f"valuation family {_fmt_pct(val_pct)} (what is already priced in).",
        f"Model read: overall z {_fmt_z(summary.overall_z)}; profile {profile} (z >= {PROFILE_THRESHOLD_Z:+.1f}).",
        f"Top +: {top_pos}. Top -: {top_neg}.",
        "Cross-sectional model read, not a recommendation; reconcile or explain a wide gap.",
    ]
    block = "\n".join(lines)
    if len(block) > PROMPT_BLOCK_MAX_CHARS:
        # Contributor names are the only unbounded part; cut there and keep
        # the closing caveat intact rather than truncating mid-sentence.
        block = "\n".join(lines[:3] + [lines[4]])
    return block[:PROMPT_BLOCK_MAX_CHARS]


# ---------------------------------------------------------------------------
# 3. Attach + disagreement
# ---------------------------------------------------------------------------

def for_memo(summary: ScorecardSummary | None, *, reconciliation: Any = None) -> ScorecardSummary | None:
    """The summary as the compose stage attaches it: a copy carrying the
    PM's reconciliation text and the server-side profile read as a note
    (so the memo section and the page quote the same call). None stays
    None — the section then says n/a."""
    if summary is None:
        return None
    out = summary.model_copy(deep=True)
    text = str(reconciliation or "").strip()
    if text:
        out.reconciliation = text[:PROMPT_BLOCK_MAX_CHARS]
    note = f"Profile read ({summary.version_key}): {profile_headline(summary)}"
    if note not in out.notes:
        out.notes = [*out.notes, note]
    return out


def detect_disagreement(summary: ScorecardSummary, memo: StockMemoOut) -> ScorecardDisagreementFlag | None:
    """Plan §5.5. Compares the memo's FINAL rating bucket centre (10/30/50/
    70/90) with the scorecard's universe percentile, and the reconciled
    valuation verdict with the valuation family's percentile.

    * ``material``: coverage >= `scorecard_min_coverage` AND (|gap| >=
      `scorecard_disagreement_material` OR a valuation contradiction). The
      two triggers are independent: a valuation contradiction is material
      even when the overall gap only reaches the watch band.
    * ``watch``: |gap| >= `scorecard_disagreement_watch`, or a material
      trigger that low coverage held back (never material under 0.6 —
      too few observed features to call the narrative wrong).
    * none otherwise, and none when the percentile is unavailable: a
      missing rank is n/a, not "agrees".

    Dimension precedence: ``overall`` when the overall gap alone is
    material, or when it is the only trigger; ``valuation`` when the
    contradiction is what carries the flag (alone, or alongside an overall
    gap in the watch band — the note then names both).
    """
    pct = summary.universe_percentile
    if pct is None:
        return None
    rating = str(memo.rating_label or "")
    rating_score = score_from_rating_label(rating)
    gap = float(rating_score) - float(pct)
    coverage = float(summary.coverage or 0.0)
    enough_coverage = coverage >= float(settings.scorecard_min_coverage)
    material_at = float(settings.scorecard_disagreement_material)
    watch_at = float(settings.scorecard_disagreement_watch)

    # Valuation contradiction: the verdict word against the family rank.
    val = _valuation_category(summary)
    val_pct = val.percentile if val is not None else None
    verdict = memo.valuation_verdict.verdict if memo.valuation_verdict is not None else "fairly_priced"
    val_direction: str | None = None
    if val_pct is not None:
        if verdict == "undervalued" and float(val_pct) < VALUATION_LOW_PCT:
            val_direction = "narrative_above_quant"
        elif verdict == "overvalued" and float(val_pct) > VALUATION_HIGH_PCT:
            val_direction = "narrative_below_quant"

    overall_material = abs(gap) >= material_at
    overall_watch = abs(gap) >= watch_at
    # The plan's rule verbatim: either trigger makes the flag material when
    # coverage allows; low coverage holds it at watch rather than dropping it.
    material_trigger = overall_material or val_direction is not None
    if not (material_trigger or overall_watch):
        return None
    severity = "material" if (material_trigger and enough_coverage) else "watch"
    held = " — held at watch: coverage below the material floor" if (material_trigger and not enough_coverage) else ""
    coverage_note = f"coverage {coverage:.0%}"
    if overall_material or val_direction is None:
        direction = "narrative_above_quant" if gap > 0 else "narrative_below_quant"
        note = (
            f"Memo rates {rating} (bucket {rating_score:.0f}) vs scorecard universe percentile {float(pct):.1f} "
            f"(gap {gap:+.1f}; {coverage_note}){held}."
        )
        return ScorecardDisagreementFlag(severity=severity, dimension="overall", direction=direction, gap=round(gap, 2), note=note)
    assert val_pct is not None
    val_gap = float(rating_score) - float(val_pct)
    overall_note = (
        f"; overall gap {gap:+.1f} vs universe percentile {float(pct):.1f} also in the watch band" if overall_watch else ""
    )
    note = (
        f"Valuation verdict '{verdict}' vs valuation-family universe percentile {float(val_pct):.1f} "
        f"(rating {rating}; {coverage_note}{overall_note}){held}."
    )
    return ScorecardDisagreementFlag(
        severity=severity, dimension="valuation", direction=val_direction, gap=round(val_gap, 2), note=note,
    )


def summarize(summary: ScorecardSummary | None, memo: StockMemoOut) -> ScorecardSummary | None:
    """The memo's scorecard summary with the disagreement flag filled from
    the FINAL memo (post rating blend, post valuation verdict). Pure: returns
    a copy; the caller assigns it. None in, None out."""
    if summary is None:
        return None
    out = summary.model_copy(deep=True)
    out.disagreement = detect_disagreement(summary, memo)
    return out


# Plan §5.5: the review question goes to the valuation analyst (the
# "what is priced in" leg) AND the earnings analyst (earnings quality and
# profitability are where the quant read and the narrative most often part
# ways). One stored question text; one CritiqueQuestion per target.
SEED_TARGETS: tuple[str, ...] = ("valuation", "earnings")

_SEED_WHY: dict[str, str] = {
    "valuation": "A material score-vs-narrative gap must be reconciled with observed figures or lower conviction.",
    "earnings": (
        "The earnings-quality and profitability families must be reconciled with the observed "
        "figures before the narrative overrides the quant read."
    ),
}


def _seed_for_target(target: str, question: str, *, why: str | None = None) -> CritiqueQuestion:
    return CritiqueQuestion(
        target_agent=target,
        question=question[:600],
        why_it_matters=why if why is not None else _SEED_WHY.get(target, _SEED_WHY["valuation"]),
    )


def seed_question(ticker: str, summary: ScorecardSummary, flag: ScorecardDisagreementFlag) -> CritiqueQuestion:
    """The deep-research question a review regen re-fires with, addressed
    to the valuation analyst (the primary target; `seed_questions` fans it
    out to every `SEED_TARGETS` member). Names the observed figures so the
    specialist argues with numbers, and asks for a falsifier so the answer
    is checkable. Its text is what `scorecard_disagreements.seed_question`
    stores."""
    worst = ", ".join(
        f"{c.feature} ({c.family}) z {_fmt_z(c.z)}" for c in (summary.top_negative or [])[:2]
    ) or "no negative contributors on file"
    best = ", ".join(
        f"{c.feature} ({c.family}) z {_fmt_z(c.z)}" for c in (summary.top_positive or [])[:2]
    ) or "no positive contributors on file"
    if flag.direction == "narrative_above_quant":
        stance = "yet the memo's stance is more constructive than that rank"
        ask = f"Which observed figures justify overriding the quant read, and what would falsify that? Weakest features: {worst}."
    else:
        stance = "yet the memo's stance is more cautious than that rank"
        ask = f"Which observed figures justify the caution against the quant read, and what would falsify it? Strongest features: {best}."
    text = (
        f"The fundamental scorecard ({summary.version_key}, as of {summary.as_of.isoformat()}) places {ticker} at the "
        f"{_fmt_pct(summary.universe_percentile)} of the universe overall, {stance} ({flag.note}). {ask}"
    )
    return _seed_for_target(SEED_TARGETS[0], text)


def seed_questions(ticker: str, summary: ScorecardSummary, flag: ScorecardDisagreementFlag) -> list[CritiqueQuestion]:
    """`seed_question` addressed to every `SEED_TARGETS` member, in order
    (plan §5.5: valuation and earnings). Same text, per-target rationale."""
    primary = seed_question(ticker, summary, flag)
    return [_seed_for_target(t, primary.question) for t in SEED_TARGETS]


# ---------------------------------------------------------------------------
# 4. Persistence: disagreement rows and review seeds
# ---------------------------------------------------------------------------

def _score_row_id(db: Any, ticker: str, summary: ScorecardSummary) -> int | None:
    from sqlalchemy import select

    from ..models import ScorecardRun, ScorecardScore
    q = (
        select(ScorecardScore.id)
        .join(ScorecardRun, ScorecardRun.id == ScorecardScore.run_id)
        .where(
            ScorecardScore.ticker == ticker.upper(),
            ScorecardScore.version_key == summary.version_key,
            ScorecardScore.as_of == summary.as_of,
        )
        .order_by(ScorecardScore.id.desc())
    )
    if summary.run_id:
        q = q.where(ScorecardRun.run_id == summary.run_id)
    got = db.execute(q.limit(1)).first()
    return int(got[0]) if got is not None else None


def persist_disagreement(memo: StockMemoOut, memo_snapshot_id: int | None) -> dict[str, Any] | None:
    """Write the `scorecard_disagreements` row for a flagged memo.

    Keyed on the persisted snapshot so a reviewer sees exactly which memo
    version disagreed with which score row; idempotent on
    `(memo_snapshot_id, scorecard_score_id, dimension)`. Returns the row as
    a dict, or None when the memo carries no flag or the score row it was
    compared against cannot be found (logged, not raised — the memo is
    already saved and a finding record must not fail it).
    """
    summary = memo.scorecard
    if summary is None or summary.disagreement is None:
        return None
    flag = summary.disagreement
    from ..database import SessionLocal
    from ..models import ScorecardDisagreement
    with SessionLocal() as db:
        ScorecardDisagreement.__table__.create(bind=db.get_bind(), checkfirst=True)
        score_id = _score_row_id(db, memo.ticker, summary)
        if score_id is None:
            log.warning("scorecard disagreement for %s: score row not found (version=%s as_of=%s)",
                        memo.ticker, summary.version_key, summary.as_of)
            return None
        existing = (
            db.query(ScorecardDisagreement)
            .filter(
                ScorecardDisagreement.memo_snapshot_id == memo_snapshot_id,
                ScorecardDisagreement.scorecard_score_id == score_id,
                ScorecardDisagreement.dimension == flag.dimension,
            )
            .first()
        )
        if existing is not None:
            return _disagreement_dict(existing)
        val = _valuation_category(summary)
        pct = summary.universe_percentile if flag.dimension == "overall" else (val.percentile if val is not None else None)
        row = ScorecardDisagreement(
            ticker=memo.ticker.upper(), memo_snapshot_id=memo_snapshot_id, scorecard_score_id=score_id,
            version_key=summary.version_key, as_of=summary.as_of,
            memo_rating=str(memo.rating_label or ""), memo_rating_score=score_from_rating_label(memo.rating_label),
            scorecard_percentile=pct, gap=flag.gap, severity=flag.severity, dimension=flag.dimension,
            status=STATUS_OPEN, seed_question=seed_question(memo.ticker, summary, flag).question,
            created_at=_utcnow(), note=flag.note,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _disagreement_dict(row)


def _disagreement_dict(row: Any) -> dict[str, Any]:
    return {
        "id": row.id, "ticker": row.ticker, "memo_snapshot_id": row.memo_snapshot_id,
        "scorecard_score_id": row.scorecard_score_id, "version_key": row.version_key, "as_of": row.as_of,
        "memo_rating": row.memo_rating, "memo_rating_score": row.memo_rating_score,
        "scorecard_percentile": row.scorecard_percentile, "gap": row.gap, "severity": row.severity,
        "dimension": row.dimension, "status": row.status, "seed_question": row.seed_question,
        "created_at": row.created_at, "resolved_at": row.resolved_at, "note": row.note,
    }


def pending_seed_questions(ticker: str) -> list[CritiqueQuestion]:
    """Seed questions from rows a review regen was queued for
    (`status=queued_review`). Empty when the feature is off or nothing is
    queued. The deep-research loop re-fires these on round 1 regardless
    of the PM critique (`deep_research.run_dialog_loop(seed_questions=)`).
    Each row yields one question per `SEED_TARGETS` member.
    """
    if not settings.enable_scorecard:
        return []
    from ..database import SessionLocal
    from ..models import ScorecardDisagreement
    with SessionLocal() as db:
        ScorecardDisagreement.__table__.create(bind=db.get_bind(), checkfirst=True)
        rows = (
            db.query(ScorecardDisagreement)
            .filter(ScorecardDisagreement.ticker == ticker.upper(), ScorecardDisagreement.status == STATUS_QUEUED_REVIEW)
            .order_by(ScorecardDisagreement.created_at.asc(), ScorecardDisagreement.id.asc())
            .all()
        )
        # One question per (row, target): the stored text is asked of every
        # `SEED_TARGETS` specialist (plan §5.5), de-duplicated per target.
        out: list[CritiqueQuestion] = []
        seen: set[tuple[str, str]] = set()
        for r in rows:
            q = (r.seed_question or "").strip()
            if not q:
                continue
            why = f"scorecard disagreement #{r.id} ({r.severity}, {r.dimension}) queued for review"
            for target in SEED_TARGETS:
                if (target, q) in seen:
                    continue
                seen.add((target, q))
                out.append(_seed_for_target(target, q, why=why))
        return out


def mark_reviewed(ticker: str, memo_snapshot_id: int | None) -> int:
    """Close the `queued_review` rows a memo run consumed. Returns the count.
    A memo written with the seed question present is the review; the row
    must not be re-queued for the same (ticker, as_of)."""
    from ..database import SessionLocal
    from ..models import ScorecardDisagreement
    now = _utcnow()
    with SessionLocal() as db:
        ScorecardDisagreement.__table__.create(bind=db.get_bind(), checkfirst=True)
        rows = (
            db.query(ScorecardDisagreement)
            .filter(ScorecardDisagreement.ticker == ticker.upper(), ScorecardDisagreement.status == STATUS_QUEUED_REVIEW)
            .all()
        )
        for r in rows:
            r.status = STATUS_REVIEWED
            r.resolved_at = now
            suffix = f"reviewed by memo snapshot {memo_snapshot_id}" if memo_snapshot_id is not None else "reviewed"
            r.note = f"{r.note} | {suffix}" if r.note else suffix
        db.commit()
        return len(rows)


def log_context_failure(what: str, exc: BaseException) -> None:
    """Warning with the redacted exception text (never a prompt or a key)."""
    log.warning("scorecard context %s failed: %s", what, safe_exc(exc))


__all__ = [
    "AGENT_NAME",
    "PROFILE_THRESHOLD_Z",
    "PROMPT_BLOCK_MAX_CHARS",
    "REGEN_SOURCE",
    "SEED_TARGETS",
    "STATUS_DISMISSED",
    "STATUS_OPEN",
    "STATUS_QUEUED_REVIEW",
    "STATUS_REVIEWED",
    "detect_disagreement",
    "for_memo",
    "load_for_memo",
    "mark_reviewed",
    "pending_seed_questions",
    "persist_disagreement",
    "profile_headline",
    "profile_reads",
    "prompt_block",
    "seed_question",
    "seed_questions",
    "summarize",
]
