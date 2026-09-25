"""W7 learned priors: the bounded, audited block a memo-run consumer reads.

Owner decision 9 (2026-09-24): memory acts as priors that update with
evidence, never as fixed instructions, and must not over-index. So a prior
reaches a prompt only as a small, fixed-budget block of whole lines, each
carrying its evidence count, under a header that says what it is:
provisional hypotheses, overridden by the run's current evidence, and never
a source for a number. Design: `design-w7-learning-final.md` §6.2-§6.3 and
§7.2, with the accepted S19 critique:

* **Only inside a live memo run.** ``render_for`` returns ``""`` and writes
  nothing unless ``safe_runner.in_memo_run()`` (contract C6), the run is not
  a backtest, and the LLM is live. Chat's ``ask_sector`` re-fires the sector
  analyst outside a memo run, so it never sees a prior — and the guard
  reports mode "off", so every caller takes today's legacy branch there.
* **Mode.** ``off`` does no I/O at all; ``shadow`` writes the audit row of
  what WOULD be shown and returns ``""``; ``inject`` writes the row FIRST and
  returns the block only if that insert succeeded (a prior that cannot be
  audited is never shown).
* **7(d).** The PM's regime and specialist-reliability blocks are not
  touched here or by the inject branch in ``pm_context``: calibration is
  frozen until the owner decides the 7(d) proposal. Only the legacy file
  memory blocks are replaced.

The DB mode stays shadow after deploy; promotion to inject is a separate,
gated owner step (``POST /api/admin/learning/mode``).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, delete, func, or_, select, update

from ..config import settings
from ..database import SessionLocal
from ..models import LearningItem, LearningRender
from . import control, ledger

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Budget:
    max_chars: int          # the whole block, header and footer included
    max_lessons: int
    max_untested: int
    max_observations: int
    scopes: tuple[str, ...]


BUDGETS: dict[str, Budget] = {
    "pm_memo": Budget(1500, 4, 2, 2, ("company", "industry_group", "sector")),
    "sector": Budget(900, 3, 2, 1, ("industry_group", "sector", "company")),
    "industry_group": Budget(700, 2, 1, 1, ("industry_group", "company")),
    "critic": Budget(700, 2, 1, 2, ("company",)),
}
# The promotion gate G3 checks audited renders against `control.RENDER_BUDGETS`
# (S18's copy of these character caps); a test pins the two equal.

SCOPE_WEIGHT = {"company": 1.0, "industry_group": 0.8, "sector": 0.6}
# Scope words never name a code or the licensed brand (owner decision 10);
# the internal scope key is never rendered at all.
SCOPE_WORD = {"company": "this company", "industry_group": "industry group peers", "sector": "sector peers"}
STANCE_TIER = {"supported": 0, "untested": 1, "contested": 2, "weakened": 3}
MAX_WEAKENED = 1

HEADER = "## Learned priors (provisional hypotheses, not instructions)"
PREAMBLE = (
    "_From past outcomes and filings. Current evidence in this run overrides them. "
    "Numbers here are not sources — do not cite them. "
    "If this run's evidence contradicts a prior, say so._"
)
PM_FOOTER = (
    '_If a prior confirmed or changed your view, add "priors_considered": '
    '[{"id": "L-12", "use": "applied|contradicted", "why": "..."}] to your JSON. '
    "Never put prior ids in memo text._"
)
CONSIDERED_USES = ("applied", "contradicted")
CONSIDERED_MAX = 10
CONSIDERED_WHY_MAX = 300

# Retention for shadow audit rows: they exist to soak the renderer before
# promotion, and G2 only reads them since the latest mode event. Inject rows
# are the record of what a published memo was shown and are kept.
SHADOW_RENDER_RETENTION_DAYS = 180
RENDER_GC_BATCH = 500
RENDER_GC_MAX_BATCHES = 20

PUBLIC_PATH = "database: learning ledger"
PUBLIC_LIMIT_MAX = 50
PUBLIC_LESSON_LABEL = "Provisional hypothesis — not investment advice."

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class RenderOutcome:
    """`text` is non-empty only when `mode == "inject"`. A guarded call
    (outside a live memo run) reports "off": learning does not apply to it,
    so the caller keeps today's legacy branch byte for byte."""
    mode: str
    text: str


# ---------------------------------------------------------------------------
# Guards (overridable in tests)
# ---------------------------------------------------------------------------

def _live_generation() -> bool:
    """A demo / no-LLM run gets nothing — the same test `graph.py` uses to
    label `generation_mode`."""
    return bool(settings.llm_enabled)


def _in_live_memo_run() -> bool:
    from ..agents.safe_runner import in_memo_run
    from ..services.data_service import current_as_of_date
    return in_memo_run() and current_as_of_date() is None and _live_generation()


def _run_id() -> str | None:
    from ..agents.llm import current_call_context
    value = current_call_context().get("run_id")
    return str(value)[:64] if value else None


def _budget(consumer: str) -> Budget:
    try:
        return BUDGETS[consumer]
    except KeyError:
        raise ValueError(f"unknown learning consumer {consumer!r}; expected one of {tuple(BUDGETS)}") from None


# ---------------------------------------------------------------------------
# Selection and rendering (pure reads)
# ---------------------------------------------------------------------------

def _one_line(text: Any) -> str:
    return _WS_RE.sub(" ", str(text or "")).strip()


def _evidence_phrase(post: ledger.Posterior) -> str:
    if post.stance == "untested":
        return "untested hypothesis"
    if post.stance == "supported":
        return f"supported {post.held} of {post.judged} later outcomes"
    if post.stance == "weakened":
        return f"has not held: {post.held} of {post.judged}"
    return f"contested: held {post.held} of {post.judged}"


def _lesson_line(item: LearningItem, post: ledger.Posterior) -> str:
    return (f"- [{ledger.ref(item.id, 'lesson')}] ({SCOPE_WORD[item.scope_type]}; "
            f"{_evidence_phrase(post)}) {_one_line(item.text)}")


def _observation_line(item: LearningItem) -> str:
    when = item.source_date.isoformat() if item.source_date else "undated"
    return (f"- [{ledger.ref(item.id, 'observation')}] ({SCOPE_WORD[item.scope_type]}; "
            f"filing observation {when}) {_one_line(item.text)}")


def _date_key(item: LearningItem) -> int:
    return item.source_date.toordinal() if item.source_date else 0


def build_block(consumer: str, *, ticker: str, sector: str | None = None,
                now: datetime | None = None, db: Any = None) -> dict[str, Any]:
    """Select, rank and pack the block for `consumer`. Reads only.

    Lessons rank by stance (supported, untested, contested, weakened — at
    most one weakened), then scope weight, then newest information, then id;
    untested lessons are capped separately so a pile of unproven hypotheses
    can never fill the block. Observations are newest first. Lines are packed
    whole: one that would push the block past the budget is dropped with
    reason "budget", never cut. When no line fits the text is ""."""
    budget = _budget(consumer)
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        raise ValueError("build_block needs a ticker")
    now = now or datetime.utcnow()
    if db is None:
        with SessionLocal() as own:
            return build_block(consumer, ticker=symbol, sector=sector, now=now, db=own)

    scopes = ledger.scopes_for(db, symbol, sector)
    pairs = [(t, k) for t, k in scopes.read() if t in budget.scopes]
    items = db.execute(
        select(LearningItem).where(
            LearningItem.status == "active",
            or_(*(and_(LearningItem.scope_type == t, LearningItem.scope_key == k) for t, k in pairs)),
            or_(LearningItem.expires_at.is_(None), LearningItem.expires_at > now),
            # No look-ahead: nothing learned after "now" (a guard, not an
            # assumption — live writers never date the future).
            or_(LearningItem.source_date.is_(None), LearningItem.source_date <= now.date()),
        )
    ).scalars().all() if pairs else []
    dropped: list[dict[str, Any]] = []
    # Defence in depth: the writer already refused codes and the brand, but
    # an item that names one never reaches a prompt either way. Checked
    # BEFORE the caps, so a refused item cannot take a slot from a clean one.
    usable = []
    for item in items:
        if ledger.text_rejection(_one_line(item.text), scope_type=item.scope_type,
                                 scope_key=item.scope_key) is not None:
            dropped.append({"ref": ledger.ref(item.id, item.kind), "reason": "label"})
        else:
            usable.append(item)
    items = usable
    post = ledger.posteriors(db, [i.id for i in items if i.kind == "lesson"], now=now)

    candidates: list[tuple[LearningItem, str, str | None]] = []   # (item, line, stance)

    lessons = sorted(
        (i for i in items if i.kind == "lesson"),
        key=lambda i: (STANCE_TIER[post[i.id].stance], -SCOPE_WEIGHT[i.scope_type], -_date_key(i), -i.id),
    )
    n_lessons = n_untested = n_weakened = 0
    for item in lessons:
        p = post[item.id]
        ref = ledger.ref(item.id, "lesson")
        if p.stance == "untested" and n_untested >= budget.max_untested:
            dropped.append({"ref": ref, "reason": "cap_untested"})
            continue
        if p.stance == "weakened" and n_weakened >= MAX_WEAKENED:
            dropped.append({"ref": ref, "reason": "cap_weakened"})
            continue
        if n_lessons >= budget.max_lessons:
            dropped.append({"ref": ref, "reason": "cap_kind"})
            continue
        n_lessons += 1
        n_untested += p.stance == "untested"
        n_weakened += p.stance == "weakened"
        candidates.append((item, _lesson_line(item, p), p.stance))

    observations = sorted((i for i in items if i.kind == "observation"),
                          key=lambda i: (-_date_key(i), -i.id))
    for n, item in enumerate(observations):
        if n >= budget.max_observations:
            dropped.append({"ref": ledger.ref(item.id, "observation"), "reason": "cap_kind"})
            continue
        candidates.append((item, _observation_line(item), None))

    frame = [HEADER, PREAMBLE]
    footer = [PM_FOOTER] if consumer == "pm_memo" else []
    fixed = len("\n".join([*frame, *footer]))
    lines: list[str] = []
    rendered: list[dict[str, Any]] = []
    for rank, (item, line, stance) in enumerate(candidates, start=1):
        ref = ledger.ref(item.id, item.kind)
        if fixed + sum(len(x) + 1 for x in lines) + len(line) + 1 > budget.max_chars:
            dropped.append({"ref": ref, "reason": "budget"})
            continue
        lines.append(line)
        rendered.append({"ref": ref, "item_id": item.id, "kind": item.kind, "scope_type": item.scope_type,
                         "stance": stance, "rank": rank, "chars": len(line)})
    text = "\n".join([*frame, *lines, *footer]) if lines else ""
    if len(text) > budget.max_chars:   # impossible by construction; guarded, not assumed
        raise ValueError(f"learned-priors block exceeds the {consumer} budget")
    return {"consumer": consumer, "ticker": symbol, "text": text, "chars": len(text),
            "items": rendered, "dropped": dropped}


# ---------------------------------------------------------------------------
# The one entry point prompts use
# ---------------------------------------------------------------------------

def render_for(consumer: str, *, ticker: str, sector: str | None = None,
               now: datetime | None = None) -> RenderOutcome:
    """The learned-priors block for one consumer in a live memo run.

    Raises ValueError only for an unknown consumer or an empty ticker (a
    caller bug). Everything the database can do wrong is absorbed: a failed
    selection is audited with its error type and shows nothing; a failed
    audit insert shows nothing."""
    _budget(consumer)
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        raise ValueError("render_for needs a ticker")
    # Contextvar checks first: a chat turn or a backtest costs no DB read.
    if not _in_live_memo_run():
        return RenderOutcome("off", "")
    mode = control.effective_mode()
    if mode not in ("shadow", "inject"):
        return RenderOutcome("off", "")
    now = now or datetime.utcnow()
    error_type: str | None = None
    try:
        block = build_block(consumer, ticker=symbol, sector=sector, now=now)
    except Exception as exc:
        error_type = type(exc).__name__[:64]
        log.warning("learning render failed for %s/%s (%s)", consumer, symbol, error_type)
        block = {"text": "", "chars": 0, "items": [], "dropped": []}
    try:
        with SessionLocal() as db:
            db.add(LearningRender(
                run_id=_run_id(), consumer=consumer, ticker=symbol[:16], mode=mode,
                chars=int(block["chars"]), items=block["items"], dropped=block["dropped"],
                error_type=error_type, created_at=now,
            ))
            db.commit()
    except Exception as exc:
        log.warning("learning render audit insert failed for %s/%s (%s); nothing injected",
                    consumer, symbol, type(exc).__name__)
        return RenderOutcome(mode, "")
    return RenderOutcome(mode, str(block["text"]) if mode == "inject" else "")


def render_safely(consumer: str, *, ticker: str, sector: str | None = None) -> RenderOutcome:
    """`render_for` for an agent's prompt builder: any failure, a caller bug
    included, is logged by type and reads as "off", so the agent keeps its
    legacy prompt. Priors must never block a memo."""
    try:
        return render_for(consumer, ticker=ticker, sector=sector)
    except Exception as exc:
        log.warning("learned priors unavailable for %s (%s)", consumer, type(exc).__name__)
        return RenderOutcome("off", "")


# ---------------------------------------------------------------------------
# After the PM call, and at persist
# ---------------------------------------------------------------------------

def record_considered(run_id: str | None, raw: Any) -> int:
    """Store the PM's `priors_considered` on this run's pm_memo INJECT render.

    Only ids that render actually showed are kept, `use` must be applied or
    contradicted, and `why` is clipped. In off / shadow there is no inject
    row, so nothing is written. Returns the number of entries kept."""
    if raw is None or not run_id:
        return 0
    with SessionLocal() as db:
        row = db.execute(
            select(LearningRender).where(
                LearningRender.run_id == str(run_id), LearningRender.consumer == "pm_memo",
                LearningRender.mode == "inject",
            ).order_by(LearningRender.id.desc()).limit(1)
        ).scalars().first()
        if row is None:
            return 0
        shown = {str(it.get("ref")) for it in (row.items or []) if isinstance(it, dict)}
        kept: list[dict[str, str]] = []
        seen: set[str] = set()
        for entry in (raw if isinstance(raw, list) else [])[:CONSIDERED_MAX]:
            if not isinstance(entry, dict):
                continue
            ref = str(entry.get("id") or "").strip()
            use = str(entry.get("use") or "").strip().lower()
            if ref not in shown or ref in seen or use not in CONSIDERED_USES:
                continue
            seen.add(ref)
            kept.append({"id": ref, "use": use, "why": _one_line(entry.get("why"))[:CONSIDERED_WHY_MAX]})
        row.considered = kept
        db.commit()
        return len(kept)


def link_run(run_id: str | None, snapshot_id: int | None) -> int:
    """Attach this run's render rows to the memo snapshot it produced (G2
    counts only linked renders). No-op with the env ceiling off: no render
    row can exist for a run this process started then."""
    if not run_id or snapshot_id is None or control.ceiling() == "off":
        return 0
    with SessionLocal() as db:
        result = db.execute(
            update(LearningRender)
            .where(LearningRender.run_id == str(run_id), LearningRender.memo_snapshot_id.is_(None))
            .values(memo_snapshot_id=int(snapshot_id))
        )
        db.commit()
        return int(result.rowcount or 0)


# ---------------------------------------------------------------------------
# Retention (called by the daily snapshot_gc loop)
# ---------------------------------------------------------------------------

def prune_shadow_renders(*, now: datetime | None = None, retention_days: int = SHADOW_RENDER_RETENTION_DAYS,
                         batch_size: int = RENDER_GC_BATCH,
                         max_batches: int = RENDER_GC_MAX_BATCHES) -> dict[str, int]:
    """Delete SHADOW render rows older than `retention_days`, in bounded
    batches (ids first, then a delete per batch, committed each time).
    Inject rows are never touched. `capped` = 1 when the pass stopped with
    more left, and the next day's run continues."""
    cutoff = (now or datetime.utcnow()) - timedelta(days=retention_days)
    deleted = 0
    capped = 0
    with SessionLocal() as db:
        for n in range(max_batches):
            ids = [row_id for (row_id,) in db.execute(
                select(LearningRender.id)
                .where(LearningRender.mode == "shadow", LearningRender.created_at < cutoff)
                .order_by(LearningRender.id).limit(batch_size)
            ).all()]
            if not ids:
                break
            result = db.execute(delete(LearningRender).where(LearningRender.id.in_(ids)))
            db.commit()
            deleted += int(result.rowcount if result.rowcount is not None else len(ids))
            if len(ids) < batch_size:
                break
            if n == max_batches - 1:
                capped = 1
    return {"deleted": deleted, "capped": capped}


# ---------------------------------------------------------------------------
# The public trail (`GET /api/stocks/{ticker}/memory`, inject mode only)
# ---------------------------------------------------------------------------

def public_trail_enabled() -> bool:
    """The ledger is public only once priors are actually being injected;
    until then the page shows the legacy trail exactly as before."""
    return control.effective_mode() == "inject"


def public_trail(ticker: str, *, limit: int = 10, now: datetime | None = None) -> dict[str, Any]:
    """Company-scope observations and non-suppressed lessons, newest first,
    in the legacy response shape `MemoryTrail.tsx` already renders.

    Never `detail` (the full postmortem narrative is audit-only), never a
    scope key, and every lesson is labelled a provisional hypothesis."""
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        raise ValueError("public_trail needs a ticker")
    limit = max(1, min(int(limit), PUBLIC_LIMIT_MAX))
    now = now or datetime.utcnow()
    with SessionLocal() as db:
        rows = db.execute(
            select(LearningItem).where(
                LearningItem.scope_type == "company", LearningItem.scope_key == symbol,
                LearningItem.status.in_(("active", "superseded", "retired")),
            )
        ).scalars().all()
        suppressed = int(db.execute(
            select(func.count(LearningItem.id)).where(
                LearningItem.scope_type == "company", LearningItem.scope_key == symbol,
                LearningItem.status == "suppressed",
            )
        ).scalar_one())
        post = ledger.posteriors(db, [r.id for r in rows if r.kind == "lesson"], now=now)

    def day(item: LearningItem) -> str:
        when = item.source_date or (item.created_at.date() if item.created_at else None)
        return when.isoformat() if when else ""

    rows = sorted(rows, key=lambda r: (day(r), r.id), reverse=True)
    entries = []
    for r in rows[:limit]:
        status = "" if r.status == "active" else f" · {r.status}"
        if r.kind == "lesson":
            trigger = f"provisional hypothesis · {_evidence_phrase(post[r.id])}{status}"
            body = f"{_one_line(r.text)}\n\n{PUBLIC_LESSON_LABEL}"
        else:
            trigger = f"filing observation{status}"
            body = _one_line(r.text)
        entries.append({"date": day(r), "trigger": trigger, "body": body, "structured_facts": None})
    lessons = [r for r in rows if r.kind == "lesson"]
    tested = sum(1 for r in lessons if post[r.id].judged > 0)
    observations = len(rows) - len(lessons)
    coverage = (
        f"Learning ledger for {symbol}: {len(lessons)} lesson(s), {tested} tested against later "
        f"benchmark-relative outcomes, and {observations} filing observation(s). Lessons are provisional "
        "hypotheses, not investment advice; each memo's current evidence takes precedence."
    )
    return {
        "ticker": symbol,
        "path": PUBLIC_PATH,
        "entry_count": len(rows),
        "suppressed_count": suppressed,
        "historical_context": coverage,
        "historical_context_suppressed": 0,
        "entries": entries,
    }
