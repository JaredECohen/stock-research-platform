"""W7 learning ledger — lessons as hypotheses, credibility as a Beta posterior.

Owner decision 9 (2026-09-24), verbatim: "backfill and enable memory but it
shouldn't over index on memory, it must update priors and learn as evidence
comes in and learn dynamically". Design: `design-w7-learning-final.md`
§3-§5 plus the accepted S18 critique:

* **Grounded grammar.** A live lesson is ``{condition, observable}``: a
  situation (<= 160 chars) a later memo's own content can show, and a
  benchmark-relative direction (outperform | underperform) over the horizon.
  The judge (cheap route) answers only whether the condition APPLIED to a
  later memo; held / failed is computed here from realized alpha with the
  postmortem verdict thresholds. A model never decides a verdict.
* **Decoupled judging.** ``judge_due`` walks every W6-eligible memo with a
  90d outcome (the horizon every lesson is stated over), regardless of
  postmortem dedupe, so evidence is not starved by the "rating unchanged"
  skip. One call per memo, capped per night in calls and dollars.
* **One row per window.** Evidence is keyed ``(item, scope_key, horizon,
  bucket)``: re-issues of one company, or three peers of one group, inside
  one window count once. Only a held / failed / mixed row closes the
  window; ``irrelevant`` is recorded per memo, so a peer the condition did
  not fit never blocks a later peer it did.
* **Nothing from templates, nothing ineligible.** Only W6-eligible outcomes
  feed learning (``outcome_eligibility.eligible_only``, fail closed); the
  learning payload goes through the W2a presenter, and writes are skipped
  when the thesis or PM view is unavailable.
* **Priors, not instructions.** Evidence decays with a 365-day half-life,
  "supported" needs three weighted later outcomes, and a lesson that keeps
  failing is retired automatically. Nothing is rewritten by an LLM.

Every write is gated on ``settings.learning_ledger_writes``. Unknown kinds,
scope types, verdicts or statuses raise ``ValueError``.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import sqrt
from typing import Any

from sqlalchemy import String, and_, cast, exists, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import settings
from ..database import SessionLocal
from ..models import (
    Company,
    LearningEvidence,
    LearningItem,
    LearningRender,
    MemoOutcome,
    MemoOutcomeEligibility,
    MemoPostmortem,
    MemoSnapshot,
)
from ..services import outcome_eligibility
from . import control

log = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------
HALF_LIFE_DAYS = 365
JUDGE_MAX = 6                      # hypotheses offered per judge call
LESSON_MAX = 240
CONDITION_MAX = 160
OBS_MAX = 320
OBS_TTL_DAYS = 400
MAX_ACTIVE_LESSONS_PER_SCOPE = 30
MAX_OBS_PER_SCOPE = 10
LESSON_MIN_HORIZON = 90
# The one horizon a live hypothesis is stated over ("... over 90 days") and
# judged on. Judging 180d and 365d outcomes too would pay the same
# horizon-independent "did the condition apply?" question up to three times
# per memo and turn one memo into three correlated evidence rows, scored on
# horizons the lesson never claimed.
LESSON_HORIZON = 90
SYNC_LIMIT = 200
DETAIL_MAX = 4000
RATIONALE_MAX = 500
MEMO_FIELD_CHARS = 1500            # each memo field is clipped on its own
JUDGE_MAX_TOKENS = 600
CONTAMINATION_LOOKBACK_DAYS = 30

KINDS = ("lesson", "observation")
SCOPE_TYPES = ("company", "industry_group", "sector")
VERDICTS = ("held", "failed", "mixed", "irrelevant")
STATUSES = ("active", "suppressed", "retired", "superseded")
OBSERVABLES = ("outperform", "underperform")
APPLIES = ("yes", "no", "unclear")
ORIGIN_KINDS = ("postmortem", "postmortem_sector", "postmortem_backfill", "filing_delta", "filing_pattern")
POSTMORTEM_ORIGINS = control.POSTMORTEM_ORIGINS

# `_deterministic_lesson` in postmortem_service writes exactly this shape
# when the LLM is unavailable; it restates numbers, it teaches nothing.
DETERMINISTIC_LESSON_RE = re.compile(r"^\d+d postmortem \((right|wrong|mixed|pending)\)\. Memo rated")
# A condition that names the outcome is circular: it could only be checked
# after the fact, which is exactly the look-ahead the grammar exists to stop.
_CIRCULAR_RE = re.compile(r"\b(outperform\w*|underperform\w*|alpha|beats? the benchmark)\b", re.IGNORECASE)
_LEADING_WHEN_RE = re.compile(r"^(when|if)\s+", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SUBJECT = {"company": "the stock", "industry_group": "group peers", "sector": "sector peers"}


def _writes_on() -> bool:
    return bool(settings.learning_ledger_writes)


def _utcnow() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Posterior (pure)
# ---------------------------------------------------------------------------

def evidence_weight(observed_at: datetime, now: datetime) -> float:
    return float(0.5 ** (max((now - observed_at).days, 0) / HALF_LIFE_DAYS))


@dataclass(frozen=True)
class Posterior:
    alpha: float
    beta: float
    n_eff: float
    mean: float
    lo80: float
    hi80: float
    held: int
    judged: int
    stance: str   # untested | supported | contested | weakened

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha": round(self.alpha, 4), "beta": round(self.beta, 4), "n_eff": round(self.n_eff, 4),
            "mean": round(self.mean, 4), "lo80": round(self.lo80, 4), "hi80": round(self.hi80, 4),
            "held": self.held, "judged": self.judged, "stance": self.stance,
        }


def posterior(evidence: Iterable[tuple[str, datetime]], *, now: datetime) -> Posterior:
    """Beta(1,1) updated by decayed verdicts. `irrelevant` is recorded but
    never moves the posterior."""
    a = b = 1.0
    held = judged = 0
    for verdict, observed_at in evidence:
        if verdict not in VERDICTS:
            raise ValueError(f"unknown learning verdict {verdict!r}")
        w = evidence_weight(observed_at, now)
        if verdict == "held":
            a += w
            held += 1
        elif verdict == "failed":
            b += w
        elif verdict == "mixed":
            a += 0.5 * w
            b += 0.5 * w
        if verdict != "irrelevant":
            judged += 1
    n_eff = a + b - 2.0
    mean = a / (a + b)
    sd = sqrt(mean * (1 - mean) / (a + b + 1))
    lo, hi = max(0.0, mean - 1.2816 * sd), min(1.0, mean + 1.2816 * sd)
    if n_eff < 1:
        stance = "untested"
    elif n_eff >= 3 and lo >= 0.5:
        stance = "supported"
    elif n_eff >= 2 and hi < 0.5:
        stance = "weakened"
    else:
        stance = "contested"
    return Posterior(a, b, n_eff, mean, lo, hi, held, judged, stance)


def should_retire(p: Posterior) -> bool:
    return p.n_eff >= 4 and p.hi80 < 0.35


def window_bucket(generated_at: datetime, horizon_days: int) -> int:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    return generated_at.date().toordinal() // horizon_days


def independence_key(scope_key: str, horizon_days: int, generated_at: datetime) -> str:
    """One evidence row per lesson per scope window. For a company lesson the
    scope key is the ticker; for a group or sector lesson it is the group or
    sector, so peers judged in the same window collapse to one row."""
    if not scope_key:
        raise ValueError("independence_key needs a scope key")
    return f"{scope_key}:{horizon_days}:{window_bucket(generated_at, horizon_days)}"


def irrelevant_key(window_key: str, snapshot_id: int) -> str:
    """Where an `irrelevant` row lives: per memo, NOT per window. A condition
    that did not apply to one peer (or one re-issue) says nothing about the
    next memo in the same window, so it must not take the window's single
    held / failed / mixed slot — that would starve the ledger of evidence."""
    return f"{window_key}:s{snapshot_id}"


def verdict_from_alpha(observable: str, alpha: float | None) -> str | None:
    """held / failed / mixed from realized alpha, with the postmortem's own
    thresholds (`_classify_verdict`): an outperform hypothesis is judged like
    a Bullish call, underperform like a Bearish one. None when alpha is
    missing (no evidence can be written)."""
    if observable not in OBSERVABLES:
        raise ValueError(f"unknown hypothesis observable {observable!r}")
    from ..services.postmortem_service import _classify_verdict
    verdict = _classify_verdict("Bullish" if observable == "outperform" else "Bearish", alpha)
    return {"right": "held", "wrong": "failed", "mixed": "mixed"}.get(verdict)


# ---------------------------------------------------------------------------
# Hypothesis grammar and text rules (pure)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Hypothesis:
    condition: str
    observable: str


def parse_hypothesis(raw: Any) -> tuple[Hypothesis | None, str | None]:
    """(hypothesis, None) or (None, reason). `reason == "empty"` means the
    model proposed nothing, which is an answer, not a rejection."""
    if raw is None or raw == "" or raw == {}:
        return None, "empty"
    if not isinstance(raw, dict):
        return None, "not_an_object"
    cond, obs = raw.get("condition"), raw.get("observable")
    if cond is None or (isinstance(cond, str) and not cond.strip()):
        return None, "empty"
    if not isinstance(cond, str):
        return None, "condition_not_text"
    cond = _LEADING_WHEN_RE.sub("", " ".join(cond.split())).rstrip(" .;,:")
    if not cond:
        return None, "empty"
    if len(cond) > CONDITION_MAX:
        return None, "condition_too_long"
    if _CIRCULAR_RE.search(cond):
        return None, "condition_names_outcome"
    if not isinstance(obs, str) or obs.strip().lower() not in OBSERVABLES:
        return None, "observable_invalid"
    return Hypothesis(cond, obs.strip().lower()), None


def lesson_text(h: Hypothesis, scope_type: str, horizon_days: int = LESSON_HORIZON) -> str:
    if scope_type not in SCOPE_TYPES:
        raise ValueError(f"unknown learning scope_type {scope_type!r}")
    text = f"When {h.condition}, expect {_SUBJECT[scope_type]} to {h.observable} the benchmark over {horizon_days} days."
    if len(text) > LESSON_MAX:  # impossible with CONDITION_MAX = 160; guarded, not assumed
        raise ValueError("rendered lesson exceeds LESSON_MAX")
    return text


def text_rejection(text: str, *, scope_type: str, scope_key: str) -> str | None:
    """Why `text` may not be stored, or None. Codes and the licensed brand
    never reach rendered text (owner decision 10); the scrubber is the one
    authority on what a code looks like, so a text it would rewrite is a
    text that names one."""
    if not isinstance(text, str) or not text.strip():
        return "empty"
    from ..services import industry_labels
    if industry_labels.has_brand(text):
        return "gics_mark"
    if scope_type == "industry_group" and scope_key and scope_key in text:
        return "scope_code"
    if industry_labels.scrub_text(text) != text:
        return "taxonomy_code"
    return None


def trailing_sentences(text: str, max_chars: int) -> str:
    """The longest run of whole trailing sentences that fits `max_chars`.

    The postmortem prompt orders "what we said, what happened, why, what to
    remember next time", so the tail is the takeaway. Whole sentences only:
    a sentence that does not fit is dropped, never cut, and "" means none fit."""
    if not isinstance(text, str):
        return ""
    lines = [re.sub(r"^\s*(?:[-*#>]+\s*)+", "", ln).strip() for ln in text.splitlines()]
    flat = " ".join(" ".join(ln.split()) for ln in lines if ln)
    flat = flat.replace("**", "").replace("__", "")
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(flat) if s.strip()]
    out = ""
    for sentence in reversed(sentences):
        candidate = f"{sentence} {out}".strip() if out else sentence
        if len(candidate) > max_chars:
            break
        out = candidate
    return out


def clip(value: Any, limit: int = MEMO_FIELD_CHARS) -> str:
    text = value if isinstance(value, str) else ("" if value is None else json.dumps(value, default=str))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def memo_block(memo: dict[str, Any], *, include_rating: bool) -> dict[str, Any]:
    """Memo-time content for a learning prompt, each field clipped on its
    own so one oversized section can never crowd out the rest (or the
    hypotheses, which are serialized BEFORE this block)."""
    def summary(key: str) -> str:
        view = memo.get(key)
        return clip(view.get("summary", "") if isinstance(view, dict) else "")

    def risks(key: str) -> list[str]:
        out = []
        for r in (memo.get(key) or [])[:8]:
            if isinstance(r, dict):
                out.append(clip(f"{r.get('title', '')}: {r.get('detail', '')}", 300))
            elif isinstance(r, str):
                out.append(clip(r, 300))
        return out

    block: dict[str, Any] = {
        "ticker": memo.get("ticker"),
        "sector": memo.get("sector"),
        "generated_at": memo.get("generated_at"),
    }
    if include_rating:
        block["rating"] = memo.get("rating_label")
        block["confidence"] = memo.get("confidence_score")
    block.update({
        "thesis": clip(memo.get("one_sentence_thesis")),
        "pm_view": clip(memo.get("final_pm_view")),
        "mispricing_thesis": clip(memo.get("mispricing_thesis")),
        "key_risks": risks("key_risks"),
        "thesis_breakers": risks("thesis_breakers"),
        "catalysts": risks("catalysts"),
        "specialist_views": {
            k: summary(k) for k in (
                "sector_agent_view", "earnings_agent_view", "filing_agent_view",
                "valuation_agent_view", "comps_agent_view", "macro_sensitivity",
            )
        },
    })
    return block


# ---------------------------------------------------------------------------
# W2a: the presented memo
# ---------------------------------------------------------------------------

@dataclass
class MemoView:
    """The presented memo (hidden sections read "Unavailable in this
    version."), and whether it may teach: thesis and PM view available."""
    available: bool
    reason: str | None
    memo: dict[str, Any] | None


_REQUIRED_SECTIONS = ("one_sentence_thesis", "final_pm_view")


def present_for_learning(snap: MemoSnapshot, db: Session | None = None) -> MemoView:
    from ..services import memo_store
    from ..services.outcome_eligibility import PM_DEGRADED_AGENT
    try:
        presented = memo_store.present_snapshot(snap, db=db)
    except memo_store.StoredMemoUnreadable:
        return MemoView(False, "memo_unreadable", None)
    av = presented.section_availability or {}
    hidden = [k for k in _REQUIRED_SECTIONS if k in av and av[k].status == "unavailable"]
    if PM_DEGRADED_AGENT in (presented.degraded_agents or []):
        reason: str | None = "template_pm"
    elif hidden:
        reason = "unavailable:" + ",".join(hidden)
    else:
        reason = None
    data = presented.model_dump(mode="json")
    data.pop("section_availability", None)
    return MemoView(reason is None, reason, data)


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scopes:
    ticker: str
    group: str | None
    sector: str | None

    def read(self) -> list[tuple[str, str]]:
        out = [("company", self.ticker)]
        if self.group:
            out.append(("industry_group", self.group))
        if self.sector:
            out.append(("sector", self.sector))
        return out

    def peer(self) -> tuple[str, str] | None:
        """Where a peer lesson or pattern goes: the industry group when the
        classification is trustworthy, else the sector, else nowhere."""
        if self.group:
            return ("industry_group", self.group)
        if self.sector:
            return ("sector", self.sector)
        return None


def _group_code(ticker: str) -> str | None:
    """The industry group only for a mapped or conflict classification with
    a code: a fallback row knows its sector, not its group (the same rule
    `industry_classification.constituents` applies to cohorts)."""
    from ..services import industry_classification as ic
    row = ic.current_for([ticker]).get(ticker)
    if not row or row.get("state") not in (ic.STATE_MAPPED, ic.STATE_CONFLICT):
        return None
    code = str(row.get("industry_group_code") or "").strip()
    return code or None


def sector_key(raw: str | None) -> str | None:
    from ..finance.scorecard_spec import normalize_sector
    from ..memory.longterm import sector_slug
    canonical = normalize_sector(raw)
    return sector_slug(canonical) if canonical else None


def scopes_for(db: Session, ticker: str, sector: str | None = None) -> Scopes:
    """Company always; group when classified; sector from `Company.sector`
    (the universe row), falling back to the caller's sector string."""
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        raise ValueError("scopes_for needs a ticker")
    company_sector = db.execute(select(Company.sector).where(Company.ticker == symbol)).scalar_one_or_none()
    return Scopes(symbol, _group_code(symbol), sector_key(company_sector) or sector_key(sector))


# ---------------------------------------------------------------------------
# Item helpers
# ---------------------------------------------------------------------------

def _history(frm: str | None, to: str, reason: str, actor: str, now: datetime) -> dict[str, Any]:
    return {"at": now.isoformat(), "from": frm, "to": to, "reason": reason, "actor": actor}


def transition(item: LearningItem, to: str, *, reason: str, actor: str, now: datetime) -> bool:
    """Change status, appending to `status_history`. False when unchanged."""
    if to not in STATUSES:
        raise ValueError(f"unknown learning item status {to!r}")
    frm = item.status
    if frm == to:
        return False
    item.status = to
    # A new list, so the JSON column registers the change.
    item.status_history = [*(item.status_history or []), _history(frm, to, reason, actor, now)]
    return True


def _new_item(
    db: Session, *, kind: str, scope_type: str, scope_key: str, text: str, origin_kind: str,
    origin_ref: str, origin_ticker: str, now: datetime, detail: str = "",
    condition: str | None = None, observable: str | None = None,
    origin_snapshot_id: int | None = None, source_date: date | None = None,
    supersede_key: str | None = None, expires_at: datetime | None = None,
) -> LearningItem:
    if kind not in KINDS:
        raise ValueError(f"unknown learning item kind {kind!r}")
    if scope_type not in SCOPE_TYPES:
        raise ValueError(f"unknown learning scope_type {scope_type!r}")
    if origin_kind not in ORIGIN_KINDS:
        raise ValueError(f"unknown learning origin_kind {origin_kind!r}")
    if observable is not None and observable not in OBSERVABLES:
        raise ValueError(f"unknown hypothesis observable {observable!r}")
    item = LearningItem(
        kind=kind, scope_type=scope_type, scope_key=scope_key, text=text, detail=detail[:DETAIL_MAX],
        condition=condition, observable=observable, origin_kind=origin_kind, origin_ref=origin_ref,
        origin_ticker=origin_ticker, origin_snapshot_id=origin_snapshot_id, source_date=source_date,
        status="active", status_history=[_history(None, "active", f"created:{origin_kind}", "system", now)],
        supersede_key=supersede_key, expires_at=expires_at, created_at=now,
    )
    db.add(item)
    return item


def _item_exists(db: Session, origin_kind: str, origin_ref: str, scope_type: str, scope_key: str) -> bool:
    return db.execute(
        select(LearningItem.id).where(
            LearningItem.origin_kind == origin_kind, LearningItem.origin_ref == origin_ref,
            LearningItem.scope_type == scope_type, LearningItem.scope_key == scope_key,
        ).limit(1)
    ).first() is not None


def _snapshot_eligible(db: Session, snapshot_id: int) -> bool:
    return db.execute(
        outcome_eligibility.eligible_only(select(MemoSnapshot.id).where(MemoSnapshot.id == snapshot_id),
                                          MemoSnapshot.id)
    ).first() is not None


def posteriors(db: Session, item_ids: Iterable[int], *, now: datetime) -> dict[int, Posterior]:
    ids = sorted(set(item_ids))
    rows: dict[int, list[tuple[str, datetime]]] = {i: [] for i in ids}
    if ids:
        for item_id, verdict, observed_at in db.execute(
            select(LearningEvidence.item_id, LearningEvidence.verdict, LearningEvidence.observed_at)
            .where(LearningEvidence.item_id.in_(ids))
        ).all():
            rows[item_id].append((verdict, observed_at))
    return {i: posterior(ev, now=now) for i, ev in rows.items()}


def ref(item_id: int, kind: str = "lesson") -> str:
    return f"{'L' if kind == 'lesson' else 'O'}-{item_id}"


# ---------------------------------------------------------------------------
# Writer 1: the postmortem (the only source of live lessons)
# ---------------------------------------------------------------------------

def postmortem_context(snap: MemoSnapshot) -> MemoView:
    """The W2a view the strong postmortem call is given when writes are on.
    A presenter failure never blocks the postmortem itself: it only means
    this memo proposes no hypothesis."""
    try:
        return present_for_learning(snap)
    except Exception as exc:
        log.warning("learning: presenting %s#%s failed (%s)", snap.ticker, snap.id, type(exc).__name__)
        return MemoView(False, f"present_failed:{type(exc).__name__}", None)


def record_postmortem(
    *, snapshot_id: int, ticker: str, horizon_days: int, evaluated_at: datetime | None,
    llm_out: dict[str, Any] | None, view: MemoView | None, now: datetime | None = None,
) -> dict[str, Any]:
    """Write the lessons one written postmortem proposed. Own session, one
    transaction, never raises: returns `{status, reason, items, rejected,
    error_type}` with status in written | skipped | duplicate | failed |
    disabled."""
    result: dict[str, Any] = {"status": "skipped", "reason": None, "items": 0, "rejected": [], "error_type": None}
    if not _writes_on():
        result.update(status="disabled", reason="learning_ledger_writes=false")
        return result
    if horizon_days != LESSON_HORIZON:
        # Lessons are stated over, and judged on, LESSON_HORIZON only.
        result["reason"] = "horizon"
        return result
    if not isinstance(llm_out, dict):
        result["reason"] = "deterministic"
        return result
    if view is None or not view.available:
        result["reason"] = (view.reason if view is not None else None) or "memo_unavailable"
        return result
    now = now or _utcnow()
    try:
        with SessionLocal() as db:
            if not _snapshot_eligible(db, snapshot_id):
                result["reason"] = "ineligible"
                return result
            pm_id = db.execute(
                select(MemoPostmortem.id).where(
                    MemoPostmortem.memo_snapshot_id == snapshot_id,
                    MemoPostmortem.horizon_days == horizon_days,
                )
            ).scalar_one_or_none()
            if pm_id is None:
                result.update(status="failed", reason="postmortem_missing", error_type="LookupError")
                return result
            scopes = scopes_for(db, ticker, (view.memo or {}).get("sector"))
            plans: list[tuple[str, str, Hypothesis, str]] = []
            for field_name, origin_kind in (("hypothesis", "postmortem"), ("peer_hypothesis", "postmortem_sector")):
                hyp, why = parse_hypothesis(llm_out.get(field_name))
                if hyp is None:
                    if why != "empty":
                        result["rejected"].append(f"{field_name}:{why}")
                    continue
                target = ("company", scopes.ticker) if origin_kind == "postmortem" else scopes.peer()
                if target is None:
                    result["rejected"].append(f"{field_name}:scope_unresolved")
                    continue
                plans.append((target[0], target[1], hyp, origin_kind))
            source = (evaluated_at or now).date()
            existing = 0
            for scope_type, scope_key, hyp, origin_kind in plans:
                text = lesson_text(hyp, scope_type, horizon_days)
                why = text_rejection(text, scope_type=scope_type, scope_key=scope_key)
                if why is not None:
                    result["rejected"].append(f"{origin_kind}:{why}")
                    continue
                if _item_exists(db, origin_kind, str(pm_id), scope_type, scope_key):
                    existing += 1
                    continue
                _new_item(
                    db, kind="lesson", scope_type=scope_type, scope_key=scope_key, text=text,
                    detail=str(llm_out.get("lesson") or ""), condition=hyp.condition,
                    observable=hyp.observable, origin_kind=origin_kind, origin_ref=str(pm_id),
                    origin_ticker=scopes.ticker, origin_snapshot_id=snapshot_id, source_date=source, now=now,
                )
                result["items"] += 1
            db.commit()
    except IntegrityError:
        result.update(status="duplicate", reason="already_recorded", items=0)
        return result
    except Exception as exc:
        log.warning("learning: postmortem write failed for %s#%s (%s)", ticker, snapshot_id, type(exc).__name__)
        result.update(status="failed", reason="exception", error_type=type(exc).__name__, items=0)
        return result
    if result["items"]:
        result["status"] = "written"
    elif existing:
        # A re-run over the same postmortem: already learned, not a failure.
        result.update(status="duplicate", reason="already_recorded")
    else:
        result["reason"] = "rejected" if result["rejected"] else "no_hypothesis"
    return result


# ---------------------------------------------------------------------------
# Writer 2: filing observations (persists the existing `_llm_diff` output)
# ---------------------------------------------------------------------------

def _is_demo_accession(accession: str | None) -> bool:
    return "DEMO" in str(accession or "").upper()


def _pack_bullets(prefix: str, bullets: list[str], limit: int) -> str:
    """Whole bullets after `prefix` up to `limit`; the first is word-cut
    only if even it alone does not fit (a filing fact is still a fact)."""
    text = prefix
    added = 0
    for b in bullets:
        piece = " ".join(str(b).split())
        if not piece:
            continue
        candidate = f"{text}{'; ' if added else ''}{piece}"
        if len(candidate) > limit:
            if added == 0:
                room = limit - len(text) - 1
                cut = piece[:room].rsplit(" ", 1)[0] if room > 0 else ""
                return f"{text}{cut}…" if cut else ""
            break
        text = candidate
        added += 1
    return text if added else ""


def record_filing(
    *, filing: Any, bullets: list[str], sector_pattern: str = "", sector: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Company observation (+ a group or sector pattern) from one filing's
    LLM diff. Returns `{status, error_type, items, superseded, reason}`;
    never raises."""
    result: dict[str, Any] = {"status": "not_requested", "error_type": None, "items": 0,
                              "superseded": 0, "reason": None}
    if not _writes_on():
        result["status"] = "disabled"
        return result
    if settings.use_demo_data_only or _is_demo_accession(getattr(filing, "accession_number", None)):
        result.update(status="refused", reason="demo")
        return result
    clean = [str(b) for b in (bullets or []) if str(b).strip()]
    pattern = sector_pattern.strip() if isinstance(sector_pattern, str) else ""
    if not clean and not pattern:
        return result
    now = now or _utcnow()
    ticker = str(filing.ticker).upper()
    ftype = str(filing.filing_type or "filing")
    filed: date | None = filing.filing_date
    stamp = filed.isoformat() if filed else "undated"
    expires = datetime.combine(filed or now.date(), datetime.min.time()) + timedelta(days=OBS_TTL_DAYS)
    try:
        with SessionLocal() as db:
            scopes = scopes_for(db, ticker, sector)
            plans: list[tuple[str, str, str, str, str | None]] = []
            if clean:
                text = _pack_bullets(f"What's new in {ftype} filed {stamp}: ", clean, OBS_MAX)
                if text:
                    plans.append(("company", ticker, text, "filing_delta", f"filing_delta:{ticker}:{ftype}"))
            if pattern:
                target = scopes.peer()
                if target is None:
                    result["reason"] = "pattern_scope_unresolved"
                else:
                    text = " ".join(pattern.split())
                    text = text if len(text) <= OBS_MAX else trailing_sentences(text, OBS_MAX)
                    if text:
                        plans.append((target[0], target[1], text, "filing_pattern", None))
            rejected = []
            for scope_type, scope_key, text, origin_kind, supersede_key in plans:
                why = text_rejection(text, scope_type=scope_type, scope_key=scope_key)
                if why is not None:
                    rejected.append(f"{origin_kind}:{why}")
                    continue
                if _item_exists(db, origin_kind, str(filing.accession_number), scope_type, scope_key):
                    continue
                # Supersede by FILING date, not by processing order. EDGAR's
                # `filings.recent` is newest-first and a batch is post-passed
                # in that order, so the newest 10-Q routinely arrives before
                # the older ones in the same batch. Only priors filed on or
                # before this filing yield to it; if a newer one is already
                # active, this older observation is born superseded (with
                # its history), so the newest fact stays the live one.
                newer: LearningItem | None = None
                if supersede_key:
                    priors = db.execute(
                        select(LearningItem).where(LearningItem.supersede_key == supersede_key,
                                                   LearningItem.status == "active")
                    ).scalars().all()
                    newer = next((p for p in priors if _filed_key(p.source_date) > _filed_key(filed)), None)
                    if newer is None:
                        for prior in priors:
                            transition(prior, "superseded", reason=f"superseded_by:{filing.accession_number}",
                                       actor="system", now=now)
                            result["superseded"] += 1
                item = _new_item(
                    db, kind="observation", scope_type=scope_type, scope_key=scope_key, text=text,
                    origin_kind=origin_kind, origin_ref=str(filing.accession_number), origin_ticker=ticker,
                    source_date=filed, supersede_key=supersede_key, expires_at=expires, now=now,
                )
                if newer is not None:
                    transition(item, "superseded", reason=f"superseded_by:{newer.origin_ref}",
                               actor="system", now=now)
                    result["superseded"] += 1
                db.flush()
                result["items"] += 1
                if origin_kind == "filing_pattern":
                    result["superseded"] += _cap_observations(db, scope_type, scope_key, now=now)
            db.commit()
            if rejected:
                result["reason"] = ";".join(rejected)
    except IntegrityError:
        result.update(status="duplicate", items=0)
        return result
    except Exception as exc:
        log.warning("learning: filing write failed for %s %s (%s)", ticker, ftype, type(exc).__name__)
        result.update(status="failed", error_type=type(exc).__name__, items=0)
        return result
    result["status"] = "written" if result["items"] else ("rejected" if result["reason"] else "not_requested")
    return result


def _filed_key(filed: date | None) -> date:
    """Sort key for an observation's information date; undated is oldest."""
    return filed or date.min


def _cap_observations(db: Session, scope_type: str, scope_key: str, *, now: datetime) -> int:
    """Keep the newest MAX_OBS_PER_SCOPE by filing date, not by when they
    were processed (a newest-first batch would otherwise keep the oldest)."""
    rows = db.execute(
        select(LearningItem).where(
            LearningItem.kind == "observation", LearningItem.scope_type == scope_type,
            LearningItem.scope_key == scope_key, LearningItem.status == "active",
        )
    ).scalars().all()
    rows = sorted(rows, key=lambda i: (_filed_key(i.source_date), i.id), reverse=True)
    n = 0
    for old in rows[MAX_OBS_PER_SCOPE:]:
        n += transition(old, "superseded", reason="scope_cap", actor="system", now=now)
    return n


# ---------------------------------------------------------------------------
# Judging: decoupled from the postmortem, capped per night
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Candidate:
    item_id: int
    scope_key: str
    condition: str
    observable: str
    key: str


def _testable_lessons_exist(db: Session) -> bool:
    return db.execute(
        select(LearningItem.id).where(
            LearningItem.kind == "lesson", LearningItem.status == "active",
            LearningItem.observable.is_not(None), LearningItem.condition.is_not(None),
        ).limit(1)
    ).first() is not None


def hypotheses_to_judge(
    db: Session, *, scopes: Scopes, snapshot_id: int, generated_at: datetime, horizon_days: int,
    limit: int = JUDGE_MAX,
) -> list[_Candidate]:
    """Active, testable lessons in this memo's scopes that were knowable when
    it was written (no look-ahead), were not learned from it (no in-sample),
    were not already judged against THIS memo, and whose window has no
    held / failed / mixed row yet. An `irrelevant` row never closes the
    window (see `irrelevant_key`)."""
    pairs = scopes.read()
    stmt = select(LearningItem).where(
        LearningItem.kind == "lesson", LearningItem.status == "active",
        LearningItem.observable.is_not(None), LearningItem.condition.is_not(None),
        or_(*(and_(LearningItem.scope_type == t, LearningItem.scope_key == k) for t, k in pairs)),
        LearningItem.source_date.is_not(None), LearningItem.source_date < generated_at.date(),
        or_(LearningItem.origin_snapshot_id.is_(None), LearningItem.origin_snapshot_id != snapshot_id),
    )
    items = db.execute(stmt).scalars().all()
    if not items:
        return []
    keys = {i.id: independence_key(i.scope_key, horizon_days, generated_at) for i in items}
    closed: set[tuple[int, str]] = set()
    seen_here: set[int] = set()
    counts: Counter[int] = Counter()
    for item_id, key, verdict, memo_id in db.execute(
        select(LearningEvidence.item_id, LearningEvidence.independence_key,
               LearningEvidence.verdict, LearningEvidence.memo_snapshot_id)
        .where(LearningEvidence.item_id.in_(keys))
    ).all():
        if memo_id == snapshot_id:
            seen_here.add(item_id)
        if verdict != "irrelevant":
            closed.add((item_id, key))
            counts[item_id] += 1
    open_items = [i for i in items if (i.id, keys[i.id]) not in closed and i.id not in seen_here]
    open_items.sort(key=lambda i: (counts[i.id], i.created_at, i.id))
    return [
        _Candidate(i.id, i.scope_key, str(i.condition), str(i.observable), keys[i.id])
        for i in open_items[:limit]
    ]


JUDGE_SYSTEM = (
    "You check whether stated conditions applied to a company at the time a research memo was "
    "written. Use only the memo text given. You never predict or judge returns."
)


def judge_prompt(cands: list[_Candidate], memo: dict[str, Any]) -> str:
    """Hypotheses first, then the memo block (each field clipped on its own),
    so an oversized memo can never truncate a hypothesis out of the prompt.
    The observable and the outcome are deliberately absent: the judge sees
    only the condition, so its answer cannot lean on what happened."""
    payload = {
        "hypotheses": [{"id": ref(c.item_id), "condition": c.condition} for c in cands],
        "memo": memo_block(memo, include_rating=False),
    }
    return (
        "For each hypothesis, decide ONLY whether its condition applied to this company at the "
        "time this memo was written, using only the memo content. Answer \"yes\", \"no\" or "
        "\"unclear\" — use \"unclear\" when the memo does not show it either way. Output JSON:\n\n"
        "{ \"judgments\": [ { \"id\": \"L-1\", \"applies\": \"yes|no|unclear\", "
        "\"why\": \"<one short sentence citing the memo>\" } ] }\n\n"
        "Hypotheses and memo:\n" + json.dumps(payload, default=str)
    )


def _parse_judgments(out: dict[str, Any], offered: set[str]) -> dict[str, tuple[str, str]]:
    judged: dict[str, tuple[str, str]] = {}
    rows = out.get("judgments") if isinstance(out, dict) else None
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or "").strip()
        if rid not in offered or rid in judged:
            continue  # unknown ids are ignored; the first answer for an id wins
        applies = str(row.get("applies") or "").strip().lower()
        judged[rid] = (applies if applies in APPLIES else "unclear", clip(row.get("why"), RATIONALE_MAX))
    return judged


def _cheap_model() -> tuple[str, str]:
    from ..agents import llm
    provider = settings.active_llm_provider
    return provider, llm.resolve_role_model("cheap", provider)


def _projected_usd(prompt: str) -> float:
    from ..services.llm_metrics import estimate_cost_usd
    provider, model = _cheap_model()
    # chars / 3 over-counts English tokens (~4 chars each) on purpose: the
    # cap must stop BEFORE a call that would cross it.
    tokens_in = (len(prompt) + len(JUDGE_SYSTEM)) // 3 + 1
    return estimate_cost_usd(provider, model, tokens_in, JUDGE_MAX_TOKENS)


def _actual_usd(usage: dict[str, Any] | None) -> float | None:
    if not usage:
        return None
    from ..services.llm_metrics import estimate_cost_usd
    return estimate_cost_usd(
        str(usage.get("provider") or ""), str(usage.get("model") or ""),
        int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0),
        cache_read_tokens=int(usage.get("cache_read_tokens") or 0),
        cache_write_tokens=int(usage.get("cache_write_tokens") or 0),
    )


def judge_due(
    *, max_calls: int, max_usd: float, now: datetime | None = None,
    chat: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Judge every eligible memo old enough to have its LESSON_HORIZON
    outcome, when it has in-scope, knowable, out-of-sample hypotheses
    without evidence — whether or not the postmortem deduped it. Only the
    LESSON_HORIZON outcome is used (every memo with a 180d or 365d outcome
    has one), so it is one cheap-route call per memo, stopping BEFORE a call
    would exceed `max_calls` or `max_usd`. Memos whose thesis or PM view the
    W2a presenter hides write nothing (counted `unavailable`). Never raises
    per item: a failed call, or a hypothesis the judge did not answer, is
    retried on a later night."""
    report: dict[str, Any] = {
        "status": "ok", "outcomes": 0, "due": 0, "calls": 0, "usd": 0.0, "evidence": 0,
        "irrelevant": 0, "unanswered": 0, "duplicates": 0, "failed": 0, "failed_memos": [], "unavailable": 0,
        "deferred": 0, "retired": 0, "stopped_reason": None,
    }
    if not _writes_on():
        report["status"] = "off"
        return report
    if not settings.llm_enabled:
        # Demo data or no key: judging a live outcome needs a live model,
        # and a demo-mode run must never spend.
        report["status"] = "no_llm"
        return report
    now = now or _utcnow()
    from ..agents import llm
    chat = chat or llm.chat_json

    with SessionLocal() as db:
        if not _testable_lessons_exist(db):
            report["status"] = "no_hypotheses"
            return report
        rows = db.execute(
            outcome_eligibility.eligible_only(
                select(
                    MemoOutcome.memo_snapshot_id, MemoOutcome.ticker, MemoOutcome.horizon_days,
                    MemoOutcome.alpha, MemoOutcome.evaluated_at, MemoSnapshot.generated_at,
                    MemoOutcomeEligibility.sector,
                )
                .join(MemoSnapshot, MemoSnapshot.id == MemoOutcome.memo_snapshot_id)
                .join(MemoOutcomeEligibility, MemoOutcomeEligibility.memo_snapshot_id == MemoOutcome.memo_snapshot_id)
                .where(MemoOutcome.horizon_days == LESSON_HORIZON, MemoOutcome.alpha.is_not(None),
                       MemoSnapshot.generated_at.is_not(None)),
                MemoOutcome.memo_snapshot_id,
            ).order_by(MemoOutcome.evaluated_at, MemoOutcome.id)
        ).all()
    report["outcomes"] = len(rows)
    scope_cache: dict[str, Scopes] = {}
    for snapshot_id, ticker, horizon, alpha, evaluated_at, generated_at, sector in rows:
        # Phase 1 (short session): candidates and the presented memo. One
        # outcome's failure is counted and named; it never ends the night.
        try:
            with SessionLocal() as db:
                if ticker not in scope_cache:
                    scope_cache[ticker] = scopes_for(db, ticker, sector)
                cands = hypotheses_to_judge(db, scopes=scope_cache[ticker], snapshot_id=snapshot_id,
                                            generated_at=generated_at, horizon_days=horizon)
                if not cands:
                    continue
                report["due"] += 1
                if report["calls"] >= max_calls:
                    report["deferred"] += 1
                    report["stopped_reason"] = report["stopped_reason"] or "max_calls"
                    continue
                snap = db.get(MemoSnapshot, snapshot_id)
                # Evidence rows are ledger writes, and the accepted W2a rule
                # is that ledger writes are skipped when the thesis or PM
                # view is unavailable (template-filled or missing): the same
                # gate as `record_postmortem` and the sync.
                view = present_for_learning(snap, db) if snap is not None else MemoView(False, "missing", None)
                snap = None
        except Exception as exc:
            report["failed"] += 1
            report["failed_memos"].append({"ticker": ticker, "memo_snapshot_id": snapshot_id,
                                           "error_type": type(exc).__name__})
            continue
        if view.memo is None or not view.available:
            report["unavailable"] += 1
            continue
        prompt = judge_prompt(cands, view.memo)
        view = None  # type: ignore[assignment]  # drop the body before the call
        projected = _projected_usd(prompt)
        if report["usd"] + projected > max_usd:
            report["deferred"] += 1
            report["stopped_reason"] = report["stopped_reason"] or "max_usd"
            continue
        # Phase 2: the call, with no session held open across it.
        report["calls"] += 1
        llm.last_usage()  # consume any stale usage so the reading below is ours
        try:
            with llm.llm_call_context(agent_name="Learning Judge", route="cheap", feature="learning_judge"):
                out = chat(prompt, system=JUDGE_SYSTEM, route="cheap", max_tokens=JUDGE_MAX_TOKENS)
        except Exception as exc:
            out = None
            log.warning("learning judge failed for %s#%s (%s)", ticker, snapshot_id, type(exc).__name__)
        spent = _actual_usd(llm.last_usage())
        report["usd"] = round(report["usd"] + (spent if spent is not None else projected), 6)
        if not isinstance(out, dict):
            report["failed"] += 1
            report["failed_memos"].append({"ticker": ticker, "memo_snapshot_id": snapshot_id})
            continue
        judged = _parse_judgments(out, {ref(c.item_id) for c in cands})
        # Phase 3 (short session): evidence rows, then the retire rule.
        try:
            with SessionLocal() as db:
                touched: list[int] = []
                for c in cands:
                    answer = judged.get(ref(c.item_id))
                    if answer is None:
                        # Not answered: no row, so neither this memo nor the
                        # window is burned. The next night offers only the
                        # still-open hypotheses, so the retry converges.
                        report["unanswered"] += 1
                        continue
                    applies, why = answer
                    verdict = verdict_from_alpha(c.observable, alpha) if applies == "yes" else "irrelevant"
                    if verdict is None:
                        continue
                    db.add(LearningEvidence(
                        item_id=c.item_id, verdict=verdict, applies=applies, postmortem_id=None,
                        memo_snapshot_id=snapshot_id, ticker=ticker, horizon_days=horizon,
                        independence_key=c.key if verdict != "irrelevant" else irrelevant_key(c.key, snapshot_id),
                        alpha=alpha, rationale=why, observed_at=evaluated_at, created_at=now,
                    ))
                    touched.append(c.item_id)
                    report["evidence" if verdict != "irrelevant" else "irrelevant"] += 1
                db.commit()
                report["retired"] += _retire_failing(db, touched, now=now)
                db.commit()
        except IntegrityError:
            report["duplicates"] += len(cands)
        except Exception as exc:
            report["failed"] += 1
            report["failed_memos"].append({"ticker": ticker, "memo_snapshot_id": snapshot_id,
                                           "error_type": type(exc).__name__})
    if report["failed"]:
        report["status"] = "partial"
    return report


def _retire_failing(db: Session, item_ids: list[int], *, now: datetime) -> int:
    if not item_ids:
        return 0
    post = posteriors(db, item_ids, now=now)
    n = 0
    for item in db.execute(
        select(LearningItem).where(LearningItem.id.in_(item_ids), LearningItem.status == "active")
    ).scalars().all():
        if should_retire(post[item.id]):
            n += transition(item, "retired", reason="auto:posterior", actor="auto", now=now)
    return n


# ---------------------------------------------------------------------------
# Nightly: epoch, backfill, integrity (K1), expiry, capacity
# ---------------------------------------------------------------------------

def ensure_epoch(*, now: datetime | None = None) -> datetime:
    with SessionLocal() as db:
        at = control.ensure_epoch(db, now=now)
        db.commit()
    return at


def sync_from_postmortems(*, limit: int | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Backfill company lessons from postmortems written BEFORE the ledger
    epoch (DB-only, no LLM, idempotent).

    Only pre-epoch rows, and only those with no item of any postmortem
    origin: a live postmortem whose hypothesis was empty or rejected was
    deliberately not learned, and re-learning its narrative here would be
    exactly the duplication the epoch exists to prevent. These historical
    lessons are narrative (no observable), so they stay "untested" and are
    never judged; they are capped, suppressible and retire by capacity."""
    report: dict[str, Any] = {"backfilled": 0, "skipped": {}, "failed": 0}
    limit = SYNC_LIMIT if limit is None else limit   # read at call time, not bound at import
    if not _writes_on():
        report["status"] = "off"
        return report
    now = now or _utcnow()
    skipped: Counter[str] = Counter()
    with SessionLocal() as db:
        epoch = control.ensure_epoch(db, now=now)
        db.commit()
        learned = exists(
            select(LearningItem.id).where(
                LearningItem.origin_kind.in_(POSTMORTEM_ORIGINS),
                LearningItem.origin_ref == cast(MemoPostmortem.id, String),
            )
        )
        rows = db.execute(
            outcome_eligibility.eligible_only(
                select(MemoPostmortem.id, MemoPostmortem.memo_snapshot_id, MemoPostmortem.ticker,
                       MemoPostmortem.lesson, MemoPostmortem.created_at)
                .where(MemoPostmortem.horizon_days >= LESSON_MIN_HORIZON,
                       MemoPostmortem.created_at < epoch, ~learned),
                MemoPostmortem.memo_snapshot_id,
            ).order_by(MemoPostmortem.id)
        ).all()
    for pm_id, snapshot_id, ticker, lesson, created_at in rows:
        if report["backfilled"] >= limit:
            skipped["pass_limit"] += 1
            continue
        text = lesson if isinstance(lesson, str) else ""
        if DETERMINISTIC_LESSON_RE.match(text.strip()):
            skipped["deterministic"] += 1
            continue
        takeaway = trailing_sentences(text, LESSON_MAX)
        if not takeaway:
            skipped["no_takeaway"] += 1
            continue
        why = text_rejection(takeaway, scope_type="company", scope_key=str(ticker).upper())
        if why is not None:
            skipped[f"rejected:{why}"] += 1
            continue
        try:
            with SessionLocal() as db:
                snap = db.get(MemoSnapshot, snapshot_id)
                view = present_for_learning(snap, db) if snap is not None else MemoView(False, "missing", None)
                snap = None
                if not view.available:
                    skipped[str(view.reason)] += 1
                    continue
                view = None  # type: ignore[assignment]
                _new_item(
                    db, kind="lesson", scope_type="company", scope_key=str(ticker).upper(), text=takeaway,
                    detail=text, origin_kind="postmortem_backfill", origin_ref=str(pm_id),
                    origin_ticker=str(ticker).upper(), origin_snapshot_id=snapshot_id,
                    source_date=created_at.date() if created_at else None, now=now,
                )
                db.commit()
                report["backfilled"] += 1
        except IntegrityError:
            skipped["duplicate"] += 1
        except Exception as exc:
            report["failed"] += 1
            log.warning("learning: backfill of postmortem %s failed (%s)", pm_id, type(exc).__name__)
    report["skipped"] = dict(skipped)
    return report


def integrity_check(*, now: datetime | None = None) -> dict[str, Any]:
    """K1: an outcome-derived item whose origin snapshot is no longer
    eligible (or unclassified) is retired; if one reached an inject render
    in the last 30 days, the DB mode is demoted to shadow automatically.
    Filing observations have no origin snapshot and are out of scope."""
    now = now or _utcnow()
    out: dict[str, Any] = {"retired": [], "demoted": False}
    with SessionLocal() as db:
        bad = db.execute(
            select(LearningItem).where(
                LearningItem.status == "active",
                LearningItem.origin_kind.in_(POSTMORTEM_ORIGINS),
                LearningItem.origin_snapshot_id.is_not(None),
                ~outcome_eligibility.eligible_exists(LearningItem.origin_snapshot_id),
            )
        ).scalars().all()
        for item in bad:
            transition(item, "retired", reason="origin_ineligible", actor="auto", now=now)
            out["retired"].append(item.id)
        if out["retired"]:
            ids = set(out["retired"])
            injected: set[int] = set()
            for (items,) in db.execute(
                select(LearningRender.items).where(
                    LearningRender.mode == "inject",
                    LearningRender.created_at >= now - timedelta(days=CONTAMINATION_LOOKBACK_DAYS),
                )
            ).all():
                for entry in items or []:
                    if isinstance(entry, dict) and entry.get("item_id") in ids:
                        injected.add(int(entry["item_id"]))
            if injected:
                out["demoted"] = control.demote_if_injecting(
                    db, reason=f"contamination: {sorted(injected)}", now=now,
                )
        db.commit()
    if out["demoted"]:
        control._cache_clear()
    return out


def expire_and_cap(*, now: datetime | None = None) -> dict[str, int]:
    """Observations past `expires_at` retire; a scope over 30 active lessons
    retires its oldest UNTESTED ones (tested lessons earned their place)."""
    now = now or _utcnow()
    out = {"expired": 0, "capacity": 0}
    with SessionLocal() as db:
        for item in db.execute(
            select(LearningItem).where(
                LearningItem.kind == "observation", LearningItem.status == "active",
                LearningItem.expires_at.is_not(None), LearningItem.expires_at < now,
            )
        ).scalars().all():
            out["expired"] += transition(item, "retired", reason="expired", actor="auto", now=now)
        crowded = db.execute(
            select(LearningItem.scope_type, LearningItem.scope_key)
            .where(LearningItem.kind == "lesson", LearningItem.status == "active")
            .group_by(LearningItem.scope_type, LearningItem.scope_key)
            .having(func.count(LearningItem.id) > MAX_ACTIVE_LESSONS_PER_SCOPE)
        ).all()
        for scope_type, scope_key in crowded:
            lessons = db.execute(
                select(LearningItem).where(
                    LearningItem.kind == "lesson", LearningItem.status == "active",
                    LearningItem.scope_type == scope_type, LearningItem.scope_key == scope_key,
                ).order_by(LearningItem.created_at, LearningItem.id)
            ).scalars().all()
            post = posteriors(db, [i.id for i in lessons], now=now)
            excess = len(lessons) - MAX_ACTIVE_LESSONS_PER_SCOPE
            for item in lessons:
                if excess <= 0:
                    break
                if post[item.id].stance == "untested":
                    out["capacity"] += transition(item, "retired", reason="capacity", actor="auto", now=now)
                    excess -= 1
        db.commit()
    return out


def nightly(*, now: datetime | None = None) -> dict[str, Any]:
    """The ledger's nightly housekeeping, run by `postmortem_loop` after
    both horizons. DB-only; the capped judge is a separate step."""
    if not _writes_on():
        return {"status": "off"}
    now = now or _utcnow()
    sync = sync_from_postmortems(now=now)
    integrity = integrity_check(now=now)
    housekeeping = expire_and_cap(now=now)
    return {
        "status": "ok",
        "backfilled": sync["backfilled"],
        "backfill_skipped": sync["skipped"],
        "backfill_failed": sync["failed"],
        "retired": len(integrity["retired"]),
        "retired_ids": integrity["retired"],
        "demoted": integrity["demoted"],
        "expired": housekeeping["expired"],
        "capacity": housekeeping["capacity"],
    }


# ---------------------------------------------------------------------------
# Admin helpers (read-only unless named otherwise)
# ---------------------------------------------------------------------------

class ItemSuperseded(Exception):
    """A superseded item's replacement is the live one; reviving it would
    put two observations of one filing type in play."""


ADMIN_STATUSES = ("active", "suppressed", "retired")


def set_status(item_id: int, status: str, *, reason: str, actor: str = "admin",
               now: datetime | None = None) -> dict[str, Any]:
    """The per-lesson kill switch. Raises LookupError (unknown item),
    ItemSuperseded, or ValueError (bad status / empty reason)."""
    if status not in ADMIN_STATUSES:
        raise ValueError(f"status must be one of {ADMIN_STATUSES}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a reason is required")
    now = now or _utcnow()
    with SessionLocal() as db:
        item = db.get(LearningItem, item_id)
        if item is None:
            raise LookupError(item_id)
        if item.status == "superseded":
            raise ItemSuperseded(item_id)
        transition(item, status, reason=reason.strip()[:500], actor=actor, now=now)
        db.commit()
        return item_dict(item, None)


def item_dict(item: LearningItem, post: Posterior | None) -> dict[str, Any]:
    return {
        "id": item.id, "ref": ref(item.id, item.kind), "kind": item.kind, "scope_type": item.scope_type,
        "scope_key": item.scope_key, "text": item.text, "detail": item.detail,
        "condition": item.condition, "observable": item.observable,
        "origin_kind": item.origin_kind, "origin_ref": item.origin_ref, "origin_ticker": item.origin_ticker,
        "origin_snapshot_id": item.origin_snapshot_id,
        "source_date": item.source_date.isoformat() if item.source_date else None,
        "status": item.status, "status_history": list(item.status_history or []),
        "expires_at": item.expires_at.isoformat() if item.expires_at else None,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "posterior": post.as_dict() if post is not None else None,
    }


PREVIEW_CONSUMERS = tuple(control.RENDER_BUDGETS)


def preview(ticker: str, *, consumer: str = "pm_memo", now: datetime | None = None) -> dict[str, Any]:
    """What the ledger holds for `ticker` in `consumer`'s scopes, ranked the
    way a block would consider them, with posteriors. Reads only — it writes
    no render row — so the owner can evaluate priors without a memo run.
    (The bounded text block itself is S19's renderer.)"""
    if consumer not in PREVIEW_CONSUMERS:
        raise ValueError(f"consumer must be one of {PREVIEW_CONSUMERS}")
    now = now or _utcnow()
    tier = {"supported": 0, "untested": 1, "contested": 2, "weakened": 3}
    weight = {"company": 1.0, "industry_group": 0.8, "sector": 0.6}
    with SessionLocal() as db:
        scopes = scopes_for(db, ticker)
        pairs = scopes.read()
        items = db.execute(
            select(LearningItem).where(
                LearningItem.status == "active",
                or_(*(and_(LearningItem.scope_type == t, LearningItem.scope_key == k) for t, k in pairs)),
            )
        ).scalars().all()
        post = posteriors(db, [i.id for i in items if i.kind == "lesson"], now=now)
        lessons = sorted(
            (i for i in items if i.kind == "lesson"),
            key=lambda i: (tier[post[i.id].stance], -weight[i.scope_type],
                           -(i.source_date.toordinal() if i.source_date else 0), -i.id),
        )
        observations = sorted((i for i in items if i.kind == "observation"),
                              key=lambda i: (-(i.source_date.toordinal() if i.source_date else 0), -i.id))
        return {
            "ticker": scopes.ticker, "consumer": consumer,
            "scopes": [{"scope_type": t, "scope_key": k} for t, k in pairs],
            "lessons": [item_dict(i, post[i.id]) for i in lessons],
            "observations": [item_dict(i, None) for i in observations],
        }
