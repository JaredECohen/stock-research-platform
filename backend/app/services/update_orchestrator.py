"""Wave 5B — update orchestrator.

Wires monitoring loops to memo refresh logic:

- New filing/earnings observed by EDGAR poller → enqueue a
  `full_reanalysis` job on the durable `regen_jobs` queue
  (`services/regen_worker.py`) — the same queue POST /analyze uses, so
  scheduler-driven and user-driven regens share one worker thread (one
  memo in memory at a time; this service has a history of OOM kills
  during regen, see render.yaml) and one failure-telemetry trail. The
  queue coalesces per ticker, so an EDGAR event landing while a user
  regen is queued/running attaches to that job instead of doubling up.
  Enqueue-and-forget: handlers return the job id, not the finished
  memo — rating/outcome lives in the job row once the worker drains it.
- Material/breaking news from the news loop → call `news_impact_agent`
  on the latest memo. If it returns `material=true`, build an
  `incremental_patch` snapshot inheriting from the prior version with
  the rating / confidence / risks patched. Critic is skipped on patches
  (locked in MASTER_PLAN); `revision_log` carries `critic_skipped: true`.

Patch frequency cap: max 2 patches per ticker per day; further material
events queue but don't fire until the next refresh.

Why a separate service vs. inlining in news_loop:
- News and EDGAR both produce events that may need a memo refresh —
  one orchestrator owns the policy (full vs. patch, dedup, throttle).
- Easier to test in isolation: build a fake event stream, assert the
  state transitions.
- Future schedulers (queue worker, etc.) can reuse the same entry points
  without re-implementing the policy.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from typing import Any

from ..schemas import NewsAlert, StockMemoOut

log = logging.getLogger(__name__)


# Locked policy: max patches per ticker per UTC day.
MAX_PATCHES_PER_DAY = 2

# How recently a memo must have been generated/viewed for new
# filings/transcripts to trigger automatic regeneration. Outside this
# window, the polling jobs still ingest + persist the raw data, but
# memo regen waits for a user request. Configurable via env;
# essentially the cost ceiling on the universe expansion.
AUTO_REGEN_RECENCY_DAYS = 30

# News patch path (FIX-017). One story is assessed once per window: the
# same headline used to be re-assessed on every 2-hourly fetch until the
# daily cap, and GOOG was patched 4 times on one headline with confidence
# stepping up each time. The same window is the age gate: an alert older
# than this is not news the memo is waiting on.
NEWS_ASSESSED_WINDOW = timedelta(hours=72)
_NEWS_ASSESSED_KIND = "news_assessed"

# " - Reuters", " | Bloomberg", " — The Wall Street Journal": a trailing
# publisher tag of at most five words after a spaced dash or bar.
_PUBLISHER_SUFFIX_RE = re.compile(r"\s+[-–—|]\s+(?:\S+\s+){0,4}\S+\s*$")

# Per-ticker FIFO queue (singleton). Largely superseded for
# full_reanalysis by the durable `regen_jobs` queue — kept because the
# /api/admin/update-queue inspector reads it and future in-process
# event types may still want it.
_QUEUES: dict[str, deque[dict[str, Any]]] = defaultdict(deque)


def _patch_count_today(ticker: str) -> int:
    """Count `incremental_patch` snapshots created today (UTC) for `ticker`.

    `MemoSnapshot.generated_at` is `datetime.utcnow()` so we compare in
    UTC — local-tz `date.today()` would mis-bucket snapshots written
    near midnight UTC.
    """
    from . import memo_store
    today = datetime.utcnow().date()
    history = memo_store.memo_history(ticker, limit=20)
    n = 0
    for snap in history:
        if snap.trigger != "incremental_patch":
            continue
        gen = snap.generated_at
        if isinstance(gen, datetime):
            gen = gen.date()
        if gen == today:
            n += 1
    return n


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def should_auto_regen(
    ticker: str, *, window_days: int = AUTO_REGEN_RECENCY_DAYS,
) -> dict[str, Any]:
    """Decide whether a polling event should trigger memo regeneration.

    Two conditions, OR'd together:
      1. `Company.auto_update_memo` is True (explicit pin — top mega-caps
         by default, user-curable via the admin endpoint).
      2. A memo exists for this ticker and was generated within the
         last `window_days` days (proxy for "user is actively watching").

    Returns a dict with the decision + the reason so callers can log
    without re-implementing the logic. Never raises — DB failure
    returns `{should: False, reason: 'db_error'}` (fail-safe — better
    to skip a regen than to over-spend).
    """
    ticker = ticker.upper()
    try:
        from ..database import SessionLocal
        from ..models import Company
        from . import memo_store
        with SessionLocal() as db:
            company = db.get(Company, ticker)
            if company is not None and company.auto_update_memo:
                return {"should": True, "reason": "auto_update_pinned"}
        snap = memo_store.latest_memo(ticker)
        if snap is None or snap.generated_at is None:
            return {"should": False, "reason": "no_memo_on_file"}
        gen_at = snap.generated_at
        if isinstance(gen_at, datetime):
            age_days = (datetime.utcnow() - gen_at).total_seconds() / 86400.0
        else:
            return {"should": False, "reason": "no_generated_at"}
        if age_days <= window_days:
            return {
                "should": True,
                "reason": f"within_window_{int(age_days)}d",
            }
        return {
            "should": False,
            "reason": f"stale_memo_{int(age_days)}d_old",
        }
    except Exception as exc:  # pragma: no cover — fail-safe to no-regen
        from ..agents.log_safety import log_safely  # lazy: agents imports services
        log_safely(log, f"should_auto_regen failed for {ticker}", exc)
        return {"should": False, "reason": "db_error"}


# `reason` values from `should_auto_regen` that mean the gate *crashed*
# rather than decided. The event handlers report these as `kind="gate_error"`
# instead of `kind="skipped"`, so a poller summarising its run cannot fold a
# dead database into "nothing was due" (RP-001 class (c), kept fail-safe).
GATE_ERROR_REASONS = frozenset({"db_error"})


def _skip_kind(decision: dict[str, Any]) -> str:
    return "gate_error" if decision.get("reason") in GATE_ERROR_REASONS else "skipped"


def _persist_raw_data_only(ticker: str) -> dict[str, Any]:
    """Ingest fresh filings + transcripts into FilingDoc / EarningsTranscript
    + the vector store, WITHOUT running the LLM memo synthesis.

    Used by the polling event handlers when the auto-regen gate
    decides "no memo yet" — we still want the raw data in the DB
    + indexed for future on-demand retrieval. Cheap (no LLM calls
    on this path; filing_memory.post_pass DOES use the LLM for the
    diff bullets, but that's a separate gate inside that function).
    """
    counts: dict[str, Any] = {"filings": 0, "transcripts": 0}
    committed = False
    try:
        from ..database import SessionLocal
        from . import history_service
        from .data_service import get_data_service
        ds = get_data_service()
        filings = ds.get_filings(ticker) or []
        transcripts = ds.get_earnings_transcripts(ticker) or []
        filing_ids: list[int] = []
        transcript_ids: list[int] = []
        with SessionLocal() as db:
            history_service._ensure_tables(db)
            counts["filings"] = history_service._ingest_filings(
                db, ticker, filings, post_pass_ids=filing_ids,
            )
            counts["transcripts"] = history_service._ingest_transcripts(
                db, ticker, transcripts, post_pass_ids=transcript_ids,
            )
            db.commit()
            committed = True
        failures = history_service.run_ingest_post_passes(filing_ids, transcript_ids)
        if failures:
            counts["post_pass_failures"] = failures
        fetch_failures = history_service.filing_fetch_failures(ticker, filings)
        if fetch_failures:
            counts["filing_fetch_failures"] = fetch_failures
            counts["persist_error"] = {
                "ticker": ticker, "stage": "filing_fetch", "error_type": "IncompleteFilingBody",
            }
            log.warning("filing fetch failures count=%d: %s", len(fetch_failures),
                        history_service.filing_fetch_failure_note(fetch_failures))
        truncated = history_service.truncated_filing_sources(ticker, filings)
        if truncated:
            counts["truncated_filings"] = truncated
            log.warning("bounded filing sources count=%d: %s",
                        len(truncated), history_service.truncated_filing_note(truncated))
    except Exception as exc:  # report failures without exposing provider request details
        if not committed:
            counts = {"filings": 0, "transcripts": 0}
        error = {
            "ticker": ticker,
            "stage": "post_pass_dispatch" if committed else "raw_ingest",
            "error_type": type(exc).__name__,
        }
        counts["persist_error"] = error
        log.warning("raw persistence failed ticker=%s stage=%s error_type=%s",
                    ticker, error["stage"], error["error_type"])
    return counts


def on_transcript_event(ticker: str, *, period: str = "") -> dict[str, Any]:
    """A new earnings transcript landed → maybe regenerate the memo.

    Two-phase:
      1. Always persist+index the raw transcript (so future memo runs
         can retrieve it). Cheap; no LLM call beyond the embed batch.
      2. Conditionally enqueue a full memo regen on the durable
         `regen_jobs` queue, gated by `should_auto_regen` (auto_update
         pin + recency window). Enqueue-and-forget: the single worker
         thread runs the memo; outcome telemetry lives in the job row.
    """
    ticker = ticker.upper()
    persist_counts = _persist_raw_data_only(ticker)
    if persist_counts.get("persist_error"):
        return {"ticker": ticker, "kind": "persist_error", "persisted": persist_counts}
    decision = should_auto_regen(ticker)
    if not decision["should"]:
        return {
            "ticker": ticker, "period": period, "kind": _skip_kind(decision),
            "reason": decision["reason"],
            "persisted": persist_counts,
        }

    from . import regen_worker
    job, created = regen_worker.enqueue(ticker, source="transcript_event")
    log.info(
        "transcript event for %s → regen job %d (%s)",
        ticker, job["id"], "created" if created else "coalesced",
    )
    return {
        "ticker": ticker, "period": period,
        "kind": "full_reanalysis",
        "job_id": job["id"], "job_created": created,
        "trigger_reason": decision["reason"],
        "persisted": persist_counts,
    }


def on_filing_event(ticker: str, *, source: str = "filing_event") -> dict[str, Any]:
    """A new filing was observed → persist + index, then maybe memo regen.

    Two-phase:
      1. Always persist+index the raw filing (so future on-demand memo
         runs can retrieve it). Cheap; ~$0.02 in embed cost per 10-K.
      2. Conditionally enqueue a full memo regen on the durable
         `regen_jobs` queue, gated by `should_auto_regen` (auto_update
         pin + recency window). Enqueue-and-forget: scheduler-driven
         regens share the single worker thread (and its one-at-a-time
         memory ceiling) with user-triggered POST /analyze jobs, and
         coalesce onto any job already queued/running for the ticker.

    `source` flows into the job's telemetry waypoints — the regime-shift
    and admin-rerun callers override it so `regen_jobs` shows what
    actually triggered each regen.
    """
    ticker = ticker.upper()
    persist_counts = _persist_raw_data_only(ticker)
    if persist_counts.get("persist_error"):
        return {"ticker": ticker, "kind": "persist_error", "persisted": persist_counts}
    decision = should_auto_regen(ticker)
    if not decision["should"]:
        return {
            "ticker": ticker, "kind": _skip_kind(decision),
            "reason": decision["reason"],
            "persisted": persist_counts,
        }

    from . import regen_worker
    job, created = regen_worker.enqueue(ticker, source=source)
    log.info(
        "filing event for %s → regen job %d (%s)",
        ticker, job["id"], "created" if created else "coalesced",
    )
    return {
        "ticker": ticker, "kind": "full_reanalysis",
        "job_id": job["id"], "job_created": created,
        "trigger_reason": decision["reason"],
        "persisted": persist_counts,
    }


def news_fingerprint(ticker: str, title: str) -> str:
    """Stable id for "the same story" about `ticker`.

    Title-based: lowercased, a trailing publisher tag stripped, punctuation
    removed, whitespace collapsed. So "Southern Co signs deal with Google -
    Reuters" and "southern co. signs deal with google" are one story. A
    genuinely reworded duplicate is missed; the window bounds that cost.
    """
    t = _PUBLISHER_SUFFIX_RE.sub("", (title or "").strip())
    t = re.sub(r"[^\w\s]+", "", t.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return hashlib.sha1(f"{ticker.upper()}|{t}".encode(), usedforsecurity=False).hexdigest()


def _assessed_subject(ticker: str) -> str:
    return f"news_assessed:{ticker.upper()}"


def _assessed_map(ticker: str, *, now: datetime) -> dict[str, dict[str, Any]]:
    """fingerprint → {verdict, at} for stories assessed within the window.

    ONE row per ticker, rewritten whole: `snapshot_gc` never deletes the
    newest row per (subject, kind), so a subject per alert would grow
    without bound.
    """
    from ..cache import cache_get
    snap = cache_get(_assessed_subject(ticker), _NEWS_ASSESSED_KIND)
    entries = (snap.payload or {}).get("assessed") if snap is not None and isinstance(snap.payload, dict) else None
    if not isinstance(entries, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for fp, entry in entries.items():
        try:
            at = datetime.fromisoformat(str(entry.get("at")))
        except (AttributeError, TypeError, ValueError):
            continue
        if now - at <= NEWS_ASSESSED_WINDOW:
            out[str(fp)] = entry
    return out


def _remember_assessment(ticker: str, fingerprint: str, verdict: str, *, now: datetime) -> None:
    from ..cache import cache_put
    entries = _assessed_map(ticker, now=now)
    entries[fingerprint] = {"verdict": verdict, "at": now.isoformat()}
    cache_put(
        _assessed_subject(ticker), _NEWS_ASSESSED_KIND,
        payload={"assessed": entries},
        sources_used=[f"news_assessed:{ticker.upper()}"],
        generated_by="update_orchestrator", cost_tokens=0,
        ttl_seconds=int(NEWS_ASSESSED_WINDOW.total_seconds()),
    )


def _news_age_gate(alert: NewsAlert, memo_generated_at: datetime | None, *, now: datetime) -> str | None:
    """Why an alert is too old to assess against this memo, or None.

    - older than the memo: the full run that wrote it could already see
      the story (patches copy the full run's `generated_at`);
    - older than the window: not news any more; the 60-day Gemini prompt
      used to re-surface weeks-old "material" stories on every pass;
    - a Gemini alert with no readable date: the date is model-written, so
      "unknown" cannot be taken for "fresh". Provider rows with no date are
      publisher feed items and keep today's leniency.
    """
    from ..agents.news_agent import parse_published_at
    published = parse_published_at(alert.published_at, now=now)
    if published is None:
        return "undated_model_alert" if alert.source == "gemini" else None
    if now - published > NEWS_ASSESSED_WINDOW:
        return "stale_alert"
    if memo_generated_at is not None:
        generated = memo_generated_at
        if generated.tzinfo is not None:
            generated = generated.astimezone(UTC).replace(tzinfo=None)
        if published < generated:
            return "older_than_memo"
    return None


def on_news_alert(ticker: str, alert: NewsAlert) -> dict[str, Any]:
    """A material/breaking news alert came in → run news_impact_agent
    against the latest memo, persist a patch if material.

    Gates, in order: the daily patch cap, a prior memo, the alert's age
    (`_news_age_gate`), and whether the story was already assessed within
    `NEWS_ASSESSED_WINDOW`. A verdict is remembered after `not_material`
    and after a published patch, never after `assessment_error`, so a
    crashed assessment is retried on the next pass.

    Returns `{patched: bool, version: int|None, reason: str}` so callers
    can log what happened.
    """
    ticker = ticker.upper()
    # Frequency cap.
    if _patch_count_today(ticker) >= MAX_PATCHES_PER_DAY:
        return {"patched": False, "ticker": ticker, "reason": "daily_cap_reached"}

    from . import memo_store
    snap = memo_store.latest_memo(ticker)
    if snap is None:
        return {"patched": False, "ticker": ticker, "reason": "no_prior_memo"}
    prior_memo = memo_store.memo_to_pydantic(snap)

    now = _utcnow()
    too_old = _news_age_gate(alert, prior_memo.generated_at, now=now)
    if too_old:
        return {"patched": False, "ticker": ticker, "reason": too_old}
    fingerprint = news_fingerprint(ticker, alert.title)
    if fingerprint in _assessed_map(ticker, now=now):
        return {"patched": False, "ticker": ticker, "reason": "already_assessed"}

    from ..agents.news_impact_agent import apply_patch, assess
    assessment = assess(prior_memo, alert)
    if assessment.get("error"):
        # (b) RP-001: the agent crashed or got nothing back, so the alert
        # was never judged — reporting it as "not material" would read as a
        # verdict. The memo is left alone either way (safe side).
        log.warning(
            "news impact assessment failed for %s (%s); alert %r left unassessed",
            ticker, assessment["error"], alert.title[:80],
        )
        return {
            "patched": False, "ticker": ticker, "reason": "assessment_error",
            "error": assessment["error"],
        }
    if not assessment.get("material"):
        _remember_assessment(ticker, fingerprint, "not_material", now=now)
        return {"patched": False, "ticker": ticker, "reason": "not_material"}

    patched_memo: StockMemoOut = apply_patch(prior_memo, assessment["patch"])
    # W2b: a patch runs neither the PM nor the critic, so it can neither
    # state a valuation-divergence reason nor earn confidence. Re-apply 7(b)
    # against the STORED evidence verdict and hold confidence at or below
    # the last full run's earned value. A patch runs no number check
    # either (7(a)): every field it rewrote or appended is labelled
    # "figures not source-checked". A no-op on memos written before these
    # guards (no `quality`): they behave exactly as before.
    from ..agents.memo_quality import enforce_after_patch
    from ..config import settings
    patched_memo, guard = enforce_after_patch(
        prior_memo, patched_memo, assessment["patch"].keys(),
        enforce=settings.rating_reconciliation_mode != "record",
    )
    if guard.rating_downgraded or guard.confidence_clamped:
        log.info(
            "patch guard %s: rating_downgraded=%s confidence_clamped=%s",
            ticker, guard.rating_downgraded, guard.confidence_clamped,
        )
    revision_log = [
        {
            "version": (snap.version or 0) + 1,
            "trigger": "incremental_patch",
            "at": datetime.utcnow().isoformat(),
            "parent_version": snap.version,
            "fields_patched": sorted(assessment["patch"].keys()),
            "rationales": assessment.get("rationales") or {},
            "delta_summary": assessment.get("delta_summary", ""),
            # Locked decision in MASTER_PLAN: critic doesn't run on patches.
            "critic_skipped": True,
            "quality_guard": {
                "rating_downgraded": guard.rating_downgraded,
                "confidence_clamped": guard.confidence_clamped,
                "fields_unchecked": guard.unchecked_fields,
            },
            "alert": {
                "title": alert.title, "severity": alert.severity,
                "source": alert.source, "published_at": alert.published_at,
            },
        }
    ]
    new_snap = memo_store.save_memo(
        patched_memo,
        trigger="incremental_patch",
        parent_version=snap.version,
        revision_log=revision_log,
    )
    # After the publish, not before: a patch that failed to save is retried.
    _remember_assessment(ticker, fingerprint, "patched", now=now)
    return {
        "patched": True,
        "ticker": ticker,
        "version": new_snap.version,
        "delta_summary": assessment.get("delta_summary", ""),
    }


def _utcnow() -> datetime:
    """Clock seam for the scorecard review cap and the news-alert age and
    dedup windows (tests pin the UTC day)."""
    return datetime.utcnow()


def _scorecard_regens_today(db: Any, *, now: datetime) -> int:
    """Regen jobs this feature enqueued since 00:00 UTC today.

    Counted from the durable queue's enqueue waypoint (`progress[0].source
    == "scorecard_disagreement"`) rather than from a module counter, so the
    cap holds across worker restarts and is visible to both processes.
    The day's rows are a handful at most, so the JSON filter runs in
    Python instead of a dialect-specific JSON query.
    """
    from ..models import RegenJob
    start = datetime.combine(now.date(), datetime.min.time())
    rows = db.query(RegenJob.progress).filter(RegenJob.enqueued_at >= start).all()
    n = 0
    for (progress,) in rows:
        first = (progress or [{}])[0] if isinstance(progress, list) and progress else {}
        if isinstance(first, dict) and first.get("source") == "scorecard_disagreement":
            n += 1
    return n


def handle_scorecard_disagreements(cap: int | None = None) -> dict[str, Any]:
    """Phase 6 — queue a deep-research review regen for open MATERIAL
    scorecard disagreements, behind `ENABLE_SCORECARD_DISAGREEMENT_REGEN`.

    This is the ONLY path by which the scorecard can spend LLM money, so:
    the flag defaults off; at most `cap` (default
    `scorecard_disagreement_regen_daily_cap`, 3) regens per UTC day,
    counted from the durable queue; each ticker must pass
    `should_auto_regen` (pinned or recently viewed); and a (ticker,
    version, as_of) that already has a review — queued or written — is
    never re-queued, so a memo that still disagrees after answering its
    seed question does not loop. A queued row flips to `queued_review`
    and the regen (`source="scorecard_disagreement"`) picks up its seed
    question through `scorecard_context.pending_seed_questions`.

    Called from the worker's scorecard loop after scoring; never from a
    page request. Returns counts for the loop note; never raises.
    """
    from ..config import settings
    out: dict[str, Any] = {
        "enabled": bool(settings.enable_scorecard_disagreement_regen),
        "cap": int(cap if cap is not None else settings.scorecard_disagreement_regen_daily_cap),
        "used_today": 0, "queued": 0, "skipped_gate": 0, "skipped_reviewed": 0, "open_material": 0,
    }
    if not out["enabled"]:
        return out
    try:
        from ..agents import scorecard_context
        from ..database import SessionLocal
        from ..models import ScorecardDisagreement
        from . import regen_worker
        now = _utcnow()
        with SessionLocal() as db:
            ScorecardDisagreement.__table__.create(bind=db.get_bind(), checkfirst=True)
            regen_worker._ensure_table(db)
            out["used_today"] = _scorecard_regens_today(db, now=now)
            budget = max(0, out["cap"] - out["used_today"])
            open_rows = (
                db.query(ScorecardDisagreement)
                .filter(
                    ScorecardDisagreement.status == scorecard_context.STATUS_OPEN,
                    ScorecardDisagreement.severity == "material",
                )
                .order_by(ScorecardDisagreement.created_at.asc(), ScorecardDisagreement.id.asc())
                .all()
            )
            out["open_material"] = len(open_rows)
            reviewed_keys = {
                (r.ticker, r.version_key, r.as_of)
                for r in db.query(ScorecardDisagreement)
                .filter(ScorecardDisagreement.status.in_(
                    (scorecard_context.STATUS_QUEUED_REVIEW, scorecard_context.STATUS_REVIEWED),
                ))
                .all()
            }
            queued_tickers: set[str] = set()
            for row in open_rows:
                if budget <= 0:
                    break
                key = (row.ticker, row.version_key, row.as_of)
                if key in reviewed_keys or row.ticker in queued_tickers:
                    out["skipped_reviewed"] += 1
                    continue
                decision = should_auto_regen(row.ticker)
                if not decision.get("should"):
                    out["skipped_gate"] += 1
                    row.note = f"{row.note} | regen skipped: {decision.get('reason', 'gate')}" if row.note else \
                        f"regen skipped: {decision.get('reason', 'gate')}"
                    continue
                job, created = regen_worker.enqueue(row.ticker, source=scorecard_context.REGEN_SOURCE)
                row.status = scorecard_context.STATUS_QUEUED_REVIEW
                suffix = f"review regen job {job['id']} ({'created' if created else 'coalesced'}) at {now.isoformat()}"
                row.note = f"{row.note} | {suffix}" if row.note else suffix
                queued_tickers.add(row.ticker)
                reviewed_keys.add(key)
                out["queued"] += 1
                if created:
                    budget -= 1
            db.commit()
    except Exception as exc:  # pragma: no cover — the loop note carries it
        from ..agents.log_safety import log_safely
        log_safely(log, "handle_scorecard_disagreements failed", exc)
        out["error"] = type(exc).__name__
    return out


def queue_depth(ticker: str | None = None) -> dict[str, int]:
    """Inspect the in-process FIFO queue (for /api/admin).

    Note: `full_reanalysis` no longer flows through this queue — it
    lives in the durable `regen_jobs` table (see
    /api/admin/regen-jobs for that telemetry)."""
    if ticker:
        return {ticker.upper(): len(_QUEUES.get(ticker.upper(), []))}
    return {t: len(q) for t, q in _QUEUES.items() if q}


# ---------------------------------------------------------------------------
# Wave 10 — macro regime shift trigger
# ---------------------------------------------------------------------------

# Regime shifts can move a lot of names at once; cap how many memos
# we'll re-run per event to keep cost predictable. Names selected =
# the most rate-sensitive (theme_exposure) names whose sector flipped
# from favored → pressured (or vice versa) under the new regime.
MAX_TICKERS_PER_REGIME_SHIFT = 10


def _affected_tickers_for_regime_shift(
    prior_regime: str, new_regime: str,
) -> list[str]:
    """Pick tickers most likely to need a memo refresh after the
    regime flipped from `prior_regime` → `new_regime`.

    Strategy:
    1. Tickers with high `long_rates_sensitivity` theme exposure are
       always candidates (rate-regime shifts hit these first).
    2. Tickers in sectors that flipped between favored/pressured
       under the new vs. prior regime are also candidates.

    Cross-references the macro_loop's `_REGIME_FAVORED` /
    `_REGIME_PRESSURED` maps via a deferred import to avoid circular
    imports at module load.
    """
    try:
        from ..monitoring.macro_loop import _REGIME_FAVORED, _REGIME_PRESSURED
    except Exception:  # pragma: no cover
        _REGIME_FAVORED, _REGIME_PRESSURED = {}, {}

    # Sectors that meaningfully changed status.
    prior_set = set(_REGIME_FAVORED.get(prior_regime, []) + _REGIME_PRESSURED.get(prior_regime, []))
    new_set = set(_REGIME_FAVORED.get(new_regime, []) + _REGIME_PRESSURED.get(new_regime, []))
    affected_sectors = (prior_set - new_set) | (new_set - prior_set)

    candidates: list[str] = []

    # 1) Long-rates-sensitive names — the first to feel a regime change.
    try:
        from .theme_exposure_service import top_for_theme
        for row in top_for_theme("long_rates_sensitivity", min_score=20.0, limit=15):
            t = row.get("ticker")
            if t and t not in candidates:
                candidates.append(t)
    except Exception as exc:  # pragma: no cover
        log.debug("theme_exposure read failed for regime shift: %s", exc)

    # 2) Sector flippers — tickers in sectors that crossed favored/pressured.
    if affected_sectors:
        try:
            from sqlalchemy import select

            from ..database import SessionLocal
            from ..models import Company
            with SessionLocal() as db:
                rows = db.execute(
                    select(Company.ticker, Company.sector)
                    .where(Company.universe_tier == "auto_analysis")
                ).all()
                for ticker, sector in rows:
                    if sector and sector in affected_sectors and ticker not in candidates:
                        candidates.append(ticker)
                    if len(candidates) >= MAX_TICKERS_PER_REGIME_SHIFT * 3:
                        break
        except Exception as exc:  # pragma: no cover
            log.debug("companies read failed for regime shift: %s", exc)

    return candidates[:MAX_TICKERS_PER_REGIME_SHIFT]


def on_regime_shift(prior_regime: str, new_regime: str) -> dict[str, Any]:
    """Wave 10 — fire when macro_loop detects the regime classification
    changed. Re-runs memos for the most affected names.

    Strategy: full_reanalysis for the top N by exposure (forces fresh
    valuation + rating; light patches wouldn't reflect the regime
    change properly). Bounded by `MAX_TICKERS_PER_REGIME_SHIFT` so a
    single regime flip can't blow the budget. The regens are enqueued
    on the `regen_jobs` queue and drained one at a time by the worker
    thread — a 10-ticker shift queues instantly instead of holding the
    scheduler thread for 10 sequential memo runs.

    Returns {prior, new, refreshed: List[str], gate_errors: List[str]} for
    cron logging (refreshed = handed to the gate; outcomes land in
    `regen_jobs`; gate_errors = tickers whose auto-regen gate crashed, so
    "refreshed" must not be read as "regenerated" for those).
    """
    if prior_regime == new_regime:
        return {"prior": prior_regime, "new": new_regime, "refreshed": [], "gate_errors": []}
    affected = _affected_tickers_for_regime_shift(prior_regime, new_regime)
    refreshed: list[str] = []
    gate_errors: list[str] = []
    for ticker in affected:
        try:
            # Reuses the full_reanalysis path (gating + enqueue).
            res = on_filing_event(ticker, source="regime_shift")
            refreshed.append(ticker)
            if res.get("kind") == "gate_error":
                gate_errors.append(ticker)
        except Exception as exc:  # pragma: no cover
            from ..agents.log_safety import log_safely  # lazy: agents imports services
            log_safely(log, f"regime-shift refresh failed for {ticker}", exc)
    log.info(
        "regime shift %s → %s: enqueued regen for %d ticker(s) — %s%s",
        prior_regime, new_regime, len(refreshed), refreshed,
        f"; gate errors on {gate_errors}" if gate_errors else "",
    )
    return {
        "prior": prior_regime, "new": new_regime,
        "refreshed": refreshed, "gate_errors": gate_errors,
    }
