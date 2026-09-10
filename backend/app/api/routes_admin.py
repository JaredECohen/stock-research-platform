"""Admin endpoints (re-seed, debug, monitoring status, LLM metrics).

Wave 8C: operational surfaces for the durable state shipped in earlier
waves — DCF version history, update orchestrator queue, news allow-list
governance, bull/bear lopsidedness audit. None of these add new
business logic; they expose what's already in the DB / service layer
so a UI or admin script can reason about platform state without
SQL-level access.

Wave 8G: UI-trace ingest + read endpoints. Frontend posts route
changes / API calls / clicks / errors; backend HTTP middleware writes
its own rows; one timeline I can query.
"""
from __future__ import annotations

import logging
from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..monitoring import KNOWN_LOOPS, _process_role, status_snapshot
from ..rate_limit import LIMITS, limiter
from ..schemas.scorecard import (
    ScorecardBackfillOut,
    ScorecardBackfillRequest,
    ScorecardEnqueueOut,
    ScorecardEvaluateRequest,
    ScorecardRefreshRequest,
    ScorecardRunOut,
)
from ..seed_universe import run_full_seed
from ..services import dcf_store, llm_metrics, memo_store, outcome_service, update_orchestrator
from .gating import enforce_global

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/seed-universe")
@limiter.limit(LIMITS["seed_universe"])
def seed_universe_endpoint(
    request: Request, response: Response, refresh: bool = False,
) -> dict:
    """Re-seed the curated screener universe (S&P 500 + curated extensions) from FMP.

    Upserts a `companies` row per ticker in `data/sp500.json` and tags it
    `auto_analysis`; any ticker outside the file is still researchable
    on demand. `refresh=true` re-fetches every profile (slower; use after
    FMP data corrections). `refresh=false` (default) only inserts missing
    rows and is cheap to call.

    This does NOT change which tickers are in the universe. Refreshing
    the constituent list is a separate, manual step —
    `python -m app.scripts.refresh_universe_lists` — and
    `GET /api/admin/universe-review` shows whether that is due.
    """
    return run_full_seed(refresh=refresh)


@router.post("/api/admin/run-backfill")
@limiter.limit(LIMITS["admin_backfill"])
def run_backfill_endpoint(
    request: Request,
    response: Response,
    ticker: str | None = Query(None, description="Single ticker; omit for full universe"),
) -> dict:
    """Trigger the heavy history backfill on demand.

    Synchronous — for the curated universe (S&P 500 + extensions) budget
    ~6 provider calls per ticker, so minutes at the 170-name starter
    list and longer at full 500. For a single ticker (`?ticker=NVDA`)
    it's ~5 seconds. Idempotent.

    Use this after a fresh deploy when the database is empty (Postgres
    on first boot has 0 financial_periods rows; the `seed_universe`
    that runs at startup only fills `companies`). Without this call,
    the screener stays empty until the nightly cron at 03:15 UTC.
    """
    from ..monitoring.history_backfill import run_once
    return run_once(ticker=ticker)


@router.get("/api/admin/monitoring/status")
def monitoring_status() -> dict:
    """Last-run timestamps + notes per registered monitoring loop."""
    return {"loops": status_snapshot()}


@router.get("/api/admin/abuse-telemetry")
def abuse_telemetry_endpoint() -> dict[str, Any]:
    """FEAT-002 phase 6 — the numbers behind "is the login wall being
    abused, and is it refusing real customers": 429s by scope / plan /
    route, trial creations per bootstrap IP hash, and the share of
    non-public API requests that were refused with 429, over the trailing
    24h. Admin-token protected by prefix (`admin_auth`); the customer
    middleware never consults a JWT for this path. Cross-process by
    construction — every input is a database row, so it reads the same
    from the web and the worker.
    """
    from ..auth.analytics import abuse_report
    from ..database import SessionLocal
    with SessionLocal() as db:
        return abuse_report(db, hours=24)


@router.get("/api/admin/llm-metrics")
def llm_metrics_endpoint(
    run_id: str | None = None,
    since_days: int = Query(7, ge=1, le=365),
) -> dict[str, Any]:
    """LLM call audit endpoint (Wave 1A).

    With `run_id`: detailed per-call trace for one memo run.
    Without `run_id`: aggregated summary over the last `since_days`.
    """
    if run_id:
        return llm_metrics.cost_per_run(run_id)
    since = datetime.utcnow() - timedelta(days=since_days)
    return {
        "since_days": since_days,
        "by_agent": llm_metrics.cost_per_agent(since=since),
        "by_provider": llm_metrics.cost_per_provider(since=since),
        "slowest": llm_metrics.slowest_calls(since=since, n=10),
    }


@router.get("/api/admin/sdk-traces")
def list_sdk_traces(
    ticker: str | None = None,
    surface: str | None = Query(None, description="memo|chat"),
    limit: int = Query(20, ge=1, le=200),
) -> dict[str, Any]:
    """Wave 10 — list recent SDK exchange traces.

    Use this to spot-check whether the SDK is firing as expected and
    pull a `run_id` to deep-link into `/api/admin/sdk-traces/{run_id}`
    for the joined trace + LLM-call view.
    """
    from ..database import SessionLocal
    from ..models import SDKTrace
    with SessionLocal() as session:
        q = session.query(SDKTrace).order_by(SDKTrace.generated_at.desc())
        if ticker:
            q = q.filter(SDKTrace.ticker == ticker.upper())
        if surface:
            q = q.filter(SDKTrace.surface == surface)
        rows = q.limit(limit).all()
        return {
            "count": len(rows),
            "traces": [
                {
                    "run_id": r.run_id,
                    "ticker": r.ticker,
                    "surface": r.surface,
                    "duration_ms": r.duration_ms,
                    "items_count": len(r.new_items or []),
                    "final_output_preview": (r.final_output or "")[:200],
                    "error": r.error or None,
                    "generated_at": r.generated_at.isoformat() + "Z",
                }
                for r in rows
            ],
        }


@router.get("/api/admin/sdk-traces/{run_id}")
def get_sdk_trace(run_id: str) -> dict[str, Any]:
    """Wave 10 — joined view: SDK exchange trace + the legacy graph's
    LLMCallLog rows for the same `run_id`.

    The SDK runs in parallel with the graph; both share `run_id` so
    reviewers can see both timelines side-by-side and diff "what the
    SDK did" vs "what the graph did" on the same input.

    Returns 404 if no SDK trace exists for the run_id (it may still
    have LLMCallLog rows from a graph-only run; query that endpoint
    directly via `/api/admin/llm-metrics?run_id=...`).
    """
    from ..database import SessionLocal
    from ..models import SDKTrace
    with SessionLocal() as session:
        trace = session.query(SDKTrace).filter(
            SDKTrace.run_id == run_id,
        ).order_by(SDKTrace.generated_at.desc()).first()
        if trace is None:
            raise HTTPException(
                status_code=404,
                detail=f"No SDK trace found for run_id={run_id}",
            )
        # Joined LLMCallLog rows — same shape as `/api/admin/llm-metrics`
        # so the frontend can reuse one renderer for both.
        llm_calls = llm_metrics.cost_per_run(run_id)
        return {
            "run_id": run_id,
            "trace": {
                "id": trace.id,
                "ticker": trace.ticker,
                "surface": trace.surface,
                "duration_ms": trace.duration_ms,
                "final_output": trace.final_output,
                "new_items": trace.new_items,
                "error": trace.error or None,
                "generated_at": trace.generated_at.isoformat() + "Z",
            },
            "llm_calls": llm_calls,
        }


@router.get("/api/admin/track-record")
def track_record_endpoint(
    horizon_days: int = Query(90, ge=1, le=365),
    ticker: str | None = None,
    sector: str | None = None,
) -> dict[str, Any]:
    """Wave 4A: aggregate realized-outcome stats over evaluated memos.

    Filters: `ticker` (single name), `sector`, `horizon_days` (which forward
    window to look at). Returns hit rate + avg alpha + total evaluated.
    """
    return outcome_service.track_record(
        ticker=ticker, sector=sector, horizon_days=horizon_days,
    )


@router.post("/api/admin/evaluate-outcomes")
def evaluate_outcomes_now(request: Request, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Manual trigger for the daily outcome loop. Useful in dev / for
    backfilling the table after deploys; production runs the scheduled
    job via APScheduler.

    Browser-called (`TrackRecord.tsx`), so it is exempt from the admin
    token and Pro under the customer policy. The work is platform-wide
    and identical whoever asks, so behind the login wall it also sits in
    one GLOBAL window (`evaluate_outcomes`, 1 per 10 minutes, shared by
    every caller) rather than a per-user one — a per-user limit would let
    N accounts run the loop N times. With the wall off the route is
    unchanged.
    """
    enforce_global(request, db, "evaluate_outcomes")
    return outcome_service.evaluate_all_due()


# ---------------------------------------------------------------------------
# Wave 10 — calibration + per-agent attribution + regime-conditional accuracy
# ---------------------------------------------------------------------------

@router.get("/api/admin/calibration")
def calibration_endpoint(
    horizon_days: int = Query(90, ge=1, le=365),
) -> dict[str, Any]:
    """Wave 10 — calibration plot data: per-rating realized excess
    return distribution. A well-calibrated PM has Strong-Buy realizations
    clearly higher than Buy realizations. Powers the upcoming
    track-record dashboard."""
    from ..services.calibration_service import calibration_by_rating
    return calibration_by_rating(horizon_days=horizon_days)


@router.get("/api/admin/per-agent-attribution")
def per_agent_attribution_endpoint(
    horizon_days: int = Query(90, ge=1, le=365),
) -> dict[str, Any]:
    """Wave 10 — per-specialist attribution stats from `memo_postmortems`.
    Surfaces systematic strengths and weaknesses ('our valuation analyst
    consistently picks the right names; our macro is pulling the wrong
    direction')."""
    from ..services.calibration_service import per_agent_attribution
    return per_agent_attribution(horizon_days=horizon_days)


@router.get("/api/admin/regime-accuracy")
def regime_accuracy_endpoint(
    horizon_days: int = Query(90, ge=1, le=365),
) -> dict[str, Any]:
    """Wave 10 — accuracy bucketed by macro regime at memo creation.
    Catches regime-specific blind spots ('we're great in soft-landing
    regimes, terrible in recessions')."""
    from ..services.calibration_service import regime_conditional_accuracy
    return regime_conditional_accuracy(horizon_days=horizon_days)


@router.get("/api/admin/calibration-summary")
def calibration_summary_endpoint(
    horizon_days: int = Query(90, ge=1, le=365),
) -> dict[str, Any]:
    """Wave 10 — one-call aggregator returning calibration + per-agent
    + regime stats. Powers the upcoming track-record dashboard with a
    single fetch."""
    from ..services.calibration_service import summary
    return summary(horizon_days=horizon_days)


@router.post("/api/admin/run-postmortems")
def run_postmortems_endpoint(
    horizon_days: int = Query(90, ge=1, le=365),
    limit: int = Query(25, ge=1, le=200),
) -> dict[str, Any]:
    """Wave 10 — manual trigger for postmortem_loop; equivalent to
    running `python -m scripts.postmortem_backfill`. Useful for
    seeding the system or recovering after a cron outage."""
    from ..services.postmortem_service import run_postmortems
    return run_postmortems(horizon_days=horizon_days, limit=limit)


@router.get("/api/admin/cron-health")
def cron_health_endpoint() -> dict[str, Any]:
    """Wave 10 — aggregated cron health.

    Returns each registered loop's last-run timestamp + freshness flag.
    A loop that hasn't reported in >24h is flagged as stale (32h for
    weekly loops). Powers the operational dashboard for surfacing
    silent cron failures.
    """
    snap = status_snapshot()
    # Start from the registry, not from what happens to have reported.
    # A loop that has never completed once has no record, so it used to be
    # absent from this response entirely — and "absent" reads as "fine".
    # That hid `postmortem_loop` failing on every 03:00 UTC run for as long
    # as `memo_outcomes.regime_at_memo` was missing: it raised before
    # reaching `record_run`, so it never appeared here at all. An endpoint
    # whose whole job is surfacing silent cron failures has to name the
    # loops it has heard nothing from.
    for name in KNOWN_LOOPS:
        snap.setdefault(name, {"last_run_at": None, "success": None, "note": "never run"})
    out_loops: list[dict[str, Any]] = []
    now = datetime.utcnow()
    weekly_loops = {"weekly_digest_loop", "sector_digest_loop", "sample_build_loop"}
    monthly_loops = {"theme_exposure_loop"}
    for loop_name, info in snap.items():
        last_run_str = info.get("last_run_at") if isinstance(info, dict) else None
        stale = True
        age_seconds = None
        if last_run_str:
            try:
                last_run = datetime.fromisoformat(last_run_str)
                age_seconds = (now - last_run).total_seconds()
                if loop_name in monthly_loops:
                    stale = age_seconds > 32 * 24 * 3600
                elif loop_name in weekly_loops:
                    stale = age_seconds > 8 * 24 * 3600
                else:
                    stale = age_seconds > 26 * 3600  # daily loops + slack
            except Exception:
                stale = True
        out_loops.append({
            "loop": loop_name,
            "last_run_at": last_run_str,
            "age_seconds": age_seconds,
            "success": (info or {}).get("success"),
            "note": (info or {}).get("note"),
            "stale": stale,
        })
    out_loops.sort(key=lambda r: r["loop"])
    n_stale = sum(1 for r in out_loops if r["stale"])
    # The universe file is not a loop — nothing refreshes it on a
    # schedule, by design — but "the snapshot is past its review date" is
    # exactly the kind of quiet rot this endpoint exists to surface, and
    # ops already looks here. Kept out of `stale_count`, which counts
    # loops; a stale file is a review task, not a cron failure.
    from ..services.universe_review import file_status
    uf = file_status()
    return {
        "loops": out_loops,
        "stale_count": n_stale,
        "universe_review": {
            "last_reviewed": uf["last_reviewed"],
            "days_since_review": uf["days_since_review"],
            "stale": uf["stale"],
        },
    }


@router.get("/api/admin/universe-review")
def universe_review_endpoint(
    compare_feed: bool = Query(
        False,
        description="Also diff data/sp500.json against the live FMP constituent "
                    "list. Read-only; needs FMP_API_KEY and ENABLE_LIVE_DATA.",
    ),
) -> dict[str, Any]:
    """Read-only review of the curated screener universe.

    Reports the universe file's review timestamp and staleness, its
    drift against the `companies` table, and — only with
    `compare_feed=true` — the added/removed diff against FMP's S&P 500
    constituent feed. Nothing is written: the universe is a hand-reviewed
    snapshot, and changing it stays a deliberate operator step
    (`python -m app.scripts.refresh_universe_lists`, then
    `POST /api/seed-universe`).
    """
    from ..services.universe_review import review_universe
    return review_universe(compare_feed=compare_feed)


@router.post("/api/admin/run-weekly-digest")
def run_weekly_digest_endpoint(
    ticker: str | None = Query(None),
    sector: str | None = Query(None),
    days_back: int = Query(7, ge=1, le=30),
) -> dict[str, Any]:
    """Wave 10 — manual trigger for the weekly digest pipeline.
    `ticker` runs the per-name digest; `sector` runs the sector
    cohort digest. Both blank runs the full universe."""
    from ..services import filing_memory
    if ticker:
        return filing_memory.weekly_digest(ticker.upper(), days_back=days_back)
    if sector:
        return filing_memory.weekly_sector_digest(sector, days_back=days_back)
    return {
        "tickers": filing_memory.weekly_digest_universe(days_back=days_back),
        "sectors": filing_memory.weekly_sector_digest_all(days_back=days_back),
    }


@router.get("/api/admin/specialist-reliability")
def specialist_reliability_endpoint(
    lookback: int = Query(30, ge=5, le=200),
) -> dict[str, Any]:
    """Wave 10 — per-specialist reliability over the last N
    postmortemmed memos. Identifies specialists whose pulls have
    correlated with WRONG calls. Surfaces the same data the PM's
    self-improvement context now reads."""
    from ..services.influence_feedback import specialist_reliability
    return specialist_reliability(lookback=lookback)


@router.get("/api/admin/postmortems/{ticker}")
def latest_postmortems_endpoint(ticker: str, limit: int = Query(5, ge=1, le=50)) -> dict[str, Any]:
    """Wave 10 — latest postmortems for a ticker. Powers the memo page's
    "we got this {right/wrong} last time" callout. Returns most recent
    first.
    """
    from sqlalchemy import select

    from ..database import SessionLocal
    from ..models import MemoPostmortem
    with SessionLocal() as db:
        rows = db.execute(
            select(MemoPostmortem)
            .where(MemoPostmortem.ticker == ticker.upper())
            .order_by(MemoPostmortem.created_at.desc())
            .limit(limit)
        ).scalars().all()
    return {
        "ticker": ticker.upper(),
        "postmortems": [
            {
                "id": r.id,
                "horizon_days": r.horizon_days,
                "verdict": r.verdict,
                "lesson": r.lesson,
                "agent_attribution": r.agent_attribution or {},
                "realized_return": r.realized_return,
                "benchmark_return": r.benchmark_return,
                "regime_at_memo": r.regime_at_memo,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.post("/api/admin/rerun-memos")
def rerun_memos_endpoint(tickers: list[str]) -> dict[str, Any]:
    """Wave 10 — admin bulk rerun. Useful when a Wave shipped that
    materially changes memo outputs (new schema field, prompt update,
    DCF default change) and you want to refresh a curated subset
    without waiting for organic triggers.

    Goes through the on_filing_event path — the same gated
    full-reanalysis route the EDGAR poller uses — which enqueues each
    ticker on the durable `regen_jobs` queue. The response returns
    job ids immediately (no more holding the request open while up to
    50 memos run); watch progress via /api/admin/regen-jobs.
    """
    from ..services.update_orchestrator import on_filing_event
    if not tickers:
        raise HTTPException(status_code=400, detail="tickers list cannot be empty")
    if len(tickers) > 50:
        raise HTTPException(
            status_code=400, detail="cap of 50 tickers per request",
        )
    results: list[dict[str, Any]] = []
    for raw in tickers:
        ticker = (raw or "").strip().upper()
        if not ticker:
            continue
        try:
            results.append(on_filing_event(ticker, source="admin_rerun"))
        except Exception as exc:  # pragma: no cover
            results.append({"ticker": ticker, "error": str(exc)})
    return {"requested": len(tickers), "results": results}


@router.post("/api/admin/mispricing-audit")
def mispricing_audit_endpoint(
    limit: int = Query(20, ge=1, le=50),
    persist: bool = Query(True),
) -> dict[str, Any]:
    """Wave 10 — audit the quality of the PM's mispricing theses
    across recent memos. Returns per-memo scores (specificity /
    differentiation / falsifiability) + a corpus-wide failure-
    mode observation. Feeds PM prompt iteration.

    When `persist=true` (default) the run is written to
    `mispricing_audits` so the PM can read the most-recent
    `pattern_observation` from its self-improvement context block.
    """
    from ..services.mispricing_audit import (
        aggregate_scores,
        persist_audit,
        run_audit,
    )
    audit = run_audit(limit=limit)
    audit["aggregate"] = aggregate_scores(audit)
    if persist:
        row_id = persist_audit(audit, audit["aggregate"])
        audit["audit_id"] = row_id
    return audit


# ---------------------------------------------------------------------------
# Wave 8C — DCF version history
# ---------------------------------------------------------------------------

@router.get("/api/admin/dcf-versions/{ticker}")
def dcf_version_history(
    ticker: str, limit: int = Query(25, ge=1, le=200),
) -> dict[str, Any]:
    """Wave 5A — DCF assumption drift over time.

    Returns the version chain newest-first with `assumption_changes`
    per version (the diff vs. parent_version) so reviewers can audit
    the LLM-driven updater's proposals before flipping it fully
    autonomous (locked decision in MASTER_PLAN §7).
    """
    rows = dcf_store.version_history(ticker.upper(), limit=limit)
    return {
        "ticker": ticker.upper(),
        "versions": [
            {
                "version": r.version,
                "parent_version": r.parent_version,
                "trigger": r.trigger,
                "generated_at": r.generated_at.isoformat(),
                "assumption_changes": r.assumption_changes or [],
                # Don't ship full DCFResult per row — too heavy for the
                # timeline view. Caller fetches one specific version
                # via the singular DCF endpoint when they need detail.
                "has_result": bool(r.dcf_result),
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Wave 8C — Update orchestrator queue inspection
# ---------------------------------------------------------------------------

@router.get("/api/admin/update-queue")
def update_queue_status(
    ticker: str | None = None,
) -> dict[str, Any]:
    """Wave 5B — in-process FIFO queue for the update orchestrator.

    Useful for diagnosing "is the loop wedged?" without shell access.
    Per-ticker FIFO means depth is usually 0; non-zero indicates an
    in-flight `full_reanalysis` or backed-up alerts.
    """
    return {
        "queue_depth_by_ticker": update_orchestrator.queue_depth(ticker),
    }


# ---------------------------------------------------------------------------
# Wave 8C — News domain governance reload
# ---------------------------------------------------------------------------

@router.post("/api/admin/news-domains/reload")
def reload_news_domains() -> dict[str, Any]:
    """Wave 6C — reload `news_domains.json` without bouncing the server.

    The agent caches the lists via `lru_cache`; this clears it so a
    just-edited governance file takes effect immediately.
    """
    from ..agents.news_agent import reload_domain_lists
    allowed, blocked = reload_domain_lists()
    return {
        "allowed_count": len(allowed),
        "blocked_count": len(blocked),
        "allowed_sample": sorted(allowed)[:5],
        "blocked_sample": sorted(blocked)[:5],
    }


# ---------------------------------------------------------------------------
# Wave 8C — Bull/bear lopsidedness audit
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Wave 8G — UI trace ingest + read
# ---------------------------------------------------------------------------

class UILogEvent(BaseModel):
    """Single trace event from the frontend.

    `kind` is the high-level category: route / api_call / click / error.
    Everything else lives in `payload` so we can ship new event types
    without bumping the schema.
    """
    kind: str
    path: str | None = None
    method: str | None = None
    status_code: int | None = None
    duration_ms: int | None = None
    session_id: str | None = None
    ts: str | None = None  # client wall-clock; not authoritative
    payload: dict[str, Any] = Field(default_factory=dict)


class UILogBatch(BaseModel):
    events: list[UILogEvent]


@router.post("/api/admin/ui-log")
def post_ui_log(batch: UILogBatch) -> dict[str, Any]:
    """Ingest a batch of UI trace events. Always returns 200 — logging
    must never block the user."""
    from ..database import SessionLocal
    from ..models import UILog
    written = 0
    try:
        with SessionLocal() as db:
            UILog.__table__.create(bind=db.get_bind(), checkfirst=True)
            for e in batch.events:
                db.add(UILog(
                    ts=datetime.utcnow(),
                    source="frontend",
                    kind=e.kind[:32],
                    path=(e.path or "")[:256] or None,
                    method=(e.method or "")[:8] or None,
                    status_code=e.status_code,
                    duration_ms=e.duration_ms,
                    session_id=(e.session_id or "")[:64] or None,
                    payload=e.payload or {},
                ))
                written += 1
            db.commit()
    except Exception:
        # RP-001 class (c), deliberately NOT made to raise: this is a
        # browser-called telemetry sink (exempt from admin auth) whose
        # contract is "always 200, never block the user". Raising would
        # turn a logging hiccup into a 500 on every page the UI traces
        # from. Log with the traceback and keep returning ok=False so the
        # failure is visible in the server log instead of nowhere.
        log.exception("ui-log ingest failed after %d event(s)", written)
        return {"written": written, "ok": False}
    return {"written": written, "ok": True}


@router.get("/api/admin/ui-log")
def get_ui_log(
    limit: int = Query(200, ge=1, le=2000),
    since_minutes: int = Query(60, ge=1, le=1440),
    source: str | None = None,
    kind: str | None = None,
    path_contains: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Read recent UI trace events newest-first. Use this to see what a
    user was doing in the UI."""
    from sqlalchemy import select

    from ..database import SessionLocal
    from ..models import UILog
    cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
    with SessionLocal() as db:
        UILog.__table__.create(bind=db.get_bind(), checkfirst=True)
        stmt = select(UILog).where(UILog.ts >= cutoff)
        if source:
            stmt = stmt.where(UILog.source == source)
        if kind:
            stmt = stmt.where(UILog.kind == kind)
        if session_id:
            stmt = stmt.where(UILog.session_id == session_id)
        if path_contains:
            stmt = stmt.where(UILog.path.like(f"%{path_contains}%"))
        stmt = stmt.order_by(UILog.ts.desc()).limit(limit)
        rows = db.execute(stmt).scalars().all()
    return {
        "since_minutes": since_minutes,
        "n": len(rows),
        "events": [
            {
                "id": r.id,
                "ts": r.ts.isoformat() if r.ts else None,
                "source": r.source,
                "kind": r.kind,
                "path": r.path,
                "method": r.method,
                "status_code": r.status_code,
                "duration_ms": r.duration_ms,
                "session_id": r.session_id,
                "payload": r.payload,
            }
            for r in rows
        ],
    }


@router.delete("/api/admin/ui-log")
def clear_ui_log() -> dict[str, Any]:
    """Wipe the trace table. Useful before starting a fresh test session."""
    from ..database import SessionLocal
    from ..models import UILog
    with SessionLocal() as db:
        UILog.__table__.create(bind=db.get_bind(), checkfirst=True)
        n = db.query(UILog).delete()
        db.commit()
    return {"deleted": n}


@router.get("/api/admin/lopsidedness-audit")
def lopsidedness_audit(
    n: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:
    """Wave 3A risk-register mitigation: telemetry on whether the
    sector-integrated bull/bear is actually balanced.

    Walks the `n` most recent `MemoSnapshot` rows and reports per-memo
    bull/bear key-point counts + `sector_lean` distribution. A
    persistent skew toward one side across many tickers is the signal
    to revisit the prompt structure or add a devil's-advocate amplifier
    (deferred per locked decision until lopsidedness shows up in practice).
    """
    rows: list[dict[str, Any]] = []
    bull_kp_total = 0
    bear_kp_total = 0
    lean_counts = {"bull": 0, "bear": 0, "balanced": 0}
    falsifiable_total = 0
    inspected = 0

    history_seen: set[str] = set()
    # Pull latest memo per ticker (skip duplicates) up to n unique tickers.
    from sqlalchemy import select

    from ..database import SessionLocal
    from ..models import MemoSnapshot
    with SessionLocal() as db:
        memo_store._ensure_table(db)
        all_rows = db.execute(
            select(MemoSnapshot)
            .order_by(MemoSnapshot.generated_at.desc())
            .limit(n * 4)  # over-fetch since we dedup by ticker
        ).scalars().all()
        for r in all_rows:
            if r.ticker in history_seen:
                continue
            history_seen.add(r.ticker)
            inspected += 1
            memo = r.memo_json or {}
            bull = memo.get("bull_case") or {}
            bear = memo.get("bear_case") or {}
            bull_kp = len(bull.get("key_points") or [])
            bear_kp = len(bear.get("key_points") or [])
            bull_kp_total += bull_kp
            bear_kp_total += bear_kp
            sector_view = memo.get("sector_agent_view") or {}
            sector_data = sector_view.get("data") or {}
            bb = sector_data.get("bull_bear_analysis") or {}
            lean = bb.get("sector_lean") or "balanced"
            if lean in lean_counts:
                lean_counts[lean] += 1
            ftests = bb.get("falsifiable_tests") or []
            falsifiable_total += len(ftests)
            rows.append({
                "ticker": r.ticker,
                "version": r.version,
                "rating": memo.get("rating_label"),
                "sector_lean": lean,
                "bull_kp": bull_kp,
                "bear_kp": bear_kp,
                "falsifiable_tests": len(ftests),
            })
            if inspected >= n:
                break

    avg_bull_kp = bull_kp_total / inspected if inspected else 0.0
    avg_bear_kp = bear_kp_total / inspected if inspected else 0.0
    skew = (avg_bull_kp - avg_bear_kp) if inspected else 0.0
    lean_skew = lean_counts["bull"] - lean_counts["bear"]
    return {
        "inspected": inspected,
        "avg_bull_key_points": round(avg_bull_kp, 2),
        "avg_bear_key_points": round(avg_bear_kp, 2),
        "key_point_skew": round(skew, 2),
        "sector_lean_counts": lean_counts,
        "lean_skew": lean_skew,
        "avg_falsifiable_tests_per_memo": round(
            falsifiable_total / inspected if inspected else 0.0, 2,
        ),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Auto-update memo gating (Phase A — universe expansion cost control)
# ---------------------------------------------------------------------------

class AutoUpdateToggle(BaseModel):
    auto_update_memo: bool = Field(
        ..., description="True = regenerate memo automatically on new filings / transcripts."
    )


@router.get("/api/admin/auto-update")
def list_auto_update_tickers() -> dict[str, Any]:
    """List tickers eligible for automatic memo regeneration.

    A ticker is eligible when its `Company.auto_update_memo` is True
    OR a memo was generated/viewed within the recency window (currently
    30 days — see `update_orchestrator.AUTO_REGEN_RECENCY_DAYS`). The
    `pinned` list is the explicit subset that always auto-regens
    regardless of recency. Users curate this list to keep the marginal
    LLM spend of the SP500 expansion predictable.
    """
    from ..database import SessionLocal
    from ..models import Company
    with SessionLocal() as db:
        rows = db.query(Company.ticker, Company.company_name).filter(
            Company.auto_update_memo == True,  # noqa: E712 — sqlalchemy
        ).order_by(Company.ticker).all()
    return {
        "pinned": [{"ticker": t, "company_name": n} for (t, n) in rows],
        "recency_window_days": update_orchestrator.AUTO_REGEN_RECENCY_DAYS,
        "note": (
            "Pinned tickers always auto-regen on new filings/transcripts. "
            "Other tickers auto-regen only if their memo was generated "
            "or viewed within the recency window."
        ),
    }


@router.put("/api/admin/auto-update/{ticker}")
def set_auto_update_memo(ticker: str, payload: AutoUpdateToggle) -> dict[str, Any]:
    """Pin or unpin a ticker for automatic memo regeneration.

    Returns 404 when the ticker isn't in the companies table. Idempotent.
    """
    from ..database import session_scope
    from ..models import Company
    ticker = ticker.upper()
    with session_scope() as db:
        company = db.get(Company, ticker)
        if company is None:
            raise HTTPException(404, f"Ticker {ticker} not in universe")
        company.auto_update_memo = payload.auto_update_memo
        return {
            "ticker": ticker,
            "auto_update_memo": company.auto_update_memo,
        }


@router.post("/api/admin/auto-update/check/{ticker}")
def check_auto_regen_decision(ticker: str) -> dict[str, Any]:
    """Dry-run the gating logic for a specific ticker — useful for
    debugging when a filing landed but no memo regenerated."""
    return update_orchestrator.should_auto_regen(ticker)


# ---------------------------------------------------------------------------
# LLM circuit-breaker ops surface
# ---------------------------------------------------------------------------

@router.get("/api/admin/llm-breakers")
def get_llm_breakers() -> dict[str, Any]:
    """Inspect the current circuit-breaker state for each LLM provider.

    Three failures in a row open the breaker — subsequent calls return
    None instantly until the cooldown elapses or the breaker is reset.
    Use this to debug "everything's silently failing" — if a provider
    shows `is_open: true`, that's the cause.

    **Scope: THIS PROCESS ONLY.** The breaker lives in module-level dicts
    in `agents.llm`, so this describes whichever process served the
    request — the web service. Since the 2026-08-12 worker split, memo
    regen (and therefore most LLM calls) runs in `marketmosaic-worker`,
    whose breakers are NOT visible here; web only makes LLM calls for
    `/api/chat`. The response says so explicitly rather than returning
    clean-looking numbers about the wrong process.

    For cross-process diagnosis use `/api/admin/llm-recent-failures`,
    which reads `LLMCallLog` from the database and therefore sees every
    process. A tripped breaker also writes a `provider_failure` row to
    `CacheCostLog`. Note the breaker self-heals after 120s idle, so a
    permanently stuck breaker — the 2026-05-31 incident — is no longer
    possible; this endpoint is for catching one while it is open.
    """
    from ..agents.llm import get_breaker_state
    return {
        "reported_by": _process_role(),
        "scope_note": (
            "Per-process state. Memo regen runs on marketmosaic-worker, whose "
            "breakers are not visible here — use /api/admin/llm-recent-failures "
            "(DB-backed) for a cross-process view."
        ),
        "providers": get_breaker_state(),
    }


@router.post("/api/admin/llm-breakers/reset")
def reset_llm_breakers(provider: str | None = Query(None)) -> dict[str, Any]:
    """Manually reset LLM circuit breakers. Pass `?provider=openai|anthropic|
    gemini` to reset one; omit for all. Useful after fixing a transient
    issue (auth, model swap, etc.) to skip the auto-reset cooldown."""
    from ..agents.llm import get_breaker_state, reset_circuit_breaker
    reset_circuit_breaker(provider)
    return {
        "reset": provider or "all",
        # Same per-process caveat as the GET: this resets the breaker in
        # the process that served the request (web), NOT the worker's.
        # Rarely matters — the breaker self-heals after 120s idle, so this
        # endpoint only skips the remainder of a cooldown — but returning
        # a bare success would imply a fleet-wide reset it cannot perform.
        "reported_by": _process_role(),
        "scope_note": (
            "Reset applied to this process only; marketmosaic-worker's breakers "
            "are unaffected and self-heal after 120s idle."
        ),
        "providers": get_breaker_state(),
    }


@router.get("/api/admin/llm-recent-failures")
def get_recent_llm_failures(
    limit: int = Query(20, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Return the most recent LLM call failures from LLMCallLog.

    Surfaces the actual provider error message — needed to diagnose
    "regen completes silently with no memo work done" cases where the
    breaker trips because of repeated provider errors but the user
    never sees the underlying reason (e.g., bad model name, auth
    rejection, rate limit).
    """
    from sqlalchemy import desc

    from ..database import SessionLocal
    from ..models import LLMCallLog
    with SessionLocal() as db:
        LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
        rows = (
            db.query(LLMCallLog)
            .filter(LLMCallLog.success == False)  # noqa: E712
            .order_by(desc(LLMCallLog.generated_at))
            .limit(limit)
            .all()
        )
        return [
            {
                "generated_at": r.generated_at.isoformat() if r.generated_at else None,
                "agent_name": r.agent_name,
                "provider": r.provider,
                "model": r.model,
                "duration_ms": r.duration_ms,
                "error": (r.error or "")[:500],
                "run_id": r.run_id,
            }
            for r in rows
        ]


@router.get("/api/admin/regen-jobs")
def list_regen_jobs(
    ticker: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Theme 5 — memo-regen queue telemetry.

    Newest-first `RegenJob` rows: status, attempt count, timings, the
    memo version produced, full error fields on failure, and the
    worker's waypoint trace. Each row's `run_id` joins against
    `/api/admin/llm-recent-failures` (LLMCallLog) and the checkpoint
    store for per-step drill-down. This is the durable replacement for
    the in-memory regen registry that OOM kills used to erase — a
    killed regen now shows up here as `failed` (WorkerRestart) instead
    of vanishing.
    """
    from ..services import regen_worker
    jobs = regen_worker.recent_jobs(ticker=ticker, limit=limit)
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    return {"count": len(jobs), "status_counts": counts, "jobs": jobs}


# ---------------------------------------------------------------------------
# Postgres sequence repair
# ---------------------------------------------------------------------------

@router.post("/api/admin/fix-sequences")
def fix_postgres_sequences() -> dict[str, Any]:
    """Reset Postgres autoincrement sequences to MAX(id)+1 for every
    table that has an `id` primary key.

    Why this exists: a Postgres sequence falls behind the table's
    actual MAX(id) when rows are inserted with EXPLICIT id values
    (bulk seed, SQLite-to-Postgres migration, manual SQL inserts).
    Postgres' nextval() then returns ids that already exist, and
    every INSERT hits `UniqueViolation` on the primary key. Seen
    on prod's `memo_snapshots` table — id=48 conflict was blocking
    every memo regen from persisting.

    Idempotent and SQLite-safe (SQLite has no sequences; the function
    returns an empty diff there).

    Returns `{fixed: [{table, sequence, old_value, new_value}, ...]}`.
    Run after any bulk import or after observing UniqueViolation on
    a `_pkey` constraint.
    """
    from sqlalchemy import inspect, text

    from ..database import engine

    if engine.dialect.name != "postgresql":
        return {"fixed": [], "note": f"no-op on {engine.dialect.name} (sequences are Postgres-only)"}

    fixed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table_name in inspector.get_table_names():
            cols = inspector.get_columns(table_name)
            id_col = next((c for c in cols if c["name"] == "id"), None)
            if id_col is None or not id_col.get("autoincrement"):
                continue
            # Probe for the sequence — typically `{table}_id_seq`
            seq_name = f"{table_name}_id_seq"
            try:
                # Check sequence exists
                seq_exists = conn.execute(text(
                    "SELECT 1 FROM information_schema.sequences "
                    "WHERE sequence_name = :seq"
                ), {"seq": seq_name}).first()
                if not seq_exists:
                    skipped.append({"table": table_name, "reason": "no sequence found"})
                    continue
                max_id = conn.execute(text(
                    f"SELECT COALESCE(MAX(id), 0) FROM {table_name}"
                )).scalar() or 0
                old_val = conn.execute(text(
                    f"SELECT last_value FROM {seq_name}"
                )).scalar()
                # is_called=true tells Postgres "advance past this value
                # before returning it"; setting it to false plus value=N
                # means nextval() returns exactly N. We want nextval()
                # to return max_id+1, so set with is_called=true and
                # value=max_id.
                new_val = max_id + 1
                conn.execute(text(
                    "SELECT setval(:seq, :val, false)"
                ), {"seq": seq_name, "val": new_val})
                fixed.append({
                    "table": table_name,
                    "sequence": seq_name,
                    "old_value": int(old_val) if old_val is not None else None,
                    "max_id": int(max_id),
                    "new_value": int(new_val),
                })
            except Exception as exc:
                skipped.append({"table": table_name, "reason": str(exc)[:200]})
    return {"fixed": fixed, "skipped": skipped}


# ---------------------------------------------------------------------------
# Phase 6 — Fundamental Factor Scorecard: enqueue-only ops surface
# ---------------------------------------------------------------------------
#
# Every endpoint here inserts a `scorecard_runs` row and returns 202. The
# web process never scores: the worker's `scorecard_loop` claims the row
# at its next interval tick (every few minutes; the daily scoring itself
# is gated to 03:45 UTC inside the loop). Protected by the admin token
# through the `/api/admin` prefix; none of these is browser-called.

def _scorecard_version_for(version_key: str | None) -> str:
    from ..services import scorecard_service
    try:
        return scorecard_service.resolve_version(version_key)["version_key"]
    except scorecard_service.UnknownVersion as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


@router.post("/api/admin/scorecard/refresh", status_code=202)
def scorecard_refresh_endpoint(payload: ScorecardRefreshRequest | None = None) -> ScorecardEnqueueOut:
    """Queue one scoring run.

    `as_of` defaults to yesterday UTC (the last complete close);
    `tickers` restricts the run to a subset (still normalised only
    against that subset — use it for diagnostics, not for the product
    cross-section). Coalesces on `(version_key, as_of, kind)`, so a repeat
    request returns the pending row with `created=false`.
    """
    from ..monitoring.scorecard_loop import scheduled_as_of
    from ..services import scorecard_queue
    payload = payload or ScorecardRefreshRequest()
    vk = _scorecard_version_for(payload.version_key)
    as_of = payload.as_of or scheduled_as_of(datetime.utcnow())
    params: dict[str, Any] = {}
    if payload.tickers is not None:
        params["tickers"] = sorted({t.strip().upper() for t in payload.tickers if t.strip()})
    run, created = scorecard_queue.enqueue_run(
        version_key=vk, as_of=as_of, kind=payload.kind, params=params, requested_by="admin",
    )
    return ScorecardEnqueueOut(
        run=ScorecardRunOut(**run), created=created,
        note="queued; the worker's scorecard_loop drains the queue at its next interval tick (minutes)",
    )


@router.post("/api/admin/scorecard/evaluate", status_code=202)
def scorecard_evaluate_endpoint(payload: ScorecardEvaluateRequest | None = None) -> ScorecardEnqueueOut:
    """Queue the evaluation job (quintile long/short, FF5+MOM regression,
    double-selection LASSO) over every month-end row on file. `as_of` is
    only the run's label (default: the last completed month end)."""
    from ..monitoring.scorecard_loop import last_completed_month_end
    from ..services import scorecard_queue
    payload = payload or ScorecardEvaluateRequest()
    vk = _scorecard_version_for(payload.version_key)
    as_of = payload.as_of or last_completed_month_end(datetime.utcnow().date())
    run, created = scorecard_queue.enqueue_run(
        version_key=vk, as_of=as_of, kind=scorecard_queue.KIND_EVALUATE, requested_by="admin",
    )
    return ScorecardEnqueueOut(
        run=ScorecardRunOut(**run), created=created,
        note="queued; results land in scorecard_evaluations and GET /api/scorecard/evaluation",
    )


@router.post("/api/admin/scorecard/backfill", status_code=202)
def scorecard_backfill_endpoint(payload: ScorecardBackfillRequest | None = None) -> ScorecardBackfillOut:
    """Queue the month-end history: one `pit_prepare` run (availability
    backfill + month-end price sync for the universe) followed by one
    `backfill` run per month end, oldest first, for `months` months ending
    at `end` (default: the last completed month end, `SCORECARD_BACKFILL_MONTHS`
    months). Month ends that already have a succeeded run are skipped.

    Honest limit: the price store is fed from the app's 252-day cached
    series, so month ends older than ~12 months score without a price —
    every valuation feature there is n/a (`missing:price`) and coverage is
    lower. The rows are still written so the fundamentals-only families
    have history; the run note reports `no_price=`.
    """
    from ..monitoring.scorecard_loop import last_completed_month_end
    from ..services import scorecard_queue
    payload = payload or ScorecardBackfillRequest()
    vk = _scorecard_version_for(payload.version_key)
    months = int(payload.months or settings.scorecard_backfill_months)
    end = payload.end or last_completed_month_end(datetime.utcnow().date())
    end = date(end.year, end.month, monthrange(end.year, end.month)[1])
    month_ends: list[date] = []
    y, m = end.year, end.month
    for _ in range(months):
        month_ends.append(date(y, m, monthrange(y, m)[1]))
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    month_ends.reverse()

    prep, _prep_created = scorecard_queue.enqueue_run(
        version_key=vk, as_of=end, kind=scorecard_queue.KIND_PIT_PREPARE, requested_by="admin",
    )
    enqueued: list[ScorecardRunOut] = []
    skipped: list[date] = []
    for me in month_ends:
        if scorecard_queue.succeeded_run_exists(vk, me):
            skipped.append(me)
            continue
        run, created = scorecard_queue.enqueue_run(
            version_key=vk, as_of=me, kind=scorecard_queue.KIND_BACKFILL, requested_by="admin",
        )
        if created:
            enqueued.append(ScorecardRunOut(**run))
    return ScorecardBackfillOut(
        pit_prepare=ScorecardRunOut(**prep), enqueued=enqueued, skipped_existing=skipped,
        note=(f"{len(enqueued)} month-end runs queued behind pit_prepare; {len(skipped)} already scored. "
              "Month ends older than the cached 252-day price window score without a price (valuation n/a)."),
    )




@router.get("/api/admin/scorecard/disagreements")
def scorecard_disagreements_endpoint(
    status: str = Query("open", pattern="^(open|queued_review|reviewed|dismissed|all)$"),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """Open (by default) memo-vs-scorecard disagreements, newest first.
    Read-only; the rows are written by the memo pipeline
    (`agents/scorecard_context.persist_disagreement`)."""
    from ..agents import scorecard_context
    from ..database import SessionLocal
    from ..models import ScorecardDisagreement
    with SessionLocal() as db:
        ScorecardDisagreement.__table__.create(bind=db.get_bind(), checkfirst=True)
        q = db.query(ScorecardDisagreement)
        if status != "all":
            q = q.filter(ScorecardDisagreement.status == status)
        rows = q.order_by(ScorecardDisagreement.created_at.desc(), ScorecardDisagreement.id.desc()).limit(limit).all()
        items = [scorecard_context._disagreement_dict(r) for r in rows]
    return {"status": status, "count": len(items), "items": items}


@router.post("/api/admin/scorecard/disagreements/{disagreement_id}/dismiss")
def scorecard_disagreement_dismiss_endpoint(disagreement_id: int) -> dict[str, Any]:
    """Close a disagreement without a review regen. Idempotent: a row that
    is already dismissed comes back unchanged; a row a review already
    closed (`reviewed`) is left as it is and reported as such."""
    from ..agents import scorecard_context
    from ..database import SessionLocal
    from ..models import ScorecardDisagreement
    with SessionLocal() as db:
        ScorecardDisagreement.__table__.create(bind=db.get_bind(), checkfirst=True)
        row = db.get(ScorecardDisagreement, disagreement_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no scorecard disagreement #{disagreement_id}")
        changed = False
        if row.status in (scorecard_context.STATUS_OPEN, scorecard_context.STATUS_QUEUED_REVIEW):
            row.status = scorecard_context.STATUS_DISMISSED
            row.resolved_at = scorecard_context._utcnow()
            db.commit()
            db.refresh(row)
            changed = True
        return {"changed": changed, "item": scorecard_context._disagreement_dict(row)}
