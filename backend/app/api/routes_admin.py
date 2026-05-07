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

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from ..monitoring import status_snapshot
from ..rate_limit import LIMITS, limiter
from ..seed_universe import run_full_seed
from ..services import dcf_store, llm_metrics, memo_store, outcome_service, update_orchestrator

router = APIRouter()


@router.post("/api/seed-universe")
@limiter.limit(LIMITS["seed_universe"])
def seed_universe_endpoint(
    request: Request, response: Response, refresh: bool = False,
) -> Dict:
    """Re-seed the S&P 100 screener universe from FMP.

    `refresh=true` re-fetches every profile (slower; use after FMP data
    corrections). `refresh=false` (default) only inserts missing rows
    and is cheap to call.
    """
    return run_full_seed(refresh=refresh)


@router.post("/api/admin/run-backfill")
@limiter.limit(LIMITS["admin_backfill"])
def run_backfill_endpoint(
    request: Request,
    response: Response,
    ticker: Optional[str] = Query(None, description="Single ticker; omit for full universe"),
) -> Dict:
    """Trigger the heavy history backfill on demand.

    Synchronous — for the curated S&P 100 this is ~3-5 minutes (~600
    provider calls). For a single ticker (`?ticker=NVDA`) it's ~5
    seconds. Idempotent.

    Use this after a fresh deploy when the database is empty (Postgres
    on first boot has 0 financial_periods rows; the `seed_universe`
    that runs at startup only fills `companies`). Without this call,
    the screener stays empty until the nightly cron at 03:15 UTC.
    """
    from ..monitoring.history_backfill import run_once
    return run_once(ticker=ticker)


@router.get("/api/admin/monitoring/status")
def monitoring_status() -> Dict:
    """Last-run timestamps + notes per registered monitoring loop."""
    return {"loops": status_snapshot()}


@router.get("/api/admin/llm-metrics")
def llm_metrics_endpoint(
    run_id: Optional[str] = None,
    since_days: int = Query(7, ge=1, le=365),
) -> Dict[str, Any]:
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
    ticker: Optional[str] = None,
    surface: Optional[str] = Query(None, description="memo|chat"),
    limit: int = Query(20, ge=1, le=200),
) -> Dict[str, Any]:
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
def get_sdk_trace(run_id: str) -> Dict[str, Any]:
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
    ticker: Optional[str] = None,
    sector: Optional[str] = None,
) -> Dict[str, Any]:
    """Wave 4A: aggregate realized-outcome stats over evaluated memos.

    Filters: `ticker` (single name), `sector`, `horizon_days` (which forward
    window to look at). Returns hit rate + avg alpha + total evaluated.
    """
    return outcome_service.track_record(
        ticker=ticker, sector=sector, horizon_days=horizon_days,
    )


@router.post("/api/admin/evaluate-outcomes")
def evaluate_outcomes_now() -> Dict[str, Any]:
    """Manual trigger for the daily outcome loop. Useful in dev / for
    backfilling the table after deploys; production runs the scheduled
    job via APScheduler."""
    return outcome_service.evaluate_all_due()


# ---------------------------------------------------------------------------
# Wave 10 — calibration + per-agent attribution + regime-conditional accuracy
# ---------------------------------------------------------------------------

@router.get("/api/admin/calibration")
def calibration_endpoint(
    horizon_days: int = Query(90, ge=1, le=365),
) -> Dict[str, Any]:
    """Wave 10 — calibration plot data: per-rating realized excess
    return distribution. A well-calibrated PM has Strong-Buy realizations
    clearly higher than Buy realizations. Powers the upcoming
    track-record dashboard."""
    from ..services.calibration_service import calibration_by_rating
    return calibration_by_rating(horizon_days=horizon_days)


@router.get("/api/admin/per-agent-attribution")
def per_agent_attribution_endpoint(
    horizon_days: int = Query(90, ge=1, le=365),
) -> Dict[str, Any]:
    """Wave 10 — per-specialist attribution stats from `memo_postmortems`.
    Surfaces systematic strengths and weaknesses ('our valuation analyst
    consistently picks the right names; our macro is pulling the wrong
    direction')."""
    from ..services.calibration_service import per_agent_attribution
    return per_agent_attribution(horizon_days=horizon_days)


@router.get("/api/admin/regime-accuracy")
def regime_accuracy_endpoint(
    horizon_days: int = Query(90, ge=1, le=365),
) -> Dict[str, Any]:
    """Wave 10 — accuracy bucketed by macro regime at memo creation.
    Catches regime-specific blind spots ('we're great in soft-landing
    regimes, terrible in recessions')."""
    from ..services.calibration_service import regime_conditional_accuracy
    return regime_conditional_accuracy(horizon_days=horizon_days)


@router.get("/api/admin/calibration-summary")
def calibration_summary_endpoint(
    horizon_days: int = Query(90, ge=1, le=365),
) -> Dict[str, Any]:
    """Wave 10 — one-call aggregator returning calibration + per-agent
    + regime stats. Powers the upcoming track-record dashboard with a
    single fetch."""
    from ..services.calibration_service import summary
    return summary(horizon_days=horizon_days)


@router.post("/api/admin/run-postmortems")
def run_postmortems_endpoint(
    horizon_days: int = Query(90, ge=1, le=365),
    limit: int = Query(25, ge=1, le=200),
) -> Dict[str, Any]:
    """Wave 10 — manual trigger for postmortem_loop; equivalent to
    running `python -m scripts.postmortem_backfill`. Useful for
    seeding the system or recovering after a cron outage."""
    from ..services.postmortem_service import run_postmortems
    return run_postmortems(horizon_days=horizon_days, limit=limit)


@router.post("/api/admin/mispricing-audit")
def mispricing_audit_endpoint(
    limit: int = Query(20, ge=1, le=50),
) -> Dict[str, Any]:
    """Wave 10 — audit the quality of the PM's mispricing theses
    across recent memos. Returns per-memo scores (specificity /
    differentiation / falsifiability) + a corpus-wide failure-
    mode observation. Feeds PM prompt iteration."""
    from ..services.mispricing_audit import aggregate_scores, run_audit
    audit = run_audit(limit=limit)
    audit["aggregate"] = aggregate_scores(audit)
    return audit


# ---------------------------------------------------------------------------
# Wave 8C — DCF version history
# ---------------------------------------------------------------------------

@router.get("/api/admin/dcf-versions/{ticker}")
def dcf_version_history(
    ticker: str, limit: int = Query(25, ge=1, le=200),
) -> Dict[str, Any]:
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
    ticker: Optional[str] = None,
) -> Dict[str, Any]:
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
def reload_news_domains() -> Dict[str, Any]:
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
    path: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    duration_ms: Optional[int] = None
    session_id: Optional[str] = None
    ts: Optional[str] = None  # client wall-clock; not authoritative
    payload: Dict[str, Any] = Field(default_factory=dict)


class UILogBatch(BaseModel):
    events: List[UILogEvent]


@router.post("/api/admin/ui-log")
def post_ui_log(batch: UILogBatch) -> Dict[str, Any]:
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
        return {"written": written, "ok": False}
    return {"written": written, "ok": True}


@router.get("/api/admin/ui-log")
def get_ui_log(
    limit: int = Query(200, ge=1, le=2000),
    since_minutes: int = Query(60, ge=1, le=1440),
    source: Optional[str] = None,
    kind: Optional[str] = None,
    path_contains: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Read recent UI trace events newest-first. Use this to see what a
    user was doing in the UI."""
    from ..database import SessionLocal
    from ..models import UILog
    from sqlalchemy import select
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
def clear_ui_log() -> Dict[str, Any]:
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
) -> Dict[str, Any]:
    """Wave 3A risk-register mitigation: telemetry on whether the
    sector-integrated bull/bear is actually balanced.

    Walks the `n` most recent `MemoSnapshot` rows and reports per-memo
    bull/bear key-point counts + `sector_lean` distribution. A
    persistent skew toward one side across many tickers is the signal
    to revisit the prompt structure or add a devil's-advocate amplifier
    (deferred per locked decision until lopsidedness shows up in practice).
    """
    rows: List[Dict[str, Any]] = []
    bull_kp_total = 0
    bear_kp_total = 0
    lean_counts = {"bull": 0, "bear": 0, "balanced": 0}
    falsifiable_total = 0
    inspected = 0

    history_seen: set[str] = set()
    # Pull latest memo per ticker (skip duplicates) up to n unique tickers.
    from ..database import SessionLocal
    from ..models import MemoSnapshot
    from sqlalchemy import select
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
