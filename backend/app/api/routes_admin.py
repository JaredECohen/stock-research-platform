"""Admin endpoints (re-seed, debug, monitoring status, LLM metrics).

Wave 8C: operational surfaces for the durable state shipped in earlier
waves — DCF version history, update orchestrator queue, news allow-list
governance, bull/bear lopsidedness audit. None of these add new
business logic; they expose what's already in the DB / service layer
so a UI or admin script can reason about platform state without
SQL-level access.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..monitoring import status_snapshot
from ..seed_demo_data import run_full_seed
from ..services import dcf_store, llm_metrics, memo_store, outcome_service, update_orchestrator

router = APIRouter()


@router.post("/api/seed-demo-data")
def seed_demo_data() -> Dict:
    return run_full_seed()


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
