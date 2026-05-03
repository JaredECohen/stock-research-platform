"""Admin endpoints (re-seed, debug, monitoring status, LLM metrics)."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

from ..monitoring import status_snapshot
from ..seed_demo_data import run_full_seed
from ..services import llm_metrics, outcome_service

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
