"""Admin endpoints (re-seed, debug, monitoring status, LLM metrics)."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

from ..monitoring import status_snapshot
from ..seed_demo_data import run_full_seed
from ..services import llm_metrics

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
