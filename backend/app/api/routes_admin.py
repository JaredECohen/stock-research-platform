"""Admin endpoints (re-seed, debug, monitoring status)."""
from __future__ import annotations

from typing import Dict

from fastapi import APIRouter

from ..monitoring import status_snapshot
from ..seed_demo_data import run_full_seed

router = APIRouter()


@router.post("/api/seed-demo-data")
def seed_demo_data() -> Dict:
    return run_full_seed()


@router.get("/api/admin/monitoring/status")
def monitoring_status() -> Dict:
    """Last-run timestamps + notes per registered monitoring loop."""
    return {"loops": status_snapshot()}
