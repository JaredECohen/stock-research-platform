"""`GET /api/admin/unit-economics` — the observed cost of a metered operation.

One read-only endpoint over `services/unit_economics.py`: what a memo run, a
pm_chat call, a chart commentary and an industry report actually cost in the
window (median and p90, with the sample size behind each), and what each
plan's allowance therefore costs per month against `auth/features.py`.

Read-only and bounded by construction:

  - it runs two statements — a COUNT and a newest-first SELECT capped at
    `unit_economics.MAX_ROWS_SCANNED` rows — both served by the
    `generated_at` index, so the work does not grow with the log;
  - `window_days` is capped at 90 (the `llm_call_logs` GC keeps 90 days, so a
    longer window could only ever return the same rows);
  - nothing is written, and no provider or model is called. The tool that
    spends money to measure a controlled action is
    `scripts/audit_unit_costs.py`, and it is not reachable from here.

Admin-token protected by prefix (`api/admin_auth`) like every other
`/api/admin/*` path; no browser calls it, so it is not on the exempt list.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from ..services import unit_economics

router = APIRouter()


@router.get("/api/admin/unit-economics")
def unit_economics_endpoint(
    window_days: int = Query(unit_economics.DEFAULT_WINDOW_DAYS, ge=1, le=90),
) -> dict[str, Any]:
    """Per-operation observed cost and the per-plan allowance projection.

    Every figure carries its sample size; a figure with too thin a sample is
    `null` with the count and the threshold in `reasons`, never a confident
    number. Rows the scan cap dropped, calls that belong to no unit and units
    whose model is missing from the price table are each counted in the
    response rather than silently omitted.
    """
    return unit_economics.build_report(window_days=window_days)
