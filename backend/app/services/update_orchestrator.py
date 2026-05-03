"""Wave 5B — update orchestrator.

Wires monitoring loops to memo refresh logic:

- New filing/earnings observed by EDGAR poller → enqueue
  `full_reanalysis(ticker)` (the existing `run_stock_memo` path).
- Material/breaking news from the news loop → call `news_impact_agent`
  on the latest memo. If it returns `material=true`, build an
  `incremental_patch` snapshot inheriting from the prior version with
  the rating / confidence / risks patched. Critic is skipped on patches
  (locked in MASTER_PLAN); `revision_log` carries `critic_skipped: true`.

Per-ticker FIFO queue prevents two events on the same ticker racing.
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

import logging
from collections import defaultdict, deque
from datetime import date as _date, datetime
from typing import Any, Deque, Dict, List, Optional

from ..schemas import NewsAlert, StockMemoOut

log = logging.getLogger(__name__)


# Locked policy: max patches per ticker per UTC day.
MAX_PATCHES_PER_DAY = 2

# Per-ticker FIFO queue (singleton). Process state — for production
# multi-process deployments we'd back this with Redis; for now the
# in-process queue is enough for the demo + tests.
_QUEUES: Dict[str, Deque[Dict[str, Any]]] = defaultdict(deque)


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

def on_filing_event(ticker: str) -> Dict[str, Any]:
    """A new filing was observed → enqueue a `full_reanalysis(ticker)`.

    Synchronous re-run is fine at our scale (one ticker per call from
    the EDGAR poller); a future async worker can pop these off the
    FIFO queue.
    """
    ticker = ticker.upper()
    _QUEUES[ticker].append({
        "kind": "full_reanalysis", "ticker": ticker,
        "enqueued_at": datetime.utcnow().isoformat(),
    })
    try:
        from ..agents.graph import run_stock_memo
        memo = run_stock_memo(ticker, force_refresh=True)
        return {
            "ticker": ticker, "kind": "full_reanalysis",
            "rating_label": memo.rating_label,
        }
    finally:
        if _QUEUES[ticker]:
            _QUEUES[ticker].popleft()


def on_news_alert(ticker: str, alert: NewsAlert) -> Dict[str, Any]:
    """A material/breaking news alert came in → run news_impact_agent
    against the latest memo, persist a patch if material.

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

    from ..agents.news_impact_agent import apply_patch, assess
    assessment = assess(prior_memo, alert)
    if not assessment.get("material"):
        return {"patched": False, "ticker": ticker, "reason": "not_material"}

    patched_memo: StockMemoOut = apply_patch(prior_memo, assessment["patch"])
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
    return {
        "patched": True,
        "ticker": ticker,
        "version": new_snap.version,
        "delta_summary": assessment.get("delta_summary", ""),
    }


def queue_depth(ticker: Optional[str] = None) -> Dict[str, int]:
    """Inspect the in-process FIFO queue (for /api/admin)."""
    if ticker:
        return {ticker.upper(): len(_QUEUES.get(ticker.upper(), []))}
    return {t: len(q) for t, q in _QUEUES.items() if q}
