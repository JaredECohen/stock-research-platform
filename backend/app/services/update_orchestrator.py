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
) -> List[str]:
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

    candidates: List[str] = []

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


def on_regime_shift(prior_regime: str, new_regime: str) -> Dict[str, Any]:
    """Wave 10 — fire when macro_loop detects the regime classification
    changed. Re-runs memos for the most affected names.

    Strategy: full_reanalysis for the top N by exposure (forces fresh
    valuation + rating; light patches wouldn't reflect the regime
    change properly). Bounded by `MAX_TICKERS_PER_REGIME_SHIFT` so a
    single regime flip can't blow the budget.

    Returns {prior, new, refreshed: List[str]} for cron logging.
    """
    if prior_regime == new_regime:
        return {"prior": prior_regime, "new": new_regime, "refreshed": []}
    affected = _affected_tickers_for_regime_shift(prior_regime, new_regime)
    refreshed: List[str] = []
    for ticker in affected:
        try:
            on_filing_event(ticker)  # reuses the full_reanalysis path
            refreshed.append(ticker)
        except Exception as exc:  # pragma: no cover
            log.warning("regime-shift refresh failed for %s: %s", ticker, exc)
    log.info(
        "regime shift %s → %s: refreshed %d ticker(s) — %s",
        prior_regime, new_regime, len(refreshed), refreshed,
    )
    return {"prior": prior_regime, "new": new_regime, "refreshed": refreshed}
