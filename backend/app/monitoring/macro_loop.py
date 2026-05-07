"""Hourly macro snapshot + regime detection.

Pulls the FRED snapshot and writes a `MacroBroadcast` to the cache. Sector
agents read this snapshot at the top of every run (Phase 6) so their prompt
context is always macro-aware.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from ..cache import cache_get, cache_put
from ..schemas import MacroBroadcast
from ..services import macro_service
from . import record_run

log = logging.getLogger(__name__)


def _detect_regime(snapshot: Dict[str, float]) -> str:
    """Heuristic regime detection — purposely simple; tune via prompts later."""
    cpi = snapshot.get("CORESTICKM159SFRBATL") or snapshot.get("CPIAUCSL") or 0.0
    fed = snapshot.get("FEDFUNDS") or 0.0
    unrate = snapshot.get("UNRATE") or 0.0
    hy = snapshot.get("BAMLH0A0HYM2") or 0.0
    if cpi > 4.0:
        return "sticky_inflation"
    if fed > 4.5 and unrate > 4.0:
        return "late_cycle_slowdown"
    if hy > 6.0:
        return "credit_stress"
    if fed < 3.0 and unrate < 4.5:
        return "soft_landing"
    return "mixed"


_REGIME_FAVORED = {
    "sticky_inflation": ["Energy", "Financials"],
    "late_cycle_slowdown": ["Utilities", "Consumer Staples", "Healthcare"],
    "credit_stress": ["Utilities", "Consumer Staples"],
    "soft_landing": ["Technology", "Industrials", "Consumer Discretionary"],
    "mixed": [],
}
_REGIME_PRESSURED = {
    "sticky_inflation": ["Consumer Discretionary", "Real Estate"],
    "late_cycle_slowdown": ["Consumer Discretionary", "Industrials"],
    "credit_stress": ["Financials"],
    "soft_landing": ["Utilities"],
    "mixed": [],
}


def run_once() -> Dict:
    snapshot = macro_service.macro_snapshot()
    regime = _detect_regime(snapshot)
    broadcast = MacroBroadcast(
        snapshot=snapshot,
        regime=regime,
        favored_sectors=_REGIME_FAVORED.get(regime, []),
        pressured_sectors=_REGIME_PRESSURED.get(regime, []),
        note=f"Regime: {regime}.",
    )

    # Detect regime change vs prior broadcast — cheap signal for the PM cache.
    prior = cache_get("macro:global", "macro_broadcast")
    prior_regime = (prior.payload.get("regime") if prior and isinstance(prior.payload, dict) else None)
    cache_put(
        "macro:global", "macro_broadcast",
        payload=broadcast.model_dump(mode="json"),
        sources_used=[f"macro:{k}" for k in snapshot.keys()],
        generated_by="macro_loop", cost_tokens=10,
        ttl_seconds=2 * 3600,
    )
    # Wave 10 — fire memo invalidation when the regime classification
    # changes. Bounded at MAX_TICKERS_PER_REGIME_SHIFT (10 by default)
    # so a flip can't run away on cost. Skipped on cold start
    # (prior_regime is None) so first-boot doesn't refresh the
    # universe.
    refreshed: List[str] = []
    if prior_regime and prior_regime != regime:
        try:
            from ..services.update_orchestrator import on_regime_shift
            res = on_regime_shift(prior_regime, regime)
            refreshed = res.get("refreshed") or []
        except Exception as exc:  # pragma: no cover
            log.warning("regime-shift trigger failed: %s", exc)

    note = f"regime={regime}" + (
        f" (shifted from {prior_regime}; refreshed {len(refreshed)})"
        if refreshed else ""
    )
    record_run("macro_loop", note=note)
    return {
        "regime": regime,
        "regime_changed": prior_regime != regime,
        "refreshed_tickers": refreshed,
        "snapshot": snapshot,
    }


def register(scheduler) -> None:
    scheduler.add_job(run_once, "interval", hours=1, id="macro_loop", replace_existing=True)
