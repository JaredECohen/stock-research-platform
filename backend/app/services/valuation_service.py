"""Valuation service: bridges fundamentals + DCF/comps engines."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..cache import cache_get, cache_put
from ..finance import comps as comps_engine
from ..finance import dcf as dcf_engine
from ..schemas import CompsResult, CompsRow, DCFAssumptions, DCFResult
from .fundamentals_service import get_full_financials


_PEERS_CACHE: Optional[Dict[str, List[str]]] = None


def _peer_groups() -> Dict[str, List[str]]:
    global _PEERS_CACHE
    if _PEERS_CACHE is None:
        path = Path(__file__).resolve().parent.parent / "data" / "peer_groups.json"
        with open(path) as f:
            _PEERS_CACHE = json.load(f)
    return _PEERS_CACHE


def get_peers(ticker: str) -> List[str]:
    return _peer_groups().get(ticker.upper(), [])


def build_comps(target_ticker: str, *, force_refresh: bool = False) -> Optional[CompsResult]:
    """Cache-backed comps. Snapshots stored as kind='company_warm:comps'.

    TTL 7d. Invalidation cascades from each peer's company_cold snapshot via
    `parent_snapshot_ids`, so a new peer 10-K naturally restales.
    """
    from ..cache import cache_get, cache_put

    if not force_refresh:
        cached = cache_get(target_ticker, "company_warm:comps", max_age_seconds=7 * 24 * 3600)
        if cached and isinstance(cached.payload, dict):
            try:
                payload = dict(cached.payload)
                payload.pop("schema_version", None)
                return CompsResult.model_validate(payload)
            except Exception:
                pass

    target = get_full_financials(target_ticker)
    if not target.get("income"):
        return None
    target_inc = sorted(target["income"], key=lambda r: r.get("period", ""))[-1]
    target_bs = sorted(target["balance"], key=lambda r: r.get("period", ""))[-1]
    target_cf = sorted(target["cash"], key=lambda r: r.get("period", ""))[-1]
    prior_inc = sorted(target["income"], key=lambda r: r.get("period", ""))[-2] if len(target["income"]) >= 2 else None
    target_row = comps_engine.build_row(
        target_ticker, target["profile"].get("company_name", target_ticker),
        target["profile"].get("market_cap"),
        target_inc, target_bs, target_cf, prior_inc,
    )

    peer_rows: List[CompsRow] = []
    for peer in get_peers(target_ticker):
        p = get_full_financials(peer)
        if not p.get("income"):
            continue
        p_inc = sorted(p["income"], key=lambda r: r.get("period", ""))[-1]
        p_bs = sorted(p["balance"], key=lambda r: r.get("period", ""))[-1]
        p_cf = sorted(p["cash"], key=lambda r: r.get("period", ""))[-1]
        p_prior = sorted(p["income"], key=lambda r: r.get("period", ""))[-2] if len(p["income"]) >= 2 else None
        peer_rows.append(comps_engine.build_row(
            peer, p["profile"].get("company_name", peer),
            p["profile"].get("market_cap"),
            p_inc, p_bs, p_cf, p_prior,
        ))
    if not peer_rows:
        return None

    result = comps_engine.compute_comps(target_row, peer_rows)

    # Snapshot for re-use; lineage = each peer's company_cold so a peer-side
    # 10-K refresh stales us.
    parent_ids: List[int] = []
    cold_target = cache_get(target_ticker, "company_cold")
    if cold_target:
        parent_ids.append(cold_target.id)
    for peer in get_peers(target_ticker):
        cold = cache_get(peer, "company_cold")
        if cold:
            parent_ids.append(cold.id)
    cache_put(
        target_ticker, "company_warm:comps",
        payload=result.model_dump(mode="json"),
        sources_used=[f"peer:{p.ticker}" for p in result.peers] + [f"target:{target_ticker}"],
        generated_by="valuation_service.build_comps",
        cost_tokens=80,
        parent_snapshots=parent_ids,
        ttl_seconds=7 * 24 * 3600,
    )
    return result


def default_dcf_assumptions(ticker: str) -> Optional[DCFAssumptions]:
    fin = get_full_financials(ticker)
    if not fin.get("income"):
        return None
    profile = fin["profile"]
    return dcf_engine.derive_default_assumptions(
        income_statements=fin["income"],
        cash_flows=fin["cash"],
        balance_sheets=fin["balance"],
        current_price=profile.get("last_price") or 0.0,
        diluted_shares=profile.get("shares_outstanding") or 0.0,
        beta=profile.get("beta") or 1.0,
    )


def build_dcf(
    ticker: str,
    assumptions: Optional[DCFAssumptions] = None,
    *,
    force_refresh: bool = False,
) -> Optional[DCFResult]:
    """Cache-backed DCF (kind='company_warm:dcf').

    Note: only the *default-assumption* DCF (assumptions=None) is cached, since
    user-supplied scenarios from the UI are essentially per-request and would
    pollute the cache. TTL 7d; lineage parent = ticker's company_cold snapshot.
    """
    use_cache = (assumptions is None) and (not force_refresh)
    if use_cache:
        cached = cache_get(ticker, "company_warm:dcf", max_age_seconds=7 * 24 * 3600)
        if cached and isinstance(cached.payload, dict):
            try:
                payload = dict(cached.payload)
                payload.pop("schema_version", None)
                return DCFResult.model_validate(payload)
            except Exception:
                pass

    if assumptions is None:
        assumptions = default_dcf_assumptions(ticker)
    if assumptions is None:
        return None
    result = dcf_engine.build_full_dcf(ticker, assumptions)

    if assumptions is not None and not force_refresh and result is not None:
        # Only cache the default-assumption build to avoid per-call thrash.
        # Detect "default" by hashing the assumptions and comparing.
        try:
            default = default_dcf_assumptions(ticker)
            same = default and default.model_dump() == assumptions.model_dump()
        except Exception:
            same = False
        if same:
            parent_ids: List[int] = []
            cold = cache_get(ticker, "company_cold")
            if cold:
                parent_ids.append(cold.id)
            cache_put(
                ticker, "company_warm:dcf",
                payload=result.model_dump(mode="json"),
                sources_used=[f"target:{ticker}", f"assumptions:default"],
                generated_by="valuation_service.build_dcf",
                cost_tokens=60,
                parent_snapshots=parent_ids,
                ttl_seconds=7 * 24 * 3600,
            )
    return result
