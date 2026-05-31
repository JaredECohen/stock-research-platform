"""Valuation service: bridges fundamentals + DCF/comps engines."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..cache import cache_get, cache_put, resolved_cost_tokens
from ..finance import comps as comps_engine
from ..finance import dcf as dcf_engine
from ..schemas import CompsResult, CompsRow, DCFAssumptions, DCFResult
from .fundamentals_service import get_full_financials


_PEERS_CACHE: Optional[Dict[str, List[str]]] = None
_EXPOSURE_PEERS_TTL_SECONDS = 30 * 24 * 3600  # Wave 10 — refresh monthly


def _peer_groups() -> Dict[str, List[str]]:
    global _PEERS_CACHE
    if _PEERS_CACHE is None:
        path = Path(__file__).resolve().parent.parent / "data" / "peer_groups.json"
        with open(path) as f:
            _PEERS_CACHE = json.load(f)
    return _PEERS_CACHE


def get_peers(ticker: str) -> List[str]:
    """Direct competitors — Track A in the two-track design.

    Two-tier resolution:
      1. Curated override in `peer_groups.json` (hand-picked
         sub-industry / direct-competitor sets — biotech same-therapy,
         semis same-sub-segment, etc.).
      2. Auto-derived by sub-industry (or industry fallback) when no
         curated entry exists. Pulls the top market-cap names in the
         same GICS bucket from the `companies` table; excludes the
         target itself. Up to 5 peers.

    Returns [] only when the target has no usable classification AND
    no curated entry, in which case the comps agent renders the
    "no peer data" stub.
    """
    t = ticker.upper()
    curated = _peer_groups().get(t)
    if curated:
        return curated
    return _derive_peers_by_classification(t, top_n=5)


def _derive_peers_by_classification(
    ticker: str, *, top_n: int = 5,
) -> List[str]:
    """Auto-derive peers from the companies table.

    Order of preference:
      1. Same `sub_industry` (GICS Level 4) — tightest match.
      2. Same `industry` (GICS Level 3) — fallback when sub-industry
         is sparse (often the case outside the curated mega-caps).

    Picks the top-N by market cap; excludes the target and any rows
    missing market cap. Result is empty when the target itself isn't
    in `companies` or has no classification.
    """
    try:
        from sqlalchemy import select, desc
        from ..database import SessionLocal
        from ..models import Company
        with SessionLocal() as db:
            target = db.execute(
                select(Company).where(Company.ticker == ticker)
            ).scalar_one_or_none()
            if target is None:
                return []
            for field, value in (
                ("sub_industry", target.sub_industry),
                ("industry", target.industry),
            ):
                if not value:
                    continue
                stmt = (
                    select(Company.ticker)
                    .where(
                        getattr(Company, field) == value,
                        Company.ticker != ticker,
                        Company.market_cap.is_not(None),
                    )
                    .order_by(desc(Company.market_cap))
                    .limit(top_n)
                )
                peers = [t for (t,) in db.execute(stmt).all()]
                if peers:
                    return peers
    except Exception:  # pragma: no cover — fail-safe to no-peers
        pass
    return []


def get_exposure_peers(ticker: str, *, force_refresh: bool = False) -> List[str]:
    """Cross-sector exposure peers — Track B in the two-track design.

    Wave 10. The user's specific request: AMZN / GOOGL / MSFT all
    share AI capex exposure even though they live in different GICS
    sectors. This function asks an LLM "name 3-5 companies that share
    material exposure to {ticker}'s key drivers, regardless of sector"
    and caches the answer with a 30-day TTL on the company-warm cache
    surface.

    LLM-optional: when no API key is configured, returns the
    intersection of `top_for_theme` for the ticker's strongest themes.
    """
    cache_key = "company_warm:exposure_peers"
    if not force_refresh:
        cached = cache_get(ticker.upper(), cache_key, max_age_seconds=_EXPOSURE_PEERS_TTL_SECONDS)
        if cached and isinstance(cached.payload, dict):
            peers = cached.payload.get("peers") or []
            if isinstance(peers, list):
                return [str(p).upper() for p in peers]

    # Try the LLM path first.
    peers: List[str] = _llm_exposure_peers(ticker.upper()) or []
    if not peers:
        peers = _theme_exposure_peers(ticker.upper())

    cache_put(
        ticker.upper(), cache_key,
        payload={"peers": peers},
        sources_used=[f"target:{ticker.upper()}"],
        generated_by="valuation_service.get_exposure_peers",
        cost_tokens=resolved_cost_tokens(15),
        parent_snapshots=[],
        ttl_seconds=_EXPOSURE_PEERS_TTL_SECONDS,
    )
    return peers


def _llm_exposure_peers(ticker: str) -> List[str]:
    """Ask the LLM to nominate 3-5 cross-sector exposure peers."""
    try:
        from ..config import settings as _settings
        if not getattr(_settings, "openai_api_key", None):
            return []
        target = get_full_financials(ticker)
        profile = target.get("profile") or {}
        if not profile:
            return []
        from ..agents import llm
        from ..database import SessionLocal
        from ..models import Company
        with SessionLocal() as db:
            universe = [t for (t,) in db.query(Company.ticker).all()]
        prompt = (
            f"Target: {ticker} ({profile.get('company_name', '')}, "
            f"{profile.get('sector', '')} / {profile.get('industry', '')}).\n"
            f"Business: {(profile.get('business_description') or '')[:600]}\n\n"
            "Name 3-5 companies that share material EXPOSURE with the "
            "target — same key drivers (e.g. AI capex, China consumer, "
            "long-rate sensitivity, GLP-1 disruption, energy "
            "transition) — REGARDLESS of GICS sector. These should be "
            "DIFFERENT from typical sub-industry peers and chosen "
            "because they will be moved by the same forces. Pick "
            "tickers from this universe only:\n"
            f"{', '.join(universe)}\n\n"
            "Return JSON: { \"peers\": [\"TICKER\", ...], \"rationale\": "
            "\"<one sentence explaining the shared exposure thesis>\" }."
        )
        out = llm.chat_json(
            prompt,
            system="You are a buy-side PM. Pick non-obvious peers that share underlying drivers.",
            route="cheap",
        )
        peers = (out or {}).get("peers") or []
        return [
            str(p).upper() for p in peers
            if isinstance(p, str) and p.upper() != ticker
        ][:5]
    except Exception:  # pragma: no cover
        return []


def _theme_exposure_peers(ticker: str) -> List[str]:
    """Fallback: pull peers via the theme_exposure table.

    Take this ticker's top-3 themes; collect the top scorers in each;
    return the unique set excluding self.
    """
    try:
        from sqlalchemy import select
        from ..database import SessionLocal
        from ..models import ThemeExposure
        from .theme_exposure_service import top_for_theme
        with SessionLocal() as db:
            rows = db.execute(
                select(ThemeExposure)
                .where(ThemeExposure.ticker == ticker.upper())
                .order_by(ThemeExposure.score.desc())
                .limit(3)
            ).scalars().all()
        peers: List[str] = []
        for r in rows:
            if r.score < 25.0:
                continue
            for top in top_for_theme(r.theme, min_score=25.0, limit=8):
                if top["ticker"] != ticker.upper() and top["ticker"] not in peers:
                    peers.append(top["ticker"])
                if len(peers) >= 5:
                    return peers
        return peers
    except Exception:  # pragma: no cover
        return []


def build_comps(target_ticker: str, *, force_refresh: bool = False) -> Optional[CompsResult]:
    """Cache-backed comps. Snapshots stored as kind='company_warm:comps'.

    TTL 7d. Invalidation cascades from each peer's company_cold snapshot via
    `parent_snapshot_ids`, so a new peer 10-K naturally restales.
    """
    from ..cache import cache_get, cache_put, resolved_cost_tokens

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

    # Wave 3E: self-historical context. Best-effort backfill of the
    # history tables (no-op if data is unchanged), then build the
    # self-historical distribution. None when the target lacks enough
    # usable history — degrades gracefully.
    try:
        from .history_service import backfill_ticker
        backfill_ticker(target_ticker)
    except Exception:  # pragma: no cover — diagnostic only
        pass
    try:
        from ..finance.comps_history import build_history_stats
        result.history = build_history_stats(target_ticker, target_row)
    except Exception:  # pragma: no cover — defensive
        result.history = None

    # Wave 10 — Track B exposure peers. Cross-sector names that share
    # the target's key exposures. Resolved from a 30-day-cached LLM
    # call (or theme_exposure fallback); we then build CompsRow for
    # each one so the same UI / chat surfaces can render them.
    try:
        exposure_tickers = get_exposure_peers(target_ticker)
        for peer in exposure_tickers:
            if peer in {p.ticker for p in result.peers}:
                continue
            p = get_full_financials(peer)
            if not p.get("income"):
                continue
            p_inc = sorted(p["income"], key=lambda r: r.get("period", ""))[-1]
            p_bs = sorted(p["balance"], key=lambda r: r.get("period", ""))[-1]
            p_cf = sorted(p["cash"], key=lambda r: r.get("period", ""))[-1]
            p_prior = sorted(p["income"], key=lambda r: r.get("period", ""))[-2] if len(p["income"]) >= 2 else None
            result.exposure_peers.append(comps_engine.build_row(
                peer, p["profile"].get("company_name", peer),
                p["profile"].get("market_cap"),
                p_inc, p_bs, p_cf, p_prior,
            ))
        if result.exposure_peers:
            result.exposure_rationale = (
                "Cross-sector names that share the target's key drivers. "
                "Selected at runtime — sector-mechanical comps still "
                "live in the `peers` list above."
            )
    except Exception:  # pragma: no cover
        pass

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
        cost_tokens=resolved_cost_tokens(80),
        parent_snapshots=parent_ids,
        ttl_seconds=7 * 24 * 3600,
    )
    return result


# Wave 10 — sectors flagged as cyclical for default-margin reversion.
# A cyclical company at peak earnings has a trailing-3yr margin that
# overstates the steady-state margin a DCF should bake in. For these
# names the DCF defaults to the mean-reversion glide rather than the
# hold-flat default. Software / staples / utilities stay on hold-flat
# because their margin profiles are more stable.
_CYCLICAL_SECTORS: set[str] = {
    "Energy", "Materials", "Industrials",
    "Consumer Discretionary",  # autos + housing-driven; high-beta cyclicals
    "Real Estate",  # cyclical re: interest rates / occupancy
    "Financials",  # NIM cyclicality
}


def _is_cyclical(profile: dict) -> bool:
    sector = (profile.get("sector") or "").strip()
    return sector in _CYCLICAL_SECTORS


def _cohort_op_margin(ticker: str) -> Optional[float]:
    """Pull the cohort median operating margin used as the mean-
    reversion target for cyclical names.

    Reads the cached `company_warm:comps` snapshot (already populated
    on memo runs) so we don't pay an extra peer-financials walk just
    to compute the target. Returns None when no cached comps exist
    or the cohort lacks operating margins; caller falls back to
    dcf_engine's 0.18 generic anchor.
    """
    try:
        cached = cache_get(ticker, "company_warm:comps", max_age_seconds=14 * 24 * 3600)
        if not cached or not isinstance(cached.payload, dict):
            return None
        median = cached.payload.get("median") or {}
        v = median.get("operating_margin")
        if isinstance(v, (int, float)) and 0.0 < v < 1.0:
            return float(v)
    except Exception:  # pragma: no cover — never block default DCF
        pass
    return None


def default_dcf_assumptions(ticker: str) -> Optional[DCFAssumptions]:
    """Default DCF assumptions for `ticker`.

    Wave 8I: pulls analyst consensus revenue estimates (when the
    provider chain has them) and feeds them to the DCF engine as the
    starting point for the 5-year growth path. The agent layer
    (`agents/dcf_updater.py`) is the only thing allowed to diverge
    from consensus, and only with a per-field rationale + ±20% cap.

    Wave 10 — for cyclical sectors (Energy, Materials, Industrials,
    Cons-Disc, Real Estate, Financials) auto-enable the margin
    mean-reversion glide using the cohort median operating margin as
    the target. Without this, the trailing-3yr-flat default
    systematically over-values cyclicals at peak earnings.
    """
    fin = get_full_financials(ticker)
    if not fin.get("income"):
        return None
    profile = fin["profile"]
    estimates = None
    try:
        from .data_service import get_data_service
        estimates = get_data_service().get_estimates(ticker)
    except Exception:  # pragma: no cover — estimates are optional
        estimates = None
    # Live intraday quote (60s TTL) feeds the DCF's `current_price`
    # so `upside_pct` reflects today's tape, not the 7-day-cached
    # `profile.last_price`. Falls back to profile, then to
    # market_cap/shares (Wave 8L) if neither chain has a quote.
    from .market_data_service import get_current_price
    live_price = get_current_price(ticker)
    last_price = live_price or profile.get("last_price") or 0.0
    diluted_shares = profile.get("shares_outstanding") or 0.0
    if not last_price and diluted_shares and profile.get("market_cap"):
        try:
            last_price = float(profile["market_cap"]) / float(diluted_shares)
        except (TypeError, ValueError, ZeroDivisionError):
            last_price = 0.0

    # Wave 10 — cyclical sector → margin mean reversion default ON.
    # Also flip on for ANY ticker classified as at-peak by cycle
    # fingerprinting (peak earnings → margin reverts), regardless of
    # sector. A software name at unprecedented operating leverage
    # benefits from the same cohort-median anchor.
    use_reversion = _is_cyclical(profile)
    cycle_pos: Optional[str] = None
    try:
        from .cycle_position import cycle_position
        pos = cycle_position(ticker)
        cycle_pos = pos.get("position")
        if cycle_pos == "peak":
            use_reversion = True
    except Exception:  # pragma: no cover — defensive
        pass
    cohort_target = _cohort_op_margin(ticker) if use_reversion else None

    baseline = dcf_engine.derive_default_assumptions(
        income_statements=fin["income"],
        cash_flows=fin["cash"],
        balance_sheets=fin["balance"],
        current_price=last_price,
        diluted_shares=diluted_shares,
        beta=profile.get("beta") or 1.0,
        analyst_estimates=estimates,
        margin_mean_reversion=use_reversion,
        cohort_op_margin=cohort_target,
    )

    # Wave 10i — layer LLM-judged sector / company overrides on top of
    # the deterministic baseline. Adjusts exit_ebitda_multiple,
    # terminal_growth, capex / D&A / NWC, and WACC adjustment to the
    # company's sector + business model. Falls back to the baseline
    # when no API key or LLM error.
    try:
        from .sector_dcf_defaults import apply_sector_overrides
        return apply_sector_overrides(profile, baseline, cycle_position=cycle_pos)
    except Exception:  # pragma: no cover — never block DCF
        return baseline


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
    # Wave 10k — pass profile through so bull/bear scenarios run with
    # LLM-driven sector-aware drivers + assumption changes (instead
    # of the prior symmetric ±400bp mechanical bumps).
    profile_for_dcf: Optional[Dict] = None
    try:
        fin = get_full_financials(ticker)
        profile_for_dcf = fin.get("profile")
    except Exception:  # pragma: no cover — defensive
        profile_for_dcf = None
    result = dcf_engine.build_full_dcf(ticker, assumptions, profile=profile_for_dcf)

    if assumptions is not None and result is not None:
        # Only cache the default-assumption build to avoid per-call thrash.
        # Detect "default" by hashing the assumptions and comparing.
        try:
            default = default_dcf_assumptions(ticker)
            same = default and default.model_dump() == assumptions.model_dump()
        except Exception:
            same = False
        if same:
            if not force_refresh:
                parent_ids: List[int] = []
                cold = cache_get(ticker, "company_cold")
                if cold:
                    parent_ids.append(cold.id)
                cache_put(
                    ticker, "company_warm:dcf",
                    payload=result.model_dump(mode="json"),
                    sources_used=[f"target:{ticker}", f"assumptions:default"],
                    generated_by="valuation_service.build_dcf",
                    cost_tokens=resolved_cost_tokens(60),
                    parent_snapshots=parent_ids,
                    ttl_seconds=7 * 24 * 3600,
                )
            # Wave 5A: persist as a versioned DCFModel so the assumption
            # updater + reviewers can chain off prior versions. First save
            # for a ticker is `initial`; subsequent default-rebuilds tag
            # `memo_rebuild` (vs. the LLM-driven `earnings_update` path).
            # Runs even on force_refresh — that's the user signaling they
            # want a fresh authoritative version recorded.
            try:
                from . import dcf_store
                prior = dcf_store.latest_version(ticker)
                trigger = "initial" if prior is None else "memo_rebuild"
                parent_version = prior.version if prior is not None else None
                dcf_store.save_version(
                    ticker, assumptions=assumptions, dcf_result=result,
                    trigger=trigger, parent_version=parent_version,
                )
            except Exception as exc:  # pragma: no cover — diagnostic only
                # DCF build must not be blocked by a persistence hiccup.
                pass
    return result
