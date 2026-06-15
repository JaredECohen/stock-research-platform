"""Seeder: populate the curated screener universe from `data/sp500.json`.

Reads the static S&P 500 ticker list (refreshable via
`app.scripts.refresh_universe_lists`), hits the live data provider
chain (FMP first) for each company's profile, upserts a row into the
`companies` table, and tags it as `auto_analysis` so the nightly
history backfill picks it up.

Auto-update memo gating — separately from the screener universe, a
small `_top_10_by_market_cap` list inside sp500.json names tickers
eligible for *automatic memo regeneration* on new filings/transcripts.
Everything else only regenerates memos on user request (or when a
prior memo was viewed within the recency window). Keeps incremental
LLM spend predictable when expanding the universe.

Idempotent. Cheap on warm starts: existing rows are skipped unless
`refresh=True`. Cold start hits ~500 FMP `/profile` calls (well within
Premium-tier rate limits; for Starter-tier, run the seeder in batches).

Heavy work — financial statement backfill, screener score recompute —
is intentionally NOT done here. Those run via:
    - `monitoring/history_backfill.py` (nightly cron, when monitoring on)
    - `services/screener_service.compute_universe_scores` (admin endpoint
      / scheduler)
so app boot stays fast even with 500+ tickers.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from .database import init_db, session_scope
from .models import Company, ScreenerScore
from .services.data_service import get_data_service

log = logging.getLogger(__name__)


def _load_universe() -> tuple[List[str], List[str]]:
    """Load (universe_tickers, auto_update_tickers) from sp500.json.

    Falls back to sp100.json when sp500.json is missing so a stale
    deployment keeps working. Returns ([], []) when both are missing.
    """
    data_dir = Path(__file__).resolve().parent / "data"
    sp500_path = data_dir / "sp500.json"
    sp100_path = data_dir / "sp100.json"
    path = sp500_path if sp500_path.exists() else sp100_path
    if not path.exists():
        log.warning("No universe file at %s — seeder is a no-op", data_dir)
        return [], []
    cfg = json.loads(path.read_text())
    universe = [t.upper() for t in (cfg.get("tickers") or [])]
    # auto-update list — explicit top-N override. Empty list means "no
    # tickers pinned to auto-update", and gating falls back to the
    # recency-window check alone.
    auto_update = [
        t.upper() for t in (cfg.get("_top_10_by_market_cap_2026_05") or [])
    ]
    return universe, auto_update


def _profile_to_company_kwargs(profile: Dict, ticker: str) -> Dict:
    """Map the data_service profile shape onto Company columns."""
    return dict(
        company_name=profile.get("company_name") or ticker,
        exchange=profile.get("exchange") or "NASDAQ",
        sector=profile.get("sector") or "",
        industry=profile.get("industry") or "",
        sub_industry=profile.get("sub_industry"),
        country=profile.get("country") or "US",
        currency=profile.get("currency") or "USD",
        market_cap=profile.get("market_cap"),
        cik=profile.get("cik"),
        business_description=profile.get("business_description") or "",
        fiscal_year_end=profile.get("fiscal_year_end"),
        is_active=profile.get("is_active", True),
        is_etf=profile.get("is_etf", False),
        beta=profile.get("beta"),
        shares_outstanding=profile.get("shares_outstanding"),
        last_price=profile.get("last_price"),
        last_price_at=datetime.utcnow(),
    )


def seed_universe(refresh: bool = False) -> Dict[str, int]:
    """Upsert S&P 100 companies from the live provider chain.

    Behavior:
      - Existing rows are skipped unless `refresh=True` (warm start = no
        provider calls).
      - Missing rows trigger a `data_service.get_company_profile` call;
        on success the row is inserted with `universe_tier=auto_analysis`.
      - Tickers the provider rejects are skipped with a warning so one
        bad symbol doesn't block the rest.
      - Existing rows still in the S&P 100 list are re-tagged
        `auto_analysis` (cheap correction if a previous tier got stuck).

    Returns counts: `{inserted, refreshed, skipped, missing_profile,
    auto_analysis, total_in_db}`.
    """
    tickers, auto_update_tickers = _load_universe()
    if not tickers:
        return {"inserted": 0, "refreshed": 0, "skipped": 0,
                "missing_profile": 0, "auto_analysis": 0, "total_in_db": 0}

    ds = get_data_service()
    universe_set: Set[str] = set(tickers)
    auto_update_set: Set[str] = set(auto_update_tickers)
    inserted = refreshed = skipped = missing = 0

    with session_scope() as db:
        for ticker in tickers:
            existing: Optional[Company] = db.get(Company, ticker)
            wants_auto_update = ticker in auto_update_set
            if existing and not refresh:
                # Warm start — leave row alone, just ensure tier is right
                # and the auto-update flag matches the current top-N list.
                if existing.universe_tier != "auto_analysis":
                    existing.universe_tier = "auto_analysis"
                if existing.auto_update_memo != wants_auto_update:
                    existing.auto_update_memo = wants_auto_update
                skipped += 1
                continue
            profile = ds.get_company_profile(ticker)
            if not profile:
                log.warning("seed_universe: no profile for %s", ticker)
                missing += 1
                continue
            kwargs = _profile_to_company_kwargs(profile, ticker)
            if existing is None:
                db.add(Company(
                    ticker=ticker, universe_tier="auto_analysis",
                    auto_update_memo=wants_auto_update, **kwargs,
                ))
                inserted += 1
            else:
                for k, v in kwargs.items():
                    setattr(existing, k, v)
                existing.universe_tier = "auto_analysis"
                existing.auto_update_memo = wants_auto_update
                refreshed += 1

        # Demote companies no longer in the universe file (kept in DB so
        # saved memos / DCFs remain readable, but they drop off the
        # screener). Preserve `analyzed_on_demand` rows the user
        # explicitly researched; only demote anything stuck on
        # `auto_analysis` from a prior universe definition.
        for row in db.query(Company).all():
            if row.ticker in universe_set:
                continue
            if row.universe_tier == "auto_analysis":
                row.universe_tier = "data_only"
            # And never auto-regenerate memos for demoted names.
            if row.auto_update_memo:
                row.auto_update_memo = False

        # Flush so the in-progress inserts are visible to the count
        # below; otherwise (autoflush=False) the count under-reports.
        db.flush()
        auto_count = db.query(Company).filter(
            Company.universe_tier == "auto_analysis"
        ).count()
        total = db.query(Company).count()

    # Scaffold empty memory files for the universe so the long-term
    # memory directory is in sync from day one. Defensive — a disk
    # hiccup must never block startup seeding.
    try:
        from .config import settings
        if settings.enable_long_term_memory:
            from .memory import CompanyMemory
            for t in tickers:
                cm = CompanyMemory.for_ticker(t)
                if not cm.path.exists():
                    cm.save()
    except Exception:  # pragma: no cover — diagnostic only
        log.debug("memory scaffold skipped during universe seed")

    return {
        "inserted": inserted,
        "refreshed": refreshed,
        "skipped": skipped,
        "missing_profile": missing,
        "auto_analysis": auto_count,
        "total_in_db": total,
    }


def ensure_company_in_universe(ticker: str) -> Optional[Dict]:
    """Lazy-introduce a ticker that isn't in the curated universe.

    Used by the research route when the user submits an arbitrary
    ticker (e.g., "PYPL" after the screener universe has settled on
    the S&P 100). Behavior:

    - Returns immediately when the row already exists.
    - Otherwise hits the live profile chain. Returns None if the
      provider chain rejects the symbol (caller should 404 the
      request).
    - Inserts a `companies` row tagged `analyzed_on_demand` so the
      memo flow has the metadata it needs and the screener stays
      curated.

    The caller is responsible for kicking off the heavy backfill
    (`history_service.backfill_ticker`); this function is fast.
    """
    ticker = ticker.upper()
    with session_scope() as db:
        existing: Optional[Company] = db.get(Company, ticker)
        if existing is not None:
            return _profile_to_company_kwargs(
                {
                    "company_name": existing.company_name,
                    "exchange": existing.exchange,
                    "sector": existing.sector,
                    "industry": existing.industry,
                    "sub_industry": existing.sub_industry,
                    "country": existing.country,
                    "currency": existing.currency,
                    "market_cap": existing.market_cap,
                    "cik": existing.cik,
                    "business_description": existing.business_description,
                    "fiscal_year_end": existing.fiscal_year_end,
                    "is_active": existing.is_active,
                    "is_etf": existing.is_etf,
                    "beta": existing.beta,
                    "shares_outstanding": existing.shares_outstanding,
                    "last_price": existing.last_price,
                },
                ticker,
            )
    ds = get_data_service()
    profile = ds.get_company_profile(ticker)
    if not profile:
        log.info("ensure_company_in_universe: provider rejected %s", ticker)
        return None
    kwargs = _profile_to_company_kwargs(profile, ticker)
    with session_scope() as db:
        # Re-check inside the transaction in case a concurrent request
        # introduced the row.
        if db.get(Company, ticker) is None:
            db.add(Company(
                ticker=ticker, universe_tier="analyzed_on_demand", **kwargs,
            ))
    return kwargs


def recompute_screener_scores() -> int:
    """Recompute screener scores against the current `auto_analysis` set.

    Reads from whatever financial data is currently in the cache /
    history tables. Cold-start with no backfill yet → returns 0 rows.
    Run after `monitoring/history_backfill.run_once()` finishes.
    """
    from .services.screener_service import compute_universe_scores
    result = compute_universe_scores(theme=None)
    with session_scope() as db:
        db.query(ScreenerScore).filter(ScreenerScore.theme.is_(None)).delete()
        for row in result.rows:
            db.add(ScreenerScore(
                ticker=row.ticker,
                quality=row.quality,
                growth=row.growth,
                valuation=row.valuation,
                earnings_momentum=row.earnings_momentum,
                catalyst=row.macro_fit,
                macro_fit=row.macro_fit,
                risk=row.risk,
                pm_conviction=row.pm_score,
                one_line_thesis=row.one_line_thesis,
                main_catalyst=row.main_catalyst,
                main_risk=row.main_risk,
                theme=row.theme,
            ))
    return len(result.rows)


def run_full_seed(refresh: bool = False) -> dict:
    """Boot-time entry point. Lightweight by default.

    Order:
      1. `init_db` — create any missing tables.
      2. `seed_universe` — upsert S&P 100 profiles from FMP.
      3. `recompute_screener_scores` — refresh AI-composite scores
         from whatever's currently in the history tables (will be
         empty on first boot until history_backfill has run).
      4. `snapshot_screener_metrics` — refresh the raw-metric snapshot
         used by the rule-based custom screener.

    Heavy work (financial backfill across all 100 names) is NOT done
    here — call `monitoring.history_backfill.run_once()` from the admin
    endpoint or the nightly cron.
    """
    init_db()
    universe = seed_universe(refresh=refresh)
    n_scores = recompute_screener_scores()
    from .services.screener_metrics_service import snapshot_universe as snap_metrics
    metrics = snap_metrics()
    return dict(universe=universe, screener_rows=n_scores, screener_metrics=metrics)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    summary = run_full_seed(refresh=True)
    log.info("Seeded universe: %s", summary)
    print(summary)
