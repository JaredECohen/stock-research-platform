"""Seeder: populate the curated screener universe from `data/sp100.json`.

Replaces the legacy demo seeder. Reads the static S&P 100 ticker list,
hits the live data provider chain (FMP first) for each company's
profile, upserts a row into the `companies` table, and tags it as
`auto_analysis` so the nightly history backfill picks it up.

Idempotent. Cheap on warm starts: existing rows are skipped unless
`refresh=True`. Cold start hits ~100 FMP `/profile` calls (well within
Starter-tier rate limits).

Heavy work — financial statement backfill, screener score recompute —
is intentionally NOT done here. Those run via:
    - `monitoring/history_backfill.py` (nightly cron, when monitoring on)
    - `services/screener_service.compute_universe_scores` (admin endpoint
      / scheduler)
so app boot stays fast even with 100+ tickers.
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


def _load_sp100() -> List[str]:
    path = Path(__file__).resolve().parent / "data" / "sp100.json"
    if not path.exists():
        log.warning("sp100.json missing at %s — seeder is a no-op", path)
        return []
    cfg = json.loads(path.read_text())
    return [t.upper() for t in (cfg.get("tickers") or [])]


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
    tickers = _load_sp100()
    if not tickers:
        return {"inserted": 0, "refreshed": 0, "skipped": 0,
                "missing_profile": 0, "auto_analysis": 0, "total_in_db": 0}

    ds = get_data_service()
    sp100_set: Set[str] = set(tickers)
    inserted = refreshed = skipped = missing = 0

    with session_scope() as db:
        for ticker in tickers:
            existing: Optional[Company] = db.get(Company, ticker)
            if existing and not refresh:
                # Warm start — leave row alone, just ensure tier is right.
                if existing.universe_tier != "auto_analysis":
                    existing.universe_tier = "auto_analysis"
                skipped += 1
                continue
            profile = ds.get_company_profile(ticker)
            if not profile:
                log.warning("seed_universe: no profile for %s", ticker)
                missing += 1
                continue
            kwargs = _profile_to_company_kwargs(profile, ticker)
            if existing is None:
                db.add(Company(ticker=ticker, universe_tier="auto_analysis", **kwargs))
                inserted += 1
            else:
                for k, v in kwargs.items():
                    setattr(existing, k, v)
                existing.universe_tier = "auto_analysis"
                refreshed += 1

        # Demote any companies no longer in the S&P 100 list (kept in DB
        # so any saved memos / DCFs remain readable, but they drop off
        # the screener universe).
        for row in db.query(Company).all():
            if row.ticker in sp100_set:
                continue
            # Preserve `analyzed_on_demand` rows the user explicitly
            # researched; only demote anything stuck on `auto_analysis`
            # from a prior universe definition.
            if row.universe_tier == "auto_analysis":
                row.universe_tier = "data_only"

        # Final count for the response.
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
      3. `recompute_screener_scores` — refresh scores from whatever's
         currently in the history tables (will be empty on first boot
         until history_backfill has run).

    Heavy work (financial backfill across all 100 names) is NOT done
    here — call `monitoring.history_backfill.run_once()` from the admin
    endpoint or the nightly cron.
    """
    init_db()
    universe = seed_universe(refresh=refresh)
    n_scores = recompute_screener_scores()
    return dict(universe=universe, screener_rows=n_scores)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    summary = run_full_seed(refresh=True)
    log.info("Seeded universe: %s", summary)
    print(summary)
