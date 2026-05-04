"""Test-fixture seeder (Wave 9b — moved out of production).

Populates the SQLite DB with the deterministic demo dataset for unit
tests. Production seeding lives in `app.seed_universe`. Tests that
need pre-populated companies / screener scores / tier markings call
`run_full_seed()` from this module after the `demo_provider` autouse
fixture has registered the in-memory provider with `data_service`.

Idempotent. The legacy `universe_tier1.json` was inlined as
`_LEGACY_TIER1` so this fixture has no on-disk dependencies.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

from ...database import init_db, session_scope
from ...models import Company, ScreenerScore
from ...services.screener_service import compute_universe_scores
from .demo_dataset import build_dataset

log = logging.getLogger(__name__)


# Inlined from the retired `data/universe_tier1.json`. The 32-ticker
# legacy universe used by the demo dataset; matches `COMPANY_PROFILES`
# in `demo_dataset.py`. Tests that assert tier-1 membership read from
# this constant.
_LEGACY_TIER1: Set[str] = {
    "NVDA", "AAPL", "MSFT", "PLTR", "AVGO", "AMD", "CRM",
    "GOOGL", "META", "NFLX",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX",
    "COST", "WMT",
    "JPM", "V", "MA", "BAC", "GS", "MS",
    "LLY", "JNJ", "MRK", "UNH",
    "XOM",
    "CAT",
    "NEE",
    "LIN",
    "AMT",
}


def seed_companies(refresh: bool = False) -> int:
    dataset = build_dataset()
    n_written = 0
    with session_scope() as db:
        for ticker, payload in dataset.items():
            if ticker.startswith("_"):
                continue
            profile = payload["profile"]
            existing: Optional[Company] = db.get(Company, ticker)
            if existing and not refresh:
                continue
            row = existing or Company(ticker=ticker)
            row.company_name = profile["company_name"]
            row.exchange = profile["exchange"]
            row.sector = profile["sector"]
            row.industry = profile["industry"]
            row.sub_industry = profile.get("sub_industry")
            row.country = profile.get("country", "US")
            row.currency = profile.get("currency", "USD")
            row.market_cap = profile.get("market_cap")
            row.cik = profile.get("cik")
            row.business_description = profile.get("business_description", "")
            row.fiscal_year_end = profile.get("fiscal_year_end")
            row.is_active = profile.get("is_active", True)
            row.is_etf = profile.get("is_etf", False)
            row.beta = profile.get("beta")
            row.shares_outstanding = profile.get("shares_outstanding")
            row.last_price = profile.get("last_price")
            row.last_price_at = datetime.utcnow()
            db.add(row)
            n_written += 1
    return n_written


def seed_universe_tiers() -> Dict[str, int]:
    """Mark legacy tier-1 names as `auto_analysis`.

    Idempotent. `analyzed_on_demand` is preserved across calls (runtime
    state set by the on-demand-analysis flow, not by this fixture).
    """
    counts: Dict[str, int] = {"auto_analysis": 0, "data_only": 0, "analyzed_on_demand": 0}
    auto_tickers: List[str] = []
    with session_scope() as db:
        for row in db.query(Company).all():
            current = row.universe_tier or "data_only"
            if row.ticker in _LEGACY_TIER1:
                row.universe_tier = "auto_analysis"
                auto_tickers.append(row.ticker)
            elif current == "analyzed_on_demand":
                pass
            else:
                row.universe_tier = "data_only"
            counts[row.universe_tier] = counts.get(row.universe_tier, 0) + 1
    try:
        from ...config import settings
        if settings.enable_long_term_memory:
            from ...memory import CompanyMemory
            for t in auto_tickers:
                cm = CompanyMemory.for_ticker(t)
                if not cm.path.exists():
                    cm.save()
    except Exception:  # pragma: no cover
        log.debug("memory scaffold skipped during fixture seed")
    return counts


def seed_screener_scores() -> int:
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


def run_full_seed() -> dict:
    init_db()
    n_companies = seed_companies(refresh=True)
    tier_counts = seed_universe_tiers()
    n_scores = seed_screener_scores()
    return dict(
        companies=n_companies,
        universe_tiers=tier_counts,
        screener_rows=n_scores,
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    summary = run_full_seed()
    log.info("Test-fixture seeded: %s", summary)
    print(summary)
