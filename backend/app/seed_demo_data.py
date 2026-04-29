"""Seeder: write demo dataset to JSON + database tables.

Idempotent — safe to run repeatedly. Used at app startup so the SQLite/Postgres
instance has the company universe and pre-computed screener scores ready
even on a cold start.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from .data.demo_dataset import build_dataset, export_to_disk
from .database import init_db, session_scope
from .models import Company, ScreenerScore
from .services.screener_service import compute_universe_scores

log = logging.getLogger(__name__)


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


def seed_screener_scores() -> int:
    result = compute_universe_scores(theme=None)
    with session_scope() as db:
        # Wipe prior un-themed scores for cleanliness
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
    paths = export_to_disk()
    n_companies = seed_companies(refresh=True)
    n_scores = seed_screener_scores()
    return dict(
        companies=n_companies,
        screener_rows=n_scores,
        json_files=[str(p) for p in paths.values()],
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    summary = run_full_seed()
    log.info("Seeded: %s", summary)
    print(summary)
