"""Seeder: write demo dataset to JSON + database tables.

Idempotent — safe to run repeatedly. Used at app startup so the SQLite/Postgres
instance has the company universe and pre-computed screener scores ready
even on a cold start.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

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


def _load_tier1_universe() -> Set[str]:
    """Read the curated tier-1 watch list (sectors → tickers) into a flat set."""
    path = Path(__file__).resolve().parent / "data" / "universe_tier1.json"
    if not path.exists():
        return set()
    cfg = json.loads(path.read_text())
    out: Set[str] = set()
    for tickers in (cfg.get("sectors") or {}).values():
        for t in tickers or []:
            out.add(t.upper())
    return out


def seed_universe_tiers() -> Dict[str, int]:
    """Mark tier-1 names as `auto_analysis`; everyone else stays `data_only`.

    Idempotent. Safe to re-run after edits to `universe_tier1.json` — every
    company's tier is reset on each pass except for `analyzed_on_demand`,
    which we preserve as runtime state set by the on-demand-analysis flow.
    """
    tier1 = _load_tier1_universe()
    counts: Dict[str, int] = {"auto_analysis": 0, "data_only": 0, "analyzed_on_demand": 0}
    with session_scope() as db:
        for row in db.query(Company).all():
            current = row.universe_tier or "data_only"
            if row.ticker in tier1:
                row.universe_tier = "auto_analysis"
            elif current == "analyzed_on_demand":
                # Keep on-demand promotion intact across seeds — that's
                # runtime state set by the UI, not config.
                pass
            else:
                row.universe_tier = "data_only"
            counts[row.universe_tier] = counts.get(row.universe_tier, 0) + 1
    return counts


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
    tier_counts = seed_universe_tiers()
    n_scores = seed_screener_scores()
    return dict(
        companies=n_companies,
        universe_tiers=tier_counts,
        screener_rows=n_scores,
        json_files=[str(p) for p in paths.values()],
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    summary = run_full_seed()
    log.info("Seeded: %s", summary)
    print(summary)
